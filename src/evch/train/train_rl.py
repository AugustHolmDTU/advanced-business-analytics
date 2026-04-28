from __future__ import annotations

import contextlib
import copy
import io
import logging
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.envs.factory import make_env
from evch.rl.evaluation import evaluate_policy
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.sim.line_corridor import LineCorridorQueueSimulator
from evch.train.eval_suites import build_seed_list, evaluate_policy_suites, resolve_train_seed_range
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.torch_runtime import configure_torch_runtime, resolve_torch_device
from evch.utils.wandb import DummyRun, init_wandb, log_artifact

LOGGER = logging.getLogger(__name__)

SIM_WANDB_COLUMNS = {
    "step",
    "global_hour",
    "arrivals_total_station_ab",
    "arrivals_total_station_bc",
    "queue_length",
    "queue_length_station_ab",
    "queue_length_station_bc",
    "queue_wait_mean_minutes",
    "queue_wait_mean_minutes_station_ab",
    "queue_wait_mean_minutes_station_bc",
    "utilization",
    "active_plugs",
    "effective_num_plugs",
    "num_active_mobile_stations",
    "num_active_mobile_stations_station_ab",
    "num_active_mobile_stations_station_bc",
    "committed_mobile_stations_station_ab",
    "committed_mobile_stations_station_bc",
    "committed_mobile_stations_bias_ab_minus_bc",
    "num_mobile_stations_middle_available",
    "num_mobile_stations_middle_charging",
    "num_mobile_stations_in_transit_to_ab",
    "num_mobile_stations_in_transit_to_bc",
    "num_mobile_stations_in_transit_to_middle",
    "unused_mobile_chargers",
    "unused_mobile_stations_estimate",
    "unused_mobile_stations_estimate_station_ab",
    "unused_mobile_stations_estimate_station_bc",
    "expected_station_arrivals_ab",
    "expected_station_arrivals_bc",
    "expected_demand_share_ab",
    "expected_demand_share_bc",
    "allocation_share_ab",
    "allocation_share_bc",
    "allocation_vs_demand_alignment",
    "allocation_bias_ab_minus_bc",
    "expected_demand_bias_ab_minus_bc",
    "local_deficit_without_mcs_station_ab",
    "local_deficit_without_mcs_station_bc",
    "local_deficit_bias_ab_minus_bc",
    "mobile_service_capacity_station_ab",
    "mobile_service_capacity_station_bc",
    "spatial_deficit_coverage",
    "alignment_on_station_ab_disruption",
    "alignment_on_station_bc_disruption",
    "expected_passing_total",
    "expected_passing_od_ab",
    "expected_passing_od_ba",
    "expected_passing_od_bc",
    "expected_passing_od_cb",
    "expected_passing_od_ac",
    "expected_passing_od_ca",
    "disruption_active",
    "disruption_type_code",
    "disruption_target",
    "disruption_target_code",
    "reward_local_queue_peak_penalty_term",
    "reward_local_wait_peak_penalty_term",
    "reward",
}
SIM_WANDB_COLUMNS.update(LineCorridorQueueSimulator.target_indicator_columns())


def _deep_merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _maybe_plot_training_curve(history: list[dict[str, float]], path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        def _metric_series(metric_name: str) -> tuple[list[float], list[float]]:
            x_values: list[float] = []
            y_values: list[float] = []
            for entry in history:
                raw_value = entry.get(metric_name)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if not np.isfinite(value):
                    continue
                x_values.append(float(entry["episode"]))
                y_values.append(value)
            return x_values, y_values

        episodes, rewards = _metric_series("reward")
        _, losses = _metric_series("loss")
        eval_episodes, eval_rewards = _metric_series("eval_mean_reward")
        eval_loss_episodes, eval_losses = _metric_series("eval_td_loss")

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(episodes, rewards, color="#1f77b4", linewidth=1.6, label="Train reward")
        if eval_rewards:
            axes[0].plot(
                eval_episodes,
                eval_rewards,
                color="#d62728",
                marker="o",
                linewidth=1.8,
                label="Eval reward",
            )
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Train vs evaluation reward")
        axes[0].legend()

        axes[1].plot(episodes, losses, color="#2ca02c", linewidth=1.6, label="Train TD loss")
        if eval_losses:
            axes[1].plot(
                eval_loss_episodes,
                eval_losses,
                color="#9467bd",
                marker="o",
                linewidth=1.8,
                label="Eval TD loss",
            )
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Train vs evaluation TD loss")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping RL training plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def _maybe_plot_station_demand_vs_mcs(metrics: pd.DataFrame, path: Path) -> bool:
    required_columns = {
        "global_hour",
        "arrivals_total_station_ab",
        "arrivals_total_station_bc",
        "expected_station_arrivals_ab",
        "expected_station_arrivals_bc",
        "num_active_mobile_stations_station_ab",
        "num_active_mobile_stations_station_bc",
    }
    if not required_columns.issubset(metrics.columns):
        return False

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        from evch.train.run_simple_corridor_sim import _disruption_windows, _shade_disruptions

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        _shade_disruptions(axes, _disruption_windows(metrics))

        station_specs = (
            ("AB", "arrivals_total_station_ab", "expected_station_arrivals_ab", "num_active_mobile_stations_station_ab"),
            ("BC", "arrivals_total_station_bc", "expected_station_arrivals_bc", "num_active_mobile_stations_station_bc"),
        )
        demand_colors = {"AB": ("#1f77b4", "#0b3c5d"), "BC": ("#2ca02c", "#145a32")}
        mcs_colors = {"AB": "#d62728", "BC": "#ff7f0e"}

        for axis, (label, arrivals_col, expected_col, mcs_col) in zip(axes, station_specs):
            raw_color, expected_color = demand_colors[label]
            axis.plot(
                metrics["global_hour"],
                metrics[arrivals_col],
                color=raw_color,
                linewidth=1.1,
                alpha=0.35,
                label=f"{label} realized demand",
            )
            axis.plot(
                metrics["global_hour"],
                metrics[expected_col],
                color=expected_color,
                linewidth=2.0,
                label=f"{label} expected demand",
            )
            axis.set_ylabel("Demand")

            twin_axis = axis.twinx()
            twin_axis.step(
                metrics["global_hour"],
                metrics[mcs_col],
                where="post",
                color=mcs_colors[label],
                linewidth=2.0,
                label=f"{label} MCS",
            )
            twin_axis.set_ylabel("Active MCS")

            lines, labels = axis.get_legend_handles_labels()
            twin_lines, twin_labels = twin_axis.get_legend_handles_labels()
            axis.legend(lines + twin_lines, labels + twin_labels, loc="upper right")
            axis.set_title(f"Station {label}: charging demand versus mobile-station allocation")

        axes[-1].set_xlabel("Global hour")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping station demand plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def _maybe_plot_station_allocation_over_time(metrics: pd.DataFrame, path: Path) -> bool:
    required_columns = {
        "global_hour",
        "num_active_mobile_stations_station_ab",
        "num_active_mobile_stations_station_bc",
    }
    if not required_columns.issubset(metrics.columns):
        return False

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        from evch.train.run_simple_corridor_sim import _disruption_windows, _shade_disruptions

        fig, axis = plt.subplots(1, 1, figsize=(12, 4.8))
        _shade_disruptions([axis], _disruption_windows(metrics))

        axis.step(
            metrics["global_hour"],
            metrics["num_active_mobile_stations_station_ab"],
            where="post",
            color="#d62728",
            linewidth=2.0,
            label="Allocated MCS at AB",
        )
        axis.step(
            metrics["global_hour"],
            metrics["num_active_mobile_stations_station_bc"],
            where="post",
            color="#ff7f0e",
            linewidth=2.0,
            label="Allocated MCS at BC",
        )
        axis.set_title("Mobile charging-station allocation by station over time")
        axis.set_xlabel("Global hour")
        axis.set_ylabel("Active MCS")
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping station allocation plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def _log_station_overlay_panels(run: Any, metrics: pd.DataFrame) -> None:
    if isinstance(run, DummyRun):
        return

    required_columns = {
        "global_hour",
        "expected_station_arrivals_ab",
        "expected_station_arrivals_bc",
        "num_active_mobile_stations_station_ab",
        "num_active_mobile_stations_station_bc",
        "allocation_bias_ab_minus_bc",
        "expected_demand_bias_ab_minus_bc",
    }
    if not required_columns.issubset(metrics.columns):
        return

    try:
        import wandb  # type: ignore

        run.log(
            {
                "sim_overlay/station_ab_demand_vs_mcs": wandb.plot.line_series(
                    xs=metrics["global_hour"].tolist(),
                    ys=[
                        metrics["expected_station_arrivals_ab"].tolist(),
                        metrics["num_active_mobile_stations_station_ab"].tolist(),
                    ],
                    keys=["AB expected charging demand", "AB allocated MCS"],
                    title="Station AB: demand vs allocated MCS",
                    xname="Global hour",
                ),
                "sim_overlay/station_bc_demand_vs_mcs": wandb.plot.line_series(
                    xs=metrics["global_hour"].tolist(),
                    ys=[
                        metrics["expected_station_arrivals_bc"].tolist(),
                        metrics["num_active_mobile_stations_station_bc"].tolist(),
                    ],
                    keys=["BC expected charging demand", "BC allocated MCS"],
                    title="Station BC: demand vs allocated MCS",
                    xname="Global hour",
                ),
                "sim_overlay/allocation_bias_vs_expected_bias": wandb.plot.line_series(
                    xs=metrics["global_hour"].tolist(),
                    ys=[
                        metrics["expected_demand_bias_ab_minus_bc"].tolist(),
                        metrics["allocation_bias_ab_minus_bc"].tolist(),
                    ],
                    keys=["Expected demand bias (AB - BC)", "Allocated MCS bias (AB - BC)"],
                    title="Allocation bias vs expected demand bias",
                    xname="Global hour",
                ),
                "sim_overlay/mcs_allocation_by_station": wandb.plot.line_series(
                    xs=metrics["global_hour"].tolist(),
                    ys=[
                        metrics["num_active_mobile_stations_station_ab"].tolist(),
                        metrics["num_active_mobile_stations_station_bc"].tolist(),
                    ],
                    keys=["Allocated MCS at AB", "Allocated MCS at BC"],
                    title="Allocated MCS by station over time",
                    xname="Global hour",
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Skipping WandB station overlay panels: %s", exc)


class WandbSb3Callback:
    def __init__(self, run: Any, log_interval: int, step_log_interval: int) -> None:
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

        class _Callback(BaseCallback):
            def __init__(self, parent: "WandbSb3Callback") -> None:
                super().__init__(verbose=0)
                self.parent = parent

            def _on_step(self) -> bool:
                return self.parent.on_step(self)

        self.callback = _Callback(self)
        self.run = run
        self.log_interval = max(log_interval, 1)
        self.step_log_interval = max(step_log_interval, 1)
        self.episode_index = 0
        self.latest_train_loss = 0.0
        self._reset_episode_accumulators()

    def _reset_episode_accumulators(self) -> None:
        self.episode_reward = 0.0
        self.served_total = 0.0
        self.unmet_total = 0.0
        self.invalid_actions = 0.0
        self.action_trace: list[float] = []
        self.active_chargers_trace: list[float] = []
        self.mobile_stations_trace: list[float] = []
        self.utilization_trace: list[float] = []
        self.idle_capacity_trace: list[float] = []
        self.effective_capacity_trace: list[float] = []
        self.unused_mobile_chargers_trace: list[float] = []
        self.unused_mobile_stations_trace: list[float] = []
        self.activated_trace: list[float] = []
        self.adjusted_trace: list[float] = []
        self.queue_length_trace: list[float] = []
        self.queue_wait_trace: list[float] = []
        self.queue_wait_station_ab_trace: list[float] = []
        self.queue_wait_station_bc_trace: list[float] = []
        self.queue_wait_excess_trace: list[float] = []
        self.queue_wait_breach_trace: list[float] = []
        self.disruption_trace: list[float] = []

    @staticmethod
    def _scalar_at(value: Any, index: int = 0) -> float:
        array = np.asarray(value)
        if array.ndim == 0:
            return float(array)
        return float(array.reshape(-1)[index])

    def _update_episode_accumulators(self, reward: float, action: float, info: dict[str, Any]) -> None:
        self.episode_reward += reward
        self.served_total += float(info.get("served_demand", 0.0))
        self.unmet_total += float(info.get("unmet_demand", 0.0))
        self.invalid_actions += float(not info.get("action_valid", True))
        self.action_trace.append(action)
        self.active_chargers_trace.append(float(info.get("num_active_chargers", 0.0)))
        self.mobile_stations_trace.append(float(info.get("num_active_mobile_stations", 0.0)))
        self.utilization_trace.append(float(info.get("utilization", 0.0)))
        self.idle_capacity_trace.append(float(info.get("idle_capacity", 0.0)))
        self.effective_capacity_trace.append(float(info.get("effective_capacity_total", 0.0)))
        self.unused_mobile_chargers_trace.append(float(info.get("unused_mobile_chargers", 0.0)))
        self.unused_mobile_stations_trace.append(float(info.get("unused_mobile_stations_estimate", 0.0)))
        self.activated_trace.append(float(info.get("activated_mobile_stations", 0.0)))
        self.adjusted_trace.append(float(info.get("adjusted_mobile_stations", 0.0)))
        self.queue_length_trace.append(float(info.get("queue_length", 0.0)))
        self.queue_wait_trace.append(float(info.get("queue_wait_mean_minutes", 0.0)))
        self.queue_wait_station_ab_trace.append(float(info.get("queue_wait_mean_minutes_station_ab", 0.0)))
        self.queue_wait_station_bc_trace.append(float(info.get("queue_wait_mean_minutes_station_bc", 0.0)))
        self.queue_wait_excess_trace.append(float(info.get("queue_wait_excess_minutes", 0.0)))
        self.queue_wait_breach_trace.append(float(info.get("queue_wait_target_breached", 0.0)))
        self.disruption_trace.append(float(info.get("disruption_active", 0.0)))

    def _log_step_metrics(self, timesteps: int, action: float, reward: float, info: dict[str, Any]) -> None:
        self.run.log(
            {
                "rl_step/action": float(action),
                "rl_step/reward": float(reward),
                "rl_step/served_demand": float(info.get("served_demand", 0.0)),
                "rl_step/unmet_demand": float(info.get("unmet_demand", 0.0)),
                "rl_step/active_mobile_stations": float(info.get("num_active_mobile_stations", 0.0)),
                "rl_step/active_mobile_stations_station_ab": float(info.get("num_active_mobile_stations_station_ab", 0.0)),
                "rl_step/active_mobile_stations_station_bc": float(info.get("num_active_mobile_stations_station_bc", 0.0)),
                "rl_step/active_chargers": float(info.get("num_active_chargers", 0.0)),
                "rl_step/queue_length": float(info.get("queue_length", 0.0)),
                "rl_step/queue_wait_mean_minutes": float(info.get("queue_wait_mean_minutes", 0.0)),
                "rl_step/queue_length_station_ab": float(info.get("queue_length_station_ab", 0.0)),
                "rl_step/queue_length_station_bc": float(info.get("queue_length_station_bc", 0.0)),
                "rl_step/queue_wait_mean_minutes_station_ab": float(info.get("queue_wait_mean_minutes_station_ab", 0.0)),
                "rl_step/queue_wait_mean_minutes_station_bc": float(info.get("queue_wait_mean_minutes_station_bc", 0.0)),
                "rl_step/unused_mobile_chargers": float(info.get("unused_mobile_chargers", 0.0)),
                "rl_step/unused_mobile_stations_estimate": float(info.get("unused_mobile_stations_estimate", 0.0)),
                "rl_step/utilization": float(info.get("utilization", 0.0)),
                "rl_step/utilization_station_ab": float(info.get("utilization_station_ab", 0.0)),
                "rl_step/utilization_station_bc": float(info.get("utilization_station_bc", 0.0)),
                "rl_step/disruption_active": float(info.get("disruption_active", 0.0)),
                "rl_step/disruption_type_code": float(info.get("disruption_type_code", 0.0)),
                "rl_step/activated_mobile_stations": float(info.get("activated_mobile_stations", 0.0)),
                "rl_step/adjusted_mobile_stations": float(info.get("adjusted_mobile_stations", 0.0)),
            },
            step=int(timesteps),
        )

    def _log_episode_metrics(self, timesteps: int, epsilon: float | None) -> None:
        if not self.action_trace:
            return

        record = {
            "episode": float(self.episode_index),
            "reward": float(self.episode_reward),
            "served_demand": float(self.served_total),
            "unmet_demand": float(self.unmet_total),
            "invalid_actions": float(self.invalid_actions),
            "epsilon": float(epsilon if epsilon is not None else 0.0),
            "loss": float(self.latest_train_loss),
            "mean_action": float(np.mean(self.action_trace)),
            "max_action": float(np.max(self.action_trace)),
            "mean_active_chargers": float(np.mean(self.active_chargers_trace)),
            "max_active_chargers": float(np.max(self.active_chargers_trace)),
            "mean_active_mobile_stations": float(np.mean(self.mobile_stations_trace)),
            "max_active_mobile_stations": float(np.max(self.mobile_stations_trace)),
            "total_activated_mobile_stations": float(np.sum(self.activated_trace)),
            "total_adjusted_mobile_stations": float(np.sum(self.adjusted_trace)),
            "mean_utilization": float(np.mean(self.utilization_trace)),
            "mean_idle_capacity": float(np.mean(self.idle_capacity_trace)),
            "mean_effective_capacity": float(np.mean(self.effective_capacity_trace)),
            "mean_unused_mobile_chargers": float(np.mean(self.unused_mobile_chargers_trace)),
            "mean_unused_mobile_stations_estimate": float(np.mean(self.unused_mobile_stations_trace)),
            "mean_queue_length": float(np.mean(self.queue_length_trace)),
            "max_queue_length": float(np.max(self.queue_length_trace)),
            "mean_queue_wait_minutes": float(np.mean(self.queue_wait_trace)),
            "mean_queue_wait_minutes_station_ab": float(np.mean(self.queue_wait_station_ab_trace)),
            "mean_queue_wait_minutes_station_bc": float(np.mean(self.queue_wait_station_bc_trace)),
            "mean_queue_wait_excess_minutes": float(np.mean(self.queue_wait_excess_trace)),
            "queue_wait_target_breach_fraction": float(np.mean(self.queue_wait_breach_trace)),
            "disruption_step_fraction": float(np.mean(self.disruption_trace)),
        }
        self.run.log({f"rl/{key}": value for key, value in record.items()}, step=int(timesteps))
        self.episode_index += 1
        self._reset_episode_accumulators()

    def on_step(self, callback: Any) -> bool:
        metrics: dict[str, float] = {"rl/timesteps": float(callback.num_timesteps)}
        exploration_rate = getattr(callback.model, "exploration_rate", None)
        epsilon = float(exploration_rate) if exploration_rate is not None else None
        if exploration_rate is not None:
            metrics["rl/exploration_rate"] = float(exploration_rate)
            metrics["rl/epsilon"] = float(exploration_rate)

        replay_buffer = getattr(callback.model, "replay_buffer", None)
        if replay_buffer is not None:
            with contextlib.suppress(TypeError):
                metrics["rl/replay_buffer_size"] = float(len(replay_buffer))

        for key, value in getattr(callback.model.logger, "name_to_value", {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                metrics[f"sb3/{key}"] = float(value)
                if key == "train/loss":
                    self.latest_train_loss = float(value)
                elif key == "rollout/exploration_rate":
                    metrics["rl/epsilon"] = float(value)
                    epsilon = float(value)

        infos = callback.locals.get("infos") or []
        rewards = callback.locals.get("rewards")
        dones = callback.locals.get("dones")
        actions = callback.locals.get("actions")
        if infos:
            info = infos[0]
            reward = self._scalar_at(rewards) if rewards is not None else 0.0
            action = self._scalar_at(actions) if actions is not None else 0.0
            self._update_episode_accumulators(reward=reward, action=action, info=info)
            if callback.num_timesteps % self.step_log_interval == 0:
                self._log_step_metrics(
                    timesteps=int(callback.num_timesteps),
                    action=action,
                    reward=reward,
                    info=info,
                )
            if dones is not None and bool(np.asarray(dones).reshape(-1)[0]):
                self._log_episode_metrics(timesteps=int(callback.num_timesteps), epsilon=epsilon)

        if callback.num_timesteps % self.log_interval == 0:
            self.run.log(metrics, step=int(callback.num_timesteps))
        return True


def _train_with_sb3(env: Any, rl_cfg: dict[str, Any], seed: int, output_dir: Path, run: Any) -> tuple[str, str]:
    from stable_baselines3 import DQN  # type: ignore

    sb3_cfg = rl_cfg["sb3"]
    device = resolve_torch_device(str(rl_cfg.get("device", "auto")))
    hidden_dims = [int(dim) for dim in rl_cfg.get("hidden_dims", [128, 128])]
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(rl_cfg["learning_rate"]),
        buffer_size=int(sb3_cfg["buffer_size"]),
        learning_starts=int(sb3_cfg["learning_starts"]),
        batch_size=int(sb3_cfg["batch_size"]),
        train_freq=int(sb3_cfg.get("train_freq", 4)),
        gradient_steps=int(sb3_cfg.get("gradient_steps", 1)),
        gamma=float(rl_cfg["gamma"]),
        tau=float(sb3_cfg["tau"]),
        target_update_interval=int(sb3_cfg["target_update_interval"]),
        exploration_fraction=float(sb3_cfg.get("exploration_fraction", 0.4)),
        exploration_initial_eps=float(sb3_cfg.get("exploration_initial_eps", rl_cfg.get("epsilon_start", 1.0))),
        exploration_final_eps=float(sb3_cfg.get("exploration_final_eps", rl_cfg.get("epsilon_end", 0.05))),
        policy_kwargs={"net_arch": hidden_dims},
        verbose=0,
        seed=seed,
        device=str(device),
    )
    callback = WandbSb3Callback(
        run=run,
        log_interval=int(rl_cfg.get("wandb_log_interval", 100)),
        step_log_interval=int(rl_cfg.get("wandb_step_log_interval", 1)),
    ).callback
    model.learn(total_timesteps=int(sb3_cfg["total_timesteps"]), callback=callback)
    checkpoint = output_dir / "best_model.zip"
    model.save(checkpoint)
    return "sb3_dqn", str(checkpoint)


def _train_with_torch_dqn(
    env: Any,
    rl_cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
    run: Any,
    eval_env_factory: Callable[[int], Any] | None = None,
    eval_episode_seeds: list[int] | None = None,
) -> tuple[str, str, list[dict[str, float]]]:
    configured_max_steps = int(rl_cfg.get("max_steps_per_episode", 0))
    env_max_steps = int(getattr(env, "max_steps", 0))
    if getattr(env, "duration_days_range", None) is not None and env_max_steps > 0:
        train_max_steps = max(configured_max_steps, env_max_steps)
    else:
        train_max_steps = configured_max_steps
    agent = SimpleDQNAgent(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        config=rl_cfg,
        seed=seed,
    )
    eval_interval_steps = int(rl_cfg.get("eval_interval_steps", 0))
    if "eval_interval_episodes" in rl_cfg:
        eval_interval_episodes = int(rl_cfg["eval_interval_episodes"])
    else:
        eval_interval_episodes = 0 if eval_interval_steps > 0 else max(1, int(rl_cfg["episodes"]) // 8)
    history = agent.train(
        env=env,
        episodes=int(rl_cfg["episodes"]),
        max_steps=train_max_steps,
        run=run,
        eval_env_factory=eval_env_factory,
        eval_interval=eval_interval_episodes,
        eval_interval_steps=eval_interval_steps,
        eval_episodes=int(rl_cfg.get("eval_during_training_episodes", max(1, int(rl_cfg.get("evaluation_episodes", 1))))),
        eval_seed=seed + 10_000,
        eval_episode_seeds=eval_episode_seeds,
    )
    checkpoint = output_dir / str(rl_cfg["checkpoint_name"])
    agent.save(checkpoint)
    return "torch_dql", str(checkpoint), history


def _make_rl_policy(backend: str, checkpoint_path: str) -> Callable[[np.ndarray, Any, bool], int]:
    if backend == "sb3_dqn":
        from stable_baselines3 import DQN  # type: ignore

        model = DQN.load(checkpoint_path)

        def policy(observation: np.ndarray, _env: Any, deterministic: bool = True) -> int:
            action, _ = model.predict(observation, deterministic=deterministic)
            return int(action)

        return policy

    agent = SimpleDQNAgent.load(checkpoint_path)

    def policy(observation: np.ndarray, _env: Any, deterministic: bool = True) -> int:
        return agent.act(observation, deterministic=deterministic, env=_env)

    return policy


def _build_mobile_comparison_rollout(
    config: dict[str, Any],
    policy: Callable[[np.ndarray, Any, bool], int],
    output_dir: Path,
    run: Any,
) -> dict[str, Any] | None:
    rollout_cfg = dict(config.get("comparison_rollout", {}))
    if not bool(rollout_cfg.get("enabled", True)):
        return None
    env_type = str(config.get("environment", {}).get("env_type", "")).lower()
    if env_type not in {
        "mobile_station_capacity",
        "corridor_mobile_mcs",
        "corridor_mobile_station",
        "line_corridor_mobile_mcs",
        "line_corridor_mobile_station",
        "corridor_mobile_mcs_abc",
    }:
        return None

    env_cfg = copy.deepcopy(config["environment"])
    seed_source = str(rollout_cfg.get("seed_source", "")).strip().lower()
    seed_index = int(rollout_cfg.get("seed_index", 0))
    if seed_source == "test_id":
        test_id_cfg = dict(config.get("train_val_test", {}).get("test_id", {}))
        test_id_overrides = test_id_cfg.get("environment_overrides")
        if isinstance(test_id_overrides, dict) and test_id_overrides:
            env_cfg = _deep_merge_dicts(env_cfg, test_id_overrides)

    env_overrides = rollout_cfg.get("environment_overrides")
    if isinstance(env_overrides, dict) and env_overrides:
        env_cfg = _deep_merge_dicts(env_cfg, env_overrides)

    use_seeded_episode = bool(rollout_cfg.get("use_seeded_episode", False))
    raw_num_days = rollout_cfg.get("num_days")
    num_days = int(raw_num_days) if raw_num_days is not None else None
    if env_type == "mobile_station_capacity":
        if num_days is not None:
            horizon = int(env_cfg["horizon"])
            env_cfg["max_steps"] = horizon * num_days
        disruption_cfg = dict(env_cfg.get("disruption", {}))
        scripted_events = [dict(event) for event in disruption_cfg.get("scripted_events", [])]
        if num_days is not None and scripted_events and bool(rollout_cfg.get("repeat_daily_disruptions", True)):
            for event in scripted_events:
                event.setdefault("repeat_daily", True)
                event.pop("day_index", None)
            disruption_cfg["scripted_events"] = scripted_events
            env_cfg["disruption"] = disruption_cfg
    else:
        sim_cfg = dict(env_cfg.get("simulation", {}))
        if num_days is not None:
            sim_cfg["duration_hours"] = 24.0 * num_days
            sim_cfg.pop("duration_days_range", None)
        elif not use_seeded_episode:
            fallback_num_days = 3
            sim_cfg["duration_hours"] = 24.0 * fallback_num_days
            sim_cfg.pop("duration_days_range", None)
        disruption_cfg = dict(sim_cfg.get("disruption", {}))
        comparison_events = rollout_cfg.get("scripted_events")
        if comparison_events:
            disruption_cfg["enabled"] = True
            disruption_cfg["mode"] = "scripted"
            disruption_cfg["scripted_events"] = [dict(event) for event in comparison_events]
        elif num_days is not None and disruption_cfg.get("mode", "random") == "scripted":
            scripted_events = [dict(event) for event in disruption_cfg.get("scripted_events", [])]
            if scripted_events and bool(rollout_cfg.get("repeat_daily_disruptions", True)):
                repeated_events: list[dict[str, Any]] = []
                template_events = sorted(scripted_events, key=lambda event: (int(event.get("day_index", 0)), float(event["start_hour"])))
                for day_index in range(num_days):
                    for event in template_events:
                        copied_event = dict(event)
                        copied_event["day_index"] = day_index
                        repeated_events.append(copied_event)
                disruption_cfg["scripted_events"] = repeated_events
        sim_cfg["disruption"] = disruption_cfg
        env_cfg["simulation"] = sim_cfg

    rollout_seed = int(rollout_cfg.get("seed", 123))
    if seed_source == "test_id":
        test_id_cfg = dict(config.get("train_val_test", {}).get("test_id", {}))
        test_id_seeds = build_seed_list(test_id_cfg.get("seeds", test_id_cfg))
        if test_id_seeds:
            bounded_index = min(max(seed_index, 0), len(test_id_seeds) - 1)
            rollout_seed = int(test_id_seeds[bounded_index])
    env = make_env(env_cfg, config["demand"], seed=rollout_seed)
    observation, _ = env.reset(seed=rollout_seed)
    rows: list[dict[str, Any]] = []
    while True:
        action = int(policy(observation, env, True))
        observation, reward, terminated, truncated, info = env.step(action)
        current_step = int(env.step_index - 1)
        global_hour = current_step * float(env.planning_step_minutes) / 60.0
        hour_of_day = global_hour % 24.0
        day_index = int(global_hour // 24.0)
        rows.append(
            {
                "step": current_step,
                "hour": global_hour,
                "global_hour": global_hour,
                "day_index": day_index,
                "hour_of_day": hour_of_day,
                "time_label": f"{int(hour_of_day):02d}:{int((hour_of_day % 1.0) * 60):02d}",
                "expected_passing_total": float(info.get("expected_passing_total", info.get("expected_arrivals_vehicles", 0.0))),
                "expected_station_arrivals_ab": float(info.get("expected_station_arrivals_ab", 0.0)),
                "expected_station_arrivals_bc": float(info.get("expected_station_arrivals_bc", 0.0)),
                "expected_passing_ab": float(info.get("expected_passing_ab", 0.0)),
                "expected_passing_ba": float(info.get("expected_passing_ba", 0.0)),
                "arrivals_total": float(info.get("arrivals_total", info.get("arrivals_vehicles", 0.0))),
                "arrivals_total_station_ab": float(info.get("arrivals_total_station_ab", 0.0)),
                "arrivals_total_station_bc": float(info.get("arrivals_total_station_bc", 0.0)),
                "arrivals_ab": float(info.get("arrivals_ab", 0.0)),
                "arrivals_ba": float(info.get("arrivals_ba", 0.0)),
                "starts_total": float(info.get("starts_total", info.get("served_demand", 0.0))),
                "starts_ab": float(info.get("starts_ab", 0.0)),
                "starts_ba": float(info.get("starts_ba", 0.0)),
                "completions_total": float(info.get("completions_total", info.get("served_demand", 0.0))),
                "completions_ab": float(info.get("completions_ab", 0.0)),
                "completions_ba": float(info.get("completions_ba", 0.0)),
                "queue_length": float(info.get("queue_length", 0.0)),
                "queue_wait_mean_minutes": float(info.get("queue_wait_mean_minutes", 0.0)),
                "queue_wait_target_minutes": float(info.get("queue_wait_target_minutes", 0.0)),
                "queue_wait_excess_minutes": float(info.get("queue_wait_excess_minutes", 0.0)),
                "queue_wait_target_breached": int(info.get("queue_wait_target_breached", 0)),
                "active_plugs": float(info.get("active_plugs", info.get("num_active_chargers", 0.0))),
                "effective_num_plugs": float(info.get("effective_num_plugs", info.get("effective_capacity_total", info.get("num_active_chargers", 0.0)))),
                "unused_mobile_chargers": float(info.get("unused_mobile_chargers", 0.0)),
                "unused_mobile_stations_estimate": float(info.get("unused_mobile_stations_estimate", 0.0)),
                "utilization": float(info.get("utilization", 0.0)),
                "started_wait_mean_minutes": float(info.get("started_wait_mean_minutes", info.get("queue_wait_mean_minutes", 0.0))),
                "completed_wait_mean_minutes": float(info.get("completed_wait_mean_minutes", info.get("queue_wait_mean_minutes", 0.0))),
                "started_service_mean_minutes": float(info.get("started_service_mean_minutes", env.mean_service_minutes)),
                "disruption_active": int(info.get("disruption_active", 0)),
                "disruption_type": str(info.get("disruption_type", "none")),
                "disruption_type_code": int(info.get("disruption_type_code", 0)),
                "disruption_target": str(info.get("disruption_target", "none")),
                "disruption_target_code": int(info.get("disruption_target_code", 0)),
                "disruption_day_index": int(info.get("disruption_day_index", -1)),
                "disruption_remaining_minutes": float(info.get("disruption_remaining_steps", 0.0)) * float(env.planning_step_minutes),
                "rl_action_mcs": float(action),
                "num_active_mobile_stations": float(info.get("num_active_mobile_stations", 0.0)),
                "num_active_mobile_stations_station_ab": float(info.get("num_active_mobile_stations_station_ab", 0.0)),
                "num_active_mobile_stations_station_bc": float(info.get("num_active_mobile_stations_station_bc", 0.0)),
                "committed_mobile_stations_station_ab": float(info.get("committed_mobile_stations_station_ab", 0.0)),
                "committed_mobile_stations_station_bc": float(info.get("committed_mobile_stations_station_bc", 0.0)),
                "committed_mobile_stations_bias_ab_minus_bc": float(info.get("committed_mobile_stations_bias_ab_minus_bc", 0.0)),
                "num_mobile_stations_middle_available": float(info.get("num_mobile_stations_middle_available", 0.0)),
                "num_mobile_stations_middle_charging": float(info.get("num_mobile_stations_middle_charging", 0.0)),
                "num_mobile_stations_in_transit_to_ab": float(info.get("num_mobile_stations_in_transit_to_ab", 0.0)),
                "num_mobile_stations_in_transit_to_bc": float(info.get("num_mobile_stations_in_transit_to_bc", 0.0)),
                "num_mobile_stations_in_transit_to_middle": float(info.get("num_mobile_stations_in_transit_to_middle", 0.0)),
                "queue_length_station_ab": float(info.get("queue_length_station_ab", 0.0)),
                "queue_length_station_bc": float(info.get("queue_length_station_bc", 0.0)),
                "queue_wait_mean_minutes_station_ab": float(info.get("queue_wait_mean_minutes_station_ab", 0.0)),
                "queue_wait_mean_minutes_station_bc": float(info.get("queue_wait_mean_minutes_station_bc", 0.0)),
                "local_deficit_without_mcs_station_ab": float(info.get("local_deficit_without_mcs_station_ab", 0.0)),
                "local_deficit_without_mcs_station_bc": float(info.get("local_deficit_without_mcs_station_bc", 0.0)),
                "local_deficit_bias_ab_minus_bc": float(info.get("local_deficit_bias_ab_minus_bc", 0.0)),
                "mobile_service_capacity_station_ab": float(info.get("mobile_service_capacity_station_ab", 0.0)),
                "mobile_service_capacity_station_bc": float(info.get("mobile_service_capacity_station_bc", 0.0)),
                "spatial_deficit_coverage": float(info.get("spatial_deficit_coverage", 0.0)),
                "unused_mobile_stations_estimate_station_ab": float(info.get("unused_mobile_stations_estimate_station_ab", 0.0)),
                "unused_mobile_stations_estimate_station_bc": float(info.get("unused_mobile_stations_estimate_station_bc", 0.0)),
                "expected_passing_od_ab": float(info.get("expected_passing_od_ab", 0.0)),
                "expected_passing_od_ba": float(info.get("expected_passing_od_ba", 0.0)),
                "expected_passing_od_bc": float(info.get("expected_passing_od_bc", 0.0)),
                "expected_passing_od_cb": float(info.get("expected_passing_od_cb", 0.0)),
                "expected_passing_od_ac": float(info.get("expected_passing_od_ac", 0.0)),
                "expected_passing_od_ca": float(info.get("expected_passing_od_ca", 0.0)),
                "reward_local_queue_peak_penalty_term": float(info.get("reward_local_queue_peak_penalty_term", 0.0)),
                "reward_local_wait_peak_penalty_term": float(info.get("reward_local_wait_peak_penalty_term", 0.0)),
                "reward": float(reward),
            }
        )
        if terminated or truncated:
            break

    frame = pd.DataFrame(rows)
    if {
        "expected_station_arrivals_ab",
        "expected_station_arrivals_bc",
        "num_active_mobile_stations_station_ab",
        "num_active_mobile_stations_station_bc",
    }.issubset(frame.columns):
        expected_total = frame["expected_station_arrivals_ab"] + frame["expected_station_arrivals_bc"]
        expected_total = expected_total.where(expected_total > 0.0, np.nan)
        frame["expected_demand_share_ab"] = (frame["expected_station_arrivals_ab"] / expected_total).fillna(0.5)
        frame["expected_demand_share_bc"] = (frame["expected_station_arrivals_bc"] / expected_total).fillna(0.5)

        allocated_total = frame["num_active_mobile_stations_station_ab"] + frame["num_active_mobile_stations_station_bc"]
        positive_allocation_mask = allocated_total > 0.0
        frame["allocation_share_ab"] = np.where(
            positive_allocation_mask,
            frame["num_active_mobile_stations_station_ab"] / allocated_total.where(positive_allocation_mask, np.nan),
            0.0,
        )
        frame["allocation_share_bc"] = np.where(
            positive_allocation_mask,
            frame["num_active_mobile_stations_station_bc"] / allocated_total.where(positive_allocation_mask, np.nan),
            0.0,
        )

        frame["allocation_demand_gap_ab"] = (frame["allocation_share_ab"] - frame["expected_demand_share_ab"]).abs()
        frame["allocation_demand_gap_bc"] = (frame["allocation_share_bc"] - frame["expected_demand_share_bc"]).abs()
        raw_alignment = 1.0 - 0.5 * (
            frame["allocation_demand_gap_ab"] + frame["allocation_demand_gap_bc"]
        )
        frame["allocation_vs_demand_alignment"] = np.where(
            positive_allocation_mask,
            raw_alignment,
            np.where(expected_total.fillna(0.0) > 0.0, 0.0, 1.0),
        )

        frame["allocation_bias_ab_minus_bc"] = (
            frame["num_active_mobile_stations_station_ab"] - frame["num_active_mobile_stations_station_bc"]
        )
        frame["expected_demand_bias_ab_minus_bc"] = (
            frame["expected_station_arrivals_ab"] - frame["expected_station_arrivals_bc"]
        )

        disruption_target_series = frame["disruption_target"] if "disruption_target" in frame else pd.Series(["none"] * len(frame))
        frame["alignment_on_station_ab_disruption"] = np.where(
            (frame["disruption_active"] == 1) & (disruption_target_series == "station_ab"),
            frame["allocation_bias_ab_minus_bc"],
            np.nan,
        )
        frame["alignment_on_station_bc_disruption"] = np.where(
            (frame["disruption_active"] == 1) & (disruption_target_series == "station_bc"),
            -frame["allocation_bias_ab_minus_bc"],
            np.nan,
        )
    disruption_targets = [target for target in LineCorridorQueueSimulator.DISRUPTION_TARGET_CODES if target != "none"]
    for target in disruption_targets:
        frame[f"disruption_target_is_{target}"] = ((frame["disruption_target"] == target) & (frame["disruption_active"] == 1)).astype(np.int32)
    from evch.train.run_simple_corridor_sim import _add_derived_metrics, _maybe_plot_daily_patterns, _maybe_plot_queue_dynamics

    frame = _add_derived_metrics(frame, step_minutes=int(env.planning_step_minutes))
    metrics_path = output_dir / "comparison_timestep_metrics.csv"
    summary_path = output_dir / "comparison_rollout_summary.json"
    plot_path = output_dir / "comparison_queue_dynamics.png"
    daily_plot_path = output_dir / "comparison_daily_patterns.png"
    station_plot_path = output_dir / "comparison_station_demand_vs_mcs.png"
    allocation_plot_path = output_dir / "comparison_mcs_allocation_by_station.png"
    frame.to_csv(metrics_path, index=False)
    summary = {
        "num_days": int(getattr(env.simulator, "num_days", num_days or 0)),
        "seed": rollout_seed,
        "mean_queue_length": float(frame["queue_length"].mean()),
        "peak_queue_length": float(frame["queue_length"].max()),
        "mean_queue_wait_minutes": float(frame["queue_wait_mean_minutes"].mean()),
        "mean_queue_wait_minutes_station_ab": float(frame["queue_wait_mean_minutes_station_ab"].mean())
        if "queue_wait_mean_minutes_station_ab" in frame
        else 0.0,
        "mean_queue_wait_minutes_station_bc": float(frame["queue_wait_mean_minutes_station_bc"].mean())
        if "queue_wait_mean_minutes_station_bc" in frame
        else 0.0,
        "peak_queue_wait_minutes": float(frame["queue_wait_mean_minutes"].max()),
        "queue_wait_target_minutes": float(frame["queue_wait_target_minutes"].max()) if "queue_wait_target_minutes" in frame else 0.0,
        "queue_wait_target_breach_fraction": float(frame["queue_wait_target_breached"].mean()) if "queue_wait_target_breached" in frame else 0.0,
        "mean_queue_wait_excess_minutes": float(frame["queue_wait_excess_minutes"].mean()) if "queue_wait_excess_minutes" in frame else 0.0,
        "mean_utilization": float(frame["utilization"].mean()),
        "mean_active_mobile_stations": float(frame["num_active_mobile_stations"].mean()),
        "mean_active_mobile_stations_normal": float(frame.loc[frame["disruption_active"] == 0, "num_active_mobile_stations"].mean())
        if (frame["disruption_active"] == 0).any()
        else 0.0,
        "mean_active_mobile_stations_disrupted": float(frame.loc[frame["disruption_active"] == 1, "num_active_mobile_stations"].mean())
        if (frame["disruption_active"] == 1).any()
        else 0.0,
        "mean_spatial_deficit_coverage": float(frame["spatial_deficit_coverage"].mean())
        if "spatial_deficit_coverage" in frame
        else 0.0,
        "mean_spatial_deficit_coverage_disrupted": float(frame.loc[frame["disruption_active"] == 1, "spatial_deficit_coverage"].mean())
        if ("spatial_deficit_coverage" in frame and (frame["disruption_active"] == 1).any())
        else 0.0,
        "mean_allocation_vs_demand_alignment": float(frame["allocation_vs_demand_alignment"].mean())
        if "allocation_vs_demand_alignment" in frame
        else 0.0,
        "mean_allocation_vs_demand_alignment_disrupted": float(
            frame.loc[frame["disruption_active"] == 1, "allocation_vs_demand_alignment"].mean()
        )
        if ("allocation_vs_demand_alignment" in frame and (frame["disruption_active"] == 1).any())
        else 0.0,
        "mean_ab_minus_bc_allocation_bias_on_station_ab_disruptions": float(frame["alignment_on_station_ab_disruption"].mean())
        if "alignment_on_station_ab_disruption" in frame
        else 0.0,
        "mean_bc_minus_ab_allocation_bias_on_station_bc_disruptions": float(frame["alignment_on_station_bc_disruption"].mean())
        if "alignment_on_station_bc_disruption" in frame
        else 0.0,
    }
    write_json(summary_path, summary)
    _maybe_plot_queue_dynamics(frame, plot_path)
    _maybe_plot_daily_patterns(frame, daily_plot_path)
    _maybe_plot_station_demand_vs_mcs(frame, station_plot_path)
    _maybe_plot_station_allocation_over_time(frame, allocation_plot_path)

    wandb_frame = frame.loc[:, [column for column in frame.columns if column in SIM_WANDB_COLUMNS]].copy()
    for column in wandb_frame.columns:
        if pd.api.types.is_numeric_dtype(wandb_frame[column].dtype):
            metric_name = f"sim/{column}"
            if column != "global_hour":
                run.define_metric(metric_name, step_metric="sim/global_hour")
    run.define_metric("sim_summary/*")
    for row in wandb_frame.to_dict(orient="records"):
        run.log({f"sim/{key}": value for key, value in row.items()})
    run.log({f"sim_summary/{key}": value for key, value in summary.items()})
    _log_station_overlay_panels(run, frame)

    log_artifact(run=run, path=metrics_path, artifact_name=f"{config['experiment']['name']}-comparison-rollout-metrics", artifact_type="metrics", aliases=["latest"])
    log_artifact(run=run, path=summary_path, artifact_name=f"{config['experiment']['name']}-comparison-rollout-summary", artifact_type="metrics", aliases=["latest"])
    log_artifact(run=run, path=plot_path, artifact_name=f"{config['experiment']['name']}-comparison-rollout-plot", artifact_type="plot", aliases=["latest"])
    log_artifact(run=run, path=daily_plot_path, artifact_name=f"{config['experiment']['name']}-comparison-daily-patterns", artifact_type="plot", aliases=["latest"])
    log_artifact(
        run=run,
        path=station_plot_path,
        artifact_name=f"{config['experiment']['name']}-comparison-station-demand-vs-mcs",
        artifact_type="plot",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=allocation_plot_path,
        artifact_name=f"{config['experiment']['name']}-comparison-mcs-allocation-by-station",
        artifact_type="plot",
        aliases=["latest"],
    )
    return {
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "daily_plot_path": str(daily_plot_path),
        "station_plot_path": str(station_plot_path),
        "allocation_plot_path": str(allocation_plot_path),
        "summary": summary,
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    rl_cfg = config["rl"]
    runtime_info = configure_torch_runtime(rl_cfg)
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "rl")
    run = init_wandb(config=config, job_type="train_rl", run_name=f"{experiment_cfg['name']}_rl")
    run.log({f"runtime/{key}": value for key, value in runtime_info.items()})

    split_cfg = dict(config.get("train_val_test", {}))
    training_cfg = dict(split_cfg.get("training", {}))
    val_cfg = dict(split_cfg.get("validation", {}))
    test_id_cfg = dict(split_cfg.get("test_id", {}))
    stress_cfg = dict(split_cfg.get("test_stress", {}))
    val_enabled = bool(val_cfg.get("enabled", True))
    periodic_eval_enabled = bool(val_cfg.get("periodic_enabled", val_enabled))

    val_env_overrides = val_cfg.get("environment_overrides", {})
    val_env_cfg = (
        _deep_merge_dicts(config["environment"], val_env_overrides)
        if isinstance(val_env_overrides, dict) and val_env_overrides
        else copy.deepcopy(config["environment"])
    )
    raw_val_seed_list = build_seed_list(val_cfg.get("seeds", val_cfg))
    val_seed_list = raw_val_seed_list if val_enabled else []
    val_episodes = len(val_seed_list) if val_seed_list else int(val_cfg.get("episodes", rl_cfg["evaluation_episodes"]))
    periodic_eval_seed_count = int(val_cfg.get("periodic_seed_count", min(max(len(raw_val_seed_list), 1), 32)))
    periodic_eval_seeds = raw_val_seed_list[:periodic_eval_seed_count] if (periodic_eval_enabled and raw_val_seed_list) else None

    train_env_cfg = copy.deepcopy(config["environment"])
    train_seed_range = resolve_train_seed_range(training_cfg)
    if train_seed_range is not None:
        train_env_cfg["episode_seed_range"] = [int(train_seed_range[0]), int(train_seed_range[1])]

    env = make_env(train_env_cfg, config["demand"], seed=seed)
    requested_backend = str(rl_cfg.get("backend", "auto")).strip().lower()
    has_sb3 = importlib.util.find_spec("stable_baselines3") is not None
    if requested_backend == "sb3_dqn" and not has_sb3:
        raise RuntimeError("RL backend `sb3_dqn` requires `stable_baselines3`, but it is not installed in this environment.")
    use_sb3 = has_sb3 and requested_backend == "sb3_dqn"

    history: list[dict[str, float]] = []
    if use_sb3:
        backend, checkpoint_path = _train_with_sb3(env, rl_cfg, seed=seed, output_dir=output_dir, run=run)
    else:
        backend, checkpoint_path, history = _train_with_torch_dqn(
            env,
            rl_cfg,
            seed=seed,
            output_dir=output_dir,
            run=run,
            eval_env_factory=(lambda eval_seed: make_env(val_env_cfg, config["demand"], seed=eval_seed))
            if periodic_eval_enabled
            else None,
            eval_episode_seeds=periodic_eval_seeds if periodic_eval_enabled else None,
        )

    policy = _make_rl_policy(backend, checkpoint_path)
    evaluation: dict[str, Any] | None = None
    if val_enabled and val_episodes > 0:
        eval_env = make_env(val_env_cfg, config["demand"], seed=(val_seed_list[0] if val_seed_list else seed + 17))
        evaluation = evaluate_policy(
            env=eval_env,
            policy=policy,
            episodes=val_episodes,
            seed=seed,
            deterministic=bool(rl_cfg.get("deterministic_eval", True)),
            episode_seeds=val_seed_list if val_seed_list else None,
        )
        run.log({f"validation/{key}": value for key, value in evaluation.items() if key not in {"episodes", "episode_seeds"}})
        run.log({"validation/num_seeds": float(len(val_seed_list)) if val_seed_list else float(val_episodes)})

    training_summary_path = output_dir / "training_summary.json"
    write_json(
        training_summary_path,
        {
            "backend": backend,
            "checkpoint_path": checkpoint_path,
            "runtime": runtime_info,
            "validation": (
                {key: value for key, value in evaluation.items() if key not in {"episodes", "episode_seeds"}}
                if evaluation is not None
                else None
            ),
        },
    )
    if history:
        write_json(output_dir / "history.json", {"history": history})

    if history and bool(experiment_cfg.get("save_plots", True)):
        _maybe_plot_training_curve(history, output_dir / "training_curve.png")

    comparison_rollout = _build_mobile_comparison_rollout(
        config=config,
        policy=policy,
        output_dir=output_dir,
        run=run,
    )

    suite_outputs = None
    if bool(test_id_cfg.get("enabled", False)) or bool(stress_cfg.get("enabled", False)) or (val_enabled and bool(val_seed_list)):
        suite_outputs = evaluate_policy_suites(
            config=config,
            checkpoint_path=checkpoint_path,
            run=run,
            output_dir=ensure_dir(output_dir / "evaluation_suites"),
        )

    log_artifact(
        run=run,
        path=checkpoint_path,
        artifact_name=f"{experiment_cfg['name']}-{backend}-checkpoint",
        artifact_type="model",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=training_summary_path,
        artifact_name=f"{experiment_cfg['name']}-rl-summary",
        artifact_type="metrics",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=output_dir / "history.json",
        artifact_name=f"{experiment_cfg['name']}-rl-history",
        artifact_type="metrics",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=output_dir / "training_curve.png",
        artifact_name=f"{experiment_cfg['name']}-rl-curve",
        artifact_type="plot",
        aliases=["latest"],
    )
    LOGGER.info("Finished RL training with backend=%s checkpoint=%s", backend, checkpoint_path)
    run.finish()
    return {
        "backend": backend,
        "checkpoint_path": checkpoint_path,
        "history": history,
        "evaluation": evaluation,
        "output_dir": str(output_dir),
        "training_summary_path": str(training_summary_path),
        "runtime": runtime_info,
        "comparison_rollout": comparison_rollout,
        "evaluation_suites": suite_outputs,
    }


def main() -> None:
    parser = build_config_parser("Train the EV charger placement RL baseline.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

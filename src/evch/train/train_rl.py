from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evch.config.loader import build_config_parser, load_config
from evch.envs.factory import make_env
from evch.rl.evaluation import evaluate_policy
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.torch_runtime import configure_torch_runtime, resolve_torch_device
from evch.utils.wandb import init_wandb, log_artifact

LOGGER = logging.getLogger(__name__)


def _maybe_plot_training_curve(history: list[dict[str, float]], path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot([entry["reward"] for entry in history], label="Episode reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Torch DQN training curve")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping RL training plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


class WandbSb3Callback:
    def __init__(self, run: Any, log_interval: int) -> None:
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

    def on_step(self, callback: Any) -> bool:
        if callback.num_timesteps % self.log_interval != 0:
            return True

        metrics: dict[str, float] = {"rl/timesteps": float(callback.num_timesteps)}
        exploration_rate = getattr(callback.model, "exploration_rate", None)
        if exploration_rate is not None:
            metrics["rl/exploration_rate"] = float(exploration_rate)

        replay_buffer = getattr(callback.model, "replay_buffer", None)
        if replay_buffer is not None:
            with contextlib.suppress(TypeError):
                metrics["rl/replay_buffer_size"] = float(len(replay_buffer))

        for key, value in getattr(callback.model.logger, "name_to_value", {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                metrics[f"sb3/{key}"] = float(value)

        self.run.log(metrics, step=int(callback.num_timesteps))
        return True


def _train_with_sb3(env: Any, rl_cfg: dict[str, Any], seed: int, output_dir: Path, run: Any) -> tuple[str, str]:
    from stable_baselines3 import DQN  # type: ignore

    sb3_cfg = rl_cfg["sb3"]
    device = resolve_torch_device(str(rl_cfg.get("device", "auto")))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(rl_cfg["learning_rate"]),
        buffer_size=int(sb3_cfg["buffer_size"]),
        learning_starts=int(sb3_cfg["learning_starts"]),
        batch_size=int(sb3_cfg["batch_size"]),
        gamma=float(rl_cfg["gamma"]),
        tau=float(sb3_cfg["tau"]),
        target_update_interval=int(sb3_cfg["target_update_interval"]),
        verbose=0,
        seed=seed,
        device=str(device),
    )
    callback = WandbSb3Callback(run=run, log_interval=int(rl_cfg.get("wandb_log_interval", 100))).callback
    model.learn(total_timesteps=int(sb3_cfg["total_timesteps"]), callback=callback)
    checkpoint = output_dir / "best_model.zip"
    model.save(checkpoint)
    return "sb3_dqn", str(checkpoint)


def _train_with_torch_dqn(env: Any, rl_cfg: dict[str, Any], seed: int, output_dir: Path, run: Any) -> tuple[str, str, list[dict[str, float]]]:
    agent = SimpleDQNAgent(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        config=rl_cfg,
        seed=seed,
    )
    history = agent.train(
        env=env,
        episodes=int(rl_cfg["episodes"]),
        max_steps=int(rl_cfg["max_steps_per_episode"]),
        run=run,
    )
    checkpoint = output_dir / str(rl_cfg["checkpoint_name"])
    agent.save(checkpoint)
    return "torch_dqn", str(checkpoint), history


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

    env = make_env(config["environment"], config["demand"], seed=seed)
    use_sb3 = False
    try:
        import stable_baselines3  # noqa: F401

        use_sb3 = rl_cfg.get("backend", "auto") == "sb3_dqn"
    except ImportError:
        use_sb3 = False

    history: list[dict[str, float]] = []
    if use_sb3:
        backend, checkpoint_path = _train_with_sb3(env, rl_cfg, seed=seed, output_dir=output_dir, run=run)
    else:
        backend, checkpoint_path, history = _train_with_torch_dqn(env, rl_cfg, seed=seed, output_dir=output_dir, run=run)

    eval_env = make_env(config["environment"], config["demand"], seed=seed + 17)
    policy = _make_rl_policy(backend, checkpoint_path)
    evaluation = evaluate_policy(
        env=eval_env,
        policy=policy,
        episodes=int(rl_cfg["evaluation_episodes"]),
        seed=seed,
        deterministic=bool(rl_cfg.get("deterministic_eval", True)),
    )
    run.log({f"rl_eval/{key}": value for key, value in evaluation.items() if not isinstance(value, list)})

    if history and bool(experiment_cfg.get("save_plots", True)):
        _maybe_plot_training_curve(history, output_dir / "training_curve.png")

    training_summary_path = output_dir / "training_summary.json"
    write_json(
        training_summary_path,
        {
            "backend": backend,
            "checkpoint_path": checkpoint_path,
            "runtime": runtime_info,
            "evaluation": {key: value for key, value in evaluation.items() if key != "episodes"},
        },
    )
    if history:
        write_json(output_dir / "history.json", {"history": history})

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
    }


def main() -> None:
    parser = build_config_parser("Train the EV charger placement RL baseline.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

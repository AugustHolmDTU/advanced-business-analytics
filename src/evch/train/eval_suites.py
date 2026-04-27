from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from evch.baselines.policies import BASELINE_POLICIES
from evch.envs.factory import make_env
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.utils.io import ensure_dir, write_json
from evch.utils.wandb import DummyRun, log_artifact

LOGGER = logging.getLogger(__name__)

PolicyFn = Callable[[np.ndarray, Any, bool], int]

POLICY_EXPORT_ALIASES = {
    "mobile_noop": "fixed",
    "fixed_only": "fixed",
    "mobile_threshold": "threshold",
    "rl": "rl",
}


def _deep_merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_seed_list(seed_cfg: Any) -> list[int]:
    if isinstance(seed_cfg, list):
        return [int(value) for value in seed_cfg]
    if isinstance(seed_cfg, dict):
        if isinstance(seed_cfg.get("seeds"), list):
            return [int(value) for value in seed_cfg["seeds"]]
        if "start" in seed_cfg and "count" in seed_cfg:
            start = int(seed_cfg["start"])
            count = int(seed_cfg["count"])
            step = int(seed_cfg.get("step", 1))
            return [start + index * step for index in range(max(count, 0))]
        if "seed_start" in seed_cfg and "count" in seed_cfg:
            start = int(seed_cfg["seed_start"])
            count = int(seed_cfg["count"])
            step = int(seed_cfg.get("step", 1))
            return [start + index * step for index in range(max(count, 0))]
    if isinstance(seed_cfg, tuple) and len(seed_cfg) == 2:
        low = int(seed_cfg[0])
        high = int(seed_cfg[1])
        if high < low:
            low, high = high, low
        return list(range(low, high + 1))
    return []


def resolve_train_seed_range(training_cfg: dict[str, Any]) -> tuple[int, int] | None:
    raw_range = training_cfg.get("episode_seed_range") or training_cfg.get("seed_range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return None
    low = int(raw_range[0])
    high = int(raw_range[1])
    if high < low:
        low, high = high, low
    return low, high


def load_rl_policy(checkpoint_path: str) -> PolicyFn:
    if checkpoint_path.endswith(".zip"):
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


def build_policy_specs(policy_names: list[str], checkpoint_path: str = "", seed: int = 0) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for policy_name in policy_names:
        if policy_name == "rl":
            if not checkpoint_path:
                continue
            policy = load_rl_policy(checkpoint_path)
        else:
            builder = BASELINE_POLICIES[policy_name]
            policy = builder(seed=seed) if policy_name == "random" else builder  # type: ignore[misc]
        specs.append(
            {
                "policy_name": policy_name,
                "export_alias": POLICY_EXPORT_ALIASES.get(policy_name, policy_name),
                "policy": policy,
            }
        )
    return specs


def _prepare_env_config(base_env_cfg: dict[str, Any], suite_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    env_cfg = copy.deepcopy(base_env_cfg)
    suite_cfg = {} if suite_cfg is None else dict(suite_cfg)
    env_overrides = suite_cfg.get("environment_overrides")
    if isinstance(env_overrides, dict) and env_overrides:
        env_cfg = _deep_merge_dicts(env_cfg, env_overrides)

    num_days = suite_cfg.get("num_days")
    if num_days is not None:
        sim_cfg = dict(env_cfg.get("simulation", {}))
        sim_cfg["duration_hours"] = 24.0 * int(num_days)
        sim_cfg.pop("duration_days_range", None)
        disruption_cfg = dict(sim_cfg.get("disruption", {}))
        scripted_events = suite_cfg.get("scripted_events")
        if scripted_events:
            disruption_cfg["enabled"] = True
            disruption_cfg["mode"] = "scripted"
            disruption_cfg["scripted_events"] = [dict(event) for event in scripted_events]
        sim_cfg["disruption"] = disruption_cfg
        env_cfg["simulation"] = sim_cfg
    return env_cfg


def _format_manifest_list(values: list[str]) -> str:
    return "|".join(values) if values else "none"


def _rollout_policy_for_seed(
    env_cfg: dict[str, Any],
    demand_cfg: dict[str, Any],
    policy: PolicyFn,
    seed: int,
    scenario_id: str,
) -> dict[str, Any]:
    env = make_env(env_cfg, demand_cfg, seed=seed)
    observation, _ = env.reset(seed=seed)
    disruption_schedule = list(getattr(env, "current_disruption_schedule", []))
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
                "global_hour": global_hour,
                "hour_of_day": hour_of_day,
                "day_index": day_index,
                "queue_length": float(info.get("queue_length", 0.0)),
                "queue_wait_mean_minutes": float(info.get("queue_wait_mean_minutes", 0.0)),
                "queue_wait_target_minutes": float(info.get("queue_wait_target_minutes", 0.0)),
                "queue_wait_excess_minutes": float(info.get("queue_wait_excess_minutes", 0.0)),
                "queue_wait_target_breached": float(info.get("queue_wait_target_breached", 0.0)),
                "reward": float(reward),
                "served_demand": float(info.get("served_demand", 0.0)),
                "unmet_demand": float(info.get("unmet_demand", 0.0)),
                "num_active_mobile_stations": float(info.get("num_active_mobile_stations", 0.0)),
                "num_active_mobile_stations_station_ab": float(info.get("num_active_mobile_stations_station_ab", 0.0)),
                "num_active_mobile_stations_station_bc": float(info.get("num_active_mobile_stations_station_bc", 0.0)),
                "activated_mobile_stations": float(info.get("activated_mobile_stations", 0.0)),
                "adjusted_mobile_stations": float(info.get("adjusted_mobile_stations", 0.0)),
                "queue_length_station_ab": float(info.get("queue_length_station_ab", 0.0)),
                "queue_length_station_bc": float(info.get("queue_length_station_bc", 0.0)),
                "queue_wait_mean_minutes_station_ab": float(info.get("queue_wait_mean_minutes_station_ab", 0.0)),
                "queue_wait_mean_minutes_station_bc": float(info.get("queue_wait_mean_minutes_station_bc", 0.0)),
                "expected_station_arrivals_ab": float(info.get("expected_station_arrivals_ab", 0.0)),
                "expected_station_arrivals_bc": float(info.get("expected_station_arrivals_bc", 0.0)),
                "disruption_active": float(info.get("disruption_active", 0.0)),
                "disruption_type_code": float(info.get("disruption_type_code", 0.0)),
                "disruption_target": str(info.get("disruption_target", "none")),
            }
        )
        if terminated or truncated:
            break

    frame = pd.DataFrame(rows)
    mcs_hours = float(frame["num_active_mobile_stations"].sum() * float(env.planning_step_minutes) / 60.0)
    summary = {
        "seed": int(seed),
        "scenario_id": scenario_id,
        "num_days": int(getattr(env.simulator, "num_days", 1)),
        "mean_queue_length": float(frame["queue_length"].mean()),
        "peak_queue_length": float(frame["queue_length"].max()),
        "mean_queue_wait_minutes": float(frame["queue_wait_mean_minutes"].mean()),
        "peak_queue_wait_minutes": float(frame["queue_wait_mean_minutes"].max()),
        "queue_wait_target_breach_fraction": float(frame["queue_wait_target_breached"].mean()),
        "queue_wait_excess_minutes": float(frame["queue_wait_excess_minutes"].sum()),
        "served_demand": float(frame["served_demand"].sum()),
        "unmet_demand": float(frame["unmet_demand"].sum()),
        "mean_active_mobile_stations": float(frame["num_active_mobile_stations"].mean()),
        "total_activated_mobile_stations": float(frame["activated_mobile_stations"].sum()),
        "total_adjusted_mobile_stations": float(frame["adjusted_mobile_stations"].sum()),
        "mcs_hours": mcs_hours,
        "active_mcs_steps": float(frame["num_active_mobile_stations"].sum()),
        "reward": float(frame["reward"].sum()),
    }
    manifest = {
        "seed": int(seed),
        "scenario_id": scenario_id,
        "num_days": int(getattr(env.simulator, "num_days", 1)),
        "disruption_count": len(disruption_schedule),
        "disruption_types": _format_manifest_list([str(event.disruption_type) for event in disruption_schedule]),
        "disruption_targets": _format_manifest_list([str(event.target) for event in disruption_schedule]),
        "start_times": _format_manifest_list([f"{int(event.day_index)}@{float(event.start_hour):.2f}" for event in disruption_schedule]),
        "durations": _format_manifest_list(
            [f"{(int(event.end_step) - int(event.start_step)) * float(env.planning_step_minutes) / 60.0:.2f}" for event in disruption_schedule]
        ),
        "severities": _format_manifest_list([f"{float(event.severity):.2f}" for event in disruption_schedule]),
    }
    return {"frame": frame, "summary": summary, "manifest": manifest}


def _aggregate_policy_summary(frame: pd.DataFrame, suite_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metric_columns = [
        "mean_queue_length",
        "peak_queue_length",
        "mean_queue_wait_minutes",
        "peak_queue_wait_minutes",
        "queue_wait_target_breach_fraction",
        "queue_wait_excess_minutes",
        "served_demand",
        "unmet_demand",
        "mean_active_mobile_stations",
        "total_activated_mobile_stations",
        "total_adjusted_mobile_stations",
        "mcs_hours",
        "active_mcs_steps",
        "reward",
    ]
    grouped = frame.groupby(["suite_name", "policy_alias"], as_index=False)[metric_columns].mean()
    grouped.insert(0, "summary_type", suite_name)
    return grouped


def _build_paired_results(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    metrics = [
        "mean_queue_length",
        "mean_queue_wait_minutes",
        "queue_wait_target_breach_fraction",
        "reward",
        "mcs_hours",
    ]
    wide = summary_frame.pivot_table(index=["seed", "scenario_id"], columns="policy_alias", values=metrics, aggfunc="first")
    wide.columns = [f"{policy}_{metric}" for metric, policy in wide.columns]
    wide = wide.reset_index()

    paired = pd.DataFrame(
        {
            "seed": wide["seed"],
            "scenario_id": wide["scenario_id"],
            "fixed_mean_queue": wide.get("fixed_mean_queue_length"),
            "threshold_mean_queue": wide.get("threshold_mean_queue_length"),
            "rl_mean_queue": wide.get("rl_mean_queue_length"),
            "fixed_mean_wait": wide.get("fixed_mean_queue_wait_minutes"),
            "threshold_mean_wait": wide.get("threshold_mean_queue_wait_minutes"),
            "rl_mean_wait": wide.get("rl_mean_queue_wait_minutes"),
            "fixed_breach_rate": wide.get("fixed_queue_wait_target_breach_fraction"),
            "threshold_breach_rate": wide.get("threshold_queue_wait_target_breach_fraction"),
            "rl_breach_rate": wide.get("rl_queue_wait_target_breach_fraction"),
            "fixed_reward": wide.get("fixed_reward"),
            "threshold_reward": wide.get("threshold_reward"),
            "rl_reward": wide.get("rl_reward"),
            "fixed_mcs_hours": wide.get("fixed_mcs_hours"),
            "threshold_mcs_hours": wide.get("threshold_mcs_hours"),
            "rl_mcs_hours": wide.get("rl_mcs_hours"),
        }
    )
    paired["rl_queue_reduction_vs_fixed"] = paired["fixed_mean_queue"] - paired["rl_mean_queue"]
    paired["rl_queue_reduction_vs_threshold"] = paired["threshold_mean_queue"] - paired["rl_mean_queue"]
    paired["rl_wait_reduction_vs_fixed"] = paired["fixed_mean_wait"] - paired["rl_mean_wait"]
    paired["rl_wait_reduction_vs_threshold"] = paired["threshold_mean_wait"] - paired["rl_mean_wait"]
    paired["rl_breach_reduction_vs_fixed"] = paired["fixed_breach_rate"] - paired["rl_breach_rate"]
    paired["rl_breach_reduction_vs_threshold"] = paired["threshold_breach_rate"] - paired["rl_breach_rate"]
    paired["rl_reward_difference_vs_fixed"] = paired["rl_reward"] - paired["fixed_reward"]
    paired["rl_reward_difference_vs_threshold"] = paired["rl_reward"] - paired["threshold_reward"]
    paired["rl_extra_mcs_usage_vs_fixed"] = paired["rl_mcs_hours"] - paired["fixed_mcs_hours"]
    paired["rl_extra_mcs_usage_vs_threshold"] = paired["rl_mcs_hours"] - paired["threshold_mcs_hours"]
    return paired


def _log_policy_summary(run: Any, namespace: str, frame: pd.DataFrame) -> None:
    if isinstance(run, DummyRun) or frame.empty:
        return
    for row in frame.to_dict(orient="records"):
        policy_alias = str(row["policy_alias"])
        prefix = f"{namespace}/{policy_alias}"
        if "stress_scenario" in row and row["stress_scenario"]:
            prefix = f"{namespace}/{row['stress_scenario']}/{policy_alias}"
        for key, value in row.items():
            if key in {"summary_type", "suite_name", "policy_alias", "stress_scenario"}:
                continue
            run.log({f"{prefix}/{key}": float(value)})


def _log_paired_summary(run: Any, frame: pd.DataFrame) -> None:
    if isinstance(run, DummyRun) or frame.empty:
        return
    metric_columns = [
        "rl_queue_reduction_vs_fixed",
        "rl_queue_reduction_vs_threshold",
        "rl_wait_reduction_vs_fixed",
        "rl_wait_reduction_vs_threshold",
        "rl_breach_reduction_vs_fixed",
        "rl_breach_reduction_vs_threshold",
        "rl_reward_difference_vs_fixed",
        "rl_reward_difference_vs_threshold",
        "rl_extra_mcs_usage_vs_fixed",
        "rl_extra_mcs_usage_vs_threshold",
    ]
    for column in metric_columns:
        run.log({f"paired_test/{column}": float(frame[column].mean())})


def _run_seed_suite(
    suite_name: str,
    seeds: list[int],
    config: dict[str, Any],
    suite_cfg: dict[str, Any],
    policy_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env_cfg = _prepare_env_config(config["environment"], suite_cfg)
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for seed in seeds:
        scenario_id = f"{suite_name}_{int(seed)}"
        manifest_record: dict[str, Any] | None = None
        for policy_spec in policy_specs:
            result = _rollout_policy_for_seed(env_cfg, config["demand"], policy_spec["policy"], seed=int(seed), scenario_id=scenario_id)
            summary_rows.append(
                {
                    "suite_name": suite_name,
                    "policy_alias": policy_spec["export_alias"],
                    **result["summary"],
                }
            )
            if manifest_record is None:
                manifest_record = {"suite_name": suite_name, **result["manifest"]}
        if manifest_record is not None:
            manifest_rows.append(manifest_record)
    return pd.DataFrame(summary_rows), pd.DataFrame(manifest_rows)


def evaluate_policy_suites(config: dict[str, Any], checkpoint_path: str, run: Any | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    run = DummyRun() if run is None else run
    experiment_cfg = config["experiment"]
    suite_output_dir = ensure_dir(
        output_dir if output_dir is not None else Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "evaluation"
    )
    split_cfg = dict(config.get("train_val_test", {}))
    evaluation_cfg = dict(config.get("evaluation", {}))
    policy_names = list(evaluation_cfg.get("policies", ["rl", "mobile_threshold", "mobile_noop"]))
    policy_specs = build_policy_specs(policy_names, checkpoint_path=checkpoint_path, seed=int(config.get("seed", 0)))

    outputs: dict[str, Any] = {"output_dir": str(suite_output_dir)}

    validation_cfg = dict(split_cfg.get("validation", {}))
    validation_seeds = build_seed_list(validation_cfg.get("seeds", validation_cfg))
    validation_summary = pd.DataFrame()
    if validation_seeds:
        validation_frame, validation_manifest = _run_seed_suite(
            "validation",
            validation_seeds,
            config,
            validation_cfg,
            [spec for spec in policy_specs if spec["export_alias"] == "rl"],
        )
        validation_summary = _aggregate_policy_summary(validation_frame, "validation")
        validation_summary.to_csv(suite_output_dir / "validation_summary.csv", index=False)
        validation_manifest.to_csv(suite_output_dir / "validation_manifest.csv", index=False)
        _log_policy_summary(run, "validation", validation_summary)
        run.log({"validation/num_scenarios": float(len(validation_seeds))})
        outputs["validation_summary_path"] = str(suite_output_dir / "validation_summary.csv")

    test_id_cfg = dict(split_cfg.get("test_id", {}))
    test_id_seeds = build_seed_list(test_id_cfg.get("seeds", test_id_cfg))
    test_id_frame = pd.DataFrame()
    test_id_manifest = pd.DataFrame()
    paired_frame = pd.DataFrame()
    if bool(test_id_cfg.get("enabled", True)) and test_id_seeds:
        test_id_frame, test_id_manifest = _run_seed_suite("test_id", test_id_seeds, config, test_id_cfg, policy_specs)
        test_id_summary = _aggregate_policy_summary(test_id_frame, "test_id")
        paired_frame = _build_paired_results(test_id_frame)
        test_id_summary.to_csv(suite_output_dir / "test_id_summary.csv", index=False)
        paired_frame.to_csv(suite_output_dir / "paired_test_results.csv", index=False)
        _log_policy_summary(run, "test_id", test_id_summary)
        _log_paired_summary(run, paired_frame)
        run.log({"test_id/num_scenarios": float(len(test_id_seeds))})
        outputs["test_id_summary_path"] = str(suite_output_dir / "test_id_summary.csv")
        outputs["paired_test_results_path"] = str(suite_output_dir / "paired_test_results.csv")

    stress_cfg = dict(split_cfg.get("test_stress", {}))
    stress_summary_frame = pd.DataFrame()
    stress_manifest_rows: list[pd.DataFrame] = []
    if bool(stress_cfg.get("enabled", False)):
        stress_rows: list[pd.DataFrame] = []
        for scenario_cfg in stress_cfg.get("scenarios", []):
            scenario_dict = dict(scenario_cfg)
            scenario_name = str(scenario_dict.get("scenario_id", f"stress_{len(stress_rows)}"))
            stress_seeds = build_seed_list(scenario_dict.get("seeds", scenario_dict.get("seed_spec", scenario_dict)))
            if not stress_seeds:
                default_seed = int(scenario_dict.get("seed", 30_000 + len(stress_rows)))
                stress_seeds = [default_seed]
            frame, manifest = _run_seed_suite(f"test_stress_{scenario_name}", stress_seeds, config, scenario_dict, policy_specs)
            if not frame.empty:
                frame["stress_scenario"] = scenario_name
                stress_rows.append(frame)
            if not manifest.empty:
                manifest["stress_scenario"] = scenario_name
                stress_manifest_rows.append(manifest)
        if stress_rows:
            stress_frame = pd.concat(stress_rows, ignore_index=True)
            metric_columns = [
                "mean_queue_length",
                "peak_queue_length",
                "mean_queue_wait_minutes",
                "peak_queue_wait_minutes",
                "queue_wait_target_breach_fraction",
                "queue_wait_excess_minutes",
                "served_demand",
                "unmet_demand",
                "mean_active_mobile_stations",
                "total_activated_mobile_stations",
                "total_adjusted_mobile_stations",
                "mcs_hours",
                "active_mcs_steps",
                "reward",
            ]
            stress_summary_frame = stress_frame.groupby(["stress_scenario", "policy_alias"], as_index=False)[metric_columns].mean()
            stress_summary_frame.to_csv(suite_output_dir / "test_stress_summary.csv", index=False)
            _log_policy_summary(run, "test_stress", stress_summary_frame)
            run.log({"test_stress/num_scenarios": float(stress_summary_frame["stress_scenario"].nunique())})
            outputs["test_stress_summary_path"] = str(suite_output_dir / "test_stress_summary.csv")

    manifest_frames = [frame for frame in [validation_manifest if validation_seeds else pd.DataFrame(), test_id_manifest] + stress_manifest_rows if not frame.empty]
    if manifest_frames:
        manifest_frame = pd.concat(manifest_frames, ignore_index=True)
        manifest_frame.to_csv(suite_output_dir / "scenario_manifest.csv", index=False)
        outputs["scenario_manifest_path"] = str(suite_output_dir / "scenario_manifest.csv")

    write_json(
        suite_output_dir / "suite_outputs.json",
        {
            "validation_summary_path": outputs.get("validation_summary_path", ""),
            "test_id_summary_path": outputs.get("test_id_summary_path", ""),
            "paired_test_results_path": outputs.get("paired_test_results_path", ""),
            "test_stress_summary_path": outputs.get("test_stress_summary_path", ""),
            "scenario_manifest_path": outputs.get("scenario_manifest_path", ""),
        },
    )
    outputs["suite_outputs_path"] = str(suite_output_dir / "suite_outputs.json")

    for key in ("validation_summary_path", "test_id_summary_path", "paired_test_results_path", "test_stress_summary_path", "scenario_manifest_path", "suite_outputs_path"):
        path = outputs.get(key)
        if path:
            log_artifact(run=run, path=path, artifact_name=f"{experiment_cfg['name']}-{Path(path).stem}", artifact_type="metrics", aliases=["latest"])

    return outputs

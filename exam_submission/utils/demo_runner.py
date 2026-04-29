from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from exam_submission.utils.plotting import POLICY_LABELS, plot_policy_rollout_panel, plot_training_history


REPO_ROOT = next(
    path.parent if path.name == "exam_submission" else path
    for path in [Path.cwd(), *Path.cwd().parents]
    if path.name == "exam_submission" or (path / "exam_submission").exists()
)
PROJECT_SRC = REPO_ROOT / "src"


def _repo_root() -> Path:
    return REPO_ROOT


def _activate_project_src() -> None:
    project_src_str = str(PROJECT_SRC)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

    # The notebook normally prefers the copied `reference_code/src` tree.
    # For the optional demo, we need the full project package instead.
    for module_name in list(sys.modules):
        if module_name == "evch" or module_name.startswith("evch."):
            del sys.modules[module_name]

    importlib.invalidate_caches()


def _load_project_symbols() -> dict[str, Any]:
    _activate_project_src()

    from evch.config.loader import load_config
    from evch.train.eval_suites import _prepare_env_config, _rollout_policy_for_seed, build_policy_specs
    from evch.train.train_rl import run_training

    return {
        "load_config": load_config,
        "_prepare_env_config": _prepare_env_config,
        "_rollout_policy_for_seed": _rollout_policy_for_seed,
        "build_policy_specs": build_policy_specs,
        "run_training": run_training,
    }


def _demo_config_paths(repo_root: Path) -> list[str]:
    base = repo_root / "exam_submission" / "reference_code" / "configs"
    return [
        str(base / "env" / "mobile_mcs_line_abc.yaml"),
        str(base / "demand" / "base.yaml"),
        str(base / "rl" / "dqn_mobile_simple.yaml"),
        str(base / "debug" / "logging_disabled.yaml"),
        str(base / "experiment" / "mobile_mcs_line_abc_train_only.yaml"),
        str(base / "demo" / "mobile_mcs_line_abc_demo_short.yaml"),
    ]


def build_appendix_demo_config(
    *,
    experiment_name: str = "exam_appendix_demo",
    output_root: str = "exam_submission/demo_outputs",
    train_episodes: int = 30,
    max_steps_per_episode: int = 288,
    eval_episodes: int = 2,
    eval_interval_steps: int = 400,
    heldout_seed: int = 12000,
    heldout_num_days: int = 5,
    validation_seed_start: int = 10000,
    validation_seed_count: int = 2,
) -> dict[str, Any]:
    symbols = _load_project_symbols()
    load_config = symbols["load_config"]

    repo_root = _repo_root()
    config = load_config(_demo_config_paths(repo_root))
    config = copy.deepcopy(config)
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = (repo_root / output_root_path).resolve()

    config.setdefault("experiment", {})
    config["experiment"]["name"] = experiment_name
    config["experiment"]["output_root"] = str(output_root_path)
    config["experiment"]["save_plots"] = False

    config.setdefault("comparison_rollout", {})
    config["comparison_rollout"]["enabled"] = False

    config.setdefault("rl", {})
    config["rl"]["episodes"] = int(train_episodes)
    config["rl"]["max_steps_per_episode"] = int(max_steps_per_episode)
    config["rl"]["evaluation_episodes"] = int(eval_episodes)
    config["rl"]["eval_interval_steps"] = int(eval_interval_steps)
    config["rl"]["eval_during_training_episodes"] = int(eval_episodes)

    split_cfg = config.setdefault("train_val_test", {})
    split_cfg["validation"] = {
        "enabled": True,
        "periodic_enabled": True,
        "seeds": {
            "start": int(validation_seed_start),
            "count": int(validation_seed_count),
        },
        "periodic_seed_count": int(validation_seed_count),
    }
    split_cfg["test_id"] = {
        "enabled": True,
        "seeds": [int(heldout_seed)],
        "num_days": int(heldout_num_days),
    }
    split_cfg["test_stress"] = {"enabled": False}
    return config


def _heldout_demo_rollout(config: dict[str, Any], checkpoint_path: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    symbols = _load_project_symbols()
    prepare_env_config = symbols["_prepare_env_config"]
    rollout_policy_for_seed = symbols["_rollout_policy_for_seed"]
    build_policy_specs = symbols["build_policy_specs"]

    test_cfg = dict(config.get("train_val_test", {}).get("test_id", {}))
    seeds = test_cfg.get("seeds", [])
    heldout_seed = int(seeds[0] if seeds else 12000)
    env_cfg = prepare_env_config(config["environment"], test_cfg)
    policy_specs = build_policy_specs(
        ["mobile_noop", "rl"],
        checkpoint_path=checkpoint_path,
        seed=int(config.get("seed", 0)),
    )

    run_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for policy_spec in policy_specs:
        result = rollout_policy_for_seed(
            env_cfg=env_cfg,
            demand_cfg=config["demand"],
            policy=policy_spec["policy"],
            seed=heldout_seed,
            scenario_id=f"appendix_demo_{heldout_seed}",
        )
        alias = str(policy_spec["export_alias"])
        run_frames[alias] = result["frame"]
        summary_rows.append({"policy": alias, **result["summary"]})

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        policy_display_names = {
            "fixed": "No-agent baseline",
            "mobile_noop": "No-agent baseline",
            "rl": "RL agent",
        }
        summary = summary[
            [
                "policy",
                "mean_queue_length",
                "peak_queue_length",
                "mean_queue_wait_minutes",
                "peak_queue_wait_minutes",
                "mean_active_mobile_stations",
                "mcs_hours",
                "reward",
            ]
        ].copy()
        summary["policy"] = summary["policy"].map(lambda value: policy_display_names.get(str(value), POLICY_LABELS.get(str(value), str(value))))
        summary["policy"] = summary["policy"].fillna("No-agent baseline")
        summary["_sort"] = summary["policy"].map({"No-agent baseline": 0, "RL agent": 1}).fillna(99)
        summary = summary.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return run_frames, summary


def run_appendix_demo(**kwargs: Any) -> dict[str, Any]:
    config = build_appendix_demo_config(**kwargs)
    run_training = _load_project_symbols()["run_training"]

    training_result = run_training(config)
    history = list(training_result.get("history", []))
    training_figure, _ = plot_training_history(history, source="appendix_demo")
    heldout_frames, heldout_summary = _heldout_demo_rollout(config, str(training_result["checkpoint_path"]))
    heldout_figure, _ = plot_policy_rollout_panel(heldout_frames, title="Demo held-out test")

    seed_list = config["train_val_test"]["test_id"]["seeds"]
    heldout_seed = int(seed_list[0] if seed_list else 12000)
    description = (
        "This appendix demo calls `run_appendix_demo()` from "
        "`exam_submission/utils/demo_runner.py`. "
        f"It trains the RL agent for {config['rl']['episodes']} short episodes, "
        "tracks train and periodic evaluation diagnostics, "
        f"and then runs one held-out `test_id` rollout on seed `{heldout_seed}` "
        f"across {config['train_val_test']['test_id']['num_days']} days to compare the trained RL agent against the no-agent baseline."
    )

    return {
        "config": config,
        "training_result": training_result,
        "history": history,
        "training_figure": training_figure,
        "heldout_frames": heldout_frames,
        "heldout_summary": heldout_summary,
        "heldout_figure": heldout_figure,
        "description": description,
    }

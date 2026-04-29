from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .artifact_finder import data_root


def repo_root() -> Path:
    return data_root().parents[1]


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_csv(path: str | Path, allow_empty: bool = False) -> pd.DataFrame:
    target = Path(path)
    if allow_empty and not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except pd.errors.EmptyDataError:
        if allow_empty:
            return pd.DataFrame()
        raise


def load_core_configs() -> dict[str, dict[str, Any]]:
    config_dir = data_root() / "configs"
    return {
        "environment": load_yaml(config_dir / "env_mobile_mcs_line_abc_current.yaml"),
        "rl_simple": load_yaml(config_dir / "rl_dqn_mobile_simple.yaml"),
        "rl_sb3": load_yaml(config_dir / "rl_dqn_mobile_sb3.yaml"),
        "experiment": load_yaml(config_dir / "experiment_mobile_mcs_line_abc_current.yaml"),
        "heldout": load_yaml(config_dir / "experiment_mobile_mcs_line_abc_heldout_comparison.yaml"),
    }


def load_heldout_runs() -> dict[str, dict[str, Any]]:
    heldout_dir = data_root() / "generated" / "heldout"
    runs: dict[str, dict[str, Any]] = {}
    for policy_name in ("mobile_noop", "mobile_threshold", "mobile_reactive"):
        runs[policy_name] = {
            "metrics": load_csv(heldout_dir / f"{policy_name}_comparison_timestep_metrics.csv"),
            "summary": load_json(heldout_dir / f"{policy_name}_comparison_summary.json"),
        }
    return runs


def load_suite_outputs() -> dict[str, pd.DataFrame]:
    suite_dir = data_root() / "generated" / "evaluation_suites_small"
    return {
        "test_id_summary": load_csv(suite_dir / "test_id_summary.csv"),
        "paired_test_results": load_csv(suite_dir / "paired_test_results.csv"),
        "test_stress_summary": load_csv(suite_dir / "test_stress_summary.csv"),
        "scenario_manifest": load_csv(suite_dir / "scenario_manifest.csv"),
        "validation_summary": load_csv(suite_dir / "validation_summary.csv", allow_empty=True),
    }


def load_reward_sweep_summary() -> pd.DataFrame:
    return load_csv(data_root() / "historical" / "reward_sweep_summary.csv")


def load_generated_reward_sweeps() -> dict[str, pd.DataFrame]:
    sweep_dir = data_root() / "generated" / "reward_sweeps"
    return {
        "mcs_cost": load_csv(sweep_dir / "mcs_cost_sweep_summary.csv", allow_empty=True),
        "queue_cost": load_csv(sweep_dir / "queue_cost_sweep_summary.csv", allow_empty=True),
    }


def current_training_history_path() -> Path:
    return repo_root() / "outputs" / "mobile_mcs_line_abc" / "rl" / "history.json"


def copied_training_history_path() -> Path:
    return data_root() / "generated" / "training" / "history_current.json"


def load_training_history() -> tuple[list[dict[str, Any]], str]:
    copied_path = copied_training_history_path()
    if copied_path.exists():
        payload = load_json(copied_path)
        return list(payload.get("history", [])), "submission_copy"

    current_path = current_training_history_path()
    if current_path.exists():
        payload = load_json(current_path)
        return list(payload.get("history", [])), "current_outputs"

    payload = load_json(data_root() / "historical" / "legacy_training_history_range_c24_q2p5.json")
    return list(payload.get("history", [])), "historical"


def load_legacy_training_history() -> list[dict[str, Any]]:
    payload = load_json(data_root() / "historical" / "legacy_training_history_range_c24_q2p5.json")
    return list(payload.get("history", []))

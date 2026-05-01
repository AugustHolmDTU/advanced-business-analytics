from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def submission_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return submission_root() / "data"


def config_root() -> Path:
    return submission_root() / "configs"


def load_json(path: str | Path, allow_missing: bool = False) -> dict[str, Any]:
    target = Path(path)
    if allow_missing and not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def load_yaml(path: str | Path, allow_missing: bool = False) -> dict[str, Any]:
    target = Path(path)
    if allow_missing and not target.exists():
        return {}
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
    cfg = config_root()
    return {
        "environment": load_yaml(cfg / "env" / "mobile_mcs_line_abc.yaml"),
        "rl_simple": load_yaml(cfg / "rl" / "dqn_mobile_simple.yaml"),
        "experiment": load_yaml(cfg / "experiment" / "mobile_mcs_line_abc.yaml"),
        "heldout": load_yaml(cfg / "experiment" / "mobile_mcs_line_abc_heldout_comparison.yaml"),
    }


def load_heldout_runs() -> dict[str, pd.DataFrame]:
    heldout_dir = data_root() / "generated" / "heldout"
    filename_by_policy = {
        "mobile_noop": "heldout_no_agent_baseline_timestep_metrics.csv",
        "rl": "heldout_rl_agent_timestep_metrics.csv",
        "mobile_reactive": "heldout_reactive_timestep_metrics.csv",
    }
    runs: dict[str, pd.DataFrame] = {}
    for policy_name, filename in filename_by_policy.items():
        # reactive is optional now, so an empty frame is fine if it is missing
        runs[policy_name] = load_csv(heldout_dir / filename, allow_empty=True)
    return runs


def load_generated_reward_sweeps() -> dict[str, pd.DataFrame]:
    sweep_dir = data_root() / "generated" / "reward_sweeps"
    return {
        "mcs_cost": load_csv(sweep_dir / "reward_sweep_mcs_cost.csv", allow_empty=True),
        "queue_cost": load_csv(sweep_dir / "reward_sweep_queue_cost.csv", allow_empty=True),
    }


def current_training_history_path() -> Path:
    return submission_root() / "demo_outputs" / "history.json"


def copied_training_history_path() -> Path:
    return data_root() / "generated" / "training" / "training_history.json"


def load_training_history() -> tuple[list[dict[str, Any]], str]:
    copied_path = copied_training_history_path()
    if copied_path.exists():
        # prefer the copied history so the submission stays self-contained
        payload = load_json(copied_path)
        return list(payload.get("history", [])), "submission_copy"

    current_path = current_training_history_path()
    if current_path.exists():
        # fallback in case someone deleted the copied file but still has outputs/
        payload = load_json(current_path)
        return list(payload.get("history", [])), "current_outputs"

    raise FileNotFoundError("No copied or current training history was available for exam_submission.")

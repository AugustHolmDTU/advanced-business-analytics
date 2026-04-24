from __future__ import annotations

import copy
import os
from itertools import product
from pathlib import Path
from typing import Any

from evch.config.loader import build_config_parser, load_config
from evch.train.train_rl import run_training
from evch.utils.io import ensure_dir, write_json


def _set_nested_value(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = payload
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _sanitize_value(raw_value: Any) -> str:
    return str(raw_value).replace(".", "p")


def build_full_grid_variants(base_config: dict[str, Any], sweep_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep_name = str(sweep_cfg.get("name", "mobile_mcs_sweep"))
    experiment_name = str(base_config["experiment"]["name"])
    variants: list[dict[str, Any]] = []
    parameters = dict(sweep_cfg.get("parameters", {}))
    parameter_keys = list(parameters.keys())
    parameter_values = [list(values) for values in parameters.values()]
    for combination_index, combination in enumerate(product(*parameter_values)):
        variant = copy.deepcopy(base_config)
        combination_tokens: list[str] = []
        combination_map: dict[str, Any] = {}
        for dotted_key, raw_value in zip(parameter_keys, combination, strict=True):
            _set_nested_value(variant, dotted_key, raw_value)
            combination_map[dotted_key] = raw_value
            combination_tokens.append(f"{dotted_key.replace('.', '_')}_{_sanitize_value(raw_value)}")

        variant["experiment"]["name"] = f"{experiment_name}_grid_{combination_index:04d}"
        wandb_cfg = ((variant.get("logging") or {}).get("wandb") or {})
        if wandb_cfg:
            wandb_cfg["group"] = sweep_name
            tags = list(wandb_cfg.get("tags", []))
            tags.extend(["sweep", "full_grid"])
            wandb_cfg["tags"] = sorted(set(tags))
        variant["experiment"]["save_plots"] = False
        variant["sweep_run"] = {
            "sweep_name": sweep_name,
            "mode": "full_grid",
            "combination_index": combination_index,
            "parameters": combination_map,
            "label": "|".join(combination_tokens),
        }
        variant["_sweep_metadata"] = {
            "sweep_name": sweep_name,
            "mode": "full_grid",
            "combination_index": combination_index,
            "parameters": combination_map,
            "label": "|".join(combination_tokens),
        }
        variants.append(variant)
    return variants


def main() -> None:
    parser = build_config_parser("Run one-at-a-time sweeps for the mobile MCS RL setup.")
    args = parser.parse_args()
    config = load_config(args.config)
    sweep_cfg = dict(config.get("sweep", {}))
    variants = build_full_grid_variants(config, sweep_cfg)
    start_index = max(int(os.getenv("SWEEP_START_INDEX", "0")), 0)
    end_index = int(os.getenv("SWEEP_END_INDEX", str(len(variants))))
    selected_variants = variants[start_index:max(start_index, min(end_index, len(variants)))]

    results: list[dict[str, Any]] = []
    summary_dir = ensure_dir(Path(config["experiment"]["output_root"]) / str(sweep_cfg.get("name", "mobile_mcs_sweep")))
    for variant in selected_variants:
        metadata = dict(variant.pop("_sweep_metadata"))
        result = run_training(variant)
        results.append(
            {
                "run_name": variant["experiment"]["name"],
                "sweep_name": metadata["sweep_name"],
                "mode": metadata["mode"],
                "combination_index": metadata["combination_index"],
                "label": metadata["label"],
                "parameters": metadata["parameters"],
                "output_dir": result["output_dir"],
                "checkpoint_path": result["checkpoint_path"],
                "mean_reward": float(result["evaluation"]["mean_reward"]),
                "mean_served_demand": float(result["evaluation"]["mean_served_demand"]),
                "mean_unmet_demand": float(result["evaluation"]["mean_unmet_demand"]),
            }
        )

    write_json(
        summary_dir / "sweep_results.json",
        {
            "sweep_name": str(sweep_cfg.get("name", "mobile_mcs_sweep")),
            "mode": "full_grid",
            "total_grid_runs": len(variants),
            "start_index": start_index,
            "end_index": start_index + len(results),
            "num_runs": len(results),
            "results": results,
        },
    )


if __name__ == "__main__":
    main()

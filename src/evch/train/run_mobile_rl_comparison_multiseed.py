from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.train.train_rl import _build_mobile_comparison_rollout, _make_rl_policy
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import DummyRun, init_wandb, log_artifact

LOGGER = logging.getLogger(__name__)


def _resolve_seed_indices(seed_index_start: int, seed_count: int, explicit_indices: str) -> list[int]:
    if explicit_indices.strip():
        return [int(value.strip()) for value in explicit_indices.split(",") if value.strip()]
    return list(range(max(seed_index_start, 0), max(seed_index_start, 0) + max(seed_count, 0)))


def _aggregate_summary(summary_frame: pd.DataFrame) -> dict[str, Any]:
    numeric_frame = summary_frame.select_dtypes(include="number")
    aggregate: dict[str, Any] = {"num_rollouts": int(len(summary_frame))}
    for column in numeric_frame.columns:
        aggregate[f"{column}_mean"] = float(numeric_frame[column].mean())
        aggregate[f"{column}_std"] = float(numeric_frame[column].std(ddof=0))
        aggregate[f"{column}_min"] = float(numeric_frame[column].min())
        aggregate[f"{column}_max"] = float(numeric_frame[column].max())
    return aggregate


def run_mobile_rl_comparison_multiseed(
    config: dict[str, Any],
    checkpoint_path: str,
    seed_index_start: int,
    seed_count: int,
    explicit_seed_indices: str = "",
    output_subdir: str = "mobile_rl_comparison_multiseed",
) -> dict[str, Any]:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / output_subdir)
    run = init_wandb(
        config=config,
        job_type="mobile_rl_comparison_multiseed",
        run_name=f"{experiment_cfg['name']}_{output_subdir}",
    )
    run.log({"comparison/policy_name": "rl", "comparison/multiseed": 1})

    backend = "sb3_dqn" if checkpoint_path.endswith(".zip") else "torch_dqn"
    policy = _make_rl_policy(backend=backend, checkpoint_path=checkpoint_path)
    seed_indices = _resolve_seed_indices(
        seed_index_start=seed_index_start,
        seed_count=seed_count,
        explicit_indices=explicit_seed_indices,
    )

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for seed_index in seed_indices:
        per_seed_config = copy.deepcopy(config)
        rollout_cfg = dict(per_seed_config.get("comparison_rollout", {}))
        rollout_cfg["enabled"] = True
        rollout_cfg["seed_source"] = "test_id"
        rollout_cfg["seed_index"] = int(seed_index)
        rollout_cfg["use_seeded_episode"] = True
        per_seed_config["comparison_rollout"] = rollout_cfg

        per_seed_dir = ensure_dir(output_dir / f"seed_index_{int(seed_index):03d}")
        result = _build_mobile_comparison_rollout(
            config=per_seed_config,
            policy=policy,
            output_dir=per_seed_dir,
            run=DummyRun(),
        )
        if result is None:
            raise RuntimeError("Comparison rollout is not enabled for this environment/config.")
        summary = dict(result["summary"])
        summary["seed_index"] = int(seed_index)
        summary["checkpoint_path"] = str(checkpoint_path)
        rows.append(summary)
        artifacts.append(
            {
                "seed_index": int(seed_index),
                "metrics_path": str(result["metrics_path"]),
                "summary_path": str(result["summary_path"]),
            }
        )

    summary_frame = pd.DataFrame(rows).sort_values("seed_index").reset_index(drop=True)
    per_seed_csv = output_dir / "per_seed_summary.csv"
    summary_frame.to_csv(per_seed_csv, index=False)

    aggregate = _aggregate_summary(summary_frame)
    aggregate["seed_indices"] = [int(value) for value in summary_frame["seed_index"].tolist()]
    aggregate["checkpoint_path"] = str(checkpoint_path)
    aggregate["output_subdir"] = output_subdir

    aggregate_json = output_dir / "aggregate_summary.json"
    aggregate_csv = output_dir / "aggregate_summary.csv"
    write_json(aggregate_json, aggregate)
    pd.DataFrame([aggregate]).to_csv(aggregate_csv, index=False)
    write_json(output_dir / "artifacts.json", {"artifacts": artifacts})

    run.log({f"comparison_multiseed/{key}": value for key, value in aggregate.items() if isinstance(value, (int, float))})
    log_artifact(run, per_seed_csv, f"{experiment_cfg['name']}-comparison-multiseed-per-seed", "metrics", aliases=["latest"])
    log_artifact(run, aggregate_json, f"{experiment_cfg['name']}-comparison-multiseed-summary", "metrics", aliases=["latest"])
    log_artifact(run, aggregate_csv, f"{experiment_cfg['name']}-comparison-multiseed-summary-csv", "metrics", aliases=["latest"])
    run.finish()

    return {
        "per_seed_csv": str(per_seed_csv),
        "aggregate_json": str(aggregate_json),
        "aggregate_csv": str(aggregate_csv),
        "aggregate": aggregate,
    }


def main() -> None:
    parser = build_config_parser("Run the held-out mobile RL comparison over multiple test_id seeds for an existing checkpoint.")
    parser.add_argument("--agent-checkpoint", required=True, default="")
    parser.add_argument("--seed-index-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--seed-indices", default="")
    parser.add_argument("--output-subdir", default="mobile_rl_comparison_multiseed")
    args = parser.parse_args()
    config = load_config(args.config)
    run_mobile_rl_comparison_multiseed(
        config=config,
        checkpoint_path=str(args.agent_checkpoint),
        seed_index_start=int(args.seed_index_start),
        seed_count=int(args.seed_count),
        explicit_seed_indices=str(args.seed_indices),
        output_subdir=str(args.output_subdir),
    )


if __name__ == "__main__":
    main()

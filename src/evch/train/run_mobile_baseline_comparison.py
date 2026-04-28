from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evch.baselines.policies import BASELINE_POLICIES
from evch.config.loader import build_config_parser, load_config
from evch.train.train_rl import _build_mobile_comparison_rollout
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def run_mobile_baseline_comparison(config: dict[str, Any], policy_name: str) -> dict[str, Any] | None:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    if policy_name not in BASELINE_POLICIES:
        raise ValueError(f"Unknown baseline policy: {policy_name}")

    builder = BASELINE_POLICIES[policy_name]
    policy = builder(seed=seed) if policy_name == "random" else builder  # type: ignore[misc]

    experiment_cfg = config["experiment"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / f"{policy_name}_comparison")
    run = init_wandb(
        config=config,
        job_type=f"{policy_name}_comparison",
        run_name=f"{experiment_cfg['name']}_{policy_name}_comparison",
    )
    run.log({"comparison/policy_name": policy_name})

    result = _build_mobile_comparison_rollout(
        config=config,
        policy=policy,
        output_dir=output_dir,
        run=run,
    )
    if result is None:
        LOGGER.warning("Comparison rollout is not enabled for this environment/config.")
    else:
        write_json(output_dir / "comparison_run.json", {"policy": policy_name, "result": result})

    run.finish()
    return result


def main() -> None:
    parser = build_config_parser("Run the fixed mobile MCS comparison rollout with a named baseline policy.")
    parser.add_argument("--policy-name", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_mobile_baseline_comparison(config, policy_name=str(args.policy_name))


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evch.baselines.policies import mobile_reactive_policy
from evch.config.loader import build_config_parser, load_config
from evch.train.train_rl import _build_mobile_comparison_rollout
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def run_mobile_reactive_comparison(config: dict[str, Any]) -> dict[str, Any] | None:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "mobile_reactive_comparison")
    run = init_wandb(
        config=config,
        job_type="mobile_reactive_comparison",
        run_name=f"{experiment_cfg['name']}_mobile_reactive_comparison",
    )
    run.log({"comparison/policy_name": "mobile_reactive"})

    result = _build_mobile_comparison_rollout(
        config=config,
        policy=mobile_reactive_policy,
        output_dir=output_dir,
        run=run,
    )
    if result is None:
        LOGGER.warning("Comparison rollout is not enabled for this environment/config.")
    else:
        write_json(output_dir / "comparison_run.json", {"policy": "mobile_reactive", "result": result})

    run.finish()
    return result


def main() -> None:
    parser = build_config_parser("Run the fixed mobile MCS comparison rollout with a simple reactive baseline policy.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_mobile_reactive_comparison(config)


if __name__ == "__main__":
    main()

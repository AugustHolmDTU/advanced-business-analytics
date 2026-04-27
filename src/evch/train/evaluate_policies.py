from __future__ import annotations

import logging
from pathlib import Path

from evch.config.loader import build_config_parser, load_config
from evch.train.eval_suites import evaluate_policy_suites
from evch.utils.io import ensure_dir
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = build_config_parser("Evaluate RL and heuristic policies.")
    parser.add_argument("--agent-checkpoint", required=False, default="")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "evaluation")
    run = init_wandb(config=config, job_type="eval", run_name=f"{experiment_cfg['name']}_eval")

    suite_outputs = evaluate_policy_suites(config=config, checkpoint_path=str(args.agent_checkpoint), run=run, output_dir=output_dir)
    LOGGER.info("Saved policy evaluation suite results to %s", suite_outputs.get("output_dir", output_dir))
    run.finish()


if __name__ == "__main__":
    main()

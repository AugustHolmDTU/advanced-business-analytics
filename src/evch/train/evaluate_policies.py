from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from evch.baselines.policies import BASELINE_POLICIES
from evch.config.loader import build_config_parser, load_config
from evch.envs.factory import make_env
from evch.rl.evaluation import evaluate_policy
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def _maybe_plot_policy_comparison(frame: pd.DataFrame, path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.bar(frame["policy"], frame["mean_reward"])
        plt.ylabel("Mean cumulative reward")
        plt.title("Policy comparison")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping policy comparison plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def _load_rl_policy(checkpoint_path: str) -> Callable[[Any, Any, bool], int]:
    if checkpoint_path.endswith(".zip"):
        from stable_baselines3 import DQN  # type: ignore

        model = DQN.load(checkpoint_path)

        def policy(observation: Any, _env: Any, deterministic: bool = True) -> int:
            action, _ = model.predict(observation, deterministic=deterministic)
            return int(action)

        return policy

    agent = SimpleDQNAgent.load(checkpoint_path)

    def policy(observation: Any, _env: Any, deterministic: bool = True) -> int:
        return agent.act(observation, deterministic=deterministic, env=_env)

    return policy


def main() -> None:
    parser = build_config_parser("Evaluate RL and heuristic policies.")
    parser.add_argument("--agent-checkpoint", required=False, default="")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    evaluation_cfg = config["evaluation"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "evaluation")
    run = init_wandb(config=config, job_type="eval", run_name=f"{experiment_cfg['name']}_eval")

    results: list[dict[str, float | str]] = []
    for policy_name in evaluation_cfg["policies"]:
        env = make_env(config["environment"], config["demand"], seed=seed + 101)
        if policy_name == "rl":
            if not args.agent_checkpoint:
                LOGGER.warning("Skipping RL evaluation because --agent-checkpoint was not provided.")
                continue
            policy = _load_rl_policy(args.agent_checkpoint)
        else:
            builder = BASELINE_POLICIES[policy_name]
            policy = builder(seed=seed) if policy_name == "random" else builder  # type: ignore[misc]
        summary = evaluate_policy(
            env=env,
            policy=policy,
            episodes=int(evaluation_cfg["episodes"]),
            seed=seed,
            deterministic=True,
        )
        record = {
            "policy": policy_name,
            "mean_reward": summary["mean_reward"],
            "mean_served_demand": summary["mean_served_demand"],
            "mean_unmet_demand": summary["mean_unmet_demand"],
            "mean_true_demand_total": summary["mean_true_demand_total"],
        }
        results.append(record)
        run.log({f"eval/{policy_name}/{key}": value for key, value in record.items() if key != "policy"})

    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "policy_comparison.csv", index=False)
    write_json(output_dir / "policy_comparison.json", {"results": results})
    if not frame.empty:
        _maybe_plot_policy_comparison(frame, output_dir / "policy_comparison.png")
    LOGGER.info("Saved policy evaluation results to %s", output_dir)
    run.finish()


if __name__ == "__main__":
    main()

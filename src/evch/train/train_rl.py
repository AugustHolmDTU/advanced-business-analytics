from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from evch.config.loader import build_config_parser, load_config
from evch.envs.charging_env import ChargingPlacementEnv
from evch.rl.evaluation import evaluate_policy
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb

LOGGER = logging.getLogger(__name__)


def _train_with_sb3(env: Any, rl_cfg: dict[str, Any], seed: int, output_dir: Path) -> tuple[str, str]:
    from stable_baselines3 import DQN  # type: ignore

    sb3_cfg = rl_cfg["sb3"]
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(rl_cfg["learning_rate"]),
        buffer_size=int(sb3_cfg["buffer_size"]),
        learning_starts=int(sb3_cfg["learning_starts"]),
        batch_size=int(sb3_cfg["batch_size"]),
        gamma=float(rl_cfg["gamma"]),
        tau=float(sb3_cfg["tau"]),
        target_update_interval=int(sb3_cfg["target_update_interval"]),
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=int(sb3_cfg["total_timesteps"]))
    checkpoint = output_dir / "best_model.zip"
    model.save(checkpoint)
    return "sb3_dqn", str(checkpoint)


def _train_with_torch_dqn(env: Any, rl_cfg: dict[str, Any], seed: int, output_dir: Path, run: Any) -> tuple[str, str, list[dict[str, float]]]:
    agent = SimpleDQNAgent(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        config=rl_cfg,
        seed=seed,
    )
    history = agent.train(
        env=env,
        episodes=int(rl_cfg["episodes"]),
        max_steps=int(rl_cfg["max_steps_per_episode"]),
        run=run,
    )
    checkpoint = output_dir / str(rl_cfg["checkpoint_name"])
    agent.save(checkpoint)
    return "torch_dqn", str(checkpoint), history


def _make_rl_policy(backend: str, checkpoint_path: str) -> Callable[[np.ndarray, Any, bool], int]:
    if backend == "sb3_dqn":
        from stable_baselines3 import DQN  # type: ignore

        model = DQN.load(checkpoint_path)

        def policy(observation: np.ndarray, _env: Any, deterministic: bool = True) -> int:
            action, _ = model.predict(observation, deterministic=deterministic)
            return int(action)

        return policy

    agent = SimpleDQNAgent.load(checkpoint_path)

    def policy(observation: np.ndarray, _env: Any, deterministic: bool = True) -> int:
        return agent.act(observation, deterministic=deterministic)

    return policy


def main() -> None:
    parser = build_config_parser("Train the EV charger placement RL baseline.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    rl_cfg = config["rl"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "rl")
    run = init_wandb(config=config, job_type="train_rl", run_name=f"{experiment_cfg['name']}_rl")

    env = ChargingPlacementEnv(config["environment"], config["demand"], seed=seed)
    use_sb3 = False
    try:
        import stable_baselines3  # noqa: F401

        use_sb3 = rl_cfg.get("backend", "auto") == "sb3_dqn"
    except ImportError:
        use_sb3 = False

    history: list[dict[str, float]] = []
    if use_sb3:
        backend, checkpoint_path = _train_with_sb3(env, rl_cfg, seed=seed, output_dir=output_dir)
    else:
        backend, checkpoint_path, history = _train_with_torch_dqn(env, rl_cfg, seed=seed, output_dir=output_dir, run=run)

    eval_env = ChargingPlacementEnv(config["environment"], config["demand"], seed=seed + 17)
    policy = _make_rl_policy(backend, checkpoint_path)
    evaluation = evaluate_policy(
        env=eval_env,
        policy=policy,
        episodes=int(rl_cfg["evaluation_episodes"]),
        seed=seed,
        deterministic=bool(rl_cfg.get("deterministic_eval", True)),
    )
    run.log({f"rl_eval/{key}": value for key, value in evaluation.items() if not isinstance(value, list)})

    if history:
        plt.figure(figsize=(8, 4))
        plt.plot([entry["reward"] for entry in history], label="Episode reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Torch DQN training curve")
        plt.tight_layout()
        plt.savefig(output_dir / "training_curve.png", dpi=180)
        plt.close()

    write_json(
        output_dir / "training_summary.json",
        {
            "backend": backend,
            "checkpoint_path": checkpoint_path,
            "evaluation": {key: value for key, value in evaluation.items() if key != "episodes"},
        },
    )
    if history:
        write_json(output_dir / "history.json", {"history": history})
    LOGGER.info("Finished RL training with backend=%s checkpoint=%s", backend, checkpoint_path)
    run.finish()


if __name__ == "__main__":
    main()


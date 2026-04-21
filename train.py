from __future__ import annotations

import argparse
from collections import deque
from typing import Any

import numpy as np

try:
    import wandb  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled explicitly in _init_run
    wandb = None

from agent.dqn import DQNAgent, Transition
from env.highway_env import EnvConfig, HighwayChargingEnv


class _NoOpRun:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.summary: dict[str, Any] = {}

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        _ = data
        _ = step

    def finish(self) -> None:
        return


def _init_run(total_steps: int, seed: int, wandb_mode: str) -> Any:
    run_config = {
        "env": "HighwayChargingEnv",
        "algorithm": "DQN",
        "fixed_capacity": 100,
        "mobile_capacity": 50,
        "alpha": 2.0,
        "beta": 10,
        "episode_length": 24,
        "total_steps": total_steps,
        "seed": seed,
    }
    if wandb_mode != "disabled" and wandb is None:
        raise RuntimeError(
            "wandb is required for logging in online/offline mode. "
            "Install it with: pip install wandb"
        )
    if wandb is None:
        return _NoOpRun(config=run_config)
    return wandb.init(project="ev_resilience_rl", mode=wandb_mode, config=run_config)


def _hist(values: np.ndarray) -> Any:
    if wandb is None:
        return values.tolist()
    return wandb.Histogram(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal DQN for EV charging resilience.")
    parser.add_argument("--total-steps", type=int, default=5000, help="Total environment interaction steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode.",
    )
    return parser.parse_args()


def train(total_steps: int, seed: int, wandb_mode: str) -> None:
    config = EnvConfig(
        fixed_station_capacity=100.0,
        mobile_charger_capacity=50.0,
        alpha=2.0,
        beta=10.0,
        episode_length=24,
    )

    run = _init_run(total_steps=total_steps, seed=seed, wandb_mode=wandb_mode)

    env = HighwayChargingEnv(config=config, seed=seed)
    agent = DQNAgent(
        state_dim=4,
        action_dim=3,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_size=10000,
        target_update_interval=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=total_steps,
        seed=seed,
    )

    global_step = 0
    episode = 0
    best_reward = -float("inf")

    reward_window: deque[float] = deque(maxlen=100)
    unmet_window: deque[float] = deque(maxlen=100)

    while global_step < total_steps:
        state, _ = env.reset(seed=seed + episode)
        done = False

        episode_reward = 0.0
        episode_unmet = 0.0
        episode_actions: list[int] = []
        episode_demands: list[float] = []

        while not done and global_step < total_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=float(done),
                )
            )
            agent.env_steps += 1
            loss = agent.train_step()

            reward_window.append(float(reward))
            unmet_window.append(float(info["unmet_demand"]))
            episode_reward += float(reward)
            episode_unmet += float(info["unmet_demand"])
            episode_actions.append(int(action))
            episode_demands.append(float(info["demand"]))

            logs = {
                "rl/reward": float(reward),
                "rl/served_demand": float(info["served_demand"]),
                "rl/unmet_demand": float(info["unmet_demand"]),
                "rl/action": int(action),
                "rl/epsilon": float(agent.epsilon()),
                "env/demand": float(info["demand"]),
                "env/capacity": float(info["capacity"]),
                "env/time": float(info["time_of_day"]),
                "env/disruption": int(info["disruption_flag"]),
                "env/utilization": float(info["utilization"]),
                "metrics/reward_rolling_100": float(np.mean(reward_window)) if reward_window else 0.0,
                "metrics/unmet_moving_avg_100": float(np.mean(unmet_window)) if unmet_window else 0.0,
            }
            if loss is not None:
                logs["rl/loss"] = float(loss)
            if int(info["disruption_flag"]) == 1:
                logs["event/disruption_triggered"] = 1
            if int(action) == 2:
                logs["event/max_action"] = 1

            run.log(logs, step=global_step)

            state = next_state
            global_step += 1

        action_counts = np.bincount(np.array(episode_actions, dtype=np.int32), minlength=3)
        action_freq = action_counts / max(1, len(episode_actions))

        run.log(
            {
                "episode/index": episode,
                "episode/total_reward": float(episode_reward),
                "episode/total_unmet_demand": float(episode_unmet),
                "episode/action_hist": _hist(np.array(episode_actions, dtype=np.int32)),
                "episode/demand_hist": _hist(np.array(episode_demands, dtype=np.float32)),
                "episode/action_freq_0": float(action_freq[0]),
                "episode/action_freq_1": float(action_freq[1]),
                "episode/action_freq_2": float(action_freq[2]),
            },
            step=global_step,
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            run.summary["best_reward"] = float(best_reward)

        episode += 1

    run.finish()


def main() -> None:
    args = parse_args()
    train(total_steps=args.total_steps, seed=args.seed, wandb_mode=args.wandb_mode)


if __name__ == "__main__":
    main()

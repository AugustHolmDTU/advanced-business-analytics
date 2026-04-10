from __future__ import annotations

from typing import Any, Callable

import numpy as np

PolicyCallable = Callable[[np.ndarray, Any, bool], int]


def evaluate_policy(env: Any, policy: PolicyCallable, episodes: int, seed: int, deterministic: bool = True) -> dict[str, Any]:
    episode_records: list[dict[str, float]] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        totals = {
            "reward": 0.0,
            "served_demand": 0.0,
            "unmet_demand": 0.0,
            "true_demand_total": 0.0,
        }
        while True:
            action = int(policy(observation, env, deterministic))
            observation, reward, terminated, truncated, info = env.step(action)
            totals["reward"] += float(reward)
            totals["served_demand"] += float(info["served_demand"])
            totals["unmet_demand"] += float(info["unmet_demand"])
            totals["true_demand_total"] += float(info["true_demand_total"])
            if terminated or truncated:
                break
        episode_records.append(totals)

    summary = {
        "mean_reward": float(np.mean([record["reward"] for record in episode_records])),
        "mean_served_demand": float(np.mean([record["served_demand"] for record in episode_records])),
        "mean_unmet_demand": float(np.mean([record["unmet_demand"] for record in episode_records])),
        "mean_true_demand_total": float(np.mean([record["true_demand_total"] for record in episode_records])),
        "episodes": episode_records,
    }
    return summary


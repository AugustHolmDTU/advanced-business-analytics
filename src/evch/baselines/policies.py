from __future__ import annotations

from typing import Callable

import numpy as np

from evch.envs.charging_env import ChargingPlacementEnv

PolicyFn = Callable[[np.ndarray, ChargingPlacementEnv, bool], int]


def _prefer_unoccupied(scores: np.ndarray, allocations: np.ndarray, noop_action: int | None = None, relocate_tolerance: float = 0.05) -> int:
    unoccupied = np.flatnonzero(allocations <= 0.0)
    if unoccupied.size > 0:
        occupied = np.flatnonzero(allocations > 0.0)
        if noop_action is not None and occupied.size > 0 and unoccupied.size + occupied.size == allocations.size:
            best_unoccupied = float(np.max(scores[unoccupied]))
            weakest_occupied = float(np.min(scores[occupied]))
            if best_unoccupied <= weakest_occupied * (1.0 + relocate_tolerance):
                return int(noop_action)
        best_local = unoccupied[np.argmax(scores[unoccupied])]
        return int(best_local)
    if noop_action is not None:
        return int(noop_action)
    return int(np.argmax(scores))


def make_random_policy(seed: int = 0) -> PolicyFn:
    rng = np.random.default_rng(seed)

    def policy(_obs: np.ndarray, env: ChargingPlacementEnv, _deterministic: bool = True) -> int:
        return int(rng.integers(env.num_sites))

    return policy


def greedy_highest_demand_policy(obs: np.ndarray, env: ChargingPlacementEnv, deterministic: bool = True) -> int:
    del obs, deterministic
    demand = env.last_observed_demand if env.last_observed_demand.sum() > 0.0 else env.expected_zone_demand()
    scores = env.candidate_site_scores(demand)
    return _prefer_unoccupied(scores, env.allocations, noop_action=getattr(env, "noop_action", None))


def coverage_policy(obs: np.ndarray, env: ChargingPlacementEnv, deterministic: bool = True) -> int:
    del obs, deterministic
    occupied = np.flatnonzero(env.allocations > 0.0)
    if occupied.size == 0:
        return int(np.argmax(env.static_site_scores))
    min_travel = env.city.travel_time_matrix[:, occupied].min(axis=1)
    uncovered_pressure = env.city.zone_base_demand * min_travel
    coverage_scores = (np.exp(-env.service_decay * env.city.travel_time_matrix).T @ uncovered_pressure).astype(np.float32)
    return _prefer_unoccupied(coverage_scores, env.allocations, noop_action=getattr(env, "noop_action", None))


BASELINE_POLICIES: dict[str, PolicyFn | Callable[..., PolicyFn]] = {
    "greedy": greedy_highest_demand_policy,
    "coverage": coverage_policy,
    "random": make_random_policy,
}

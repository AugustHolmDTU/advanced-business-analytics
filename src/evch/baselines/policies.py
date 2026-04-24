from __future__ import annotations

from math import ceil
from typing import Any, Callable

import numpy as np

PolicyFn = Callable[[np.ndarray, Any, bool], int]


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

    def policy(_obs: np.ndarray, env: Any, _deterministic: bool = True) -> int:
        action_mask = getattr(env, "valid_action_mask", lambda: None)()
        if action_mask is not None:
            valid_indices = np.flatnonzero(np.asarray(action_mask, dtype=bool))
            if valid_indices.size > 0:
                return int(rng.choice(valid_indices))
        return int(rng.integers(env.action_space.n))

    return policy


def greedy_highest_demand_policy(obs: np.ndarray, env: Any, deterministic: bool = True) -> int:
    del obs, deterministic
    demand = env.last_observed_demand if env.last_observed_demand.sum() > 0.0 else env.expected_zone_demand()
    scores = env.candidate_site_scores(demand)
    return _prefer_unoccupied(scores, env.allocations, noop_action=getattr(env, "noop_action", None))


def coverage_policy(obs: np.ndarray, env: Any, deterministic: bool = True) -> int:
    del obs, deterministic
    occupied = np.flatnonzero(env.allocations > 0.0)
    if occupied.size == 0:
        return int(np.argmax(env.static_site_scores))
    min_travel = env.city.travel_time_matrix[:, occupied].min(axis=1)
    uncovered_pressure = env.city.zone_base_demand * min_travel
    coverage_scores = (np.exp(-env.service_decay * env.city.travel_time_matrix).T @ uncovered_pressure).astype(np.float32)
    return _prefer_unoccupied(coverage_scores, env.allocations, noop_action=getattr(env, "noop_action", None))


def mobile_threshold_policy(obs: np.ndarray, env: Any, deterministic: bool = True) -> int:
    del obs, deterministic
    if (
        hasattr(env, "expected_vehicle_arrivals_by_station")
        and hasattr(env, "current_service_capacity_per_step_by_station")
        and hasattr(env, "action_from_mobile_station_allocation")
    ):
        expected_by_station = np.asarray(env.expected_vehicle_arrivals_by_station(), dtype=np.float32)
        if hasattr(env, "queue_lengths_by_station"):
            expected_by_station = expected_by_station + np.asarray(env.queue_lengths_by_station, dtype=np.float32)
        base_capacity_by_station = np.asarray(env.current_service_capacity_per_step_by_station(), dtype=np.float32)
        mobile_capacity = max(
            float(env.mobile_station_chargers) * float(getattr(env, "planning_step_minutes", 60.0)) / max(float(getattr(env, "mean_service_minutes", 30.0)), 1e-6),
            1e-6,
        )
        deficits = np.maximum(expected_by_station - base_capacity_by_station, 0.0)
        allocation = np.zeros_like(deficits, dtype=np.int32)
        remaining = int(getattr(env, "max_mobile_stations", 0))
        while remaining > 0 and float(deficits.max()) > 0.0:
            station_index = int(np.argmax(deficits))
            allocation[station_index] += 1
            deficits[station_index] = max(float(deficits[station_index]) - mobile_capacity, 0.0)
            remaining -= 1
        return int(env.action_from_mobile_station_allocation(allocation.tolist()))

    if hasattr(env, "expected_vehicle_arrivals") and hasattr(env, "current_service_capacity_per_step"):
        expected_total = float(env.expected_vehicle_arrivals()) + float(getattr(env, "queue_length", 0.0))
        base_capacity = float(env.current_service_capacity_per_step(service_time_multiplier=1.0, effective_base_plugs=getattr(env, "base_station_plugs", 0)))
        mobile_capacity = max(
            float(env.mobile_station_chargers) * float(getattr(env, "planning_step_minutes", 60.0)) / max(float(getattr(env, "mean_service_minutes", 30.0)), 1e-6),
            1e-6,
        )
    else:
        expected_total = (
            float(env.expected_total_demand()) if hasattr(env, "expected_total_demand") else float(getattr(env, "last_expected_total_demand", 0.0))
        )
        base_capacity = float(getattr(env, "base_station_capacity", 0.0))
        mobile_capacity = max(float(getattr(env, "mobile_station_capacity", 1.0)), 1e-6)
    max_mobile_stations = int(getattr(env, "max_mobile_stations", 0))
    required_mobile_stations = max(0, ceil((expected_total - base_capacity) / mobile_capacity))
    return int(min(required_mobile_stations, max_mobile_stations))


def mobile_noop_policy(obs: np.ndarray, env: Any, deterministic: bool = True) -> int:
    del obs, env, deterministic
    return 0


BASELINE_POLICIES: dict[str, PolicyFn | Callable[..., PolicyFn]] = {
    "greedy": greedy_highest_demand_policy,
    "coverage": coverage_policy,
    "mobile_threshold": mobile_threshold_policy,
    "mobile_noop": mobile_noop_policy,
    "random": make_random_policy,
}

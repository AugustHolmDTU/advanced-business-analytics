from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evch.data.city import build_city
from evch.data.synthetic import SyntheticCity, static_site_accessibility_scores
from evch.envs.demand import DemandGenerator

try:  # pragma: no cover - covered indirectly when gymnasium is installed
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - local fallback for minimal execution
    class _Env:
        metadata: dict[str, Any] = {}

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ChargingPlacementEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> None:
        self.env_config = env_config
        self.demand_config = demand_config
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.city: SyntheticCity = build_city(env_config=env_config, demand_config=demand_config, seed=seed)
        self.num_sites = int(self.city.site_coords.shape[0])
        self.num_zones = int(self.city.zone_coords.shape[0])
        self.max_chargers = min(int(env_config["max_chargers"]), self.num_sites)
        self.charger_capacity = float(env_config["charger_capacity"])
        self.horizon = int(env_config["horizon"])
        self.max_steps = int(env_config.get("max_steps", self.horizon))
        self.service_decay = float(env_config["service_decay"])
        self.demand_generator = DemandGenerator(self.city, demand_config, horizon=self.horizon, seed=seed)
        self.static_site_scores = static_site_accessibility_scores(self.city, self.service_decay)

        obs_dim = 2 * self.num_sites + 2 * self.num_zones + 3
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_sites)

        self.allocations = np.zeros(self.num_sites, dtype=np.float32)
        self.site_availability = np.ones(self.num_sites, dtype=np.float32)
        self.last_observed_demand = np.zeros(self.num_zones, dtype=np.float32)
        self.last_true_demand = np.zeros(self.num_zones, dtype=np.float32)
        self.step_index = 0
        self.is_weekend = False

    def seed(self, seed: int | None = None) -> None:
        chosen_seed = self.base_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(chosen_seed)
        self.demand_generator = DemandGenerator(self.city, self.demand_config, horizon=self.horizon, seed=chosen_seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.seed(seed)
        self.step_index = 0
        self.is_weekend = bool(self.rng.random() < (2.0 / 7.0))
        self.site_availability = np.ones(self.num_sites, dtype=np.float32)
        self.last_observed_demand = np.zeros(self.num_zones, dtype=np.float32)
        self.last_true_demand = np.zeros(self.num_zones, dtype=np.float32)
        self.allocations = np.zeros(self.num_sites, dtype=np.float32)

        if bool(self.env_config.get("start_filled", False)):
            ranked = np.argsort(self.static_site_scores)[::-1][: self.max_chargers]
            self.allocations[ranked] = 1.0

        info = {
            "is_weekend": self.is_weekend,
            "num_active_chargers": int(self.allocations.sum()),
        }
        return self._get_observation(), info

    def _get_observation(self) -> np.ndarray:
        time_fraction = float(self.step_index % self.horizon) / float(max(self.horizon - 1, 1))
        time_sin = np.sin(2.0 * np.pi * time_fraction)
        time_cos = np.cos(2.0 * np.pi * time_fraction)
        obs = np.concatenate(
            [
                self.allocations,
                self.site_availability,
                self.last_observed_demand,
                self.city.zone_base_demand,
                np.asarray([time_sin, time_cos, float(self.is_weekend)], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def expected_zone_demand(self) -> np.ndarray:
        return self.demand_generator.expected_lambda(step=self.step_index, is_weekend=self.is_weekend)

    def candidate_site_scores(self, demand_vector: np.ndarray | None = None) -> np.ndarray:
        demand = self.expected_zone_demand() if demand_vector is None else demand_vector
        weights = np.exp(-self.service_decay * self.city.travel_time_matrix)
        return (weights.T @ demand).astype(np.float32)

    def _pick_relocation_source(self, target_site: int) -> int | None:
        occupied = np.flatnonzero(self.allocations > 0.0)
        eligible = occupied[occupied != target_site]
        if eligible.size == 0:
            return None
        scores = self.static_site_scores[eligible]
        return int(eligible[int(np.argmin(scores))])

    def _apply_action(self, action: int) -> tuple[int, int]:
        deployed = 0
        relocated = 0
        target = int(action)
        total_allocated = int(self.allocations.sum())
        if self.allocations[target] > 0.0:
            return deployed, relocated
        if total_allocated < self.max_chargers:
            self.allocations[target] = 1.0
            deployed = 1
            return deployed, relocated
        source = self._pick_relocation_source(target)
        if source is None:
            return deployed, relocated
        self.allocations[source] = 0.0
        self.allocations[target] = 1.0
        relocated = 1
        return deployed, relocated

    def _sample_disruptions(self) -> tuple[float, int | None, float]:
        cfg = self.env_config.get("disruption", {})
        demand_spike_multiplier = 1.0
        outage_site = None
        extra_noise_scale = 0.0
        if self.rng.random() < float(cfg.get("demand_spike_probability", 0.0)):
            demand_spike_multiplier = float(cfg.get("demand_spike_multiplier", 1.0))
        if self.rng.random() < float(cfg.get("outage_probability", 0.0)):
            occupied = np.flatnonzero(self.allocations > 0.0)
            if occupied.size > 0:
                outage_site = int(self.rng.choice(occupied))
            else:
                outage_site = int(self.rng.integers(self.num_sites))
        extra_noise_scale = float(cfg.get("observation_noise_scale", 0.0))
        return demand_spike_multiplier, outage_site, extra_noise_scale

    def _serve_demand(self, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        remaining_capacity = self.allocations * self.site_availability * self.charger_capacity
        served = np.zeros(self.num_zones, dtype=np.float32)
        weights = np.exp(-self.service_decay * self.city.travel_time_matrix)
        zone_order = np.argsort(demand)[::-1]
        for zone_idx in zone_order:
            remaining_demand = float(demand[zone_idx])
            site_order = np.argsort(weights[zone_idx])[::-1]
            for site_idx in site_order:
                if remaining_capacity[site_idx] <= 0.0:
                    continue
                access = float(weights[zone_idx, site_idx])
                if access <= 1e-6:
                    continue
                effective_capacity = remaining_capacity[site_idx] * access
                served_here = min(remaining_demand, effective_capacity)
                served[zone_idx] += served_here
                remaining_capacity[site_idx] -= served_here / access
                remaining_demand -= served_here
                if remaining_demand <= 1e-6:
                    break
        unmet = np.clip(demand - served, 0.0, None).astype(np.float32)
        return served, unmet

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        deployed, relocated = self._apply_action(int(action))
        demand_spike_multiplier, outage_site, extra_noise_scale = self._sample_disruptions()
        self.site_availability = np.ones(self.num_sites, dtype=np.float32)
        if outage_site is not None:
            self.site_availability[outage_site] = 0.0

        snapshot = self.demand_generator.sample(
            step=self.step_index,
            is_weekend=self.is_weekend,
            demand_spike_multiplier=demand_spike_multiplier,
            extra_noise_scale=extra_noise_scale,
        )
        served, unmet = self._serve_demand(snapshot.true_demand)
        self.last_true_demand = snapshot.true_demand
        self.last_observed_demand = snapshot.observed_demand

        served_total = float(served.sum())
        unmet_total = float(unmet.sum())
        reward = (
            served_total
            - float(self.env_config["unmet_penalty"]) * unmet_total
            - float(self.env_config["deployment_cost"]) * deployed
            - float(self.env_config["relocation_cost"]) * relocated
            - float(self.env_config["outage_penalty"]) * int(outage_site is not None)
        )

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        info = {
            "served_demand": served_total,
            "unmet_demand": unmet_total,
            "true_demand_total": float(snapshot.true_demand.sum()),
            "expected_demand_total": float(snapshot.expected_lambda.sum()),
            "num_active_chargers": int(self.allocations.sum()),
            "deployed": deployed,
            "relocated": relocated,
            "outage_site": outage_site,
            "demand_spike_multiplier": demand_spike_multiplier,
            "reward": reward,
        }
        return self._get_observation(), float(reward), terminated, False, info

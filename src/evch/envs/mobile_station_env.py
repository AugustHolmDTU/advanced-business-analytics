from __future__ import annotations

from typing import Any

import numpy as np

from evch.data.city import build_city
from evch.envs.charging_env import gym, spaces
from evch.envs.demand import DemandGenerator


class MobileStationChargingEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> None:
        self.env_config = env_config
        self.demand_config = demand_config
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.horizon = int(env_config["horizon"])
        self.max_steps = int(env_config.get("max_steps", self.horizon))
        self.randomize_on_reset = bool(env_config.get("randomize_on_reset", False))
        self.reset_seed_stride = int(env_config.get("reset_seed_stride", 97))
        self.reset_counter = 0

        self.base_station_plugs = int(env_config.get("base_station_plugs", 12))
        self.base_plug_capacity = float(env_config.get("base_plug_capacity", 1.0))
        self.base_station_capacity = float(
            env_config.get("base_station_capacity", self.base_station_plugs * self.base_plug_capacity)
        )
        self.max_mobile_stations = int(env_config.get("max_mobile_stations", 10))
        self.mobile_station_chargers = int(env_config.get("mobile_station_chargers", 2))
        self.mobile_station_capacity = float(env_config.get("mobile_station_capacity", 20.0))
        self.reward_scale = float(env_config.get("reward_scale", 1.0))

        reward_cfg = env_config.get("reward", {})
        self.served_reward_weight = float(reward_cfg.get("served_reward_weight", 1.0))
        self.unmet_penalty = float(reward_cfg.get("unmet_penalty", 2.5))
        self.active_mobile_station_cost = float(reward_cfg.get("active_mobile_station_cost", 14.0))
        self.activation_cost = float(reward_cfg.get("activation_cost", 4.0))
        self.adjustment_cost = float(reward_cfg.get("adjustment_cost", 1.0))
        self.idle_capacity_penalty = float(reward_cfg.get("idle_capacity_penalty", 0.15))
        self.utilization_bonus = float(reward_cfg.get("utilization_bonus", 1.0))

        self.city = build_city(self.env_config, self.demand_config, seed=seed)
        self.demand_generator = DemandGenerator(self.city, demand_config, horizon=self.horizon, seed=seed)

        max_capacity = self.base_station_capacity + self.max_mobile_stations * self.mobile_station_capacity
        self.demand_normalizer = float(env_config.get("demand_normalizer", max(max_capacity, 1.0)))
        self.capacity_normalizer = float(env_config.get("capacity_normalizer", max(max_capacity, 1.0)))

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_mobile_stations + 1)

        self.current_mobile_stations = 0
        self.last_observed_total_demand = 0.0
        self.last_true_total_demand = 0.0
        self.last_expected_total_demand = 0.0
        self.last_unmet_demand = 0.0
        self.last_utilization = 0.0
        self.step_index = 0
        self.is_weekend = False

    def seed(self, seed: int | None = None) -> None:
        chosen_seed = self.base_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(chosen_seed)
        if self.randomize_on_reset:
            self.city = build_city(self.env_config, self.demand_config, seed=chosen_seed)
        self.demand_generator = DemandGenerator(self.city, self.demand_config, horizon=self.horizon, seed=chosen_seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        episode_seed = self.base_seed + self.reset_counter * self.reset_seed_stride if seed is None else int(seed)
        self.reset_counter += 1
        self.seed(episode_seed)

        self.step_index = 0
        self.is_weekend = bool(self.rng.random() < (2.0 / 7.0))
        self.current_mobile_stations = 0
        self.last_observed_total_demand = 0.0
        self.last_true_total_demand = 0.0
        self.last_expected_total_demand = 0.0
        self.last_unmet_demand = 0.0
        self.last_utilization = 0.0
        return self._get_observation(), {"num_active_mobile_stations": 0, "is_weekend": self.is_weekend}

    def expected_total_demand(self) -> float:
        expected = self.demand_generator.expected_lambda(step=self.step_index, is_weekend=self.is_weekend)
        return float(np.sum(expected))

    def current_total_capacity(self) -> float:
        return float(self.base_station_capacity + self.current_mobile_stations * self.mobile_station_capacity)

    def valid_action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def _normalized(self, value: float, scale: float) -> float:
        return float(value) / max(scale, 1e-6)

    def _get_observation(self) -> np.ndarray:
        time_fraction = float(self.step_index % self.horizon) / float(max(self.horizon - 1, 1))
        time_sin = np.sin(2.0 * np.pi * time_fraction)
        time_cos = np.cos(2.0 * np.pi * time_fraction)
        expected_total = self.expected_total_demand()
        current_capacity = self.current_total_capacity()
        expected_gap_to_base = max(expected_total - self.base_station_capacity, 0.0)
        expected_gap_to_current = max(expected_total - current_capacity, 0.0)
        obs = np.asarray(
            [
                self._normalized(self.last_observed_total_demand, self.demand_normalizer),
                self._normalized(self.last_expected_total_demand, self.demand_normalizer),
                self._normalized(expected_total, self.demand_normalizer),
                self._normalized(self.last_unmet_demand, self.demand_normalizer),
                self._normalized(self.base_station_capacity, self.capacity_normalizer),
                self._normalized(current_capacity, self.capacity_normalizer),
                self._normalized(expected_gap_to_base, self.capacity_normalizer),
                self._normalized(expected_gap_to_current, self.capacity_normalizer),
                float(self.current_mobile_stations) / float(max(self.max_mobile_stations, 1)),
                time_sin,
                time_cos,
                float(self.is_weekend),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        chosen_mobile_stations = int(np.clip(action, 0, self.max_mobile_stations))
        previous_mobile_stations = self.current_mobile_stations
        self.current_mobile_stations = chosen_mobile_stations

        snapshot = self.demand_generator.sample(step=self.step_index, is_weekend=self.is_weekend)
        total_true_demand = float(np.sum(snapshot.true_demand))
        total_expected_demand = float(np.sum(snapshot.expected_lambda))
        total_observed_demand = float(np.sum(snapshot.observed_demand))

        total_capacity = self.current_total_capacity()
        served_total = min(total_true_demand, total_capacity)
        unmet_total = max(total_true_demand - served_total, 0.0)
        idle_capacity = max(total_capacity - total_true_demand, 0.0)
        utilization = served_total / max(total_capacity, 1e-6)

        activated = max(self.current_mobile_stations - previous_mobile_stations, 0)
        adjusted = abs(self.current_mobile_stations - previous_mobile_stations)
        reward = (
            self.served_reward_weight * served_total
            - self.unmet_penalty * unmet_total
            - self.active_mobile_station_cost * float(self.current_mobile_stations)
            - self.activation_cost * float(activated)
            - self.adjustment_cost * float(adjusted)
            - self.idle_capacity_penalty * idle_capacity
            + self.utilization_bonus * utilization
        )
        reward *= self.reward_scale

        self.last_true_total_demand = total_true_demand
        self.last_observed_total_demand = total_observed_demand
        self.last_expected_total_demand = total_expected_demand
        self.last_unmet_demand = unmet_total
        self.last_utilization = utilization

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        info = {
            "served_demand": served_total,
            "unmet_demand": unmet_total,
            "true_demand_total": total_true_demand,
            "expected_demand_total": total_expected_demand,
            "num_active_mobile_stations": int(self.current_mobile_stations),
            "num_active_chargers": int(self.base_station_plugs + self.current_mobile_stations * self.mobile_station_chargers),
            "base_capacity_total": float(self.base_station_capacity),
            "mobile_capacity_total": float(self.current_mobile_stations * self.mobile_station_capacity),
            "effective_capacity_total": total_capacity,
            "action_valid": True,
            "utilization": utilization,
            "activated_mobile_stations": int(activated),
            "adjusted_mobile_stations": int(adjusted),
            "idle_capacity": idle_capacity,
            "reward": reward,
        }
        return self._get_observation(), float(reward), terminated, False, info

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(slots=True)
class EnvConfig:
    fixed_station_capacity: float = 100.0
    mobile_charger_capacity: float = 50.0
    alpha: float = 2.0
    beta: float = 10.0
    episode_length: int = 24
    disruption_probability: float = 0.1
    demand_spike_multiplier: float = 1.5
    capacity_drop_fraction: float = 0.3
    demand_noise_std: float = 6.0


class HighwayChargingEnv(gym.Env):
    """Minimal EV charging resilience environment on a 1D highway."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        # State: [current_demand, current_capacity, time_of_day_normalized, disruption_flag]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([300.0, 300.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self.hour = 0
        self.current_demand = 0.0
        self.current_capacity = self.config.fixed_station_capacity
        self.disruption_flag = 0

    def _time_of_day_normalized(self) -> float:
        return float(self.hour) / float(max(self.config.episode_length - 1, 1))

    def _baseline_demand(self, hour: int) -> float:
        # Shifted sinusoid to peak in the evening around 18:00.
        phase = (2.0 * np.pi * (hour - 12)) / 24.0
        return 70.0 + 30.0 * float(np.sin(phase))

    def _sample_demand(self, hour: int) -> float:
        demand = self._baseline_demand(hour)
        demand += float(self.rng.normal(0.0, self.config.demand_noise_std))
        return max(0.0, demand)

    def _sample_disruption(self) -> tuple[int, str | None]:
        if float(self.rng.random()) >= self.config.disruption_probability:
            return 0, None

        disruption_type = "demand_spike" if float(self.rng.random()) < 0.5 else "capacity_drop"
        return 1, disruption_type

    def _state(self) -> np.ndarray:
        return np.array(
            [
                self.current_demand,
                self.current_capacity,
                self._time_of_day_normalized(),
                float(self.disruption_flag),
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.hour = 0
        self.disruption_flag = 0
        self.current_capacity = self.config.fixed_station_capacity
        self.current_demand = self._sample_demand(self.hour)

        return self._state(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        base_capacity = self.config.fixed_station_capacity
        disruption_flag, disruption_type = self._sample_disruption()

        demand = self._sample_demand(self.hour)
        if disruption_type == "demand_spike":
            demand *= self.config.demand_spike_multiplier

        capacity = base_capacity + action * self.config.mobile_charger_capacity
        if disruption_type == "capacity_drop":
            capacity *= 1.0 - self.config.capacity_drop_fraction

        served_demand = min(demand, capacity)
        unmet_demand = max(0.0, demand - capacity)

        action_cost = self.config.beta * float(action)
        reward = served_demand - self.config.alpha * unmet_demand - action_cost

        self.current_demand = float(demand)
        self.current_capacity = float(capacity)
        self.disruption_flag = int(disruption_flag)

        info = {
            "demand": float(demand),
            "capacity": float(capacity),
            "served_demand": float(served_demand),
            "unmet_demand": float(unmet_demand),
            "time_of_day": float(self.hour),
            "disruption_flag": int(disruption_flag),
            "disruption_type": disruption_type,
            "utilization": float(demand / capacity) if capacity > 0.0 else 0.0,
        }

        self.hour += 1
        terminated = self.hour >= self.config.episode_length
        truncated = False

        return self._state(), float(reward), terminated, truncated, info

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SimpleEnvConfig:
    days: int = 3
    base_demand: float = 40.0
    scale: float = 80.0
    noise_std: float = 4.0
    morning_peak_hour: float = 8.0
    afternoon_peak_hour: float = 17.0

    @property
    def total_steps(self) -> int:
        return int(self.days * 24)


class SimpleDemandEnv:
    """Minimal simulator for hourly EV charging demand at a highway bottleneck."""

    def __init__(self, config: SimpleEnvConfig | None = None, seed: int = 42) -> None:
        self.config = config or SimpleEnvConfig()
        self.rng = np.random.default_rng(seed)
        self.current_hour = 0

    def reset(self) -> None:
        self.current_hour = 0

    def _demand_without_noise(self, hour_of_day: int) -> float:
        morning_peak = np.exp(-((hour_of_day - self.config.morning_peak_hour) ** 2) / (2.0 * 1.5**2))
        afternoon_peak = np.exp(-((hour_of_day - self.config.afternoon_peak_hour) ** 2) / (2.0 * 2.0**2))
        return float(self.config.base_demand + self.config.scale * (morning_peak + afternoon_peak))

    def step(self) -> dict[str, float | int]:
        hour_of_day = int(self.current_hour % 24)
        day_index = int(self.current_hour // 24)

        demand = self._demand_without_noise(hour_of_day)
        demand += float(self.rng.normal(0.0, self.config.noise_std))
        demand = max(0.0, demand)

        obs = {
            "global_hour": int(self.current_hour),
            "hour_of_day": hour_of_day,
            "day_index": day_index,
            "demand": float(demand),
        }

        self.current_hour += 1
        return obs

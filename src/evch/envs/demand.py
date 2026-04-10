from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evch.data.features import build_feature_record, records_to_frame
from evch.data.synthetic import SyntheticCity


@dataclass(slots=True)
class DemandSnapshot:
    expected_lambda: np.ndarray
    true_demand: np.ndarray
    observed_demand: np.ndarray


class DemandGenerator:
    def __init__(self, city: SyntheticCity, config: dict, horizon: int, seed: int) -> None:
        self.city = city
        self.config = config
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def _time_profile(self, hour: int) -> float:
        cfg = self.config
        morning = cfg["morning_peak_weight"] * np.exp(
            -((hour - cfg["morning_peak_hour"]) ** 2) / (2.0 * cfg["peak_width"] ** 2)
        )
        evening = cfg["evening_peak_weight"] * np.exp(
            -((hour - cfg["evening_peak_hour"]) ** 2) / (2.0 * cfg["peak_width"] ** 2)
        )
        return cfg["background_intensity"] + morning + evening

    def expected_lambda(
        self,
        step: int,
        is_weekend: bool = False,
        demand_spike_multiplier: float = 1.0,
    ) -> np.ndarray:
        hour = int(step % self.horizon)
        weekend_factor = self.config["weekend_multiplier"] if is_weekend else self.config["weekday_multiplier"]
        profile = self._time_profile(hour)
        zone_factor = np.clip(1.0 + self.city.zone_scale, 0.2, None)
        lambdas = self.city.zone_base_demand * zone_factor * profile * weekend_factor * demand_spike_multiplier
        return np.clip(lambdas, 0.05, self.config["poisson_clip"]).astype(np.float32)

    def observe(self, true_demand: np.ndarray, extra_noise_scale: float = 0.0) -> np.ndarray:
        noise_std = self.config["observation_noise_std"] * (1.0 + extra_noise_scale)
        observed = true_demand + self.rng.normal(0.0, noise_std, size=true_demand.shape)
        return np.clip(observed, 0.0, None).astype(np.float32)

    def sample(
        self,
        step: int,
        is_weekend: bool = False,
        demand_spike_multiplier: float = 1.0,
        extra_noise_scale: float = 0.0,
    ) -> DemandSnapshot:
        expected = self.expected_lambda(step, is_weekend=is_weekend, demand_spike_multiplier=demand_spike_multiplier)
        sampled = self.rng.poisson(expected).astype(np.float32)
        observed = self.observe(sampled, extra_noise_scale=extra_noise_scale)
        return DemandSnapshot(expected_lambda=expected, true_demand=sampled, observed_demand=observed)

    def generate_supervised_frame(self, num_days: int, include_prev_observation: bool = True) -> pd.DataFrame:
        records: list[dict] = []
        previous_observed = np.zeros(len(self.city.zone_base_demand), dtype=np.float32)
        for day in range(num_days):
            is_weekend = day % 7 in (5, 6)
            for step in range(self.horizon):
                snapshot = self.sample(step=step, is_weekend=is_weekend)
                hour = step % self.horizon
                for zone_id in range(len(self.city.zone_base_demand)):
                    prev_obs = previous_observed[zone_id] if include_prev_observation else 0.0
                    records.append(
                        build_feature_record(
                            city=self.city,
                            zone_id=zone_id,
                            hour=hour,
                            is_weekend=is_weekend,
                            prev_observed_demand=float(prev_obs),
                            expected_lambda=float(snapshot.expected_lambda[zone_id]),
                            observed_demand=float(snapshot.observed_demand[zone_id]),
                            true_demand=float(snapshot.true_demand[zone_id]),
                        )
                    )
                previous_observed = snapshot.observed_demand.copy()
        return records_to_frame(records)


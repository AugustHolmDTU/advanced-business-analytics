from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evch.sim.simple_corridor import SimulationResult


@dataclass(slots=True)
class CityNode:
    name: str
    position_km: float


@dataclass(slots=True)
class ChargingStationNode:
    key: str
    position_km: float
    num_plugs: int


@dataclass(slots=True)
class CorridorTripDefinition:
    key: str
    origin_index: int
    destination_index: int
    station_weights: tuple[float, float]


@dataclass(slots=True)
class QueuedVehicle:
    arrival_step: int
    trip_key: str
    station_index: int
    base_service_minutes: float


@dataclass(slots=True)
class ActiveSession:
    arrival_step: int
    start_step: int
    end_step: int
    trip_key: str
    station_index: int
    service_minutes: float


@dataclass(slots=True)
class LineDisruptionEvent:
    disruption_type: str
    target: str
    day_index: int
    start_step: int
    end_step: int
    start_hour: float
    end_hour: float
    severity: float
    scripted: bool
    effective_num_plugs_by_station: tuple[int, int]
    demand_multipliers_by_trip: dict[str, float]
    service_time_multiplier_by_station: tuple[float, float]


class LineCorridorQueueSimulator:
    SUPPORTED_DISRUPTION_TYPES = (
        "capacity_drop",
        "station_outage",
        "demand_surge",
        "service_time_inflation",
    )

    DISRUPTION_TYPE_CODES = {
        "none": 0,
        "capacity_drop": 1,
        "station_outage": 2,
        "demand_surge": 3,
        "service_time_inflation": 4,
    }

    STATION_KEYS = ("station_ab", "station_bc")
    TRIP_KEYS = ("od_ab", "od_ba", "od_bc", "od_cb", "od_ac", "od_ca")

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.city_names = [str(name) for name in config.get("city_names", ["City A", "City B", "City C"])]
        if len(self.city_names) != 3:
            raise ValueError("simulation.city_names must contain exactly three city labels")

        self.inter_city_distance_km = float(config.get("inter_city_distance_km", 100.0))
        if self.inter_city_distance_km <= 0.0:
            raise ValueError("simulation.inter_city_distance_km must be positive")

        self.cities = [
            CityNode(name=self.city_names[0], position_km=0.0),
            CityNode(name=self.city_names[1], position_km=self.inter_city_distance_km),
            CityNode(name=self.city_names[2], position_km=2.0 * self.inter_city_distance_km),
        ]
        default_station_positions = [0.5 * self.inter_city_distance_km, 1.5 * self.inter_city_distance_km]
        station_positions = [float(value) for value in config.get("station_positions_km", default_station_positions)]
        if len(station_positions) != 2:
            raise ValueError("simulation.station_positions_km must contain exactly two station positions")

        raw_num_plugs = config["num_plugs"]
        if isinstance(raw_num_plugs, (list, tuple)):
            num_plugs = [int(value) for value in raw_num_plugs]
        else:
            num_plugs = [int(raw_num_plugs), int(raw_num_plugs)]
        if len(num_plugs) != 2 or any(value <= 0 for value in num_plugs):
            raise ValueError("simulation.num_plugs must be a positive integer or a length-2 list")

        self.stations = [
            ChargingStationNode(key=self.STATION_KEYS[0], position_km=station_positions[0], num_plugs=num_plugs[0]),
            ChargingStationNode(key=self.STATION_KEYS[1], position_km=station_positions[1], num_plugs=num_plugs[1]),
        ]
        self.num_plugs_by_station = np.asarray(num_plugs, dtype=np.int32)

        self.step_minutes = int(config["step_minutes"])
        self.duration_hours = float(config.get("duration_hours", 24.0))
        self.num_steps = int(round(self.duration_hours * 60.0 / self.step_minutes))
        self.steps_per_day = int(round(24.0 * 60.0 / self.step_minutes))
        self.num_days = max(1, int(np.ceil(self.num_steps / self.steps_per_day)))
        self.stop_probability = float(config["charging_stop_probability"])
        if self.step_minutes <= 0:
            raise ValueError("simulation.step_minutes must be positive")
        if self.num_steps <= 0:
            raise ValueError("simulation.duration_hours must produce at least one step")
        if not 0.0 <= self.stop_probability <= 1.0:
            raise ValueError("simulation.charging_stop_probability must be in [0, 1]")

        service_cfg = config["service_time"]
        self.service_mean_minutes = float(service_cfg["mean_minutes"])
        self.service_std_minutes = float(service_cfg["std_minutes"])
        self.service_min_minutes = float(service_cfg["min_minutes"])
        self.service_max_minutes = float(service_cfg["max_minutes"])
        if self.service_min_minutes <= 0.0 or self.service_max_minutes < self.service_min_minutes:
            raise ValueError("simulation.service_time bounds are invalid")

        traffic_cfg = config["traffic"]
        self.baseline_cars_per_step = float(traffic_cfg["baseline_cars_per_step"])
        self.morning_peak_hour = float(traffic_cfg["morning_peak_hour"])
        self.evening_peak_hour = float(traffic_cfg["evening_peak_hour"])
        self.morning_peak_cars_per_step = float(traffic_cfg["morning_peak_cars_per_step"])
        self.evening_peak_cars_per_step = float(traffic_cfg["evening_peak_cars_per_step"])
        self.midday_bump_hour = float(traffic_cfg.get("midday_bump_hour", 12.5))
        self.midday_bump_cars_per_step = float(traffic_cfg.get("midday_bump_cars_per_step", 0.0))
        self.peak_width_hours = float(traffic_cfg["peak_width_hours"])
        self.directional_bias_amplitude = float(traffic_cfg.get("directional_bias_amplitude", 0.22))
        self.middle_city_share = float(traffic_cfg.get("middle_city_share", 0.34))
        self.long_trip_share = float(traffic_cfg.get("long_trip_share", 0.35))
        self.middle_destination_bias_amplitude = float(
            traffic_cfg.get("middle_destination_bias_amplitude", self.directional_bias_amplitude)
        )
        if self.peak_width_hours <= 0.0:
            raise ValueError("simulation.traffic.peak_width_hours must be positive")

        self.trip_definitions = {
            "od_ab": CorridorTripDefinition("od_ab", 0, 1, (1.0, 0.0)),
            "od_ba": CorridorTripDefinition("od_ba", 1, 0, (1.0, 0.0)),
            "od_bc": CorridorTripDefinition("od_bc", 1, 2, (0.0, 1.0)),
            "od_cb": CorridorTripDefinition("od_cb", 2, 1, (0.0, 1.0)),
            "od_ac": CorridorTripDefinition("od_ac", 0, 2, (0.5, 0.5)),
            "od_ca": CorridorTripDefinition("od_ca", 2, 0, (0.5, 0.5)),
        }

        self.disruption_cfg = dict(config.get("disruption", {}))
        self.disruption_enabled = bool(self.disruption_cfg.get("enabled", False))
        self.disruption_mode = str(self.disruption_cfg.get("mode", "scripted")).lower()
        self.daily_event_probability = float(self.disruption_cfg.get("daily_event_probability", 0.0))
        self.day_disruption_count_weights = self._parse_day_disruption_count_weights(
            self.disruption_cfg.get("day_disruption_count_weights")
        )
        self.event_types = [
            str(event_type)
            for event_type in self.disruption_cfg.get("event_types", list(self.SUPPORTED_DISRUPTION_TYPES))
        ]
        for event_type in self.event_types:
            if event_type not in self.SUPPORTED_DISRUPTION_TYPES:
                raise ValueError(f"Unsupported disruption type: {event_type}")

    def _parse_day_disruption_count_weights(self, raw_value: Any) -> dict[int, float]:
        default_weights = {0: max(1.0 - self.daily_event_probability, 0.0), 1: self.daily_event_probability, 2: 0.0}
        if not isinstance(raw_value, dict) or not raw_value:
            return default_weights
        weights: dict[int, float] = {}
        for key, value in raw_value.items():
            weights[int(key)] = max(float(value), 0.0)
        return weights

    def _time_hours(self, step: int) -> float:
        return step * self.step_minutes / 60.0

    def _gaussian_peak(self, hour: float, center_hour: float, amplitude: float) -> float:
        return amplitude * np.exp(-((hour - center_hour) ** 2) / (2.0 * self.peak_width_hours**2))

    def _origin_shares(self, hour_of_day: float) -> np.ndarray:
        morning_signal = np.exp(-((hour_of_day - self.morning_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
        evening_signal = np.exp(-((hour_of_day - self.evening_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
        midday_signal = np.exp(-((hour_of_day - self.midday_bump_hour) ** 2) / (2.0 * self.peak_width_hours**2))
        middle_share = float(np.clip(self.middle_city_share, 0.1, 0.8))
        edge_share = max(1.0 - middle_share, 1e-6)
        scores = np.asarray(
            [
                edge_share * (1.0 + self.directional_bias_amplitude * (morning_signal - evening_signal)),
                middle_share * (1.0 + 0.35 * midday_signal),
                edge_share * (1.0 + self.directional_bias_amplitude * (evening_signal - morning_signal)),
            ],
            dtype=np.float64,
        )
        scores = np.clip(scores, 1e-6, None)
        return scores / scores.sum()

    def expected_traffic(self, step: int, demand_multipliers_by_trip: dict[str, float] | None = None) -> dict[str, Any]:
        multipliers = {trip_key: 1.0 for trip_key in self.TRIP_KEYS}
        if demand_multipliers_by_trip is not None:
            for trip_key, value in demand_multipliers_by_trip.items():
                if trip_key in multipliers:
                    multipliers[trip_key] = float(value)

        global_hour = self._time_hours(step)
        hour_of_day = global_hour % 24.0
        morning_peak = self._gaussian_peak(hour_of_day, self.morning_peak_hour, self.morning_peak_cars_per_step)
        evening_peak = self._gaussian_peak(hour_of_day, self.evening_peak_hour, self.evening_peak_cars_per_step)
        midday_bump = self._gaussian_peak(hour_of_day, self.midday_bump_hour, self.midday_bump_cars_per_step)
        total_flow = max(self.baseline_cars_per_step + morning_peak + evening_peak + midday_bump, 0.0)

        origin_shares = self._origin_shares(hour_of_day)
        middle_to_east = float(
            np.clip(
                0.5
                + self.middle_destination_bias_amplitude
                * (
                    np.exp(-((hour_of_day - self.morning_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
                    - np.exp(-((hour_of_day - self.evening_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
                ),
                0.15,
                0.85,
            )
        )
        long_trip_share = float(np.clip(self.long_trip_share, 0.05, 0.8))

        expected_by_trip = {
            "od_ab": total_flow * origin_shares[0] * (1.0 - long_trip_share) * multipliers["od_ab"],
            "od_ac": total_flow * origin_shares[0] * long_trip_share * multipliers["od_ac"],
            "od_ba": total_flow * origin_shares[1] * (1.0 - middle_to_east) * multipliers["od_ba"],
            "od_bc": total_flow * origin_shares[1] * middle_to_east * multipliers["od_bc"],
            "od_ca": total_flow * origin_shares[2] * long_trip_share * multipliers["od_ca"],
            "od_cb": total_flow * origin_shares[2] * (1.0 - long_trip_share) * multipliers["od_cb"],
        }
        station_expected_passing = np.zeros(2, dtype=np.float64)
        station_expected_charging = np.zeros(2, dtype=np.float64)
        for trip_key, expected in expected_by_trip.items():
            weights = np.asarray(self.trip_definitions[trip_key].station_weights, dtype=np.float64)
            station_expected_passing += expected * weights
            station_expected_charging += expected * self.stop_probability * weights

        return {
            "global_hour": global_hour,
            "hour_of_day": hour_of_day,
            "total_passing_expected": float(sum(expected_by_trip.values())),
            "expected_passing_total": float(sum(expected_by_trip.values())),
            "expected_by_trip": expected_by_trip,
            "station_expected_passing": station_expected_passing,
            "station_expected_charging": station_expected_charging,
        }

    def _sample_service_minutes(self, service_time_multiplier: float = 1.0, rng: np.random.Generator | None = None) -> float:
        local_rng = self.rng if rng is None else rng
        sampled = local_rng.normal(self.service_mean_minutes * service_time_multiplier, self.service_std_minutes)
        return float(np.clip(sampled, self.service_min_minutes, self.service_max_minutes * service_time_multiplier))

    def _apply_service_time_multiplier(self, base_service_minutes: float, service_time_multiplier: float = 1.0) -> float:
        scaled = float(base_service_minutes) * float(service_time_multiplier)
        return float(np.clip(scaled, self.service_min_minutes, self.service_max_minutes * float(service_time_multiplier)))

    def _step_rng(self, step: int, channel: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, step, channel]))

    def _sample_arrivals(self, step: int, expected_by_trip: dict[str, float]) -> list[QueuedVehicle]:
        arrival_rng = self._step_rng(step, 1)
        assignment_rng = self._step_rng(step, 2)
        service_rng = self._step_rng(step, 3)
        arrivals: list[QueuedVehicle] = []
        for trip_key, expected_passing in expected_by_trip.items():
            passing = int(arrival_rng.poisson(max(expected_passing, 0.0)))
            charging = int(arrival_rng.binomial(passing, self.stop_probability))
            if charging <= 0:
                continue
            weights = np.asarray(self.trip_definitions[trip_key].station_weights, dtype=np.float64)
            station_counts = assignment_rng.multinomial(charging, weights / max(weights.sum(), 1e-6))
            for station_index, count in enumerate(station_counts):
                arrivals.extend(
                    QueuedVehicle(
                        arrival_step=-1,
                        trip_key=trip_key,
                        station_index=station_index,
                        base_service_minutes=self._sample_service_minutes(service_time_multiplier=1.0, rng=service_rng),
                    )
                    for _ in range(int(count))
                )
        if arrivals:
            assignment_rng.shuffle(arrivals)
        return arrivals

    def _steps_from_hours(self, hours: float) -> int:
        return max(1, int(round(hours * 60.0 / self.step_minutes)))

    def _coerce_duration_hours(self, value: Any) -> float:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low = float(value[0])
            high = float(value[1])
            if high < low:
                low, high = high, low
            return float(self.rng.uniform(low, high))
        return float(value)

    def _coerce_numeric_value(self, value: Any, minimum: float | None = None, maximum: float | None = None) -> float:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            low = float(value[0])
            high = float(value[1])
            if high < low:
                low, high = high, low
            sampled = float(self.rng.uniform(low, high))
        else:
            sampled = float(value)
        if minimum is not None:
            sampled = max(sampled, minimum)
        if maximum is not None:
            sampled = min(sampled, maximum)
        return sampled

    def _resolve_start_hour(self, raw_range: Any, duration_hours: float) -> float:
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            low = float(raw_range[0])
            high = float(raw_range[1])
        else:
            low = 7.0
            high = 19.0
        if high < low:
            low, high = high, low
        latest_start = max(low, min(high, 24.0 - duration_hours))
        return float(self.rng.uniform(low, latest_start))

    def _default_targets_for_type(self, disruption_type: str) -> list[str]:
        if disruption_type in {"capacity_drop", "station_outage", "service_time_inflation"}:
            return list(self.STATION_KEYS) + ["all_stations"]
        if disruption_type == "demand_surge":
            return list(self.TRIP_KEYS) + ["eastbound", "westbound", "all_ods"]
        raise ValueError(f"Unsupported disruption type: {disruption_type}")

    def _apply_target_to_trip_multipliers(self, target: str, severity: float) -> dict[str, float]:
        multipliers = {trip_key: 1.0 for trip_key in self.TRIP_KEYS}
        if target in multipliers:
            multipliers[target] = max(severity, 1.0)
            return multipliers
        if target == "eastbound":
            for trip_key in ("od_ab", "od_bc", "od_ac"):
                multipliers[trip_key] = max(severity, 1.0)
            return multipliers
        if target == "westbound":
            for trip_key in ("od_ba", "od_cb", "od_ca"):
                multipliers[trip_key] = max(severity, 1.0)
            return multipliers
        if target == "all_ods":
            for trip_key in multipliers:
                multipliers[trip_key] = max(severity, 1.0)
            return multipliers
        raise ValueError(f"Unsupported demand_surge target: {target}")

    def _make_disruption_event(
        self,
        disruption_type: str,
        target: str,
        day_index: int,
        start_hour: float,
        duration_hours: float,
        severity: float,
        scripted: bool,
    ) -> LineDisruptionEvent:
        duration_hours = float(np.clip(duration_hours, 0.25, 24.0))
        latest_start = max(0.0, 24.0 - duration_hours)
        start_hour = float(np.clip(start_hour, 0.0, latest_start))
        start_step = day_index * self.steps_per_day + int(round(start_hour * 60.0 / self.step_minutes))
        duration_steps = self._steps_from_hours(duration_hours)
        end_step = min(self.num_steps, start_step + duration_steps)
        end_hour = min(24.0, start_hour + duration_steps * self.step_minutes / 60.0)

        effective_num_plugs = self.num_plugs_by_station.copy()
        demand_multipliers = {trip_key: 1.0 for trip_key in self.TRIP_KEYS}
        service_time_multipliers = np.ones(2, dtype=np.float32)

        if disruption_type == "capacity_drop":
            target_num_plugs = int(np.clip(round(severity), 0, int(self.num_plugs_by_station.max())))
            if target == "all_stations":
                effective_num_plugs[:] = target_num_plugs
            elif target in self.STATION_KEYS:
                effective_num_plugs[self.STATION_KEYS.index(target)] = target_num_plugs
            else:
                raise ValueError(f"Unsupported capacity_drop target: {target}")
        elif disruption_type == "station_outage":
            if target == "all_stations":
                effective_num_plugs[:] = 0
            elif target in self.STATION_KEYS:
                effective_num_plugs[self.STATION_KEYS.index(target)] = 0
            else:
                raise ValueError(f"Unsupported station_outage target: {target}")
        elif disruption_type == "service_time_inflation":
            multiplier = max(float(severity), 1.0)
            if target == "all_stations":
                service_time_multipliers[:] = multiplier
            elif target in self.STATION_KEYS:
                service_time_multipliers[self.STATION_KEYS.index(target)] = multiplier
            else:
                raise ValueError(f"Unsupported service_time_inflation target: {target}")
        elif disruption_type == "demand_surge":
            demand_multipliers = self._apply_target_to_trip_multipliers(target, float(severity))
        else:
            raise ValueError(f"Unsupported disruption type: {disruption_type}")

        return LineDisruptionEvent(
            disruption_type=disruption_type,
            target=target,
            day_index=day_index,
            start_step=start_step,
            end_step=end_step,
            start_hour=start_hour,
            end_hour=end_hour,
            severity=float(severity),
            scripted=scripted,
            effective_num_plugs_by_station=(int(effective_num_plugs[0]), int(effective_num_plugs[1])),
            demand_multipliers_by_trip=demand_multipliers,
            service_time_multiplier_by_station=(float(service_time_multipliers[0]), float(service_time_multipliers[1])),
        )

    def _sample_day_disruption_count(self) -> int:
        counts = np.asarray(sorted(self.day_disruption_count_weights.keys()), dtype=np.int32)
        weights = np.asarray([self.day_disruption_count_weights[int(count)] for count in counts], dtype=np.float64)
        if weights.sum() <= 0.0:
            return 0
        probabilities = weights / weights.sum()
        return int(self.rng.choice(counts, p=probabilities))

    def _sample_random_event(self, day_index: int) -> LineDisruptionEvent | None:
        if not self.event_types:
            return None
        disruption_type = str(self.rng.choice(self.event_types))
        if disruption_type == "capacity_drop":
            cfg = dict(self.disruption_cfg.get("capacity_drop", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 18.0]), duration_hours)
            severity = self._coerce_numeric_value(
                cfg.get("target_num_plugs", cfg.get("target_num_plugs_range", 3)),
                minimum=0.0,
                maximum=float(self.num_plugs_by_station.max()),
            )
        elif disruption_type == "station_outage":
            cfg = dict(self.disruption_cfg.get("station_outage", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 1.5))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.5]), duration_hours)
            severity = 0.0
        elif disruption_type == "demand_surge":
            cfg = dict(self.disruption_cfg.get("demand_surge", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.0]), duration_hours)
            severity = self._coerce_numeric_value(cfg.get("multiplier", cfg.get("multiplier_range", 1.8)), minimum=1.0)
        else:
            cfg = dict(self.disruption_cfg.get("service_time_inflation", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.0]), duration_hours)
            severity = self._coerce_numeric_value(cfg.get("multiplier", cfg.get("multiplier_range", 1.5)), minimum=1.0)
        target_options = [str(value) for value in cfg.get("targets", self._default_targets_for_type(disruption_type))]
        target = str(self.rng.choice(target_options))
        return self._make_disruption_event(
            disruption_type=disruption_type,
            target=target,
            day_index=day_index,
            start_hour=start_hour,
            duration_hours=duration_hours,
            severity=severity,
            scripted=False,
        )

    def _sample_random_events_for_day(self, day_index: int) -> list[LineDisruptionEvent]:
        target_count = self._sample_day_disruption_count()
        if target_count <= 0:
            return []
        sampled: list[LineDisruptionEvent] = []
        attempts = 0
        max_attempts = max(24, target_count * 16)
        while len(sampled) < target_count and attempts < max_attempts:
            attempts += 1
            event = self._sample_random_event(day_index)
            if event is None:
                continue
            overlaps_existing = any(event.start_step < existing.end_step and existing.start_step < event.end_step for existing in sampled)
            if overlaps_existing:
                continue
            sampled.append(event)
        sampled.sort(key=lambda event: (event.start_step, event.end_step))
        return sampled

    def _build_disruption_schedule(self) -> list[LineDisruptionEvent]:
        if not self.disruption_enabled:
            return []

        events: list[LineDisruptionEvent] = []
        if self.disruption_mode == "scripted":
            for payload in self.disruption_cfg.get("scripted_events", []):
                disruption_type = str(payload["disruption_type"])
                if disruption_type not in self.SUPPORTED_DISRUPTION_TYPES:
                    raise ValueError(f"Unsupported scripted disruption type: {disruption_type}")
                day_index = int(payload["day_index"])
                if day_index < 0 or day_index >= self.num_days:
                    raise ValueError(f"Scripted disruption day_index out of range: {day_index}")
                events.append(
                    self._make_disruption_event(
                        disruption_type=disruption_type,
                        target=str(payload["target"]),
                        day_index=day_index,
                        start_hour=float(payload["start_hour"]),
                        duration_hours=float(payload["duration_hours"]),
                        severity=float(payload.get("severity", 1.0)),
                        scripted=True,
                    )
                )
        elif self.disruption_mode == "random":
            for day_index in range(self.num_days):
                events.extend(self._sample_random_events_for_day(day_index))
        else:
            raise ValueError(f"Unknown disruption mode: {self.disruption_mode}")

        events.sort(key=lambda event: (event.start_step, event.end_step))
        trimmed = [event for event in events if event.start_step < self.num_steps and event.end_step > event.start_step]
        for previous, current in zip(trimmed, trimmed[1:]):
            if current.start_step < previous.end_step:
                raise ValueError("Overlapping scripted disruptions are not supported.")
        return trimmed

    def _event_map(self, events: list[LineDisruptionEvent]) -> dict[int, LineDisruptionEvent]:
        event_by_step: dict[int, LineDisruptionEvent] = {}
        for event in events:
            for step in range(event.start_step, min(event.end_step, self.num_steps)):
                event_by_step[step] = event
        return event_by_step

    def _disruption_state(self, step: int, event_by_step: dict[int, LineDisruptionEvent]) -> dict[str, Any]:
        event = event_by_step.get(step)
        if event is None:
            return {
                "disruption_active": 0,
                "disruption_type": "none",
                "disruption_type_code": 0,
                "disruption_target": "none",
                "disruption_day_index": -1,
                "disruption_remaining_minutes": 0.0,
                "effective_num_plugs_by_station": self.num_plugs_by_station.copy(),
                "demand_multipliers_by_trip": {trip_key: 1.0 for trip_key in self.TRIP_KEYS},
                "service_time_multiplier_by_station": np.ones(2, dtype=np.float32),
                "scripted_disruption": 0,
            }
        return {
            "disruption_active": 1,
            "disruption_type": event.disruption_type,
            "disruption_type_code": self.DISRUPTION_TYPE_CODES[event.disruption_type],
            "disruption_target": event.target,
            "disruption_day_index": event.day_index,
            "disruption_remaining_minutes": float(max(event.end_step - step, 0) * self.step_minutes),
            "effective_num_plugs_by_station": np.asarray(event.effective_num_plugs_by_station, dtype=np.int32),
            "demand_multipliers_by_trip": dict(event.demand_multipliers_by_trip),
            "service_time_multiplier_by_station": np.asarray(event.service_time_multiplier_by_station, dtype=np.float32),
            "scripted_disruption": int(event.scripted),
        }

    def _requeue_active_sessions(
        self,
        queue_by_station: list[deque[QueuedVehicle]],
        active_sessions_by_station: list[list[ActiveSession]],
        station_index: int,
    ) -> None:
        sessions = active_sessions_by_station[station_index]
        if not sessions:
            return
        for session in sorted(sessions, key=lambda item: (item.arrival_step, item.start_step), reverse=True):
            queue_by_station[station_index].appendleft(
                QueuedVehicle(
                    arrival_step=session.arrival_step,
                    trip_key=session.trip_key,
                    station_index=station_index,
                    base_service_minutes=session.service_minutes,
                )
            )

    def _summary_stats(self, metrics: pd.DataFrame) -> dict[str, float]:
        if metrics.empty:
            return {
                "mean_queue_length": 0.0,
                "mean_wait_minutes": 0.0,
                "mean_utilization": 0.0,
                "total_arrivals": 0.0,
                "total_completions": 0.0,
            }
        return {
            "mean_queue_length": float(metrics["queue_length"].mean()),
            "mean_wait_minutes": float(metrics["queue_wait_mean_minutes"].mean()),
            "mean_utilization": float(metrics["utilization"].mean()),
            "total_arrivals": float(metrics["arrivals_total"].sum()),
            "total_completions": float(metrics["completions_total"].sum()),
        }

    def run(self) -> SimulationResult:
        queue_by_station = [deque(), deque()]
        active_sessions_by_station: list[list[ActiveSession]] = [[], []]
        rows: list[dict[str, Any]] = []
        disruption_schedule = self._build_disruption_schedule()
        disruption_by_step = self._event_map(disruption_schedule)

        for step in range(self.num_steps):
            completed_now_by_station: list[list[ActiveSession]] = [[], []]
            for station_index in range(2):
                completed_now_by_station[station_index] = [
                    session for session in active_sessions_by_station[station_index] if session.end_step <= step
                ]
                active_sessions_by_station[station_index] = [
                    session for session in active_sessions_by_station[station_index] if session.end_step > step
                ]

            disruption_state = self._disruption_state(step, disruption_by_step)
            if disruption_state["disruption_type"] == "station_outage" and disruption_state["disruption_active"] == 1:
                event = disruption_by_step[step]
                if event.start_step == step:
                    targeted = (
                        range(2) if event.target == "all_stations" else [self.STATION_KEYS.index(str(event.target))]
                    )
                    for station_index in targeted:
                        self._requeue_active_sessions(queue_by_station, active_sessions_by_station, station_index)
                        active_sessions_by_station[station_index] = []

            expected = self.expected_traffic(
                step=step,
                demand_multipliers_by_trip=disruption_state["demand_multipliers_by_trip"],
            )
            arrivals = self._sample_arrivals(step=step, expected_by_trip=expected["expected_by_trip"])
            arrivals_by_station = np.zeros(2, dtype=np.int32)
            for vehicle in arrivals:
                vehicle.arrival_step = step
                queue_by_station[vehicle.station_index].append(vehicle)
                arrivals_by_station[vehicle.station_index] += 1

            starts_now_by_station: list[list[ActiveSession]] = [[], []]
            for station_index in range(2):
                effective_num_plugs = int(disruption_state["effective_num_plugs_by_station"][station_index])
                available_plugs = max(effective_num_plugs - len(active_sessions_by_station[station_index]), 0)
                for _ in range(min(available_plugs, len(queue_by_station[station_index]))):
                    vehicle = queue_by_station[station_index].popleft()
                    service_minutes = self._apply_service_time_multiplier(
                        base_service_minutes=vehicle.base_service_minutes,
                        service_time_multiplier=float(disruption_state["service_time_multiplier_by_station"][station_index]),
                    )
                    service_steps = max(1, int(np.ceil(service_minutes / self.step_minutes)))
                    session = ActiveSession(
                        arrival_step=vehicle.arrival_step,
                        start_step=step,
                        end_step=step + service_steps,
                        trip_key=vehicle.trip_key,
                        station_index=station_index,
                        service_minutes=service_minutes,
                    )
                    active_sessions_by_station[station_index].append(session)
                    starts_now_by_station[station_index].append(session)

            queue_lengths = np.asarray([len(queue_by_station[idx]) for idx in range(2)], dtype=np.float32)
            queue_wait_means = []
            for station_index in range(2):
                wait_minutes = [(step - vehicle.arrival_step) * self.step_minutes for vehicle in queue_by_station[station_index]]
                queue_wait_means.append(float(np.mean(wait_minutes)) if wait_minutes else 0.0)
            queue_wait_means = np.asarray(queue_wait_means, dtype=np.float32)
            active_plugs_by_station = np.asarray(
                [len(active_sessions_by_station[idx]) for idx in range(2)],
                dtype=np.float32,
            )
            effective_num_plugs_by_station = np.asarray(disruption_state["effective_num_plugs_by_station"], dtype=np.float32)
            utilization_by_station = np.divide(
                active_plugs_by_station,
                np.maximum(effective_num_plugs_by_station, 1.0),
                out=np.zeros_like(active_plugs_by_station),
                where=effective_num_plugs_by_station > 0,
            )

            rows.append(
                {
                    "step": step,
                    "global_hour": float(expected["global_hour"]),
                    "hour_of_day": float(expected["hour_of_day"]),
                    "day_index": int(expected["global_hour"] // 24.0),
                    "expected_passing_total": float(expected["total_passing_expected"]),
                    "arrivals_total": float(arrivals_by_station.sum()),
                    "starts_total": float(sum(len(items) for items in starts_now_by_station)),
                    "completions_total": float(sum(len(items) for items in completed_now_by_station)),
                    "queue_length": float(queue_lengths.sum()),
                    "queue_wait_mean_minutes": float(queue_wait_means.mean()),
                    "active_plugs": float(active_plugs_by_station.sum()),
                    "effective_num_plugs": float(effective_num_plugs_by_station.sum()),
                    "utilization": float(active_plugs_by_station.sum() / max(effective_num_plugs_by_station.sum(), 1.0)),
                    "queue_length_station_ab": float(queue_lengths[0]),
                    "queue_length_station_bc": float(queue_lengths[1]),
                    "queue_wait_mean_minutes_station_ab": float(queue_wait_means[0]),
                    "queue_wait_mean_minutes_station_bc": float(queue_wait_means[1]),
                    "arrivals_total_station_ab": float(arrivals_by_station[0]),
                    "arrivals_total_station_bc": float(arrivals_by_station[1]),
                    "starts_total_station_ab": float(len(starts_now_by_station[0])),
                    "starts_total_station_bc": float(len(starts_now_by_station[1])),
                    "effective_num_plugs_station_ab": float(effective_num_plugs_by_station[0]),
                    "effective_num_plugs_station_bc": float(effective_num_plugs_by_station[1]),
                    "utilization_station_ab": float(utilization_by_station[0]),
                    "utilization_station_bc": float(utilization_by_station[1]),
                    "expected_station_arrivals_ab": float(expected["station_expected_charging"][0]),
                    "expected_station_arrivals_bc": float(expected["station_expected_charging"][1]),
                    "expected_passing_od_ab": float(expected["expected_by_trip"]["od_ab"]),
                    "expected_passing_od_ba": float(expected["expected_by_trip"]["od_ba"]),
                    "expected_passing_od_bc": float(expected["expected_by_trip"]["od_bc"]),
                    "expected_passing_od_cb": float(expected["expected_by_trip"]["od_cb"]),
                    "expected_passing_od_ac": float(expected["expected_by_trip"]["od_ac"]),
                    "expected_passing_od_ca": float(expected["expected_by_trip"]["od_ca"]),
                    **{
                        "disruption_active": int(disruption_state["disruption_active"]),
                        "disruption_type": str(disruption_state["disruption_type"]),
                        "disruption_type_code": int(disruption_state["disruption_type_code"]),
                        "disruption_target": str(disruption_state["disruption_target"]),
                        "disruption_day_index": int(disruption_state["disruption_day_index"]),
                        "disruption_remaining_minutes": float(disruption_state["disruption_remaining_minutes"]),
                    },
                }
            )

        metrics = pd.DataFrame(rows)
        peak_row = metrics.iloc[int(metrics["queue_length"].idxmax())]
        summary = {
            "num_days": self.num_days,
            "step_minutes": self.step_minutes,
            "city_names": list(self.city_names),
            "station_positions_km": [float(station.position_km) for station in self.stations],
            "num_plugs_by_station": [int(value) for value in self.num_plugs_by_station],
            "total_arrivals": float(metrics["arrivals_total"].sum()),
            "total_starts": float(metrics["starts_total"].sum()),
            "total_completions": float(metrics["completions_total"].sum()),
            "peak_queue_length": float(peak_row["queue_length"]),
            "peak_queue_hour": float(peak_row["global_hour"]),
            "final_queue_length": float(metrics["queue_length"].iloc[-1]),
            "mean_queue_length": float(metrics["queue_length"].mean()),
            "mean_wait_minutes": float(metrics["queue_wait_mean_minutes"].mean()),
            "mean_utilization": float(metrics["utilization"].mean()),
            "non_disruption_baseline": self._summary_stats(metrics.loc[metrics["disruption_active"] == 0]),
            "disruption_metrics": self._summary_stats(metrics.loc[metrics["disruption_active"] == 1]),
            "disruption_event_count": len(disruption_schedule),
        }
        return SimulationResult(metrics=metrics, summary=summary)

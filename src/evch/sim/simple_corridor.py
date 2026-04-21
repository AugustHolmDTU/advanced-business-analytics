from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class QueuedVehicle:
    arrival_step: int
    direction: str


@dataclass(slots=True)
class ActiveSession:
    arrival_step: int
    start_step: int
    end_step: int
    direction: str
    service_minutes: float


@dataclass(slots=True)
class SimulationResult:
    metrics: pd.DataFrame
    summary: dict[str, Any]


class SimpleCorridorQueueSimulator:
    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.road_length_km = float(config.get("road_length_km", 100.0))
        self.station_position_km = float(config.get("station_position_km", 0.5 * self.road_length_km))
        self.num_plugs = int(config["num_plugs"])
        self.step_minutes = int(config["step_minutes"])
        self.duration_hours = float(config.get("duration_hours", 24.0))
        self.num_steps = int(round(self.duration_hours * 60.0 / self.step_minutes))
        self.stop_probability = float(config["charging_stop_probability"])

        if self.num_plugs <= 0:
            raise ValueError("simulation.num_plugs must be positive")
        if self.step_minutes <= 0:
            raise ValueError("simulation.step_minutes must be positive")
        if self.num_steps <= 0:
            raise ValueError("simulation.duration_hours must produce at least one step")
        if not 0.0 <= self.stop_probability <= 1.0:
            raise ValueError("simulation.charging_stop_probability must be in [0, 1]")

        city_names = config.get("city_names", ["City A", "City B"])
        if len(city_names) != 2:
            raise ValueError("simulation.city_names must contain exactly two city labels")
        self.city_a, self.city_b = [str(name) for name in city_names]
        self.direction_ab = f"{self.city_a}_to_{self.city_b}"
        self.direction_ba = f"{self.city_b}_to_{self.city_a}"

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
        self.minimum_direction_share = float(traffic_cfg.get("minimum_direction_share", 0.18))
        if self.peak_width_hours <= 0.0:
            raise ValueError("simulation.traffic.peak_width_hours must be positive")
        if not 0.0 <= self.minimum_direction_share <= 0.5:
            raise ValueError("simulation.traffic.minimum_direction_share must be in [0, 0.5]")

    def _time_hours(self, step: int) -> float:
        return step * self.step_minutes / 60.0

    def _gaussian_peak(self, hour: float, center_hour: float, amplitude: float) -> float:
        return amplitude * np.exp(-((hour - center_hour) ** 2) / (2.0 * self.peak_width_hours**2))

    def expected_traffic(self, step: int) -> dict[str, float]:
        global_hour = self._time_hours(step)
        hour_of_day = global_hour % 24.0
        morning_peak = self._gaussian_peak(hour_of_day, self.morning_peak_hour, self.morning_peak_cars_per_step)
        evening_peak = self._gaussian_peak(hour_of_day, self.evening_peak_hour, self.evening_peak_cars_per_step)
        midday_bump = self._gaussian_peak(hour_of_day, self.midday_bump_hour, self.midday_bump_cars_per_step)
        total_flow = max(self.baseline_cars_per_step + morning_peak + evening_peak + midday_bump, 0.0)

        raw_share_ab = 0.5 + self.directional_bias_amplitude * (
            np.exp(-((hour_of_day - self.morning_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
            - np.exp(-((hour_of_day - self.evening_peak_hour) ** 2) / (2.0 * self.peak_width_hours**2))
        )
        share_ab = float(np.clip(raw_share_ab, self.minimum_direction_share, 1.0 - self.minimum_direction_share))
        share_ba = 1.0 - share_ab
        return {
            "global_hour": global_hour,
            "hour_of_day": hour_of_day,
            "total_passing_expected": total_flow,
            "expected_passing_ab": total_flow * share_ab,
            "expected_passing_ba": total_flow * share_ba,
        }

    def _sample_service_minutes(self) -> float:
        sampled = self.rng.normal(self.service_mean_minutes, self.service_std_minutes)
        return float(np.clip(sampled, self.service_min_minutes, self.service_max_minutes))

    def _sample_arrivals(self, expected_ab: float, expected_ba: float) -> list[QueuedVehicle]:
        passing_ab = int(self.rng.poisson(max(expected_ab, 0.0)))
        passing_ba = int(self.rng.poisson(max(expected_ba, 0.0)))
        charging_ab = int(self.rng.binomial(passing_ab, self.stop_probability))
        charging_ba = int(self.rng.binomial(passing_ba, self.stop_probability))

        directions = [self.direction_ab] * charging_ab + [self.direction_ba] * charging_ba
        if directions:
            self.rng.shuffle(directions)
        return [QueuedVehicle(arrival_step=-1, direction=direction) for direction in directions]

    def run(self) -> SimulationResult:
        queue: deque[QueuedVehicle] = deque()
        active_sessions: list[ActiveSession] = []
        rows: list[dict[str, float | int | str]] = []
        wait_minutes_started: list[float] = []
        wait_minutes_completed: list[float] = []
        total_service_minutes_started = 0.0
        total_service_minutes_completed = 0.0

        for step in range(self.num_steps):
            completed_now = [session for session in active_sessions if session.end_step <= step]
            active_sessions = [session for session in active_sessions if session.end_step > step]
            completed_waits = [(session.start_step - session.arrival_step) * self.step_minutes for session in completed_now]
            wait_minutes_completed.extend(completed_waits)
            total_service_minutes_completed += sum(session.service_minutes for session in completed_now)

            expected = self.expected_traffic(step)
            global_hour = float(expected["global_hour"])
            day_index = int(global_hour // 24.0)
            hour_of_day = float(expected["hour_of_day"])
            arrivals = self._sample_arrivals(
                expected_ab=float(expected["expected_passing_ab"]),
                expected_ba=float(expected["expected_passing_ba"]),
            )

            arrivals_ab = 0
            arrivals_ba = 0
            for vehicle in arrivals:
                vehicle.arrival_step = step
                queue.append(vehicle)
                if vehicle.direction == self.direction_ab:
                    arrivals_ab += 1
                else:
                    arrivals_ba += 1

            starts_now: list[ActiveSession] = []
            available_plugs = self.num_plugs - len(active_sessions)
            for _ in range(min(available_plugs, len(queue))):
                vehicle = queue.popleft()
                service_minutes = self._sample_service_minutes()
                service_steps = max(1, int(np.ceil(service_minutes / self.step_minutes)))
                session = ActiveSession(
                    arrival_step=vehicle.arrival_step,
                    start_step=step,
                    end_step=step + service_steps,
                    direction=vehicle.direction,
                    service_minutes=service_minutes,
                )
                active_sessions.append(session)
                starts_now.append(session)

            started_waits = [(session.start_step - session.arrival_step) * self.step_minutes for session in starts_now]
            wait_minutes_started.extend(started_waits)
            total_service_minutes_started += sum(session.service_minutes for session in starts_now)

            queue_wait_minutes = [(step - vehicle.arrival_step) * self.step_minutes for vehicle in queue]
            rows.append(
                {
                    "step": step,
                    "hour": global_hour,
                    "global_hour": global_hour,
                    "day_index": day_index,
                    "hour_of_day": hour_of_day,
                    "time_label": f"{int(hour_of_day):02d}:{int((hour_of_day % 1.0) * 60):02d}",
                    "expected_passing_total": expected["total_passing_expected"],
                    "expected_passing_ab": expected["expected_passing_ab"],
                    "expected_passing_ba": expected["expected_passing_ba"],
                    "arrivals_total": len(arrivals),
                    "arrivals_ab": arrivals_ab,
                    "arrivals_ba": arrivals_ba,
                    "starts_total": len(starts_now),
                    "starts_ab": sum(session.direction == self.direction_ab for session in starts_now),
                    "starts_ba": sum(session.direction == self.direction_ba for session in starts_now),
                    "completions_total": len(completed_now),
                    "completions_ab": sum(session.direction == self.direction_ab for session in completed_now),
                    "completions_ba": sum(session.direction == self.direction_ba for session in completed_now),
                    "queue_length": len(queue),
                    "queue_wait_mean_minutes": float(np.mean(queue_wait_minutes)) if queue_wait_minutes else 0.0,
                    "queue_wait_max_minutes": float(np.max(queue_wait_minutes)) if queue_wait_minutes else 0.0,
                    "active_plugs": len(active_sessions),
                    "utilization": len(active_sessions) / float(self.num_plugs),
                    "started_wait_mean_minutes": float(np.mean(started_waits)) if started_waits else 0.0,
                    "completed_wait_mean_minutes": float(np.mean(completed_waits)) if completed_waits else 0.0,
                }
            )

        metrics = pd.DataFrame(rows)
        peak_idx = int(metrics["queue_length"].idxmax())
        peak_row = metrics.iloc[peak_idx]
        summary = {
            "road_length_km": self.road_length_km,
            "station_position_km": self.station_position_km,
            "num_plugs": self.num_plugs,
            "step_minutes": self.step_minutes,
            "duration_hours": self.duration_hours,
            "charging_stop_probability": self.stop_probability,
            "total_arrivals": int(metrics["arrivals_total"].sum()),
            "total_starts": int(metrics["starts_total"].sum()),
            "total_completions": int(metrics["completions_total"].sum()),
            "final_queue_length": int(metrics["queue_length"].iloc[-1]),
            "peak_queue_length": int(metrics["queue_length"].max()),
            "peak_queue_time": str(peak_row["time_label"]),
            "mean_queue_length": float(metrics["queue_length"].mean()),
            "mean_utilization": float(metrics["utilization"].mean()),
            "max_utilization": float(metrics["utilization"].max()),
            "mean_wait_started_minutes": float(np.mean(wait_minutes_started)) if wait_minutes_started else 0.0,
            "max_wait_started_minutes": float(np.max(wait_minutes_started)) if wait_minutes_started else 0.0,
            "mean_wait_completed_minutes": float(np.mean(wait_minutes_completed)) if wait_minutes_completed else 0.0,
            "max_wait_completed_minutes": float(np.max(wait_minutes_completed)) if wait_minutes_completed else 0.0,
            "mean_service_time_started_minutes": total_service_minutes_started / max(len(wait_minutes_started), 1),
            "mean_service_time_completed_minutes": total_service_minutes_completed / max(len(wait_minutes_completed), 1),
        }
        return SimulationResult(metrics=metrics, summary=summary)

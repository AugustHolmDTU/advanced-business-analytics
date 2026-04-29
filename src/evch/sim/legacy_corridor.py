from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evch.sim.common import SimulationResult


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
class DisruptionEvent:
    disruption_type: str
    day_index: int
    start_step: int
    end_step: int
    start_hour: float
    end_hour: float
    severity: float
    scripted: bool
    effective_num_plugs: int
    demand_multiplier_ab: float
    demand_multiplier_ba: float
    service_time_multiplier: float


class SimpleCorridorQueueSimulator:
    SUPPORTED_DISRUPTION_TYPES = (
        "capacity_drop",
        "station_outage",
        "demand_surge_ab",
        "demand_surge_ba",
        "service_time_inflation",
    )

    DISRUPTION_TYPE_CODES = {
        "none": 0,
        "capacity_drop": 1,
        "station_outage": 2,
        "demand_surge_ab": 3,
        "demand_surge_ba": 4,
        "service_time_inflation": 5,
    }

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
        self.steps_per_day = int(round(24.0 * 60.0 / self.step_minutes))
        self.num_days = max(1, int(np.ceil(self.num_steps / self.steps_per_day)))
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

        self.disruption_cfg = dict(config.get("disruption", {}))
        self.disruption_enabled = bool(self.disruption_cfg.get("enabled", False))
        self.disruption_mode = str(self.disruption_cfg.get("mode", "random")).lower()
        self.daily_event_probability = float(self.disruption_cfg.get("daily_event_probability", 0.0))
        self.event_types = [
            str(event_type)
            for event_type in self.disruption_cfg.get("event_types", list(self.SUPPORTED_DISRUPTION_TYPES))
        ]
        for event_type in self.event_types:
            if event_type not in self.SUPPORTED_DISRUPTION_TYPES:
                raise ValueError(f"Unsupported disruption type: {event_type}")

    def _time_hours(self, step: int) -> float:
        return step * self.step_minutes / 60.0

    def _gaussian_peak(self, hour: float, center_hour: float, amplitude: float) -> float:
        return amplitude * np.exp(-((hour - center_hour) ** 2) / (2.0 * self.peak_width_hours**2))

    def expected_traffic(
        self,
        step: int,
        demand_multiplier_ab: float = 1.0,
        demand_multiplier_ba: float = 1.0,
    ) -> dict[str, float]:
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
        expected_ab = total_flow * share_ab * demand_multiplier_ab
        expected_ba = total_flow * share_ba * demand_multiplier_ba
        return {
            "global_hour": global_hour,
            "hour_of_day": hour_of_day,
            "total_passing_expected": expected_ab + expected_ba,
            "expected_passing_ab": expected_ab,
            "expected_passing_ba": expected_ba,
        }

    def _sample_service_minutes(self, service_time_multiplier: float = 1.0) -> float:
        sampled = self.rng.normal(self.service_mean_minutes * service_time_multiplier, self.service_std_minutes)
        return float(np.clip(sampled, self.service_min_minutes, self.service_max_minutes * service_time_multiplier))

    def _sample_arrivals(self, expected_ab: float, expected_ba: float) -> list[QueuedVehicle]:
        passing_ab = int(self.rng.poisson(max(expected_ab, 0.0)))
        passing_ba = int(self.rng.poisson(max(expected_ba, 0.0)))
        charging_ab = int(self.rng.binomial(passing_ab, self.stop_probability))
        charging_ba = int(self.rng.binomial(passing_ba, self.stop_probability))

        directions = [self.direction_ab] * charging_ab + [self.direction_ba] * charging_ba
        if directions:
            self.rng.shuffle(directions)
        return [QueuedVehicle(arrival_step=-1, direction=direction) for direction in directions]

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

    def _make_disruption_event(
        self,
        disruption_type: str,
        day_index: int,
        start_hour: float,
        duration_hours: float,
        severity: float,
        scripted: bool,
    ) -> DisruptionEvent:
        duration_hours = float(np.clip(duration_hours, 0.25, 24.0))
        latest_start = max(0.0, 24.0 - duration_hours)
        start_hour = float(np.clip(start_hour, 0.0, latest_start))
        start_step = day_index * self.steps_per_day + int(round(start_hour * 60.0 / self.step_minutes))
        duration_steps = self._steps_from_hours(duration_hours)
        end_step = min(self.num_steps, start_step + duration_steps)
        end_hour = min(24.0, start_hour + duration_steps * self.step_minutes / 60.0)

        effective_num_plugs = self.num_plugs
        demand_multiplier_ab = 1.0
        demand_multiplier_ba = 1.0
        service_time_multiplier = 1.0

        if disruption_type == "capacity_drop":
            effective_num_plugs = int(np.clip(round(severity), 0, self.num_plugs))
        elif disruption_type == "station_outage":
            effective_num_plugs = 0
        elif disruption_type == "demand_surge_ab":
            demand_multiplier_ab = max(float(severity), 1.0)
        elif disruption_type == "demand_surge_ba":
            demand_multiplier_ba = max(float(severity), 1.0)
        elif disruption_type == "service_time_inflation":
            service_time_multiplier = max(float(severity), 1.0)
        else:
            raise ValueError(f"Unsupported disruption type: {disruption_type}")

        return DisruptionEvent(
            disruption_type=disruption_type,
            day_index=day_index,
            start_step=start_step,
            end_step=end_step,
            start_hour=start_hour,
            end_hour=end_hour,
            severity=float(severity),
            scripted=scripted,
            effective_num_plugs=effective_num_plugs,
            demand_multiplier_ab=demand_multiplier_ab,
            demand_multiplier_ba=demand_multiplier_ba,
            service_time_multiplier=service_time_multiplier,
        )

    def _sample_random_event(self, day_index: int) -> DisruptionEvent | None:
        if not self.event_types or self.rng.random() > self.daily_event_probability:
            return None

        disruption_type = str(self.rng.choice(self.event_types))
        if disruption_type == "capacity_drop":
            cfg = dict(self.disruption_cfg.get("capacity_drop", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 18.0]), duration_hours)
            severity = float(cfg.get("target_num_plugs", 3))
        elif disruption_type == "station_outage":
            cfg = dict(self.disruption_cfg.get("station_outage", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 1.5))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.5]), duration_hours)
            severity = 0.0
        elif disruption_type in ("demand_surge_ab", "demand_surge_ba"):
            cfg = dict(self.disruption_cfg.get("demand_surge", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.0]), duration_hours)
            severity = float(cfg.get("multiplier", 1.8))
        elif disruption_type == "service_time_inflation":
            cfg = dict(self.disruption_cfg.get("service_time_inflation", {}))
            duration_hours = self._coerce_duration_hours(cfg.get("duration_hours", 2.0))
            start_hour = self._resolve_start_hour(cfg.get("start_hour_range", [7.0, 19.0]), duration_hours)
            severity = float(cfg.get("multiplier", 1.5))
        else:  # pragma: no cover - guarded earlier
            return None

        return self._make_disruption_event(
            disruption_type=disruption_type,
            day_index=day_index,
            start_hour=start_hour,
            duration_hours=duration_hours,
            severity=severity,
            scripted=False,
        )

    def _build_disruption_schedule(self) -> list[DisruptionEvent]:
        if not self.disruption_enabled:
            return []

        mode = self.disruption_mode
        events: list[DisruptionEvent] = []

        if mode == "scripted":
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
                        day_index=day_index,
                        start_hour=float(payload["start_hour"]),
                        duration_hours=float(payload["duration_hours"]),
                        severity=float(payload.get("severity", 1.0)),
                        scripted=True,
                    )
                )
        elif mode == "random":
            for day_index in range(self.num_days):
                event = self._sample_random_event(day_index)
                if event is not None:
                    events.append(event)
        else:
            raise ValueError(f"Unknown disruption mode: {mode}")

        events.sort(key=lambda event: (event.start_step, event.end_step))
        trimmed = [event for event in events if event.start_step < self.num_steps and event.end_step > event.start_step]
        for previous, current in zip(trimmed, trimmed[1:]):
            if current.start_step < previous.end_step:
                raise ValueError("Overlapping scripted disruptions are not supported.")
        return trimmed

    def _event_map(self, events: list[DisruptionEvent]) -> dict[int, DisruptionEvent]:
        event_by_step: dict[int, DisruptionEvent] = {}
        for event in events:
            for step in range(event.start_step, min(event.end_step, self.num_steps)):
                event_by_step[step] = event
        return event_by_step

    def _disruption_state(self, step: int, event_by_step: dict[int, DisruptionEvent]) -> dict[str, int | float | str]:
        event = event_by_step.get(step)
        if event is None:
            return {
                "disruption_active": 0,
                "disruption_type": "none",
                "disruption_type_code": 0,
                "disruption_day_index": -1,
                "disruption_start_hour": -1.0,
                "disruption_end_hour": -1.0,
                "disruption_remaining_minutes": 0.0,
                "effective_num_plugs": self.num_plugs,
                "demand_multiplier_ab": 1.0,
                "demand_multiplier_ba": 1.0,
                "service_time_multiplier": 1.0,
                "scripted_disruption": 0,
            }

        return {
            "disruption_active": 1,
            "disruption_type": event.disruption_type,
            "disruption_type_code": self.DISRUPTION_TYPE_CODES[event.disruption_type],
            "disruption_day_index": event.day_index,
            "disruption_start_hour": event.start_hour,
            "disruption_end_hour": event.end_hour,
            "disruption_remaining_minutes": float(max(event.end_step - step, 0) * self.step_minutes),
            "effective_num_plugs": event.effective_num_plugs,
            "demand_multiplier_ab": event.demand_multiplier_ab,
            "demand_multiplier_ba": event.demand_multiplier_ba,
            "service_time_multiplier": event.service_time_multiplier,
            "scripted_disruption": int(event.scripted),
        }

    def _requeue_active_sessions(self, queue: deque[QueuedVehicle], active_sessions: list[ActiveSession]) -> None:
        if not active_sessions:
            return
        for session in sorted(active_sessions, key=lambda item: (item.arrival_step, item.start_step), reverse=True):
            queue.appendleft(QueuedVehicle(arrival_step=session.arrival_step, direction=session.direction))

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
        queue: deque[QueuedVehicle] = deque()
        active_sessions: list[ActiveSession] = []
        rows: list[dict[str, float | int | str]] = []
        wait_minutes_started: list[float] = []
        wait_minutes_completed: list[float] = []
        total_service_minutes_started = 0.0
        total_service_minutes_completed = 0.0
        disruption_schedule = self._build_disruption_schedule()
        disruption_by_step = self._event_map(disruption_schedule)

        for step in range(self.num_steps):
            completed_now = [session for session in active_sessions if session.end_step <= step]
            active_sessions = [session for session in active_sessions if session.end_step > step]
            completed_waits = [(session.start_step - session.arrival_step) * self.step_minutes for session in completed_now]
            wait_minutes_completed.extend(completed_waits)
            total_service_minutes_completed += sum(session.service_minutes for session in completed_now)

            disruption_state = self._disruption_state(step, disruption_by_step)
            if (
                disruption_state["disruption_type"] == "station_outage"
                and disruption_state["disruption_active"] == 1
                and disruption_by_step[step].start_step == step
            ):
                self._requeue_active_sessions(queue, active_sessions)
                active_sessions = []

            expected = self.expected_traffic(
                step=step,
                demand_multiplier_ab=float(disruption_state["demand_multiplier_ab"]),
                demand_multiplier_ba=float(disruption_state["demand_multiplier_ba"]),
            )
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
            effective_num_plugs = int(disruption_state["effective_num_plugs"])
            available_plugs = max(effective_num_plugs - len(active_sessions), 0)
            for _ in range(min(available_plugs, len(queue))):
                vehicle = queue.popleft()
                service_minutes = self._sample_service_minutes(
                    service_time_multiplier=float(disruption_state["service_time_multiplier"])
                )
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
            active_plugs = len(active_sessions)
            utilization_denominator = float(max(effective_num_plugs, 1))
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
                    "active_plugs": active_plugs,
                    "utilization": active_plugs / utilization_denominator if effective_num_plugs > 0 else 0.0,
                    "started_service_mean_minutes": float(np.mean([session.service_minutes for session in starts_now]))
                    if starts_now
                    else 0.0,
                    "started_wait_mean_minutes": float(np.mean(started_waits)) if started_waits else 0.0,
                    "completed_wait_mean_minutes": float(np.mean(completed_waits)) if completed_waits else 0.0,
                    **disruption_state,
                }
            )

        metrics = pd.DataFrame(rows)
        peak_idx = int(metrics["queue_length"].idxmax())
        peak_row = metrics.iloc[peak_idx]
        disruption_counts = {
            disruption_type: int(sum(event.disruption_type == disruption_type for event in disruption_schedule))
            for disruption_type in self.SUPPORTED_DISRUPTION_TYPES
        }
        per_type_metrics = {
            disruption_type: self._summary_stats(metrics.loc[metrics["disruption_type"] == disruption_type])
            for disruption_type in self.SUPPORTED_DISRUPTION_TYPES
            if not metrics.loc[metrics["disruption_type"] == disruption_type].empty
        }
        baseline_metrics = self._summary_stats(metrics.loc[metrics["disruption_active"] == 0])

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
            "disruption_enabled": int(self.disruption_enabled),
            "disruption_mode": self.disruption_mode,
            "disruption_event_count": len(disruption_schedule),
            "disruption_counts_by_type": disruption_counts,
            "total_disrupted_minutes": int(metrics["disruption_active"].sum() * self.step_minutes),
            "disruption_metrics_by_type": per_type_metrics,
            "non_disruption_baseline": baseline_metrics,
        }
        return SimulationResult(metrics=metrics, summary=summary)

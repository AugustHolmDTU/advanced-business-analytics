from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evch.data.city import build_city
from evch.envs.charging_env import gym, spaces
from evch.envs.demand import DemandGenerator


@dataclass(slots=True)
class MobileDisruptionEvent:
    disruption_type: str
    day_index: int
    start_step: int
    end_step: int
    severity: float
    effective_base_plugs: int
    demand_multiplier: float
    service_time_multiplier: float


class MobileStationChargingEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    DISRUPTION_TYPE_CODES = {
        "none": 0,
        "capacity_drop": 1,
        "station_outage": 2,
        "demand_surge_ab": 3,
        "demand_surge_ba": 4,
        "service_time_inflation": 5,
    }

    def __init__(self, env_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> None:
        self.env_config = env_config
        self.demand_config = demand_config
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_candidate_sites = int(env_config.get("num_candidate_sites", 1))
        if self.num_candidate_sites != 1:
            raise ValueError("MobileStationChargingEnv supports exactly one fixed charging site for MCS deployment.")

        self.horizon = int(env_config["horizon"])
        self.max_steps = int(env_config.get("max_steps", self.horizon))
        self.randomize_on_reset = bool(env_config.get("randomize_on_reset", False))
        self.reset_seed_stride = int(env_config.get("reset_seed_stride", 97))
        self.reset_counter = 0

        self.base_station_plugs = int(env_config.get("base_station_plugs", 12))
        self.max_mobile_stations = int(env_config.get("max_mobile_stations", 10))
        self.mobile_station_chargers = int(env_config.get("mobile_station_chargers", 2))
        self.mobile_station_capacity = float(env_config.get("mobile_station_capacity", 20.0))
        self.reward_scale = float(env_config.get("reward_scale", 1.0))
        self.planning_step_minutes = float(env_config.get("planning_step_minutes", 60.0))
        self.vehicle_arrival_scale = float(env_config.get("vehicle_arrival_scale", 4.0))
        self.mean_service_minutes = float((env_config.get("service_time") or {}).get("mean_minutes", 30.0))
        self.queue_normalizer = float(env_config.get("queue_normalizer", 100.0))
        self.arrival_normalizer = float(env_config.get("arrival_normalizer", 40.0))
        self.capacity_normalizer = float(env_config.get("capacity_normalizer", self.base_station_plugs + self.max_mobile_stations * self.mobile_station_chargers))

        reward_cfg = env_config.get("reward", {})
        self.served_reward_weight = float(reward_cfg.get("served_reward_weight", 1.0))
        self.unmet_penalty = float(reward_cfg.get("unmet_penalty", 2.0))
        self.active_mobile_station_cost = float(reward_cfg.get("active_mobile_station_cost", 30.0))
        self.activation_cost = float(reward_cfg.get("activation_cost", 8.0))
        self.adjustment_cost = float(reward_cfg.get("adjustment_cost", 3.0))
        self.idle_capacity_penalty = float(reward_cfg.get("idle_capacity_penalty", 0.15))
        self.utilization_bonus = float(reward_cfg.get("utilization_bonus", 1.0))
        self.queue_length_penalty = float(reward_cfg.get("queue_length_penalty", 3.0))
        self.disruption_response_bonus = float(reward_cfg.get("disruption_response_bonus", 0.0))

        self.city = build_city(self.env_config, self.demand_config, seed=seed)
        self.demand_generator = DemandGenerator(self.city, demand_config, horizon=self.horizon, seed=seed)
        self.fixed_site_index = 0
        self.fixed_site_coords_km = self.city.site_coords[self.fixed_site_index].astype(np.float32)

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_mobile_stations + 1)

        self.current_mobile_stations = 0
        self.last_arrivals = 0.0
        self.last_served = 0.0
        self.last_service_capacity = 0.0
        self.last_idle_capacity = 0.0
        self.last_expected_arrivals = 0.0
        self.last_observed_total_demand = 0.0
        self.last_true_total_demand = 0.0
        self.last_expected_total_demand = 0.0
        self.last_unmet_demand = 0.0
        self.last_utilization = 0.0
        self.queue_length = 0.0
        self.queue_wait_mean_minutes = 0.0
        self.current_event: MobileDisruptionEvent | None = None
        self.disruption_by_step: dict[int, MobileDisruptionEvent] = {}
        self.step_index = 0
        self.is_weekend = False

    def seed(self, seed: int | None = None) -> None:
        chosen_seed = self.base_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(chosen_seed)
        if self.randomize_on_reset:
            self.city = build_city(self.env_config, self.demand_config, seed=chosen_seed)
            self.fixed_site_coords_km = self.city.site_coords[self.fixed_site_index].astype(np.float32)
        self.demand_generator = DemandGenerator(self.city, self.demand_config, horizon=self.horizon, seed=chosen_seed)

    def _step_from_hours(self, hours: float) -> int:
        return max(1, int(round(hours * 60.0 / self.planning_step_minutes)))

    def _coerce_event(self, payload: dict[str, Any]) -> MobileDisruptionEvent:
        disruption_type = str(payload["disruption_type"])
        severity = float(payload.get("severity", 1.0))
        day_index = int(payload.get("day_index", 0))
        steps_per_day = int(round(self.horizon * 60.0 / self.planning_step_minutes))
        start_step = day_index * steps_per_day + int(round(float(payload.get("start_hour", 0.0)) * 60.0 / self.planning_step_minutes))
        duration_steps = self._step_from_hours(float(payload.get("duration_hours", 1.0)))
        end_step = min(self.max_steps, start_step + duration_steps)

        effective_base_plugs = self.base_station_plugs
        demand_multiplier = 1.0
        service_time_multiplier = 1.0
        if disruption_type == "capacity_drop":
            effective_base_plugs = int(np.clip(round(severity), 0, self.base_station_plugs))
        elif disruption_type == "station_outage":
            effective_base_plugs = 0
        elif disruption_type in {"demand_surge_ab", "demand_surge_ba"}:
            demand_multiplier = max(severity, 1.0)
        elif disruption_type == "service_time_inflation":
            service_time_multiplier = max(severity, 1.0)
        else:
            raise ValueError(f"Unsupported mobile disruption type: {disruption_type}")

        return MobileDisruptionEvent(
            disruption_type=disruption_type,
            day_index=day_index,
            start_step=start_step,
            end_step=end_step,
            severity=severity,
            effective_base_plugs=effective_base_plugs,
            demand_multiplier=demand_multiplier,
            service_time_multiplier=service_time_multiplier,
        )

    def _build_disruption_schedule(self) -> dict[int, MobileDisruptionEvent]:
        disruption_cfg = dict(self.env_config.get("disruption", {}))
        if not bool(disruption_cfg.get("enabled", False)):
            return {}

        schedule: dict[int, MobileDisruptionEvent] = {}
        mode = str(disruption_cfg.get("mode", "scripted")).lower()
        events: list[MobileDisruptionEvent] = []
        steps_per_day = int(round(self.horizon * 60.0 / self.planning_step_minutes))
        num_days = max(1, int(np.ceil(self.max_steps / max(steps_per_day, 1))))
        if mode == "scripted":
            for payload in disruption_cfg.get("scripted_events", []):
                raw_payload = dict(payload)
                if bool(raw_payload.get("repeat_daily", False)):
                    for day_index in range(num_days):
                        repeated = dict(raw_payload)
                        repeated["day_index"] = day_index
                        events.append(self._coerce_event(repeated))
                else:
                    events.append(self._coerce_event(raw_payload))
        elif mode == "random":
            if self.rng.random() < float(disruption_cfg.get("event_probability", 1.0)):
                event_types = list(
                    disruption_cfg.get(
                        "event_types",
                        [
                            "capacity_drop",
                            "station_outage",
                            "demand_surge_ab",
                            "demand_surge_ba",
                            "service_time_inflation",
                        ],
                    )
                )
                disruption_type = str(self.rng.choice(event_types))
                duration_hours = float(disruption_cfg.get("duration_hours", 2.0))
                start_max = max(0.0, float(self.horizon) - duration_hours)
                start_hour = float(self.rng.uniform(0.0, start_max))
                severity_defaults = {
                    "capacity_drop": float(disruption_cfg.get("capacity_drop_target_num_plugs", max(self.base_station_plugs // 3, 1))),
                    "station_outage": 0.0,
                    "demand_surge_ab": float(disruption_cfg.get("demand_surge_multiplier", 1.8)),
                    "demand_surge_ba": float(disruption_cfg.get("demand_surge_multiplier", 1.8)),
                    "service_time_inflation": float(disruption_cfg.get("service_time_multiplier", 1.5)),
                }
                events.append(
                    self._coerce_event(
                        {
                            "disruption_type": disruption_type,
                            "start_hour": start_hour,
                            "duration_hours": duration_hours,
                            "severity": severity_defaults[disruption_type],
                        }
                    )
                )
        else:
            raise ValueError(f"Unknown mobile disruption mode: {mode}")

        for event in events:
            for step in range(max(event.start_step, 0), min(event.end_step, self.max_steps)):
                schedule[step] = event
        return schedule

    def _disruption_state(self) -> dict[str, float | int | str]:
        event = self.disruption_by_step.get(self.step_index)
        self.current_event = event
        if event is None:
            return {
                "disruption_active": 0,
                "disruption_type": "none",
                "disruption_type_code": 0,
                "disruption_day_index": -1,
                "disruption_remaining_steps": 0.0,
                "effective_base_plugs": self.base_station_plugs,
                "demand_multiplier": 1.0,
                "service_time_multiplier": 1.0,
            }
        return {
            "disruption_active": 1,
            "disruption_type": event.disruption_type,
            "disruption_type_code": self.DISRUPTION_TYPE_CODES[event.disruption_type],
            "disruption_day_index": event.day_index,
            "disruption_remaining_steps": float(max(event.end_step - self.step_index, 0)),
            "effective_base_plugs": event.effective_base_plugs,
            "demand_multiplier": event.demand_multiplier,
            "service_time_multiplier": event.service_time_multiplier,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        episode_seed = self.base_seed + self.reset_counter * self.reset_seed_stride if seed is None else int(seed)
        self.reset_counter += 1
        self.seed(episode_seed)

        self.step_index = 0
        self.is_weekend = bool(self.rng.random() < (2.0 / 7.0))
        self.current_mobile_stations = 0
        self.last_arrivals = 0.0
        self.last_served = 0.0
        self.last_service_capacity = 0.0
        self.last_idle_capacity = 0.0
        self.last_expected_arrivals = 0.0
        self.last_observed_total_demand = 0.0
        self.last_true_total_demand = 0.0
        self.last_expected_total_demand = 0.0
        self.last_unmet_demand = 0.0
        self.last_utilization = 0.0
        self.queue_length = 0.0
        self.queue_wait_mean_minutes = 0.0
        self.disruption_by_step = self._build_disruption_schedule()
        self.current_event = None
        return self._get_observation(), {
            "num_active_mobile_stations": 0,
            "is_weekend": self.is_weekend,
            "fixed_site_index": self.fixed_site_index,
            "fixed_site_coords_km": self.fixed_site_coords_km.tolist(),
        }

    def expected_total_demand(self) -> float:
        expected = self.demand_generator.expected_lambda(step=self.step_index, is_weekend=self.is_weekend)
        return float(np.sum(expected))

    def expected_vehicle_arrivals(self) -> float:
        disruption_state = self._disruption_state()
        expected_total = self.expected_total_demand() * float(disruption_state["demand_multiplier"])
        return max(expected_total / max(self.vehicle_arrival_scale, 1e-6), 0.0)

    def current_total_capacity(self) -> float:
        return float(self.base_station_plugs + self.current_mobile_stations * self.mobile_station_chargers)

    def current_service_capacity_per_step(self, service_time_multiplier: float = 1.0, effective_base_plugs: int | None = None) -> float:
        base_plugs = self.base_station_plugs if effective_base_plugs is None else int(effective_base_plugs)
        total_effective_plugs = base_plugs + self.current_mobile_stations * self.mobile_station_chargers
        service_minutes = max(self.mean_service_minutes * service_time_multiplier, 1e-6)
        return float(total_effective_plugs) * float(self.planning_step_minutes) / service_minutes

    def valid_action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def _normalized(self, value: float, scale: float) -> float:
        return float(value) / max(scale, 1e-6)

    def _get_observation(self) -> np.ndarray:
        time_fraction = float(self.step_index % self.horizon) / float(max(self.horizon - 1, 1))
        time_sin = np.sin(2.0 * np.pi * time_fraction)
        time_cos = np.cos(2.0 * np.pi * time_fraction)
        disruption_state = self._disruption_state()
        expected_arrivals = self.expected_vehicle_arrivals()
        service_capacity = self.current_service_capacity_per_step(
            service_time_multiplier=float(disruption_state["service_time_multiplier"]),
            effective_base_plugs=int(disruption_state["effective_base_plugs"]),
        )
        obs = np.asarray(
            [
                self._normalized(self.queue_length, self.queue_normalizer),
                self._normalized(self.last_arrivals, self.arrival_normalizer),
                self._normalized(self.last_served, self.arrival_normalizer),
                self._normalized(self.last_service_capacity, self.arrival_normalizer),
                self._normalized(expected_arrivals, self.arrival_normalizer),
                self._normalized(self.last_idle_capacity, self.arrival_normalizer),
                self._normalized(self.queue_wait_mean_minutes, self.planning_step_minutes * 8.0),
                self._normalized(self.base_station_plugs, self.capacity_normalizer),
                self._normalized(float(disruption_state["effective_base_plugs"]), self.capacity_normalizer),
                self._normalized(float(self.current_mobile_stations * self.mobile_station_chargers), self.capacity_normalizer),
                self._normalized(service_capacity, self.arrival_normalizer),
                float(self.current_mobile_stations) / float(max(self.max_mobile_stations, 1)),
                float(disruption_state["disruption_active"]),
                float(disruption_state["disruption_type_code"]) / float(max(len(self.DISRUPTION_TYPE_CODES) - 1, 1)),
                self._normalized(float(disruption_state["disruption_remaining_steps"]), float(max(self.max_steps, 1))),
                time_sin + 0.5 * time_cos + 0.25 * float(self.is_weekend),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        chosen_mobile_stations = int(np.clip(action, 0, self.max_mobile_stations))
        previous_mobile_stations = self.current_mobile_stations
        self.current_mobile_stations = chosen_mobile_stations

        disruption_state = self._disruption_state()
        snapshot = self.demand_generator.sample(step=self.step_index, is_weekend=self.is_weekend)
        total_true_demand = float(np.sum(snapshot.true_demand))
        total_expected_demand = float(np.sum(snapshot.expected_lambda))
        total_observed_demand = float(np.sum(snapshot.observed_demand))

        demand_multiplier = float(disruption_state["demand_multiplier"])
        arrivals = int(
            self.rng.poisson(
                max(total_true_demand * demand_multiplier / max(self.vehicle_arrival_scale, 1e-6), 0.0)
            )
        )
        effective_base_plugs = int(disruption_state["effective_base_plugs"])
        service_capacity = self.current_service_capacity_per_step(
            service_time_multiplier=float(disruption_state["service_time_multiplier"]),
            effective_base_plugs=effective_base_plugs,
        )
        service_capacity_vehicles = int(np.floor(service_capacity))

        total_queue_pressure = self.queue_length + float(arrivals)
        served_total = float(min(total_queue_pressure, service_capacity_vehicles))
        self.queue_length = max(total_queue_pressure - served_total, 0.0)
        idle_capacity = max(float(service_capacity_vehicles) - served_total, 0.0)
        utilization = served_total / max(float(service_capacity_vehicles), 1e-6) if service_capacity_vehicles > 0 else 0.0
        self.queue_wait_mean_minutes = self.queue_length * self.planning_step_minutes / max(total_queue_pressure, 1.0)

        activated = max(self.current_mobile_stations - previous_mobile_stations, 0)
        adjusted = abs(self.current_mobile_stations - previous_mobile_stations)
        unmet_total = self.queue_length
        reward = (
            self.served_reward_weight * served_total
            - self.unmet_penalty * unmet_total
            - self.queue_length_penalty * self.queue_length
            - self.active_mobile_station_cost * float(self.current_mobile_stations)
            - self.activation_cost * float(activated)
            - self.adjustment_cost * float(adjusted)
            - self.idle_capacity_penalty * idle_capacity
            + self.utilization_bonus * utilization
            + self.disruption_response_bonus * float(disruption_state["disruption_active"]) * float(self.current_mobile_stations > 0)
        )
        reward *= self.reward_scale

        self.last_arrivals = float(arrivals)
        self.last_served = served_total
        self.last_service_capacity = float(service_capacity_vehicles)
        self.last_idle_capacity = idle_capacity
        self.last_expected_arrivals = total_expected_demand / max(self.vehicle_arrival_scale, 1e-6)
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
            "arrivals_vehicles": float(arrivals),
            "expected_arrivals_vehicles": float(self.last_expected_arrivals),
            "queue_length": float(self.queue_length),
            "queue_wait_mean_minutes": float(self.queue_wait_mean_minutes),
            "service_capacity_vehicles": float(service_capacity_vehicles),
            "num_active_mobile_stations": int(self.current_mobile_stations),
            "num_active_chargers": int(effective_base_plugs + self.current_mobile_stations * self.mobile_station_chargers),
            "base_capacity_total": float(effective_base_plugs),
            "mobile_capacity_total": float(self.current_mobile_stations * self.mobile_station_chargers),
            "effective_capacity_total": float(service_capacity_vehicles),
            "action_valid": True,
            "utilization": utilization,
            "activated_mobile_stations": int(activated),
            "adjusted_mobile_stations": int(adjusted),
            "idle_capacity": idle_capacity,
            "disruption_active": int(disruption_state["disruption_active"]),
            "disruption_type": str(disruption_state["disruption_type"]),
            "disruption_type_code": int(disruption_state["disruption_type_code"]),
            "disruption_day_index": int(disruption_state["disruption_day_index"]),
            "disruption_remaining_steps": float(disruption_state["disruption_remaining_steps"]),
            "effective_base_plugs": int(effective_base_plugs),
            "fixed_site_index": self.fixed_site_index,
            "fixed_site_coords_km": self.fixed_site_coords_km.tolist(),
            "reward": reward,
        }
        return self._get_observation(), float(reward), terminated, False, info

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from evch.envs.charging_env import gym, spaces
from evch.sim.simple_corridor import ActiveSession, QueuedVehicle, SimpleCorridorQueueSimulator


class CorridorMobileStationEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> None:
        del demand_config
        self.env_config = env_config
        self.base_seed = int(seed)
        self.reset_counter = 0
        self.reset_seed_stride = int(env_config.get("reset_seed_stride", 97))

        sim_cfg = dict(env_config.get("simulation", {}))
        if not sim_cfg:
            raise ValueError("CorridorMobileStationEnv requires environment.simulation config.")
        self.simulation_config = sim_cfg
        self.base_station_plugs = int(sim_cfg["num_plugs"])
        self.max_mobile_stations = int(env_config.get("max_mobile_stations", 10))
        self.mobile_station_chargers = int(env_config.get("mobile_station_chargers", 2))
        self.mobile_station_capacity = float(env_config.get("mobile_station_capacity", 20.0))
        self.reward_scale = float(env_config.get("reward_scale", 1.0))
        self.fixed_site_index = 0
        self.fixed_site_coords_km = np.asarray(
            [float(sim_cfg.get("station_position_km", 0.0)), 0.0],
            dtype=np.float32,
        )

        reward_cfg = env_config.get("reward", {})
        self.served_reward_weight = float(reward_cfg.get("served_reward_weight", 1.0))
        self.unmet_penalty = float(reward_cfg.get("unmet_penalty", 2.0))
        self.active_mobile_station_cost = float(reward_cfg.get("active_mobile_station_cost", 30.0))
        self.activation_cost = float(reward_cfg.get("activation_cost", 8.0))
        self.adjustment_cost = float(reward_cfg.get("adjustment_cost", 3.0))
        self.idle_capacity_penalty = float(reward_cfg.get("idle_capacity_penalty", 0.15))
        self.utilization_bonus = float(reward_cfg.get("utilization_bonus", 1.0))
        self.queue_length_penalty = float(reward_cfg.get("queue_length_penalty", 3.0))
        self.queue_wait_penalty = float(reward_cfg.get("queue_wait_penalty", 0.02))
        self.queue_wait_target_minutes = float(reward_cfg.get("queue_wait_target_minutes", 0.0))
        self.queue_wait_excess_penalty = float(
            reward_cfg.get("queue_wait_excess_penalty", self.queue_wait_penalty)
        )
        self.queue_wait_hard_penalty = float(reward_cfg.get("queue_wait_hard_penalty", 0.0))
        self.queue_wait_service_level_bonus = float(reward_cfg.get("queue_wait_service_level_bonus", 0.0))
        self.disruption_response_bonus = float(reward_cfg.get("disruption_response_bonus", 0.0))

        self.simulator = SimpleCorridorQueueSimulator(config=sim_cfg, seed=self.base_seed)
        self.planning_step_minutes = float(self.simulator.step_minutes)
        self.mean_service_minutes = float(self.simulator.service_mean_minutes)
        self.horizon = int(self.simulator.steps_per_day)
        self.max_steps = int(self.simulator.num_steps)
        self.queue_normalizer = float(env_config.get("queue_normalizer", 80.0))
        self.arrival_normalizer = float(env_config.get("arrival_normalizer", 20.0))
        self.capacity_normalizer = float(
            env_config.get(
                "capacity_normalizer",
                self.base_station_plugs + self.max_mobile_stations * self.mobile_station_chargers,
            )
        )

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_mobile_stations + 1)

        self.rng = np.random.default_rng(self.base_seed)
        self.queue: deque[QueuedVehicle] = deque()
        self.active_sessions: list[ActiveSession] = []
        self.disruption_by_step: dict[int, Any] = {}
        self.current_mobile_stations = 0
        self.step_index = 0

        self.last_arrivals_total = 0.0
        self.last_arrivals_ab = 0.0
        self.last_arrivals_ba = 0.0
        self.last_starts_total = 0.0
        self.last_completions_total = 0.0
        self.queue_length = 0.0
        self.queue_wait_mean_minutes = 0.0
        self.last_expected_passing_total = 0.0
        self.last_expected_passing_ab = 0.0
        self.last_expected_passing_ba = 0.0
        self.last_utilization = 0.0
        self.last_effective_num_plugs = float(self.base_station_plugs)
        self.last_idle_capacity = 0.0
        self.last_disruption_state: dict[str, Any] = {}

    def seed(self, seed: int | None = None) -> None:
        chosen_seed = self.base_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(chosen_seed)
        self.simulator = SimpleCorridorQueueSimulator(config=self.simulation_config, seed=chosen_seed)

    def valid_action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def current_total_capacity(self) -> float:
        return float(self.last_effective_num_plugs)

    def current_service_capacity_per_step(self, service_time_multiplier: float = 1.0, effective_base_plugs: int | None = None) -> float:
        base_plugs = self.base_station_plugs if effective_base_plugs is None else int(effective_base_plugs)
        total_effective_plugs = base_plugs + self.current_mobile_stations * self.mobile_station_chargers
        return float(total_effective_plugs) * float(self.planning_step_minutes) / max(
            self.mean_service_minutes * service_time_multiplier,
            1e-6,
        )

    def expected_vehicle_arrivals(self) -> float:
        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multiplier_ab=float(disruption_state["demand_multiplier_ab"]),
            demand_multiplier_ba=float(disruption_state["demand_multiplier_ba"]),
        )
        return float(expected["total_passing_expected"]) * float(self.simulator.stop_probability)

    def _normalized(self, value: float, scale: float) -> float:
        return float(value) / max(scale, 1e-6)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        episode_seed = self.base_seed + self.reset_counter * self.reset_seed_stride if seed is None else int(seed)
        self.reset_counter += 1
        self.seed(episode_seed)
        self.queue = deque()
        self.active_sessions = []
        self.current_mobile_stations = 0
        self.step_index = 0
        disruption_schedule = self.simulator._build_disruption_schedule()
        self.disruption_by_step = self.simulator._event_map(disruption_schedule)

        self.last_arrivals_total = 0.0
        self.last_arrivals_ab = 0.0
        self.last_arrivals_ba = 0.0
        self.last_starts_total = 0.0
        self.last_completions_total = 0.0
        self.queue_length = 0.0
        self.queue_wait_mean_minutes = 0.0
        self.last_expected_passing_total = 0.0
        self.last_expected_passing_ab = 0.0
        self.last_expected_passing_ba = 0.0
        self.last_utilization = 0.0
        self.last_effective_num_plugs = float(self.base_station_plugs)
        self.last_idle_capacity = 0.0
        self.last_disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        return self._get_observation(), {
            "fixed_site_index": self.fixed_site_index,
            "fixed_site_coords_km": self.fixed_site_coords_km.tolist(),
            "num_active_mobile_stations": 0,
        }

    def _get_observation(self) -> np.ndarray:
        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multiplier_ab=float(disruption_state["demand_multiplier_ab"]),
            demand_multiplier_ba=float(disruption_state["demand_multiplier_ba"]),
        )
        time_fraction = float(self.step_index % self.horizon) / float(max(self.horizon - 1, 1))
        obs = np.asarray(
            [
                self._normalized(self.queue_length, self.queue_normalizer),
                self._normalized(self.queue_wait_mean_minutes, self.planning_step_minutes * 4.0),
                self._normalized(self.last_arrivals_total, self.arrival_normalizer),
                self._normalized(self.last_arrivals_ab, self.arrival_normalizer),
                self._normalized(self.last_arrivals_ba, self.arrival_normalizer),
                self._normalized(self.last_starts_total, self.arrival_normalizer),
                self._normalized(self.last_completions_total, self.arrival_normalizer),
                self._normalized(float(expected["total_passing_expected"]) * float(self.simulator.stop_probability), self.arrival_normalizer),
                self._normalized(self.last_effective_num_plugs, self.capacity_normalizer),
                self._normalized(float(self.current_mobile_stations * self.mobile_station_chargers), self.capacity_normalizer),
                self._normalized(float(len(self.active_sessions)), self.capacity_normalizer),
                self._normalized(float(disruption_state["disruption_remaining_minutes"]), 24.0 * 60.0),
                float(disruption_state["disruption_active"]),
                float(disruption_state["disruption_type_code"]) / 5.0,
                np.sin(2.0 * np.pi * time_fraction),
                np.cos(2.0 * np.pi * time_fraction),
                self._normalized(float(expected["expected_passing_ab"]) * float(self.simulator.stop_probability), self.arrival_normalizer),
                self._normalized(float(expected["expected_passing_ba"]) * float(self.simulator.stop_probability), self.arrival_normalizer),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        chosen_mobile_stations = int(np.clip(action, 0, self.max_mobile_stations))
        previous_mobile_stations = self.current_mobile_stations
        self.current_mobile_stations = chosen_mobile_stations

        completed_now = [session for session in self.active_sessions if session.end_step <= self.step_index]
        self.active_sessions = [session for session in self.active_sessions if session.end_step > self.step_index]
        completed_waits = [(session.start_step - session.arrival_step) * self.simulator.step_minutes for session in completed_now]

        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        if (
            disruption_state["disruption_type"] == "station_outage"
            and disruption_state["disruption_active"] == 1
            and self.disruption_by_step[self.step_index].start_step == self.step_index
        ):
            self.simulator._requeue_active_sessions(self.queue, self.active_sessions)
            self.active_sessions = []

        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multiplier_ab=float(disruption_state["demand_multiplier_ab"]),
            demand_multiplier_ba=float(disruption_state["demand_multiplier_ba"]),
        )
        arrivals = self.simulator._sample_arrivals(
            expected_ab=float(expected["expected_passing_ab"]),
            expected_ba=float(expected["expected_passing_ba"]),
        )

        arrivals_ab = 0
        arrivals_ba = 0
        for vehicle in arrivals:
            vehicle.arrival_step = self.step_index
            self.queue.append(vehicle)
            if vehicle.direction == self.simulator.direction_ab:
                arrivals_ab += 1
            else:
                arrivals_ba += 1

        starts_now: list[ActiveSession] = []
        base_effective_num_plugs = int(disruption_state["effective_num_plugs"])
        total_effective_num_plugs = base_effective_num_plugs + self.current_mobile_stations * self.mobile_station_chargers
        available_plugs = max(total_effective_num_plugs - len(self.active_sessions), 0)
        for _ in range(min(available_plugs, len(self.queue))):
            vehicle = self.queue.popleft()
            service_minutes = self.simulator._sample_service_minutes(
                service_time_multiplier=float(disruption_state["service_time_multiplier"])
            )
            service_steps = max(1, int(np.ceil(service_minutes / self.simulator.step_minutes)))
            session = ActiveSession(
                arrival_step=vehicle.arrival_step,
                start_step=self.step_index,
                end_step=self.step_index + service_steps,
                direction=vehicle.direction,
                service_minutes=service_minutes,
            )
            self.active_sessions.append(session)
            starts_now.append(session)

        started_waits = [(session.start_step - session.arrival_step) * self.simulator.step_minutes for session in starts_now]
        queue_wait_minutes = [(self.step_index - vehicle.arrival_step) * self.simulator.step_minutes for vehicle in self.queue]
        active_plugs = len(self.active_sessions)
        utilization = active_plugs / float(max(total_effective_num_plugs, 1)) if total_effective_num_plugs > 0 else 0.0
        mobile_chargers_total = float(self.current_mobile_stations * self.mobile_station_chargers)
        mobile_chargers_used_estimate = float(
            np.clip(active_plugs - base_effective_num_plugs, 0, self.current_mobile_stations * self.mobile_station_chargers)
        )
        unused_mobile_chargers = max(mobile_chargers_total - mobile_chargers_used_estimate, 0.0)
        unused_mobile_stations_estimate = unused_mobile_chargers / max(float(self.mobile_station_chargers), 1e-6)

        served_total = float(len(starts_now))
        unmet_total = float(len(self.queue))
        idle_capacity = max(float(total_effective_num_plugs - active_plugs), 0.0)
        activated = max(self.current_mobile_stations - previous_mobile_stations, 0)
        adjusted = abs(self.current_mobile_stations - previous_mobile_stations)
        mean_started_wait = float(np.mean(started_waits)) if started_waits else 0.0
        mean_queue_wait = float(np.mean(queue_wait_minutes)) if queue_wait_minutes else 0.0
        queue_wait_excess = max(mean_queue_wait - self.queue_wait_target_minutes, 0.0)
        queue_wait_target_breached = float(
            self.queue_wait_target_minutes > 0.0 and mean_queue_wait > self.queue_wait_target_minutes
        )
        queue_wait_penalty_term = (
            self.queue_wait_penalty * mean_queue_wait
            if self.queue_wait_target_minutes <= 0.0
            else self.queue_wait_excess_penalty * queue_wait_excess
        )
        reward = (
            self.served_reward_weight * served_total
            - self.unmet_penalty * unmet_total
            - self.queue_length_penalty * unmet_total
            - queue_wait_penalty_term
            - self.queue_wait_hard_penalty * queue_wait_target_breached
            - self.active_mobile_station_cost * float(self.current_mobile_stations)
            - self.activation_cost * float(activated)
            - self.adjustment_cost * float(adjusted)
            - self.idle_capacity_penalty * idle_capacity
            + self.utilization_bonus * utilization
            + self.queue_wait_service_level_bonus * float(
                self.queue_wait_target_minutes > 0.0 and queue_wait_target_breached == 0.0
            )
            + self.disruption_response_bonus * float(disruption_state["disruption_active"]) * float(self.current_mobile_stations > 0)
        )
        reward *= self.reward_scale

        self.last_arrivals_total = float(len(arrivals))
        self.last_arrivals_ab = float(arrivals_ab)
        self.last_arrivals_ba = float(arrivals_ba)
        self.last_starts_total = served_total
        self.last_completions_total = float(len(completed_now))
        self.queue_length = unmet_total
        self.queue_wait_mean_minutes = mean_queue_wait
        self.last_expected_passing_total = float(expected["total_passing_expected"])
        self.last_expected_passing_ab = float(expected["expected_passing_ab"])
        self.last_expected_passing_ba = float(expected["expected_passing_ba"])
        self.last_utilization = utilization
        self.last_effective_num_plugs = float(total_effective_num_plugs)
        self.last_idle_capacity = idle_capacity
        self.last_disruption_state = dict(disruption_state)

        info = {
            "served_demand": served_total,
            "unmet_demand": unmet_total,
            "true_demand_total": float(len(arrivals)),
            "expected_demand_total": float(expected["total_passing_expected"]) * float(self.simulator.stop_probability),
            "expected_arrivals_vehicles": float(expected["total_passing_expected"]) * float(self.simulator.stop_probability),
            "arrivals_vehicles": float(len(arrivals)),
            "queue_length": unmet_total,
            "queue_wait_mean_minutes": mean_queue_wait,
            "queue_wait_target_minutes": float(self.queue_wait_target_minutes),
            "queue_wait_excess_minutes": float(queue_wait_excess),
            "queue_wait_target_breached": int(queue_wait_target_breached),
            "service_capacity_vehicles": float(total_effective_num_plugs),
            "num_active_mobile_stations": int(self.current_mobile_stations),
            "num_active_chargers": int(total_effective_num_plugs),
            "active_plugs": int(active_plugs),
            "base_capacity_total": float(base_effective_num_plugs),
            "mobile_capacity_total": mobile_chargers_total,
            "effective_capacity_total": float(total_effective_num_plugs),
            "unused_mobile_chargers": float(unused_mobile_chargers),
            "unused_mobile_stations_estimate": float(unused_mobile_stations_estimate),
            "action_valid": True,
            "utilization": utilization,
            "activated_mobile_stations": int(activated),
            "adjusted_mobile_stations": int(adjusted),
            "idle_capacity": idle_capacity,
            "disruption_active": int(disruption_state["disruption_active"]),
            "disruption_type": str(disruption_state["disruption_type"]),
            "disruption_type_code": int(disruption_state["disruption_type_code"]),
            "disruption_day_index": int(disruption_state["disruption_day_index"]),
            "disruption_remaining_steps": float(disruption_state["disruption_remaining_minutes"]) / float(self.simulator.step_minutes),
            "disruption_remaining_minutes": float(disruption_state["disruption_remaining_minutes"]),
            "effective_base_plugs": int(base_effective_num_plugs),
            "fixed_site_index": self.fixed_site_index,
            "fixed_site_coords_km": self.fixed_site_coords_km.tolist(),
            "arrivals_total": float(len(arrivals)),
            "arrivals_ab": float(arrivals_ab),
            "arrivals_ba": float(arrivals_ba),
            "starts_total": served_total,
            "starts_ab": float(sum(session.direction == self.simulator.direction_ab for session in starts_now)),
            "starts_ba": float(sum(session.direction == self.simulator.direction_ba for session in starts_now)),
            "completions_total": float(len(completed_now)),
            "completions_ab": float(sum(session.direction == self.simulator.direction_ab for session in completed_now)),
            "completions_ba": float(sum(session.direction == self.simulator.direction_ba for session in completed_now)),
            "expected_passing_total": float(expected["total_passing_expected"]),
            "expected_passing_ab": float(expected["expected_passing_ab"]),
            "expected_passing_ba": float(expected["expected_passing_ba"]),
            "started_wait_mean_minutes": mean_started_wait,
            "completed_wait_mean_minutes": float(np.mean(completed_waits)) if completed_waits else 0.0,
            "started_service_mean_minutes": float(np.mean([session.service_minutes for session in starts_now])) if starts_now else 0.0,
            "reward": reward,
        }

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        return self._get_observation(), float(reward), terminated, False, info

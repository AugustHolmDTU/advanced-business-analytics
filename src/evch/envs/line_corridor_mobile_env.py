from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from evch.envs.charging_env import gym, spaces
from evch.sim.line_corridor import ActiveSession, LineCorridorQueueSimulator, QueuedVehicle


@dataclass(slots=True)
class MobileChargingStationUnit:
    state: str
    timer_steps: int = 0


class LineCorridorMobileStationEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(self, env_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> None:
        del demand_config
        self.env_config = env_config
        self.base_seed = int(seed)
        self.reset_counter = 0
        self.reset_seed_stride = int(env_config.get("reset_seed_stride", 97))
        raw_episode_seed_range = env_config.get("episode_seed_range")
        self.episode_seed_range = (
            (int(raw_episode_seed_range[0]), int(raw_episode_seed_range[1]))
            if isinstance(raw_episode_seed_range, (list, tuple)) and len(raw_episode_seed_range) == 2
            else None
        )

        sim_cfg = dict(env_config.get("simulation", {}))
        if not sim_cfg:
            raise ValueError("LineCorridorMobileStationEnv requires environment.simulation config.")
        self.simulation_config = sim_cfg
        self.duration_days_range = self._parse_duration_days_range(sim_cfg.get("duration_days_range"))

        self.max_mobile_stations = int(env_config.get("max_mobile_stations", 10))
        self.mobile_station_chargers = int(env_config.get("mobile_station_chargers", 2))
        self.mobile_station_capacity = float(env_config.get("mobile_station_capacity", 20.0))
        self.reward_scale = float(env_config.get("reward_scale", 1.0))
        self.mcs_middle_travel_minutes = float(env_config.get("mcs_middle_travel_minutes", 30.0))
        self.mcs_charge_full_minutes = float(env_config.get("mcs_charge_full_minutes", 60.0))

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
        self.disruption_response_bonus = float(reward_cfg.get("disruption_response_bonus", 0.0))
        self.spatial_deficit_alignment_bonus = float(reward_cfg.get("spatial_deficit_alignment_bonus", 0.0))
        self.spatial_deficit_direction_bonus = float(reward_cfg.get("spatial_deficit_direction_bonus", 0.0))

        self.simulator = self._build_simulator(seed=self.base_seed, duration_hours=self._max_episode_duration_hours())
        self.station_keys = list(self.simulator.STATION_KEYS)
        self.fixed_site_coords_km = np.asarray(
            [[station.position_km, 0.0] for station in self.simulator.stations],
            dtype=np.float32,
        )
        self.base_station_plugs_by_station = self.simulator.num_plugs_by_station.astype(np.int32)
        self.planning_step_minutes = float(self.simulator.step_minutes)
        self.mean_service_minutes = float(self.simulator.service_mean_minutes)
        self.horizon = int(self.simulator.steps_per_day)
        self.max_steps = int(self.simulator.num_steps)
        self.queue_normalizer = float(env_config.get("queue_normalizer", 80.0))
        self.arrival_normalizer = float(env_config.get("arrival_normalizer", 20.0))
        self.capacity_normalizer = float(
            env_config.get(
                "capacity_normalizer",
                float(self.base_station_plugs_by_station.sum() + self.max_mobile_stations * self.mobile_station_chargers),
            )
        )
        self.mobile_count_normalizer = float(max(self.max_mobile_stations, 1))
        self.middle_travel_steps = max(1, int(round(self.mcs_middle_travel_minutes / self.planning_step_minutes)))
        self.relocation_steps = max(1, self.middle_travel_steps * 2)
        self.charge_full_steps = max(1, int(round(self.mcs_charge_full_minutes / self.planning_step_minutes)))

        self.action_map = self._build_action_map()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(28,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_map))

        self.rng = np.random.default_rng(self.base_seed)
        self.seed_stream_rng = np.random.default_rng(self.base_seed + 1_001)
        self.queue_by_station = [deque(), deque()]
        self.active_sessions_by_station: list[list[ActiveSession]] = [[], []]
        self.disruption_by_step: dict[int, Any] = {}
        self.mobile_station_units = self._build_mobile_station_units()
        self.current_mobile_stations_by_station = np.zeros(2, dtype=np.int32)
        self.step_index = 0

        self.last_arrivals_by_station = np.zeros(2, dtype=np.float32)
        self.last_starts_by_station = np.zeros(2, dtype=np.float32)
        self.last_completions_by_station = np.zeros(2, dtype=np.float32)
        self.queue_lengths_by_station = np.zeros(2, dtype=np.float32)
        self.queue_wait_mean_by_station = np.zeros(2, dtype=np.float32)
        self.last_expected_station_arrivals = np.zeros(2, dtype=np.float32)
        self.last_effective_num_plugs_by_station = self.base_station_plugs_by_station.astype(np.float32)
        self.last_disruption_state: dict[str, Any] = {}
        self.current_disruption_schedule: list[Any] = []
        self.current_episode_seed = self.base_seed

    @staticmethod
    def _parse_duration_days_range(raw_value: Any) -> tuple[int, int] | None:
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 2:
            return None
        low = int(raw_value[0])
        high = int(raw_value[1])
        if low <= 0 or high <= 0:
            raise ValueError("simulation.duration_days_range values must be positive integers")
        if high < low:
            low, high = high, low
        return low, high

    def _max_episode_duration_hours(self) -> float:
        if self.duration_days_range is None:
            return float(self.simulation_config.get("duration_hours", 24.0))
        return 24.0 * float(self.duration_days_range[1])

    def _sample_episode_duration_hours(self) -> float:
        if self.duration_days_range is None:
            return float(self.simulation_config.get("duration_hours", 24.0))
        low, high = self.duration_days_range
        sampled_days = int(self.rng.integers(low, high + 1))
        return 24.0 * float(sampled_days)

    def _build_simulator(self, seed: int, duration_hours: float) -> LineCorridorQueueSimulator:
        sim_cfg = dict(self.simulation_config)
        sim_cfg["duration_hours"] = float(duration_hours)
        return LineCorridorQueueSimulator(config=sim_cfg, seed=seed)

    def _build_action_map(self) -> list[tuple[int, int]]:
        allocations: list[tuple[int, int]] = []
        for first in range(self.max_mobile_stations + 1):
            for second in range(self.max_mobile_stations + 1 - first):
                allocations.append((first, second))
        return allocations

    def _build_mobile_station_units(self) -> list[MobileChargingStationUnit]:
        return [MobileChargingStationUnit(state="middle_available", timer_steps=0) for _ in range(self.max_mobile_stations)]

    def _refresh_mobile_station_counts(self) -> None:
        self.current_mobile_stations_by_station = np.asarray(
            [
                sum(unit.state == "station_ab" for unit in self.mobile_station_units),
                sum(unit.state == "station_bc" for unit in self.mobile_station_units),
            ],
            dtype=np.int32,
        )

    def _mobile_station_state_counts(self) -> dict[str, int]:
        return {
            "middle_available": sum(unit.state == "middle_available" for unit in self.mobile_station_units),
            "middle_charging": sum(unit.state == "middle_charging" for unit in self.mobile_station_units),
            "station_ab": int(self.current_mobile_stations_by_station[0]),
            "station_bc": int(self.current_mobile_stations_by_station[1]),
            "transit_to_ab": sum(unit.state == "transit_to_ab" for unit in self.mobile_station_units),
            "transit_to_bc": sum(unit.state == "transit_to_bc" for unit in self.mobile_station_units),
            "transit_to_middle": sum(unit.state == "transit_to_middle" for unit in self.mobile_station_units),
        }

    def _advance_mobile_station_units(self) -> None:
        for unit in self.mobile_station_units:
            if unit.timer_steps > 0:
                unit.timer_steps -= 1
            if unit.timer_steps > 0:
                continue
            if unit.state == "transit_to_ab":
                unit.state = "station_ab"
            elif unit.state == "transit_to_bc":
                unit.state = "station_bc"
            elif unit.state == "transit_to_middle":
                unit.state = "middle_charging"
                unit.timer_steps = self.charge_full_steps
            elif unit.state == "middle_charging":
                unit.state = "middle_available"
        self._refresh_mobile_station_counts()

    def _dispatch_mobile_station_units(self, desired_allocation: np.ndarray) -> None:
        desired_ab = int(desired_allocation[0])
        desired_bc = int(desired_allocation[1])

        stationed_ab = [unit for unit in self.mobile_station_units if unit.state == "station_ab"]
        stationed_bc = [unit for unit in self.mobile_station_units if unit.state == "station_bc"]
        committed_ab = len(stationed_ab) + sum(unit.state == "transit_to_ab" for unit in self.mobile_station_units)
        committed_bc = len(stationed_bc) + sum(unit.state == "transit_to_bc" for unit in self.mobile_station_units)

        if len(stationed_ab) > desired_ab:
            surplus_ab = len(stationed_ab) - desired_ab
            move_ab_to_bc = min(surplus_ab, max(desired_bc - committed_bc, 0))
            for unit in stationed_ab[:move_ab_to_bc]:
                unit.state = "transit_to_bc"
                unit.timer_steps = self.relocation_steps
            for unit in stationed_ab[move_ab_to_bc:surplus_ab]:
                unit.state = "transit_to_middle"
                unit.timer_steps = self.middle_travel_steps

        if len(stationed_bc) > desired_bc:
            surplus_bc = len(stationed_bc) - desired_bc
            move_bc_to_ab = min(surplus_bc, max(desired_ab - committed_ab, 0))
            for unit in stationed_bc[:move_bc_to_ab]:
                unit.state = "transit_to_ab"
                unit.timer_steps = self.relocation_steps
            for unit in stationed_bc[move_bc_to_ab:surplus_bc]:
                unit.state = "transit_to_middle"
                unit.timer_steps = self.middle_travel_steps

        committed_ab = sum(unit.state in {"station_ab", "transit_to_ab"} for unit in self.mobile_station_units)
        committed_bc = sum(unit.state in {"station_bc", "transit_to_bc"} for unit in self.mobile_station_units)
        middle_available = [unit for unit in self.mobile_station_units if unit.state == "middle_available"]
        need_ab = max(desired_ab - committed_ab, 0)
        need_bc = max(desired_bc - committed_bc, 0)

        use_for_ab = min(need_ab, len(middle_available))
        for unit in middle_available[:use_for_ab]:
            unit.state = "transit_to_ab"
            unit.timer_steps = self.middle_travel_steps

        middle_available = [unit for unit in self.mobile_station_units if unit.state == "middle_available"]
        use_for_bc = min(need_bc, len(middle_available))
        for unit in middle_available[:use_for_bc]:
            unit.state = "transit_to_bc"
            unit.timer_steps = self.middle_travel_steps

        self._refresh_mobile_station_counts()

    def action_to_allocation(self, action: int) -> np.ndarray:
        return np.asarray(self.action_map[int(action)], dtype=np.int32)

    def action_from_mobile_station_allocation(self, allocation: list[int] | tuple[int, int] | np.ndarray) -> int:
        first, second = [int(value) for value in allocation]
        key = (first, second)
        try:
            return self.action_map.index(key)
        except ValueError as exc:
            raise ValueError(f"Invalid mobile station allocation: {key}") from exc

    def seed(self, seed: int | None = None) -> None:
        chosen_seed = self.base_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(chosen_seed)
        self.simulator = self._build_simulator(seed=chosen_seed, duration_hours=self._sample_episode_duration_hours())
        self.planning_step_minutes = float(self.simulator.step_minutes)
        self.mean_service_minutes = float(self.simulator.service_mean_minutes)
        self.horizon = int(self.simulator.steps_per_day)
        self.max_steps = int(self.simulator.num_steps)
        self.middle_travel_steps = max(1, int(round(self.mcs_middle_travel_minutes / self.planning_step_minutes)))
        self.relocation_steps = max(1, self.middle_travel_steps * 2)
        self.charge_full_steps = max(1, int(round(self.mcs_charge_full_minutes / self.planning_step_minutes)))

    def valid_action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def expected_vehicle_arrivals(self) -> float:
        return float(self.expected_vehicle_arrivals_by_station().sum())

    def expected_vehicle_arrivals_by_station(self) -> np.ndarray:
        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multipliers_by_trip=disruption_state["demand_multipliers_by_trip"],
        )
        return np.asarray(expected["station_expected_charging"], dtype=np.float32)

    def current_service_capacity_per_step(self, service_time_multiplier: float = 1.0, effective_base_plugs: int | None = None) -> float:
        del effective_base_plugs
        return float(self.current_service_capacity_per_step_by_station(service_time_multipliers=np.full(2, service_time_multiplier)).sum())

    def current_service_capacity_per_step_by_station(
        self,
        service_time_multipliers: np.ndarray | None = None,
        effective_base_plugs: np.ndarray | None = None,
    ) -> np.ndarray:
        multipliers = np.ones(2, dtype=np.float32) if service_time_multipliers is None else np.asarray(service_time_multipliers, dtype=np.float32)
        base_plugs = self.base_station_plugs_by_station if effective_base_plugs is None else np.asarray(effective_base_plugs, dtype=np.float32)
        total_plugs = base_plugs + self.current_mobile_stations_by_station.astype(np.float32) * float(self.mobile_station_chargers)
        return total_plugs * float(self.planning_step_minutes) / np.maximum(self.mean_service_minutes * multipliers, 1e-6)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None and self.episode_seed_range is not None:
            low, high = self.episode_seed_range
            if high < low:
                low, high = high, low
            episode_seed = int(self.seed_stream_rng.integers(low, high + 1))
        else:
            episode_seed = self.base_seed + self.reset_counter * self.reset_seed_stride if seed is None else int(seed)
        self.reset_counter += 1
        self.seed(episode_seed)
        self.current_episode_seed = int(episode_seed)
        self.queue_by_station = [deque(), deque()]
        self.active_sessions_by_station = [[], []]
        self.mobile_station_units = self._build_mobile_station_units()
        self.current_mobile_stations_by_station = np.zeros(2, dtype=np.int32)
        self.step_index = 0
        disruption_schedule = self.simulator._build_disruption_schedule()
        self.current_disruption_schedule = list(disruption_schedule)
        self.disruption_by_step = self.simulator._event_map(disruption_schedule)

        self.last_arrivals_by_station = np.zeros(2, dtype=np.float32)
        self.last_starts_by_station = np.zeros(2, dtype=np.float32)
        self.last_completions_by_station = np.zeros(2, dtype=np.float32)
        self.queue_lengths_by_station = np.zeros(2, dtype=np.float32)
        self.queue_wait_mean_by_station = np.zeros(2, dtype=np.float32)
        self.last_expected_station_arrivals = np.zeros(2, dtype=np.float32)
        self.last_effective_num_plugs_by_station = self.base_station_plugs_by_station.astype(np.float32)
        self.last_disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)

        return self._get_observation(), {
            "fixed_site_indices": [0, 1],
            "fixed_site_coords_km": self.fixed_site_coords_km.tolist(),
            "num_active_mobile_stations": 0,
            "num_active_mobile_stations_station_ab": 0,
            "num_active_mobile_stations_station_bc": 0,
            "num_mobile_stations_middle_available": self.max_mobile_stations,
            "num_mobile_stations_middle_charging": 0,
            "num_mobile_stations_in_transit_to_ab": 0,
            "num_mobile_stations_in_transit_to_bc": 0,
        }

    def _normalized(self, value: float, scale: float) -> float:
        return float(value) / max(scale, 1e-6)

    def _affected_station_indices_for_target(self, disruption_target: str) -> list[int]:
        target = str(disruption_target)
        if target == "station_ab":
            return [0]
        if target == "station_bc":
            return [1]
        if target == "all_stations":
            return [0, 1]
        if target in getattr(self.simulator, "trip_definitions", {}):
            weights = np.asarray(self.simulator.trip_definitions[target].station_weights, dtype=np.float32)
            return [int(index) for index, weight in enumerate(weights) if float(weight) > 0.0]
        if target in {"eastbound", "westbound", "all_ods"}:
            return [0, 1]
        return []

    def _target_affects_station_flags(self, disruption_target: str) -> tuple[float, float]:
        affected_station_indices = self._affected_station_indices_for_target(disruption_target)
        return (
            1.0 if 0 in affected_station_indices else 0.0,
            1.0 if 1 in affected_station_indices else 0.0,
        )

    def _get_observation(self) -> np.ndarray:
        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multipliers_by_trip=disruption_state["demand_multipliers_by_trip"],
        )
        effective_base_plugs_by_station = np.asarray(disruption_state["effective_num_plugs_by_station"], dtype=np.float32)
        effective_total_plugs_by_station = (
            effective_base_plugs_by_station
            + self.current_mobile_stations_by_station.astype(np.float32) * float(self.mobile_station_chargers)
        )
        service_time_multipliers = np.asarray(disruption_state["service_time_multiplier_by_station"], dtype=np.float32)
        total_service_capacity_by_station = self.current_service_capacity_per_step_by_station(
            service_time_multipliers=service_time_multipliers,
            effective_base_plugs=effective_total_plugs_by_station,
        )
        local_deficit_by_station = np.maximum(
            self.queue_lengths_by_station.astype(np.float32)
            + np.asarray(expected["station_expected_charging"], dtype=np.float32)
            - total_service_capacity_by_station.astype(np.float32),
            0.0,
        )
        local_deficit_bias = float(local_deficit_by_station[0] - local_deficit_by_station[1])
        time_fraction = float(self.step_index % self.horizon) / float(max(self.horizon - 1, 1))
        target_affects_ab, target_affects_bc = self._target_affects_station_flags(
            str(disruption_state.get("disruption_target", "none"))
        )
        mobile_state_counts = self._mobile_station_state_counts()
        obs = np.asarray(
            [
                self._normalized(self.queue_lengths_by_station[0], self.queue_normalizer),
                self._normalized(self.queue_lengths_by_station[1], self.queue_normalizer),
                self._normalized(self.queue_wait_mean_by_station[0], self.planning_step_minutes * 4.0),
                self._normalized(self.queue_wait_mean_by_station[1], self.planning_step_minutes * 4.0),
                self._normalized(self.last_arrivals_by_station[0], self.arrival_normalizer),
                self._normalized(self.last_arrivals_by_station[1], self.arrival_normalizer),
                self._normalized(self.last_starts_by_station[0], self.arrival_normalizer),
                self._normalized(self.last_starts_by_station[1], self.arrival_normalizer),
                self._normalized(float(expected["station_expected_charging"][0]), self.arrival_normalizer),
                self._normalized(float(expected["station_expected_charging"][1]), self.arrival_normalizer),
                self._normalized(float(effective_total_plugs_by_station[0]), self.capacity_normalizer),
                self._normalized(float(effective_total_plugs_by_station[1]), self.capacity_normalizer),
                self._normalized(float(self.current_mobile_stations_by_station[0] * self.mobile_station_chargers), self.capacity_normalizer),
                self._normalized(float(self.current_mobile_stations_by_station[1] * self.mobile_station_chargers), self.capacity_normalizer),
                self._normalized(float(local_deficit_by_station[0]), self.queue_normalizer + self.arrival_normalizer),
                self._normalized(float(local_deficit_by_station[1]), self.queue_normalizer + self.arrival_normalizer),
                self._normalized(local_deficit_bias, self.queue_normalizer + self.arrival_normalizer),
                float(disruption_state["disruption_active"]),
                float(disruption_state["disruption_type_code"]) / 4.0,
                self._normalized(float(disruption_state["disruption_remaining_minutes"]), 24.0 * 60.0),
                target_affects_ab,
                target_affects_bc,
                self._normalized(float(mobile_state_counts["middle_available"]), self.mobile_count_normalizer),
                self._normalized(float(mobile_state_counts["middle_charging"]), self.mobile_count_normalizer),
                self._normalized(float(mobile_state_counts["transit_to_ab"]), self.mobile_count_normalizer),
                self._normalized(float(mobile_state_counts["transit_to_bc"]), self.mobile_count_normalizer),
                np.sin(2.0 * np.pi * time_fraction),
                np.cos(2.0 * np.pi * time_fraction),
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._advance_mobile_station_units()
        previous_mobile_stations = self.current_mobile_stations_by_station.copy()
        chosen_mobile_stations = self.action_to_allocation(action)
        self._dispatch_mobile_station_units(chosen_mobile_stations)

        completed_now_by_station: list[list[ActiveSession]] = [[], []]
        completed_waits_by_station = [[], []]
        for station_index in range(2):
            completed_now_by_station[station_index] = [
                session for session in self.active_sessions_by_station[station_index] if session.end_step <= self.step_index
            ]
            self.active_sessions_by_station[station_index] = [
                session for session in self.active_sessions_by_station[station_index] if session.end_step > self.step_index
            ]
            completed_waits_by_station[station_index] = [
                (session.start_step - session.arrival_step) * self.simulator.step_minutes
                for session in completed_now_by_station[station_index]
            ]

        disruption_state = self.simulator._disruption_state(self.step_index, self.disruption_by_step)
        if disruption_state["disruption_type"] == "station_outage" and disruption_state["disruption_active"] == 1:
            event = self.disruption_by_step[self.step_index]
            if event.start_step == self.step_index:
                targeted = range(2) if event.target == "all_stations" else [self.station_keys.index(str(event.target))]
                for station_index in targeted:
                    self.simulator._requeue_active_sessions(self.queue_by_station, self.active_sessions_by_station, station_index)
                    self.active_sessions_by_station[station_index] = []

        expected = self.simulator.expected_traffic(
            step=self.step_index,
            demand_multipliers_by_trip=disruption_state["demand_multipliers_by_trip"],
        )
        arrivals = self.simulator._sample_arrivals(self.step_index, expected["expected_by_trip"])

        arrivals_by_station = np.zeros(2, dtype=np.int32)
        for vehicle in arrivals:
            vehicle.arrival_step = self.step_index
            self.queue_by_station[vehicle.station_index].append(vehicle)
            arrivals_by_station[vehicle.station_index] += 1
        pending_queue_lengths_before_service = np.asarray([len(queue) for queue in self.queue_by_station], dtype=np.float32)

        starts_now_by_station: list[list[ActiveSession]] = [[], []]
        for station_index in range(2):
            base_effective_num_plugs = int(disruption_state["effective_num_plugs_by_station"][station_index])
            total_effective_num_plugs = base_effective_num_plugs + int(
                self.current_mobile_stations_by_station[station_index] * self.mobile_station_chargers
            )
            available_plugs = max(total_effective_num_plugs - len(self.active_sessions_by_station[station_index]), 0)
            for _ in range(min(available_plugs, len(self.queue_by_station[station_index]))):
                vehicle = self.queue_by_station[station_index].popleft()
                service_minutes = self.simulator._apply_service_time_multiplier(
                    base_service_minutes=vehicle.base_service_minutes,
                    service_time_multiplier=float(disruption_state["service_time_multiplier_by_station"][station_index]),
                )
                service_steps = max(1, int(np.ceil(service_minutes / self.simulator.step_minutes)))
                session = ActiveSession(
                    arrival_step=vehicle.arrival_step,
                    start_step=self.step_index,
                    end_step=self.step_index + service_steps,
                    trip_key=vehicle.trip_key,
                    station_index=station_index,
                    service_minutes=service_minutes,
                )
                self.active_sessions_by_station[station_index].append(session)
                starts_now_by_station[station_index].append(session)

        started_waits_by_station = [
            [(session.start_step - session.arrival_step) * self.simulator.step_minutes for session in sessions]
            for sessions in starts_now_by_station
        ]
        queue_lengths = np.asarray([len(queue) for queue in self.queue_by_station], dtype=np.float32)
        queue_wait_means = []
        for station_index in range(2):
            wait_minutes = [(self.step_index - vehicle.arrival_step) * self.simulator.step_minutes for vehicle in self.queue_by_station[station_index]]
            queue_wait_means.append(float(np.mean(wait_minutes)) if wait_minutes else 0.0)
        queue_wait_means = np.asarray(queue_wait_means, dtype=np.float32)

        active_plugs_by_station = np.asarray(
            [len(self.active_sessions_by_station[idx]) for idx in range(2)],
            dtype=np.float32,
        )
        effective_num_plugs_by_station = np.asarray(
            disruption_state["effective_num_plugs_by_station"],
            dtype=np.float32,
        ) + self.current_mobile_stations_by_station.astype(np.float32) * float(self.mobile_station_chargers)
        utilization_by_station = np.divide(
            active_plugs_by_station,
            np.maximum(effective_num_plugs_by_station, 1.0),
            out=np.zeros_like(active_plugs_by_station),
            where=effective_num_plugs_by_station > 0,
        )

        mobile_chargers_total_by_station = self.current_mobile_stations_by_station.astype(np.float32) * float(self.mobile_station_chargers)
        mobile_chargers_used_estimate_by_station = np.clip(
            active_plugs_by_station - np.asarray(disruption_state["effective_num_plugs_by_station"], dtype=np.float32),
            0.0,
            mobile_chargers_total_by_station,
        )
        unused_mobile_chargers_by_station = np.maximum(mobile_chargers_total_by_station - mobile_chargers_used_estimate_by_station, 0.0)
        unused_mobile_stations_by_station = unused_mobile_chargers_by_station / max(float(self.mobile_station_chargers), 1e-6)

        served_total = float(sum(len(sessions) for sessions in starts_now_by_station))
        unmet_total = float(queue_lengths.sum())
        idle_capacity = max(float(effective_num_plugs_by_station.sum() - active_plugs_by_station.sum()), 0.0)
        activated = np.maximum(self.current_mobile_stations_by_station - previous_mobile_stations, 0)
        adjusted = np.abs(self.current_mobile_stations_by_station - previous_mobile_stations)
        mean_queue_wait = float(queue_wait_means.mean())
        queue_wait_burden = float(np.dot(queue_lengths.astype(np.float32), queue_wait_means.astype(np.float32)))
        base_effective_num_plugs_by_station = np.asarray(disruption_state["effective_num_plugs_by_station"], dtype=np.float32)
        service_time_multipliers = np.asarray(disruption_state["service_time_multiplier_by_station"], dtype=np.float32)
        base_service_capacity_by_station = self.current_service_capacity_per_step_by_station(
            service_time_multipliers=service_time_multipliers,
            effective_base_plugs=base_effective_num_plugs_by_station,
        )
        mobile_service_capacity_by_station = np.maximum(
            effective_num_plugs_by_station.astype(np.float32) - base_effective_num_plugs_by_station,
            0.0,
        ) * float(self.planning_step_minutes) / np.maximum(self.mean_service_minutes * service_time_multipliers, 1e-6)
        local_deficit_without_mcs_by_station = np.maximum(
            pending_queue_lengths_before_service + np.asarray(expected["station_expected_charging"], dtype=np.float32) - base_service_capacity_by_station,
            0.0,
        )
        total_local_deficit_without_mcs = float(local_deficit_without_mcs_by_station.sum())
        spatial_deficit_coverage = 0.0
        if total_local_deficit_without_mcs > 0.0:
            covered_deficit = np.minimum(mobile_service_capacity_by_station, local_deficit_without_mcs_by_station)
            spatial_deficit_coverage = float(covered_deficit.sum() / total_local_deficit_without_mcs)
        spatial_deficit_direction_alignment = 0.0
        total_active_mobile_stations = float(self.current_mobile_stations_by_station.sum())
        if total_local_deficit_without_mcs > 0.0 and total_active_mobile_stations > 0.0:
            deficit_bias_norm = float(
                (local_deficit_without_mcs_by_station[0] - local_deficit_without_mcs_by_station[1])
                / total_local_deficit_without_mcs
            )
            allocation_bias_norm = float(
                (self.current_mobile_stations_by_station[0] - self.current_mobile_stations_by_station[1])
                / total_active_mobile_stations
            )
            spatial_deficit_direction_alignment = deficit_bias_norm * allocation_bias_norm
        affected_station_indices = self._affected_station_indices_for_target(disruption_state.get("disruption_target", "none"))
        disruption_response_bonus_term = 0.0
        if float(disruption_state["disruption_active"]) > 0.0 and affected_station_indices:
            has_response_on_affected_station = bool(self.current_mobile_stations_by_station[affected_station_indices].sum() > 0)
            disruption_response_bonus_term = self.disruption_response_bonus * float(has_response_on_affected_station)
        spatial_deficit_alignment_bonus_term = self.spatial_deficit_alignment_bonus * spatial_deficit_coverage
        spatial_deficit_direction_bonus_term = self.spatial_deficit_direction_bonus * spatial_deficit_direction_alignment
        reward = (
            self.served_reward_weight * served_total
            - self.unmet_penalty * unmet_total
            - self.queue_length_penalty * unmet_total
            - self.queue_wait_penalty * queue_wait_burden
            - self.active_mobile_station_cost * float(self.current_mobile_stations_by_station.sum())
            - self.activation_cost * float(activated.sum())
            - self.adjustment_cost * float(adjusted.sum())
            - self.idle_capacity_penalty * idle_capacity
            + self.utilization_bonus * float(utilization_by_station.mean())
            + disruption_response_bonus_term
            + spatial_deficit_alignment_bonus_term
            + spatial_deficit_direction_bonus_term
        )
        reward *= self.reward_scale

        self.last_arrivals_by_station = arrivals_by_station.astype(np.float32)
        self.last_starts_by_station = np.asarray([len(items) for items in starts_now_by_station], dtype=np.float32)
        self.last_completions_by_station = np.asarray([len(items) for items in completed_now_by_station], dtype=np.float32)
        self.queue_lengths_by_station = queue_lengths
        self.queue_wait_mean_by_station = queue_wait_means
        self.last_expected_station_arrivals = np.asarray(expected["station_expected_charging"], dtype=np.float32)
        self.last_effective_num_plugs_by_station = effective_num_plugs_by_station
        self.last_disruption_state = dict(disruption_state)
        mobile_state_counts = self._mobile_station_state_counts()
        target_affects_ab, target_affects_bc = self._target_affects_station_flags(str(disruption_state.get("disruption_target", "none")))

        info = {
            "served_demand": served_total,
            "unmet_demand": unmet_total,
            "true_demand_total": float(len(arrivals)),
            "expected_demand_total": float(np.asarray(expected["station_expected_charging"]).sum()),
            "expected_arrivals_vehicles": float(np.asarray(expected["station_expected_charging"]).sum()),
            "arrivals_vehicles": float(len(arrivals)),
            "queue_length": float(queue_lengths.sum()),
            "queue_wait_mean_minutes": mean_queue_wait,
            "queue_wait_burden_minutes": queue_wait_burden,
            "local_deficit_without_mcs_station_ab": float(local_deficit_without_mcs_by_station[0]),
            "local_deficit_without_mcs_station_bc": float(local_deficit_without_mcs_by_station[1]),
            "local_deficit_without_mcs_total": float(total_local_deficit_without_mcs),
            "local_deficit_bias_ab_minus_bc": float(local_deficit_without_mcs_by_station[0] - local_deficit_without_mcs_by_station[1]),
            "mobile_service_capacity_station_ab": float(mobile_service_capacity_by_station[0]),
            "mobile_service_capacity_station_bc": float(mobile_service_capacity_by_station[1]),
            "spatial_deficit_coverage": float(spatial_deficit_coverage),
            "spatial_deficit_direction_alignment": float(spatial_deficit_direction_alignment),
            "service_capacity_vehicles": float(effective_num_plugs_by_station.sum()),
            "num_active_mobile_stations": int(self.current_mobile_stations_by_station.sum()),
            "num_active_mobile_stations_station_ab": int(self.current_mobile_stations_by_station[0]),
            "num_active_mobile_stations_station_bc": int(self.current_mobile_stations_by_station[1]),
            "num_mobile_stations_middle_available": int(mobile_state_counts["middle_available"]),
            "num_mobile_stations_middle_charging": int(mobile_state_counts["middle_charging"]),
            "num_mobile_stations_in_transit_to_ab": int(mobile_state_counts["transit_to_ab"]),
            "num_mobile_stations_in_transit_to_bc": int(mobile_state_counts["transit_to_bc"]),
            "num_active_chargers": int(effective_num_plugs_by_station.sum()),
            "active_plugs": int(active_plugs_by_station.sum()),
            "base_capacity_total": float(np.asarray(disruption_state["effective_num_plugs_by_station"]).sum()),
            "mobile_capacity_total": float(mobile_chargers_total_by_station.sum()),
            "effective_capacity_total": float(effective_num_plugs_by_station.sum()),
            "effective_num_plugs": float(effective_num_plugs_by_station.sum()),
            "unused_mobile_chargers": float(unused_mobile_chargers_by_station.sum()),
            "unused_mobile_stations_estimate": float(unused_mobile_stations_by_station.sum()),
            "unused_mobile_chargers_station_ab": float(unused_mobile_chargers_by_station[0]),
            "unused_mobile_chargers_station_bc": float(unused_mobile_chargers_by_station[1]),
            "unused_mobile_stations_estimate_station_ab": float(unused_mobile_stations_by_station[0]),
            "unused_mobile_stations_estimate_station_bc": float(unused_mobile_stations_by_station[1]),
            "action_valid": True,
            "utilization": float(utilization_by_station.mean()),
            "utilization_station_ab": float(utilization_by_station[0]),
            "utilization_station_bc": float(utilization_by_station[1]),
            "activated_mobile_stations": int(activated.sum()),
            "adjusted_mobile_stations": int(adjusted.sum()),
            "idle_capacity": idle_capacity,
            "reward_queue_wait_penalty_term": float(self.queue_wait_penalty * queue_wait_burden),
            "reward_disruption_response_bonus_term": float(disruption_response_bonus_term),
            "reward_spatial_deficit_alignment_bonus_term": float(spatial_deficit_alignment_bonus_term),
            "reward_spatial_deficit_direction_bonus_term": float(spatial_deficit_direction_bonus_term),
            "disruption_active": int(disruption_state["disruption_active"]),
            "disruption_type": str(disruption_state["disruption_type"]),
            "disruption_type_code": int(disruption_state["disruption_type_code"]),
            "disruption_target": str(disruption_state.get("disruption_target", "none")),
            "disruption_target_code": int(disruption_state.get("disruption_target_code", 0)),
            "target_affects_ab": float(target_affects_ab),
            "target_affects_bc": float(target_affects_bc),
            "disruption_day_index": int(disruption_state["disruption_day_index"]),
            "disruption_remaining_steps": float(disruption_state["disruption_remaining_minutes"]) / float(self.simulator.step_minutes),
            "disruption_remaining_minutes": float(disruption_state["disruption_remaining_minutes"]),
            "effective_base_plugs": int(np.asarray(disruption_state["effective_num_plugs_by_station"]).sum()),
            "effective_base_plugs_station_ab": int(disruption_state["effective_num_plugs_by_station"][0]),
            "effective_base_plugs_station_bc": int(disruption_state["effective_num_plugs_by_station"][1]),
            "arrivals_total": float(len(arrivals)),
            "arrivals_total_station_ab": float(arrivals_by_station[0]),
            "arrivals_total_station_bc": float(arrivals_by_station[1]),
            "queue_length_station_ab": float(queue_lengths[0]),
            "queue_length_station_bc": float(queue_lengths[1]),
            "queue_wait_mean_minutes_station_ab": float(queue_wait_means[0]),
            "queue_wait_mean_minutes_station_bc": float(queue_wait_means[1]),
            "expected_station_arrivals_ab": float(expected["station_expected_charging"][0]),
            "expected_station_arrivals_bc": float(expected["station_expected_charging"][1]),
            "expected_passing_total": float(expected["total_passing_expected"]),
            "expected_passing_od_ab": float(expected["expected_by_trip"]["od_ab"]),
            "expected_passing_od_ba": float(expected["expected_by_trip"]["od_ba"]),
            "expected_passing_od_bc": float(expected["expected_by_trip"]["od_bc"]),
            "expected_passing_od_cb": float(expected["expected_by_trip"]["od_cb"]),
            "expected_passing_od_ac": float(expected["expected_by_trip"]["od_ac"]),
            "expected_passing_od_ca": float(expected["expected_by_trip"]["od_ca"]),
            "reward": float(reward),
        }

        self.step_index += 1
        terminated = self.step_index >= self.max_steps
        return self._get_observation(), float(reward), terminated, False, info

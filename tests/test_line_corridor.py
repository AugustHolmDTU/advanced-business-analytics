from collections import deque
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evch.baselines.policies import mobile_threshold_policy
from evch.envs.factory import make_env
from evch.envs.line_corridor_mobile_env import LineCorridorMobileStationEnv, MobileChargingStationUnit
from evch.sim.line_corridor import ActiveSession, LineCorridorQueueSimulator, QueuedVehicle
from evch.train.run_mobile_noop_comparison import run_mobile_noop_comparison

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _base_line_config() -> dict:
    return {
        "city_names": ["City A", "City B", "City C"],
        "inter_city_distance_km": 100.0,
        "station_positions_km": [50.0, 150.0],
        "num_plugs": [12, 12],
        "step_minutes": 5,
        "duration_hours": 24.0,
        "charging_stop_probability": 0.055,
        "service_time": {
            "mean_minutes": 30.0,
            "std_minutes": 8.0,
            "min_minutes": 15.0,
            "max_minutes": 50.0,
        },
        "traffic": {
            "baseline_cars_per_step": 18.0,
            "morning_peak_hour": 8.0,
            "evening_peak_hour": 17.0,
            "morning_peak_cars_per_step": 30.0,
            "evening_peak_cars_per_step": 32.0,
            "midday_bump_hour": 12.5,
            "midday_bump_cars_per_step": 8.0,
            "peak_width_hours": 1.6,
            "directional_bias_amplitude": 0.24,
            "middle_city_share": 0.34,
            "long_trip_share": 0.35,
            "middle_destination_bias_amplitude": 0.18,
        },
        "disruption": {
            "enabled": True,
            "mode": "scripted",
            "scripted_events": [
                {
                    "disruption_type": "capacity_drop",
                    "target": "station_ab",
                    "day_index": 0,
                    "start_hour": 6.0,
                    "duration_hours": 2.0,
                    "severity": 4.0,
                },
                {
                    "disruption_type": "demand_surge",
                    "target": "od_ac",
                    "day_index": 0,
                    "start_hour": 16.0,
                    "duration_hours": 2.0,
                    "severity": 1.8,
                },
            ],
        },
    }


class LineCorridorQueueSimulatorTest(unittest.TestCase):
    def test_expected_traffic_has_six_od_pairs_and_two_station_demands(self) -> None:
        simulator = LineCorridorQueueSimulator(config=_base_line_config(), seed=7)

        expected = simulator.expected_traffic(step=96)

        self.assertEqual(set(expected["expected_by_trip"].keys()), set(simulator.TRIP_KEYS))
        self.assertEqual(len(expected["station_expected_charging"]), 2)
        self.assertGreater(expected["expected_passing_total"], 0.0)
        self.assertGreater(expected["station_expected_charging"][0], 0.0)
        self.assertGreater(expected["station_expected_charging"][1], 0.0)

    def test_long_trip_demand_splits_across_both_stations(self) -> None:
        simulator = LineCorridorQueueSimulator(config=_base_line_config(), seed=9)

        expected = simulator.expected_traffic(step=96, demand_multipliers_by_trip={"od_ac": 2.0})

        self.assertGreater(expected["expected_by_trip"]["od_ac"], 0.0)
        self.assertGreater(expected["station_expected_charging"][0], 0.0)
        self.assertGreater(expected["station_expected_charging"][1], 0.0)

    def test_run_exposes_station_specific_columns(self) -> None:
        simulator = LineCorridorQueueSimulator(config=_base_line_config(), seed=5)

        result = simulator.run()

        self.assertIn("queue_length_station_ab", result.metrics.columns)
        self.assertIn("queue_length_station_bc", result.metrics.columns)
        self.assertIn("expected_passing_od_ac", result.metrics.columns)
        self.assertIn("disruption_target", result.metrics.columns)


class LineCorridorMobileStationEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_config = {
            "env_type": "line_corridor_mobile_mcs",
            "max_mobile_stations": 4,
            "mobile_station_chargers": 2,
            "mobile_station_capacity": 20.0,
            "reward_scale": 0.1,
            "reward": {
                "served_reward_weight": 1.0,
                "unmet_penalty": 2.0,
                "active_mobile_station_cost": 18.0,
                "activation_cost": 5.0,
                "adjustment_cost": 2.0,
                "idle_capacity_penalty": 0.1,
                "utilization_bonus": 1.0,
                "queue_length_penalty": 2.0,
                "queue_wait_penalty": 0.02,
            },
            "simulation": _base_line_config(),
        }

    def test_factory_builds_line_corridor_mobile_env(self) -> None:
        env = make_env(self.env_config, {}, seed=5)
        self.assertIsInstance(env, LineCorridorMobileStationEnv)

    def test_action_map_covers_two_station_allocations(self) -> None:
        env = LineCorridorMobileStationEnv(self.env_config, {}, seed=3)
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(env.action_from_mobile_station_allocation((0, 0)), 0)
        env.reset(seed=4)
        self.assertEqual(tuple(env.action_to_allocation(env.action_from_mobile_station_allocation((1, 0)))), (1, 0))

    def test_step_reports_station_specific_mobile_counts(self) -> None:
        env = LineCorridorMobileStationEnv(self.env_config, {}, seed=3)
        obs, info = env.reset(seed=4)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["num_active_mobile_stations"], 0)

        action = env.action_from_mobile_station_allocation((1, 0))
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["num_active_mobile_stations"], 0)
        self.assertEqual(step_info["num_active_mobile_stations_station_ab"], 0)
        self.assertEqual(step_info["num_active_mobile_stations_station_bc"], 0)
        self.assertEqual(step_info["committed_mobile_stations_station_ab"], 1)
        self.assertEqual(step_info["committed_mobile_stations_station_bc"], 0)
        self.assertEqual(step_info["committed_mobile_stations_bias_ab_minus_bc"], 1)
        self.assertEqual(step_info["num_mobile_stations_in_transit_to_ab"], 1)
        self.assertEqual(step_info["num_mobile_stations_in_transit_to_bc"], 0)
        self.assertEqual(step_info["num_mobile_stations_in_transit_to_middle"], 0)
        self.assertIn("queue_length_station_ab", step_info)
        self.assertIn("queue_length_station_bc", step_info)
        self.assertIn("unused_mobile_stations_estimate_station_ab", step_info)

    def test_observation_includes_target_affects_flags(self) -> None:
        env_config = {
            **self.env_config,
            "simulation": {
                **self.env_config["simulation"],
                "disruption": {
                    "enabled": True,
                    "mode": "scripted",
                    "scripted_events": [
                        {
                            "disruption_type": "demand_surge",
                            "target": "od_bc",
                            "day_index": 0,
                            "start_hour": 0.0,
                            "duration_hours": 1.0,
                            "severity": 3.0,
                        }
                    ],
                },
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)

        obs, _ = env.reset(seed=4)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(obs[22], 0.0)
        self.assertEqual(obs[23], 1.0)
        self.assertEqual(obs[28], 0.0)

    def test_reset_can_sample_one_to_three_day_episode_lengths(self) -> None:
        env_config = {
            **self.env_config,
            "simulation": {
                **self.env_config["simulation"],
                "duration_days_range": [1, 3],
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)

        sampled_steps = set()
        for seed in range(1, 16):
            env.reset(seed=seed)
            sampled_steps.add(env.max_steps)

        self.assertTrue(sampled_steps.issubset({288, 576, 864}))
        self.assertGreater(len(sampled_steps), 1)

    def test_reset_reproduces_same_randomized_scenario_for_same_seed(self) -> None:
        env_config = {
            **self.env_config,
            "simulation": {
                **self.env_config["simulation"],
                "duration_days_range": [1, 3],
                "disruption": {
                    "enabled": True,
                    "mode": "random",
                    "day_disruption_count_weights": {0: 0.2, 1: 0.6, 2: 0.2},
                    "event_types": ["capacity_drop", "demand_surge"],
                    "capacity_drop": {
                        "duration_hours": [1.0, 2.0],
                        "start_hour_range": [6.0, 12.0],
                        "target_num_plugs": [3, 5],
                        "targets": ["station_ab", "station_bc"],
                    },
                    "demand_surge": {
                        "duration_hours": [1.0, 2.0],
                        "start_hour_range": [8.0, 18.0],
                        "multiplier": [1.6, 2.0],
                        "targets": ["od_ab", "od_bc"],
                    },
                },
            },
        }
        env_a = LineCorridorMobileStationEnv(env_config, {}, seed=11)
        env_b = LineCorridorMobileStationEnv(env_config, {}, seed=99)

        env_a.reset(seed=1234)
        env_b.reset(seed=1234)

        schedule_a = [
            (event.disruption_type, event.target, event.day_index, round(event.start_hour, 3), event.start_step, event.end_step, round(event.severity, 3))
            for event in env_a.current_disruption_schedule
        ]
        schedule_b = [
            (event.disruption_type, event.target, event.day_index, round(event.start_hour, 3), event.start_step, event.end_step, round(event.severity, 3))
            for event in env_b.current_disruption_schedule
        ]

        self.assertEqual(env_a.max_steps, env_b.max_steps)
        self.assertEqual(schedule_a, schedule_b)

    def test_mobile_threshold_policy_returns_valid_two_station_action(self) -> None:
        env = LineCorridorMobileStationEnv(self.env_config, {}, seed=3)
        obs, _ = env.reset(seed=4)

        action = mobile_threshold_policy(obs, env)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.action_space.n)

    def test_reward_queue_wait_penalty_uses_local_queue_burden(self) -> None:
        env_config = {
            **self.env_config,
            "reward_scale": 1.0,
            "reward": {
                **self.env_config["reward"],
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 1.0,
                "disruption_response_bonus": 0.0,
            },
            "simulation": {
                **self.env_config["simulation"],
                "charging_stop_probability": 0.0,
                "num_plugs": [1, 1],
                "disruption": {"enabled": False},
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)
        env.reset(seed=4)
        env.active_sessions_by_station = [
            [],
            [
                ActiveSession(
                    arrival_step=0,
                    start_step=0,
                    end_step=100,
                    trip_key="od_bc",
                    station_index=1,
                    service_minutes=30.0,
                )
            ],
        ]
        env.queue_by_station = [
            deque(),
            deque(
                [
                    QueuedVehicle(
                        arrival_step=-4,
                        trip_key="od_bc",
                        station_index=1,
                        base_service_minutes=30.0,
                    )
                ]
            ),
        ]

        _, reward, _, _, info = env.step(env.action_from_mobile_station_allocation((0, 0)))

        self.assertAlmostEqual(info["queue_wait_mean_minutes_station_ab"], 0.0)
        self.assertAlmostEqual(info["queue_wait_mean_minutes_station_bc"], 20.0)
        self.assertAlmostEqual(info["queue_wait_burden_minutes"], 20.0)
        self.assertAlmostEqual(info["reward_queue_wait_penalty_term"], 20.0)
        self.assertAlmostEqual(reward, -20.0)

    def test_local_queue_peak_penalty_punishes_one_sided_queue_blowup(self) -> None:
        env_config = {
            **self.env_config,
            "reward_scale": 1.0,
            "reward": {
                **self.env_config["reward"],
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 0.0,
                "local_queue_peak_penalty": 2.0,
                "disruption_response_bonus": 0.0,
            },
            "simulation": {
                **self.env_config["simulation"],
                "charging_stop_probability": 0.0,
                "num_plugs": [1, 1],
                "disruption": {"enabled": False},
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)
        env.reset(seed=4)
        env.queue_by_station = [
            deque(
                [
                    QueuedVehicle(arrival_step=-1, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                    QueuedVehicle(arrival_step=-1, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                    QueuedVehicle(arrival_step=-1, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                ]
            ),
            deque(),
        ]

        _, reward, _, _, info = env.step(env.action_from_mobile_station_allocation((0, 0)))

        self.assertAlmostEqual(info["queue_length_station_ab"], 2.0)
        self.assertAlmostEqual(info["queue_length_station_bc"], 0.0)
        self.assertAlmostEqual(info["reward_local_queue_peak_penalty_term"], 4.0)
        self.assertAlmostEqual(reward, -4.0)

    def test_local_wait_peak_penalty_punishes_worst_station_wait(self) -> None:
        env_config = {
            **self.env_config,
            "reward_scale": 1.0,
            "reward": {
                **self.env_config["reward"],
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 0.0,
                "local_queue_peak_penalty": 0.0,
                "local_wait_peak_penalty": 2.0,
                "disruption_response_bonus": 0.0,
            },
            "simulation": {
                **self.env_config["simulation"],
                "charging_stop_probability": 0.0,
                "num_plugs": [1, 1],
                "disruption": {"enabled": False},
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)
        env.reset(seed=4)
        env.active_sessions_by_station = [
            [],
            [
                ActiveSession(
                    arrival_step=0,
                    start_step=0,
                    end_step=100,
                    trip_key="od_bc",
                    station_index=1,
                    service_minutes=30.0,
                )
            ],
        ]
        env.queue_by_station = [
            deque(),
            deque(
                [
                    QueuedVehicle(
                        arrival_step=-4,
                        trip_key="od_bc",
                        station_index=1,
                        base_service_minutes=30.0,
                    )
                ]
            ),
        ]

        _, reward, _, _, info = env.step(env.action_from_mobile_station_allocation((0, 0)))

        self.assertAlmostEqual(info["queue_wait_mean_minutes_station_bc"], 20.0)
        self.assertAlmostEqual(info["reward_local_wait_peak_penalty_term"], 40.0)
        self.assertAlmostEqual(reward, -40.0)

    def test_disruption_response_bonus_only_rewards_affected_station(self) -> None:
        env_config = {
            **self.env_config,
            "reward_scale": 1.0,
            "reward": {
                **self.env_config["reward"],
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 0.0,
                "disruption_response_bonus": 5.0,
            },
            "simulation": {
                **self.env_config["simulation"],
                "charging_stop_probability": 0.0,
                "disruption": {
                    "enabled": True,
                    "mode": "scripted",
                    "scripted_events": [
                        {
                            "disruption_type": "demand_surge",
                            "target": "od_bc",
                            "day_index": 0,
                            "start_hour": 0.0,
                            "duration_hours": 1.0,
                            "severity": 3.0,
                        }
                    ],
                },
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)
        env.reset(seed=4)
        env.mobile_station_units = [
            MobileChargingStationUnit(state="station_ab"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
        ]
        env._refresh_mobile_station_counts()
        _, reward_wrong, _, _, info_wrong = env.step(env.action_from_mobile_station_allocation((1, 0)))
        env.reset(seed=4)
        env.mobile_station_units = [
            MobileChargingStationUnit(state="station_bc"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
        ]
        env._refresh_mobile_station_counts()
        _, reward_right, _, _, info_right = env.step(env.action_from_mobile_station_allocation((0, 1)))

        self.assertEqual(info_wrong["disruption_target"], "od_bc")
        self.assertAlmostEqual(info_wrong["reward_disruption_response_bonus_term"], 0.0)
        self.assertAlmostEqual(info_right["reward_disruption_response_bonus_term"], 5.0)
        self.assertAlmostEqual(reward_wrong, 0.0)
        self.assertAlmostEqual(reward_right, 5.0)

    def test_spatial_deficit_alignment_bonus_rewards_covering_local_deficit(self) -> None:
        env_config = {
            **self.env_config,
            "reward_scale": 1.0,
            "reward": {
                **self.env_config["reward"],
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 0.0,
                "disruption_response_bonus": 0.0,
                "spatial_deficit_alignment_bonus": 10.0,
                "spatial_deficit_direction_bonus": 10.0,
            },
            "simulation": {
                **self.env_config["simulation"],
                "charging_stop_probability": 0.0,
                "num_plugs": [1, 1],
                "traffic": {
                    **self.env_config["simulation"]["traffic"],
                    "baseline_cars_per_step": 0.0,
                    "morning_peak_cars_per_step": 0.0,
                    "evening_peak_cars_per_step": 0.0,
                    "midday_bump_cars_per_step": 0.0,
                },
                "disruption": {"enabled": False},
            },
        }
        env = LineCorridorMobileStationEnv(env_config, {}, seed=3)
        env.reset(seed=4)
        env.mobile_station_units = [
            MobileChargingStationUnit(state="station_bc"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
        ]
        env._refresh_mobile_station_counts()
        env.queue_by_station = [
            deque(
                [
                    QueuedVehicle(arrival_step=-2, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                    QueuedVehicle(arrival_step=-2, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                ]
            ),
            deque(),
        ]

        _, reward_wrong, _, _, info_wrong = env.step(env.action_from_mobile_station_allocation((0, 1)))
        env.reset(seed=4)
        env.mobile_station_units = [
            MobileChargingStationUnit(state="station_ab"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
            MobileChargingStationUnit(state="middle_available"),
        ]
        env._refresh_mobile_station_counts()
        env.queue_by_station = [
            deque(
                [
                    QueuedVehicle(arrival_step=-2, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                    QueuedVehicle(arrival_step=-2, trip_key="od_ab", station_index=0, base_service_minutes=30.0),
                ]
            ),
            deque(),
        ]
        _, reward_right, _, _, info_right = env.step(env.action_from_mobile_station_allocation((1, 0)))

        self.assertGreater(info_right["spatial_deficit_coverage"], info_wrong["spatial_deficit_coverage"])
        self.assertGreater(info_right["reward_spatial_deficit_alignment_bonus_term"], info_wrong["reward_spatial_deficit_alignment_bonus_term"])
        self.assertGreater(info_right["reward_spatial_deficit_direction_bonus_term"], info_wrong["reward_spatial_deficit_direction_bonus_term"])
        self.assertGreater(reward_right, reward_wrong)

    def test_noop_comparison_logs_three_day_time_series_with_od_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "seed": 21,
                "experiment": {
                    "name": "mobile_mcs_line_abc",
                    "output_root": tmp_dir,
                    "save_plots": False,
                },
                "logging": {"level": "INFO", "wandb": {"enabled": False}},
                "demand": {},
                "comparison_rollout": {
                    "enabled": True,
                    "num_days": 3,
                    "seed": 123,
                    "repeat_daily_disruptions": False,
                    "scripted_events": [
                        {
                            "disruption_type": "capacity_drop",
                            "target": "station_ab",
                            "day_index": 0,
                            "start_hour": 6.0,
                            "duration_hours": 2.0,
                            "severity": 4.0,
                        },
                        {
                            "disruption_type": "demand_surge",
                            "target": "od_ac",
                            "day_index": 0,
                            "start_hour": 16.0,
                            "duration_hours": 2.0,
                            "severity": 1.8,
                        },
                        {
                            "disruption_type": "service_time_inflation",
                            "target": "station_bc",
                            "day_index": 1,
                            "start_hour": 10.0,
                            "duration_hours": 2.0,
                            "severity": 1.5,
                        },
                        {
                            "disruption_type": "station_outage",
                            "target": "station_bc",
                            "day_index": 1,
                            "start_hour": 17.0,
                            "duration_hours": 1.5,
                            "severity": 0.0,
                        },
                        {
                            "disruption_type": "demand_surge",
                            "target": "od_ca",
                            "day_index": 2,
                            "start_hour": 7.0,
                            "duration_hours": 2.0,
                            "severity": 1.8,
                        },
                        {
                            "disruption_type": "capacity_drop",
                            "target": "station_ab",
                            "day_index": 2,
                            "start_hour": 15.0,
                            "duration_hours": 2.0,
                            "severity": 5.0,
                        },
                    ],
                },
                "environment": {
                    "env_type": "line_corridor_mobile_mcs",
                    "max_mobile_stations": 4,
                    "mobile_station_chargers": 2,
                    "mobile_station_capacity": 20.0,
                    "reward_scale": 0.1,
                    "reward": {
                        "served_reward_weight": 1.0,
                        "unmet_penalty": 2.0,
                        "active_mobile_station_cost": 18.0,
                        "activation_cost": 5.0,
                        "adjustment_cost": 2.0,
                        "idle_capacity_penalty": 0.1,
                        "utilization_bonus": 1.0,
                        "queue_length_penalty": 2.0,
                        "queue_wait_penalty": 0.02,
                    },
                    "simulation": _base_line_config(),
                },
            }

            result = run_mobile_noop_comparison(config)

            self.assertIsNotNone(result)
            metrics_path = Path(tmp_dir) / "mobile_mcs_line_abc" / "mobile_noop_comparison" / "comparison_timestep_metrics.csv"
            self.assertTrue(metrics_path.exists())
            frame = pd.read_csv(metrics_path)
            self.assertEqual(len(frame), 864)
            self.assertIn("global_hour", frame.columns)
            self.assertIn("day_index", frame.columns)
            self.assertIn("expected_passing_od_ac", frame.columns)
            self.assertIn("expected_passing_od_ca", frame.columns)
            self.assertIn("num_active_mobile_stations_station_ab", frame.columns)
            self.assertIn("num_active_mobile_stations_station_bc", frame.columns)

    def test_noop_comparison_without_disruptions_stays_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            clean_simulation = _base_line_config()
            clean_simulation["disruption"] = {"enabled": False}
            config = {
                "seed": 21,
                "experiment": {
                    "name": "mobile_mcs_line_abc_no_disruptions",
                    "output_root": tmp_dir,
                    "save_plots": False,
                },
                "logging": {"level": "INFO", "wandb": {"enabled": False}},
                "demand": {},
                "comparison_rollout": {
                    "enabled": True,
                    "num_days": 3,
                    "seed": 123,
                    "repeat_daily_disruptions": False,
                },
                "environment": {
                    "env_type": "line_corridor_mobile_mcs",
                    "max_mobile_stations": 4,
                    "mobile_station_chargers": 2,
                    "mobile_station_capacity": 20.0,
                    "reward_scale": 0.1,
                    "reward": {
                        "served_reward_weight": 1.0,
                        "unmet_penalty": 2.0,
                        "active_mobile_station_cost": 18.0,
                        "activation_cost": 5.0,
                        "adjustment_cost": 2.0,
                        "idle_capacity_penalty": 0.1,
                        "utilization_bonus": 1.0,
                        "queue_length_penalty": 2.0,
                        "queue_wait_penalty": 0.02,
                    },
                    "simulation": clean_simulation,
                },
            }

            result = run_mobile_noop_comparison(config)

            self.assertIsNotNone(result)
            metrics_path = (
                Path(tmp_dir) / "mobile_mcs_line_abc_no_disruptions" / "mobile_noop_comparison" / "comparison_timestep_metrics.csv"
            )
            frame = pd.read_csv(metrics_path)
            self.assertEqual(len(frame), 864)
            self.assertEqual(int(frame["disruption_active"].sum()), 0)
            self.assertTrue((frame["disruption_type"] == "none").all())


if __name__ == "__main__":
    unittest.main()

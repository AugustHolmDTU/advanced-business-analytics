from collections import deque
import unittest

from evch.sim.simple_corridor import ActiveSession, QueuedVehicle
from evch.envs.corridor_mobile_env import CorridorMobileStationEnv
from evch.envs.factory import make_env


class CorridorMobileStationEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_config = {
            "env_type": "corridor_mobile_mcs",
            "max_mobile_stations": 10,
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
            "simulation": {
                "city_names": ["City A", "City B"],
                "road_length_km": 100.0,
                "station_position_km": 50.0,
                "num_plugs": 12,
                "step_minutes": 60,
                "duration_hours": 24.0,
                "charging_stop_probability": 0.2,
                "service_time": {
                    "mean_minutes": 30.0,
                    "std_minutes": 4.0,
                    "min_minutes": 20.0,
                    "max_minutes": 45.0,
                },
                "traffic": {
                    "baseline_cars_per_step": 6.0,
                    "morning_peak_hour": 8.0,
                    "evening_peak_hour": 17.0,
                    "morning_peak_cars_per_step": 6.0,
                    "evening_peak_cars_per_step": 7.0,
                    "midday_bump_hour": 12.0,
                    "midday_bump_cars_per_step": 2.0,
                    "peak_width_hours": 1.4,
                    "directional_bias_amplitude": 0.18,
                    "minimum_direction_share": 0.2,
                },
                "disruption": {
                    "enabled": True,
                    "mode": "scripted",
                    "scripted_events": [
                        {
                            "disruption_type": "capacity_drop",
                            "day_index": 0,
                            "start_hour": 1.0,
                            "duration_hours": 1.0,
                            "severity": 3.0,
                        },
                        {
                            "disruption_type": "station_outage",
                            "day_index": 0,
                            "start_hour": 3.0,
                            "duration_hours": 1.0,
                            "severity": 0.0,
                        },
                    ],
                },
            },
        }
        self.demand_config = {}

    def test_factory_builds_corridor_mobile_env(self) -> None:
        env = make_env(self.env_config, self.demand_config, seed=5)
        self.assertIsInstance(env, CorridorMobileStationEnv)

    def test_reset_and_step_report_corridor_sim_metrics(self) -> None:
        env = CorridorMobileStationEnv(self.env_config, self.demand_config, seed=3)
        obs, info = env.reset(seed=4)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["fixed_site_index"], 0)
        self.assertEqual(info["num_active_mobile_stations"], 0)

        next_obs, reward, terminated, truncated, step_info = env.step(2)
        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["num_active_mobile_stations"], 2)
        self.assertEqual(step_info["mobile_capacity_total"], 4.0)
        self.assertIn("expected_passing_total", step_info)
        self.assertIn("starts_total", step_info)
        self.assertIn("completions_total", step_info)
        self.assertIn("active_plugs", step_info)
        self.assertIn("effective_base_plugs", step_info)
        self.assertIn("unused_mobile_chargers", step_info)
        self.assertIn("unused_mobile_stations_estimate", step_info)

    def test_multiple_disruptions_same_day_activate_in_sequence(self) -> None:
        env = CorridorMobileStationEnv(self.env_config, self.demand_config, seed=3)
        env.reset(seed=4)

        _, _, _, _, before = env.step(0)
        _, _, _, _, first = env.step(0)
        _, _, _, _, between = env.step(0)
        _, _, _, _, second = env.step(0)

        self.assertEqual(before["disruption_active"], 0)
        self.assertEqual(first["disruption_type"], "capacity_drop")
        self.assertEqual(first["effective_base_plugs"], 3)
        self.assertEqual(between["disruption_active"], 0)
        self.assertEqual(second["disruption_type"], "station_outage")
        self.assertEqual(second["effective_base_plugs"], 0)

    def test_queue_wait_service_level_penalty_only_hits_above_target(self) -> None:
        config = {
            "env_type": "corridor_mobile_mcs",
            "max_mobile_stations": 2,
            "mobile_station_chargers": 2,
            "mobile_station_capacity": 20.0,
            "reward_scale": 1.0,
            "reward": {
                "served_reward_weight": 0.0,
                "unmet_penalty": 0.0,
                "active_mobile_station_cost": 0.0,
                "activation_cost": 0.0,
                "adjustment_cost": 0.0,
                "idle_capacity_penalty": 0.0,
                "utilization_bonus": 0.0,
                "queue_length_penalty": 0.0,
                "queue_wait_penalty": 0.0,
                "queue_wait_target_minutes": 15.0,
                "queue_wait_excess_penalty": 2.0,
                "queue_wait_hard_penalty": 7.0,
                "queue_wait_service_level_bonus": 0.0,
            },
            "simulation": {
                "city_names": ["City A", "City B"],
                "road_length_km": 100.0,
                "station_position_km": 50.0,
                "num_plugs": 1,
                "step_minutes": 5,
                "duration_hours": 2.0,
                "charging_stop_probability": 0.0,
                "service_time": {
                    "mean_minutes": 30.0,
                    "std_minutes": 1.0,
                    "min_minutes": 30.0,
                    "max_minutes": 30.0,
                },
                "traffic": {
                    "baseline_cars_per_step": 0.0,
                    "morning_peak_hour": 8.0,
                    "evening_peak_hour": 17.0,
                    "morning_peak_cars_per_step": 0.0,
                    "evening_peak_cars_per_step": 0.0,
                    "midday_bump_hour": 12.0,
                    "midday_bump_cars_per_step": 0.0,
                    "peak_width_hours": 1.0,
                    "directional_bias_amplitude": 0.0,
                    "minimum_direction_share": 0.2,
                },
                "disruption": {"enabled": False},
            },
        }
        env = CorridorMobileStationEnv(config, {}, seed=5)
        env.reset(seed=6)
        env.active_sessions = [
            ActiveSession(
                arrival_step=0,
                start_step=0,
                end_step=100,
                direction=env.simulator.direction_ab,
                service_minutes=30.0,
            )
        ]
        env.queue = deque([QueuedVehicle(arrival_step=-4, direction=env.simulator.direction_ab)])

        _, reward, _, _, info = env.step(0)

        self.assertEqual(info["queue_wait_target_minutes"], 15.0)
        self.assertEqual(info["queue_wait_excess_minutes"], 5.0)
        self.assertEqual(info["queue_wait_target_breached"], 1)
        self.assertAlmostEqual(reward, -(2.0 * 5.0 + 7.0))


if __name__ == "__main__":
    unittest.main()

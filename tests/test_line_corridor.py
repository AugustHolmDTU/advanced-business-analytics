import unittest

from evch.baselines.policies import mobile_threshold_policy
from evch.envs.factory import make_env
from evch.envs.line_corridor_mobile_env import LineCorridorMobileStationEnv
from evch.sim.line_corridor import LineCorridorQueueSimulator


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
        self.assertEqual(env.action_space.n, 15)
        self.assertEqual(env.action_from_mobile_station_allocation((0, 0)), 0)
        self.assertEqual(tuple(env.action_to_allocation(env.action_from_mobile_station_allocation((2, 1)))), (2, 1))

    def test_step_reports_station_specific_mobile_counts(self) -> None:
        env = LineCorridorMobileStationEnv(self.env_config, {}, seed=3)
        obs, info = env.reset(seed=4)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["num_active_mobile_stations"], 0)

        action = env.action_from_mobile_station_allocation((2, 1))
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["num_active_mobile_stations"], 3)
        self.assertEqual(step_info["num_active_mobile_stations_station_ab"], 2)
        self.assertEqual(step_info["num_active_mobile_stations_station_bc"], 1)
        self.assertIn("queue_length_station_ab", step_info)
        self.assertIn("queue_length_station_bc", step_info)
        self.assertIn("unused_mobile_stations_estimate_station_ab", step_info)

    def test_mobile_threshold_policy_returns_valid_two_station_action(self) -> None:
        env = LineCorridorMobileStationEnv(self.env_config, {}, seed=3)
        obs, _ = env.reset(seed=4)

        action = mobile_threshold_policy(obs, env)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.action_space.n)


if __name__ == "__main__":
    unittest.main()

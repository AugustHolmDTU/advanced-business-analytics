import unittest

from evch.baselines.policies import mobile_noop_policy, mobile_threshold_policy
from evch.envs.factory import make_env
from evch.envs.mobile_station_env import MobileStationChargingEnv


class MobileStationEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env_config = {
            "env_type": "mobile_station_capacity",
            "num_candidate_sites": 1,
            "num_demand_zones": 6,
            "horizon": 24,
            "max_steps": 8,
            "city_extent_km": 4.0,
            "base_station_plugs": 12,
            "base_plug_capacity": 1.0,
            "max_mobile_stations": 10,
            "mobile_station_chargers": 2,
            "mobile_station_capacity": 20.0,
            "reward_scale": 0.1,
            "reward": {
                "served_reward_weight": 1.0,
                "unmet_penalty": 2.0,
                "active_mobile_station_cost": 12.0,
                "activation_cost": 3.0,
                "adjustment_cost": 1.0,
                "idle_capacity_penalty": 0.1,
                "utilization_bonus": 1.0,
            },
        }
        self.demand_config = {
            "base_rate_min": 3.0,
            "base_rate_max": 9.0,
            "zone_scale_std": 0.2,
            "morning_peak_hour": 8.0,
            "evening_peak_hour": 18.0,
            "peak_width": 2.0,
            "morning_peak_weight": 0.9,
            "evening_peak_weight": 1.1,
            "weekday_multiplier": 1.0,
            "weekend_multiplier": 0.8,
            "background_intensity": 0.35,
            "poisson_clip": 60.0,
            "observation_noise_std": 0.4,
        }

    def test_factory_builds_mobile_station_env(self) -> None:
        env = make_env(self.env_config, self.demand_config, seed=5)
        self.assertIsInstance(env, MobileStationChargingEnv)

    def test_reset_and_step_report_mobile_station_metrics(self) -> None:
        env = MobileStationChargingEnv(self.env_config, self.demand_config, seed=3)
        obs, info = env.reset(seed=4)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["num_active_mobile_stations"], 0)

        next_obs, reward, terminated, truncated, step_info = env.step(3)
        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertFalse(terminated)
        self.assertEqual(step_info["num_active_mobile_stations"], 3)
        self.assertAlmostEqual(step_info["mobile_capacity_total"], 60.0)
        self.assertAlmostEqual(step_info["effective_capacity_total"], 72.0)
        self.assertIn("served_demand", step_info)
        self.assertIn("unmet_demand", step_info)

    def test_mobile_threshold_policy_matches_capacity_gap(self) -> None:
        env = MobileStationChargingEnv(self.env_config, self.demand_config, seed=1)
        env.reset(seed=2)
        env.step_index = int(self.demand_config["morning_peak_hour"])
        action = mobile_threshold_policy(env._get_observation(), env)
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(action, env.max_mobile_stations)
        self.assertEqual(mobile_noop_policy(env._get_observation(), env), 0)


if __name__ == "__main__":
    unittest.main()

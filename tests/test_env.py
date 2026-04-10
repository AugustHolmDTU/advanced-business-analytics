import unittest

import numpy as np

from evch.envs.charging_env import ChargingPlacementEnv


class ChargingEnvTest(unittest.TestCase):
    def test_env_reset_and_step(self) -> None:
        env_config = {
            "num_candidate_sites": 5,
            "num_demand_zones": 3,
            "max_chargers": 2,
            "charger_capacity": 8.0,
            "horizon": 12,
            "max_steps": 6,
            "service_decay": 0.08,
            "deployment_cost": 0.1,
            "relocation_cost": 0.2,
            "unmet_penalty": 1.0,
            "outage_penalty": 0.1,
            "city_extent_km": 5.0,
            "disruption": {},
        }
        demand_config = {
            "base_rate_min": 2.0,
            "base_rate_max": 5.0,
            "zone_scale_std": 0.2,
            "morning_peak_hour": 8.0,
            "evening_peak_hour": 18.0,
            "peak_width": 2.0,
            "morning_peak_weight": 1.0,
            "evening_peak_weight": 1.0,
            "weekday_multiplier": 1.0,
            "weekend_multiplier": 0.8,
            "background_intensity": 0.5,
            "poisson_clip": 30.0,
            "observation_noise_std": 0.3,
        }
        env = ChargingPlacementEnv(env_config, demand_config, seed=3)
        obs, info = env.reset(seed=4)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["num_active_chargers"], 0)

        next_obs, reward, terminated, truncated, step_info = env.step(1)
        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIn("served_demand", step_info)
        self.assertLessEqual(step_info["num_active_chargers"], env.max_chargers)
        self.assertTrue(np.all(env.allocations >= 0.0))
        self.assertIsInstance(terminated, bool)


if __name__ == "__main__":
    unittest.main()

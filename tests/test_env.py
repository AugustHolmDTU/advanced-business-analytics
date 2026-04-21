import unittest
import json
import tempfile

import numpy as np

from evch.data.synthetic import make_synthetic_layout
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

    def test_capacity_scale_and_baseline_availability_affect_effective_capacity(self) -> None:
        env_config = {
            "num_candidate_sites": 4,
            "num_demand_zones": 3,
            "max_chargers": 2,
            "charger_capacity": 10.0,
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
        env = ChargingPlacementEnv(env_config, demand_config, seed=2)
        env.site_capacity_scale = np.asarray([1.8, 1.0, 1.0, 1.0], dtype=np.float32)
        env.base_site_availability = np.asarray([0.5, 1.0, 1.0, 1.0], dtype=np.float32)
        env.reset(seed=3)
        env.allocations[:] = 0.0
        env.allocations[0] = 1.0

        _, _, _, _, info = env.step(env.noop_action)

        self.assertAlmostEqual(env.site_availability[0], 0.5)
        self.assertAlmostEqual(info["effective_capacity_total"], 9.0, places=5)

    def test_corridor_layout_and_randomized_reset(self) -> None:
        env_config = {
            "layout": "corridor",
            "num_candidate_sites": 9,
            "num_demand_zones": 6,
            "max_chargers": 2,
            "charger_capacity": 8.0,
            "horizon": 12,
            "max_steps": 6,
            "service_decay": 0.05,
            "deployment_cost": 0.1,
            "relocation_cost": 0.2,
            "unmet_penalty": 1.0,
            "outage_penalty": 0.1,
            "city_extent_km": 180.0,
            "corridor_length_km": 180.0,
            "randomize_on_reset": True,
            "reset_seed_stride": 13,
            "corridor": {
                "num_cities": 3,
            },
            "disruption": {
                "travel_slowdown_probability": 1.0,
                "travel_slowdown_multiplier": 1.3,
                "road_closure_probability": 1.0,
                "road_closure_penalty_minutes": 15.0,
            },
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

        city = make_synthetic_layout(env_config, demand_config, seed=7)
        self.assertEqual(city.layout_type, "corridor")
        self.assertEqual(city.travel_time_matrix.shape, (6, 9))
        self.assertEqual(city.site_travel_time_matrix.shape, (9, 9))
        self.assertIn("city_hub", city.site_types)
        self.assertIn("service_area", city.site_types)
        self.assertIn("highway_exit", city.site_types)

        env = ChargingPlacementEnv(env_config, demand_config, seed=5)
        env.reset()
        first_layout = env.city.site_coords.copy()
        _, _, _, _, info = env.step(0)
        self.assertGreaterEqual(info["travel_slowdown_multiplier"], 1.0)
        self.assertGreaterEqual(info["road_closure_penalty_minutes"], 0.0)

        env.reset()
        second_layout = env.city.site_coords.copy()
        self.assertFalse(np.allclose(first_layout, second_layout))

    def test_corridor_layout_with_calibration_priors(self) -> None:
        env_config = {
            "layout": "corridor",
            "num_candidate_sites": 8,
            "num_demand_zones": 5,
            "max_chargers": 2,
            "charger_capacity": 8.0,
            "horizon": 12,
            "max_steps": 6,
            "service_decay": 0.05,
            "deployment_cost": 0.1,
            "relocation_cost": 0.2,
            "unmet_penalty": 1.0,
            "outage_penalty": 0.1,
            "city_extent_km": 160.0,
            "corridor_length_km": 160.0,
            "corridor": {
                "num_cities": 2,
            },
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
        priors_payload = {
            "source": "tomtom_multi_snapshot",
            "location_count": 4,
            "distributions": {
                "connector_count": {"values": [2, 4, 8, 12]},
                "max_power_kw": {"values": [22.0, 50.0, 150.0]},
                "availability_ratio": {"values": [0.3, 0.5, 0.8]},
            },
            "recommended_corridor_defaults": {
                "mainline_speed_kmh": 111.0,
                "urban_speed_kmh": 69.0,
                "service_area_min_gap_km": 19.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            priors_path = f"{tmpdir}/priors.json"
            with open(priors_path, "w", encoding="utf-8") as handle:
                json.dump(priors_payload, handle)
            env_config["corridor"]["calibration_priors_path"] = priors_path
            city = make_synthetic_layout(env_config, demand_config, seed=9)

        self.assertEqual(city.metadata["calibration_source"], "tomtom_multi_snapshot")
        self.assertEqual(city.metadata["calibration_location_count"], 4)
        self.assertEqual(len(city.metadata["site_connector_proxy"]), 8)
        self.assertEqual(len(city.metadata["site_power_proxy_kw"]), 8)
        self.assertEqual(len(city.metadata["site_availability_proxy"]), 8)
        self.assertIn(111.0, city.metadata["segment_speeds_kmh"])


if __name__ == "__main__":
    unittest.main()

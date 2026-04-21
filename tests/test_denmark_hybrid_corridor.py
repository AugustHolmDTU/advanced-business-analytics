import unittest

import pandas as pd

from evch.train.build_denmark_hybrid_corridor import _build_demand_profile, _build_site_type_profiles


class DenmarkHybridCorridorTest(unittest.TestCase):
    def test_build_site_type_profiles(self) -> None:
        history = pd.DataFrame(
            {
                "site_type": ["city_hub", "city_hub", "service_area", "service_area", "highway_exit", "highway_exit"],
                "total_connectors": [10, 14, 6, 8, 4, 5],
                "max_power_kw": [250.0, 180.0, 150.0, 120.0, 75.0, 90.0],
                "availability_ratio": [0.35, 0.42, 0.58, 0.61, 0.74, 0.68],
            }
        )
        profiles = _build_site_type_profiles(history)

        self.assertIn("city_hub", profiles)
        self.assertIn("service_area", profiles)
        self.assertEqual(len(profiles["city_hub"]["connector_range"]), 2)
        self.assertGreaterEqual(profiles["city_hub"]["connector_range"][1], profiles["city_hub"]["connector_range"][0])
        self.assertGreater(profiles["service_area"]["availability_mean"], 0.0)

    def test_build_demand_profile(self) -> None:
        history = pd.DataFrame(
            {
                "snapshot_hour": [6, 8, 9, 17, 18, 20, 6, 8, 18, 20],
                "utilization_ratio": [0.28, 0.55, 0.49, 0.61, 0.78, 0.57, 0.26, 0.51, 0.75, 0.55],
                "is_weekend": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        profile = _build_demand_profile(history)

        self.assertGreaterEqual(profile["morning_peak_hour"], 5.0)
        self.assertLessEqual(profile["morning_peak_hour"], 11.0)
        self.assertGreaterEqual(profile["evening_peak_hour"], 14.0)
        self.assertLessEqual(profile["evening_peak_hour"], 21.0)
        self.assertGreater(profile["evening_peak_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()

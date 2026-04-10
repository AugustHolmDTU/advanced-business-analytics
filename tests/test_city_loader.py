import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evch.data.city import build_city


class TomTomSnapshotCityLoaderTest(unittest.TestCase):
    def test_build_city_from_tomtom_snapshot(self) -> None:
        snapshot_payload = {
            "center": {"lat": 55.678, "lon": 12.534, "freeform_address": "Frederiksberg, Denmark"},
            "site_coords_km": [[0.0, 0.2], [0.5, -0.1]],
            "zone_coords_km": [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5]],
            "site_latlon": [[55.679, 12.533], [55.677, 12.541]],
            "zone_latlon": [[55.678, 12.528], [55.678, 12.540], [55.683, 12.534]],
            "travel_time_matrix_minutes": [[6.0, 8.0], [7.0, 4.0], [5.0, 6.5]],
            "site_names": ["A", "B"],
            "site_addresses": ["Addr A", "Addr B"],
            "stations": [],
        }
        env_config = {
            "num_candidate_sites": 2,
            "num_demand_zones": 3,
            "city_extent_km": 3.0,
            "city_data": {"source": "tomtom_snapshot"},
        }
        demand_config = {
            "base_rate_min": 2.0,
            "base_rate_max": 5.0,
            "zone_scale_std": 0.2,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "snapshot.json"
            with snapshot_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot_payload, handle)
            env_config["city_data"]["snapshot_path"] = str(snapshot_path)
            city = build_city(env_config=env_config, demand_config=demand_config, seed=4)

        self.assertEqual(city.site_coords.shape, (2, 2))
        self.assertEqual(city.zone_coords.shape, (3, 2))
        self.assertEqual(city.travel_time_matrix.shape, (3, 2))
        self.assertEqual(city.site_names, ["A", "B"])
        self.assertEqual(city.site_addresses, ["Addr A", "Addr B"])
        self.assertTrue(np.all(city.zone_base_demand >= 2.0))
        self.assertTrue(np.all(city.zone_base_demand <= 5.0))


if __name__ == "__main__":
    unittest.main()

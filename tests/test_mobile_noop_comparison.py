import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evch.train.run_mobile_noop_comparison import run_mobile_noop_comparison

logging.getLogger("matplotlib").setLevel(logging.WARNING)


class MobileNoopComparisonTest(unittest.TestCase):
    def test_runs_fixed_three_day_noop_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "seed": 21,
                "experiment": {
                    "name": "mobile_mcs_simple",
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
                            "day_index": 0,
                            "start_hour": 6.0,
                            "duration_hours": 2.0,
                            "severity": 4.0,
                        },
                        {
                            "disruption_type": "demand_surge_ab",
                            "day_index": 0,
                            "start_hour": 16.0,
                            "duration_hours": 2.0,
                            "severity": 1.8,
                        },
                        {
                            "disruption_type": "service_time_inflation",
                            "day_index": 1,
                            "start_hour": 10.0,
                            "duration_hours": 2.0,
                            "severity": 1.5,
                        },
                        {
                            "disruption_type": "station_outage",
                            "day_index": 1,
                            "start_hour": 17.0,
                            "duration_hours": 1.5,
                            "severity": 0.0,
                        },
                        {
                            "disruption_type": "demand_surge_ba",
                            "day_index": 2,
                            "start_hour": 7.0,
                            "duration_hours": 2.0,
                            "severity": 1.8,
                        },
                        {
                            "disruption_type": "capacity_drop",
                            "day_index": 2,
                            "start_hour": 15.0,
                            "duration_hours": 2.0,
                            "severity": 5.0,
                        },
                    ],
                },
                "environment": {
                    "env_type": "corridor_mobile_mcs",
                    "max_mobile_stations": 10,
                    "mobile_station_chargers": 2,
                    "mobile_station_capacity": 20.0,
                    "reward_scale": 0.1,
                    "reward": {
                        "served_reward_weight": 1.0,
                        "unmet_penalty": 2.5,
                        "active_mobile_station_cost": 24.0,
                        "activation_cost": 6.0,
                        "adjustment_cost": 2.0,
                        "idle_capacity_penalty": 0.1,
                        "utilization_bonus": 1.0,
                        "queue_length_penalty": 2.5,
                        "queue_wait_penalty": 0.02,
                    },
                    "simulation": {
                        "city_names": ["City A", "City B"],
                        "road_length_km": 100.0,
                        "station_position_km": 50.0,
                        "num_plugs": 12,
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
                            "baseline_cars_per_step": 11.0,
                            "morning_peak_hour": 8.0,
                            "evening_peak_hour": 17.0,
                            "morning_peak_cars_per_step": 22.0,
                            "evening_peak_cars_per_step": 24.0,
                            "midday_bump_hour": 12.5,
                            "midday_bump_cars_per_step": 6.0,
                            "peak_width_hours": 1.6,
                            "directional_bias_amplitude": 0.24,
                            "minimum_direction_share": 0.18,
                        },
                        "disruption": {
                            "enabled": True,
                            "mode": "scripted",
                            "scripted_events": [],
                        },
                    },
                },
            }

            result = run_mobile_noop_comparison(config)

            self.assertIsNotNone(result)
            output_dir = Path(tmp_dir) / "mobile_mcs_simple" / "mobile_noop_comparison"
            metrics_path = output_dir / "comparison_timestep_metrics.csv"
            allocation_plot_path = output_dir / "comparison_mcs_allocation_by_station.png"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(allocation_plot_path.exists())
            frame = pd.read_csv(metrics_path)
            self.assertEqual(len(frame), 864)
            self.assertEqual(frame["num_active_mobile_stations"].max(), 0.0)
            self.assertIn("unused_mobile_chargers", frame.columns)
            self.assertIn("unused_mobile_stations_estimate", frame.columns)
            self.assertIn("global_hour", frame.columns)
            self.assertIn("day_index", frame.columns)
            self.assertIn("hour_of_day", frame.columns)
            self.assertTrue(any(column.startswith("starts_") for column in frame.columns))
            self.assertTrue(any(column.startswith("started_") for column in frame.columns))
            self.assertTrue(any(column.startswith("completions_") for column in frame.columns))
            self.assertIn("disruption_target_code", frame.columns)
            self.assertEqual(result["allocation_plot_path"], str(allocation_plot_path))

    def test_runs_seeded_test_id_noop_comparison(self) -> None:
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
                "train_val_test": {
                    "test_id": {
                        "enabled": True,
                        "seeds": {
                            "start": 20000,
                            "count": 3,
                        },
                        "environment_overrides": {
                            "simulation": {
                                "duration_days_range": [5, 10],
                                "disruption": {
                                    "day_disruption_count_weights": {
                                        1: 0.6,
                                        2: 0.4,
                                    },
                                    "required_events": [
                                        {
                                            "disruption_type": "demand_surge",
                                            "day_index_range": [0, 8],
                                            "duration_hours": [3.0, 4.5],
                                            "start_hour_range": [6.0, 18.0],
                                            "multiplier": [2.8, 3.2],
                                            "targets": ["od_ab", "od_ba"],
                                        },
                                        {
                                            "disruption_type": "demand_surge",
                                            "day_index_range": [1, 9],
                                            "duration_hours": [3.0, 4.5],
                                            "start_hour_range": [6.0, 18.0],
                                            "multiplier": [2.8, 3.2],
                                            "targets": ["od_bc", "od_cb"],
                                        },
                                    ],
                                },
                            }
                        },
                    }
                },
                "comparison_rollout": {
                    "enabled": True,
                    "seed_source": "test_id",
                    "seed_index": 0,
                    "use_seeded_episode": True,
                    "environment_overrides": {
                        "simulation": {
                            "disruption": {
                                "enabled": True,
                                "mode": "random",
                                "event_types": ["demand_surge"],
                            },
                        }
                    },
                },
                "environment": {
                    "env_type": "line_corridor_mobile_mcs",
                    "max_mobile_stations": 10,
                    "mobile_station_chargers": 2,
                    "mobile_station_capacity": 20.0,
                    "reward_scale": 0.1,
                    "reward": {
                        "served_reward_weight": 1.0,
                        "unmet_penalty": 2.5,
                        "active_mobile_station_cost": 24.0,
                        "activation_cost": 6.0,
                        "adjustment_cost": 2.0,
                        "idle_capacity_penalty": 0.1,
                        "utilization_bonus": 1.0,
                        "queue_length_penalty": 2.5,
                        "queue_wait_penalty": 0.02,
                    },
                    "simulation": {
                        "duration_days_range": [5, 10],
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
                            "baseline_cars_per_step": 22.0,
                            "morning_peak_hour": 8.0,
                            "evening_peak_hour": 17.0,
                            "morning_peak_cars_per_step": 45.0,
                            "evening_peak_cars_per_step": 50.0,
                            "midday_bump_hour": 12.5,
                            "midday_bump_cars_per_step": 10.0,
                            "peak_width_hours": 1.6,
                            "directional_bias_amplitude": 0.24,
                            "middle_city_share": 0.34,
                            "long_trip_share": 0.35,
                            "middle_destination_bias_amplitude": 0.18,
                        },
                        "disruption": {
                            "enabled": True,
                            "mode": "random",
                            "day_disruption_count_weights": {
                                1: 0.6,
                                2: 0.4,
                            },
                            "event_types": ["demand_surge"],
                            "demand_surge": {
                                "duration_hours": [2.0, 3.5],
                                "start_hour_range": [6.0, 20.0],
                                "multiplier": 2.4,
                                "targets": ["od_ab", "od_ba"],
                            },
                        },
                    },
                },
            }

            result = run_mobile_noop_comparison(config)

            self.assertIsNotNone(result)
            output_dir = Path(tmp_dir) / "mobile_mcs_line_abc" / "mobile_noop_comparison"
            metrics_path = output_dir / "comparison_timestep_metrics.csv"
            allocation_plot_path = output_dir / "comparison_mcs_allocation_by_station.png"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(allocation_plot_path.exists())
            frame = pd.read_csv(metrics_path)
            self.assertIn(len(frame), {288 * days for days in range(5, 11)})
            self.assertEqual(result["summary"]["seed"], 20000)
            self.assertIn(result["summary"]["num_days"], set(range(5, 11)))
            self.assertEqual(frame["num_active_mobile_stations"].max(), 0.0)
            self.assertGreaterEqual(int(frame["disruption_active"].sum()), 2)
            self.assertGreater(len(frame.loc[frame["disruption_active"] == 1, "day_index"].dropna().unique()), 1)
            self.assertIn("disruption_target_code", frame.columns)
            self.assertIn("disruption_target_is_od_ab", frame.columns)
            self.assertIn("disruption_target_is_od_bc", frame.columns)
            active_targets = set(frame.loc[frame["disruption_active"] == 1, "disruption_target"].dropna().astype(str).unique())
            self.assertTrue(active_targets.intersection({"od_ab", "od_ba"}))
            self.assertTrue(active_targets.intersection({"od_bc", "od_cb"}))
            self.assertEqual(result["allocation_plot_path"], str(allocation_plot_path))


if __name__ == "__main__":
    unittest.main()

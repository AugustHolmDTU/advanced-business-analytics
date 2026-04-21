import unittest

from evch.sim.simple_corridor import SimpleCorridorQueueSimulator


def _base_config() -> dict:
    return {
        "city_names": ["City A", "City B"],
        "road_length_km": 100.0,
        "station_position_km": 50.0,
        "num_plugs": 10,
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
    }


class SimpleCorridorQueueSimulatorTest(unittest.TestCase):
    def test_run_returns_metrics_and_summary(self) -> None:
        simulator = SimpleCorridorQueueSimulator(config=_base_config(), seed=5)

        result = simulator.run()

        self.assertEqual(len(result.metrics), 288)
        self.assertIn("queue_length", result.metrics.columns)
        self.assertIn("utilization", result.metrics.columns)
        self.assertIn("global_hour", result.metrics.columns)
        self.assertIn("hour_of_day", result.metrics.columns)
        self.assertIn("day_index", result.metrics.columns)
        self.assertGreater(result.summary["total_arrivals"], 0)
        self.assertLessEqual(result.metrics["active_plugs"].max(), 10)
        self.assertGreater(result.summary["peak_queue_length"], 0)

    def test_time_axes_roll_over_across_multiple_days(self) -> None:
        config = _base_config()
        config["duration_hours"] = 49.0
        simulator = SimpleCorridorQueueSimulator(config=config, seed=3)

        result = simulator.run()

        self.assertEqual(result.metrics["day_index"].iloc[0], 0)
        self.assertEqual(result.metrics["day_index"].iloc[-1], 2)
        self.assertAlmostEqual(result.metrics["hour_of_day"].iloc[0], 0.0)
        self.assertAlmostEqual(result.metrics["hour_of_day"].iloc[288], 0.0)
        self.assertAlmostEqual(result.metrics["global_hour"].iloc[-1], 48.916666666666664)

    def test_expected_traffic_repeats_each_day(self) -> None:
        simulator = SimpleCorridorQueueSimulator(config=_base_config(), seed=4)

        morning_day_1 = simulator.expected_traffic(96)
        morning_day_2 = simulator.expected_traffic(96 + 288)
        evening_day_1 = simulator.expected_traffic(204)
        evening_day_3 = simulator.expected_traffic(204 + 2 * 288)

        self.assertAlmostEqual(morning_day_1["hour_of_day"], morning_day_2["hour_of_day"])
        self.assertAlmostEqual(evening_day_1["hour_of_day"], evening_day_3["hour_of_day"])
        self.assertAlmostEqual(morning_day_1["total_passing_expected"], morning_day_2["total_passing_expected"])
        self.assertAlmostEqual(evening_day_1["expected_passing_ab"], evening_day_3["expected_passing_ab"])
        self.assertAlmostEqual(evening_day_1["expected_passing_ba"], evening_day_3["expected_passing_ba"])

    def test_peak_profile_builds_and_then_drains_queue(self) -> None:
        config = _base_config()
        config["num_plugs"] = 2
        config["step_minutes"] = 10
        config["duration_hours"] = 12.0
        config["charging_stop_probability"] = 1.0
        config["service_time"] = {
            "mean_minutes": 20.0,
            "std_minutes": 2.0,
            "min_minutes": 10.0,
            "max_minutes": 25.0,
        }
        config["traffic"] = {
            "baseline_cars_per_step": 0.0,
            "morning_peak_hour": 1.5,
            "evening_peak_hour": 9.0,
            "morning_peak_cars_per_step": 4.0,
            "evening_peak_cars_per_step": 0.0,
            "midday_bump_hour": 6.0,
            "midday_bump_cars_per_step": 0.0,
            "peak_width_hours": 0.35,
            "directional_bias_amplitude": 0.0,
            "minimum_direction_share": 0.1,
        }

        simulator = SimpleCorridorQueueSimulator(config=config, seed=2)
        result = simulator.run()

        self.assertGreater(result.summary["peak_queue_length"], 0)
        self.assertEqual(result.summary["final_queue_length"], 0)
        self.assertEqual(result.summary["total_arrivals"], result.summary["total_starts"])
        self.assertLessEqual(result.summary["total_completions"], result.summary["total_starts"])


if __name__ == "__main__":
    unittest.main()

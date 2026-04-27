import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evch.envs.line_corridor_mobile_env import LineCorridorMobileStationEnv
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.train.eval_suites import evaluate_policy_suites


def _suite_config(tmp_dir: str) -> dict:
    return {
        "seed": 21,
        "experiment": {
            "name": "mobile_mcs_line_abc",
            "output_root": tmp_dir,
            "save_plots": False,
        },
        "logging": {"level": "INFO", "wandb": {"enabled": False}},
        "demand": {},
        "evaluation": {
            "policies": ["rl", "mobile_threshold", "mobile_noop"],
        },
        "train_val_test": {
            "training": {
                "episode_seed_range": [0, 99],
            },
            "validation": {
                "seeds": {
                    "start": 100,
                    "count": 2,
                }
            },
            "test_id": {
                "enabled": True,
                "seeds": {
                    "start": 200,
                    "count": 3,
                },
            },
            "test_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "scenario_id": "ab_capacity_drop",
                        "seeds": [300],
                        "num_days": 1,
                        "scripted_events": [
                            {
                                "disruption_type": "capacity_drop",
                                "target": "station_ab",
                                "day_index": 0,
                                "start_hour": 8.0,
                                "duration_hours": 2.0,
                                "severity": 4.0,
                            }
                        ],
                    }
                ],
            },
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
            "simulation": {
                "duration_days_range": [1, 2],
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
                    "mode": "random",
                    "day_disruption_count_weights": {
                        0: 0.2,
                        1: 0.7,
                        2: 0.1,
                    },
                    "event_types": ["capacity_drop", "demand_surge"],
                    "capacity_drop": {
                        "duration_hours": [1.0, 2.0],
                        "start_hour_range": [6.0, 20.0],
                        "target_num_plugs": [3, 6],
                        "targets": ["station_ab", "station_bc"],
                    },
                    "demand_surge": {
                        "duration_hours": [1.0, 3.0],
                        "start_hour_range": [6.0, 20.0],
                        "multiplier": [1.7, 2.2],
                        "targets": ["od_ab", "od_bc", "od_ac"],
                    },
                },
            },
        },
        "rl": {
            "gamma": 0.98,
            "batch_size": 8,
            "learning_starts": 1,
            "target_update_interval": 2,
            "train_frequency": 1,
            "gradient_steps": 1,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay_steps": 10,
            "tau": 1.0,
            "reward_clip": 25.0,
            "heuristic_prior_strength": 0.0,
            "heuristic_prior_mode": "placement",
            "wandb_step_log_interval": 10,
            "hidden_dims": [32, 32],
            "learning_rate": 0.001,
            "replay_capacity": 64,
        },
    }


class EvalSuitesTest(unittest.TestCase):
    def test_evaluate_policy_suites_exports_paired_seed_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _suite_config(tmp_dir)
            env = LineCorridorMobileStationEnv(config["environment"], {}, seed=5)
            agent = SimpleDQNAgent(
                obs_dim=int(env.observation_space.shape[0]),
                action_dim=int(env.action_space.n),
                config=config["rl"],
                seed=5,
            )
            checkpoint_path = Path(tmp_dir) / "dummy_agent.pt"
            agent.save(checkpoint_path)

            result = evaluate_policy_suites(config=config, checkpoint_path=str(checkpoint_path))

            output_dir = Path(result["output_dir"])
            self.assertTrue((output_dir / "validation_summary.csv").exists())
            self.assertTrue((output_dir / "test_id_summary.csv").exists())
            self.assertTrue((output_dir / "paired_test_results.csv").exists())
            self.assertTrue((output_dir / "test_stress_summary.csv").exists())
            self.assertTrue((output_dir / "scenario_manifest.csv").exists())

            paired = pd.read_csv(output_dir / "paired_test_results.csv")
            self.assertEqual(len(paired), 3)
            self.assertIn("rl_queue_reduction_vs_fixed", paired.columns)
            self.assertIn("rl_reward_difference_vs_threshold", paired.columns)

            manifest = pd.read_csv(output_dir / "scenario_manifest.csv")
            self.assertEqual(set(manifest["suite_name"].unique()), {"validation", "test_id", "test_stress_ab_capacity_drop"})
            self.assertTrue({"disruption_count", "disruption_types", "disruption_targets", "start_times", "durations", "severities"}.issubset(manifest.columns))

            test_summary = pd.read_csv(output_dir / "test_id_summary.csv")
            self.assertEqual(set(test_summary["policy_alias"].unique()), {"fixed", "threshold", "rl"})

            stress_summary = pd.read_csv(output_dir / "test_stress_summary.csv")
            self.assertEqual(set(stress_summary["stress_scenario"].unique()), {"ab_capacity_drop"})
            self.assertEqual(set(stress_summary["policy_alias"].unique()), {"fixed", "threshold", "rl"})


if __name__ == "__main__":
    unittest.main()

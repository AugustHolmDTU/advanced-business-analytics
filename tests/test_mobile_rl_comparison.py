import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evch.envs.line_corridor_mobile_env import LineCorridorMobileStationEnv
from evch.rl.simple_dqn import SimpleDQNAgent
from evch.train.run_mobile_rl_comparison import run_mobile_rl_comparison

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _base_config(tmp_dir: str) -> dict:
    return {
        "seed": 21,
        "experiment": {
            "name": "mobile_mcs_line_abc",
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
                {
                    "disruption_type": "service_time_inflation",
                    "target": "station_bc",
                    "day_index": 1,
                    "start_hour": 10.0,
                    "duration_hours": 2.0,
                    "severity": 1.5,
                },
                {
                    "disruption_type": "station_outage",
                    "target": "station_bc",
                    "day_index": 1,
                    "start_hour": 17.0,
                    "duration_hours": 1.5,
                    "severity": 0.0,
                },
                {
                    "disruption_type": "demand_surge",
                    "target": "od_ca",
                    "day_index": 2,
                    "start_hour": 7.0,
                    "duration_hours": 2.0,
                    "severity": 1.8,
                },
                {
                    "disruption_type": "capacity_drop",
                    "target": "station_ab",
                    "day_index": 2,
                    "start_hour": 15.0,
                    "duration_hours": 2.0,
                    "severity": 5.0,
                },
            ],
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
                    "mode": "scripted",
                    "scripted_events": [],
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


class MobileRlComparisonTest(unittest.TestCase):
    def test_runs_fixed_three_day_rl_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            env = LineCorridorMobileStationEnv(config["environment"], {}, seed=5)
            agent = SimpleDQNAgent(
                obs_dim=int(env.observation_space.shape[0]),
                action_dim=int(env.action_space.n),
                config=config["rl"],
                seed=5,
            )
            checkpoint_path = Path(tmp_dir) / "dummy_agent.pt"
            agent.save(checkpoint_path)

            result = run_mobile_rl_comparison(config, checkpoint_path=str(checkpoint_path))

            self.assertIsNotNone(result)
            output_dir = Path(tmp_dir) / "mobile_mcs_line_abc" / "mobile_rl_comparison"
            metrics_path = output_dir / "comparison_timestep_metrics.csv"
            self.assertTrue(metrics_path.exists())
            frame = pd.read_csv(metrics_path)
            self.assertEqual(len(frame), 864)
            self.assertIn("num_active_mobile_stations_station_ab", frame.columns)
            self.assertIn("expected_passing_od_ac", frame.columns)

    def test_runs_longer_test_id_seeded_heldout_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _base_config(tmp_dir)
            config["train_val_test"] = {
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
                                "day_disruption_count_weights": {1: 0.6, 2: 0.4},
                            },
                        }
                    },
                }
            }
            config["comparison_rollout"] = {
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
                        }
                    }
                },
            }

            env = LineCorridorMobileStationEnv(config["environment"], {}, seed=5)
            agent = SimpleDQNAgent(
                obs_dim=int(env.observation_space.shape[0]),
                action_dim=int(env.action_space.n),
                config=config["rl"],
                seed=5,
            )
            checkpoint_path = Path(tmp_dir) / "dummy_agent.pt"
            agent.save(checkpoint_path)

            result = run_mobile_rl_comparison(config, checkpoint_path=str(checkpoint_path))

            self.assertIsNotNone(result)
            output_dir = Path(tmp_dir) / "mobile_mcs_line_abc" / "mobile_rl_comparison"
            metrics_path = output_dir / "comparison_timestep_metrics.csv"
            self.assertTrue(metrics_path.exists())
            frame = pd.read_csv(metrics_path)
            self.assertIn(len(frame), {1440, 2880})
            self.assertEqual(result["summary"]["seed"], 20000)
            self.assertIn(result["summary"]["num_days"], {5, 10})
            self.assertGreaterEqual(int(frame["disruption_active"].sum()), 2)
            self.assertIn("allocation_vs_demand_alignment", frame.columns)
            self.assertGreater(len(frame.loc[frame["disruption_active"] == 1, "day_index"].dropna().unique()), 1)


if __name__ == "__main__":
    unittest.main()

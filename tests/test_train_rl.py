import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from evch.train.train_rl import WandbSb3Callback


class _FakeBaseCallback:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose


class _FakeReplayBuffer:
    def __len__(self) -> int:
        return 7


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int | None]] = []

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logs.append((metrics, step))


class WandbSb3CallbackTest(unittest.TestCase):
    def test_callback_logs_episode_and_step_metrics_with_simple_namespaces(self) -> None:
        fake_callbacks = ModuleType("stable_baselines3.common.callbacks")
        fake_callbacks.BaseCallback = _FakeBaseCallback
        fake_common = ModuleType("stable_baselines3.common")
        fake_common.callbacks = fake_callbacks
        fake_sb3 = ModuleType("stable_baselines3")
        fake_sb3.common = fake_common

        with mock.patch.dict(
            sys.modules,
            {
                "stable_baselines3": fake_sb3,
                "stable_baselines3.common": fake_common,
                "stable_baselines3.common.callbacks": fake_callbacks,
            },
        ):
            run = _FakeRun()
            callback_wrapper = WandbSb3Callback(run=run, log_interval=2, step_log_interval=2)

        model = SimpleNamespace(
            exploration_rate=0.25,
            replay_buffer=_FakeReplayBuffer(),
            logger=SimpleNamespace(
                name_to_value={
                    "train/loss": 1.5,
                    "rollout/exploration_rate": 0.25,
                }
            ),
        )

        first_step = SimpleNamespace(
            num_timesteps=1,
            model=model,
            locals={
                "infos": [
                    {
                        "served_demand": 2.0,
                        "unmet_demand": 1.0,
                        "num_active_mobile_stations": 3.0,
                        "num_active_mobile_stations_station_ab": 1.0,
                        "num_active_mobile_stations_station_bc": 2.0,
                        "num_active_chargers": 9.0,
                        "queue_length": 4.0,
                        "queue_wait_mean_minutes": 6.0,
                        "queue_length_station_ab": 1.0,
                        "queue_length_station_bc": 3.0,
                        "queue_wait_mean_minutes_station_ab": 2.0,
                        "queue_wait_mean_minutes_station_bc": 8.0,
                        "unused_mobile_chargers": 0.0,
                        "unused_mobile_stations_estimate": 1.0,
                        "utilization": 0.7,
                        "utilization_station_ab": 0.4,
                        "utilization_station_bc": 0.9,
                        "disruption_active": 1.0,
                        "disruption_type_code": 3.0,
                        "activated_mobile_stations": 1.0,
                        "adjusted_mobile_stations": 1.0,
                        "idle_capacity": 0.5,
                        "effective_capacity_total": 9.0,
                        "queue_wait_excess_minutes": 1.5,
                        "queue_wait_target_breached": 1.0,
                        "action_valid": True,
                    }
                ],
                "rewards": np.asarray([1.25], dtype=np.float32),
                "dones": np.asarray([False]),
                "actions": np.asarray([3], dtype=np.int64),
            },
        )
        second_step = SimpleNamespace(
            num_timesteps=2,
            model=model,
            locals={
                "infos": [
                    {
                        "served_demand": 1.0,
                        "unmet_demand": 0.0,
                        "num_active_mobile_stations": 2.0,
                        "num_active_mobile_stations_station_ab": 2.0,
                        "num_active_mobile_stations_station_bc": 0.0,
                        "num_active_chargers": 8.0,
                        "queue_length": 2.0,
                        "queue_wait_mean_minutes": 4.0,
                        "queue_length_station_ab": 2.0,
                        "queue_length_station_bc": 0.0,
                        "queue_wait_mean_minutes_station_ab": 4.0,
                        "queue_wait_mean_minutes_station_bc": 0.0,
                        "unused_mobile_chargers": 1.0,
                        "unused_mobile_stations_estimate": 0.0,
                        "utilization": 0.8,
                        "utilization_station_ab": 0.8,
                        "utilization_station_bc": 0.0,
                        "disruption_active": 0.0,
                        "disruption_type_code": 0.0,
                        "activated_mobile_stations": 0.0,
                        "adjusted_mobile_stations": 1.0,
                        "idle_capacity": 0.0,
                        "effective_capacity_total": 8.0,
                        "queue_wait_excess_minutes": 0.0,
                        "queue_wait_target_breached": 0.0,
                        "action_valid": True,
                    }
                ],
                "rewards": np.asarray([0.75], dtype=np.float32),
                "dones": np.asarray([True]),
                "actions": np.asarray([2], dtype=np.int64),
            },
        )

        self.assertTrue(callback_wrapper.on_step(first_step))
        self.assertTrue(callback_wrapper.on_step(second_step))

        logged_metrics = [metrics for metrics, _ in run.logs]

        step_log = next(metrics for metrics in logged_metrics if "rl_step/action" in metrics)
        self.assertEqual(step_log["rl_step/action"], 2.0)
        self.assertEqual(step_log["rl_step/queue_length_station_ab"], 2.0)
        self.assertEqual(step_log["rl_step/queue_length_station_bc"], 0.0)

        episode_log = next(metrics for metrics in logged_metrics if "rl/reward" in metrics)
        self.assertAlmostEqual(episode_log["rl/reward"], 2.0)
        self.assertAlmostEqual(episode_log["rl/served_demand"], 3.0)
        self.assertAlmostEqual(episode_log["rl/unmet_demand"], 1.0)
        self.assertAlmostEqual(episode_log["rl/loss"], 1.5)
        self.assertAlmostEqual(episode_log["rl/epsilon"], 0.25)
        self.assertAlmostEqual(episode_log["rl/mean_queue_wait_minutes_station_ab"], 3.0)
        self.assertAlmostEqual(episode_log["rl/mean_queue_wait_minutes_station_bc"], 4.0)

        summary_log = next(metrics for metrics in logged_metrics if "rl/timesteps" in metrics)
        self.assertEqual(summary_log["rl/timesteps"], 2.0)
        self.assertEqual(summary_log["rl/replay_buffer_size"], 7.0)
        self.assertEqual(summary_log["rl/epsilon"], 0.25)
        self.assertEqual(summary_log["sb3/train/loss"], 1.5)


if __name__ == "__main__":
    unittest.main()

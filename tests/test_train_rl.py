import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from evch.train.train_rl import WandbSb3Callback, _maybe_plot_training_curve, _train_with_torch_dqn, run_training


class _FakeBaseCallback:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose


class _FakeReplayBuffer:
    def __len__(self) -> int:
        return 7


class _FakeRun:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int | None]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class _FakeEnv:
    def __init__(self) -> None:
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=5)


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


class RunTrainingTrainOnlyTest(unittest.TestCase):
    def test_train_only_mode_skips_post_training_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "outputs" / "demo" / "rl" / "best_model.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("checkpoint", encoding="utf-8")

            config = {
                "seed": 7,
                "experiment": {
                    "name": "demo",
                    "output_root": str(Path(tmp_dir) / "outputs"),
                    "save_plots": False,
                },
                "logging": {
                    "level": "INFO",
                    "wandb": {"enabled": False},
                },
                "demand": {},
                "environment": {
                    "env_type": "line_corridor_mobile_mcs",
                },
                "comparison_rollout": {
                    "enabled": False,
                },
                "train_val_test": {
                    "training": {"episode_seed_range": [0, 99]},
                    "validation": {"enabled": False, "periodic_enabled": False},
                    "test_id": {"enabled": False},
                    "test_stress": {"enabled": False},
                },
                "rl": {
                    "backend": "torch_dqn",
                    "checkpoint_name": "best_model.pt",
                    "episodes": 2,
                    "evaluation_episodes": 2,
                },
            }

            run = _FakeRun()
            with (
                mock.patch("evch.train.train_rl.configure_logging"),
                mock.patch("evch.train.train_rl.set_global_seed"),
                mock.patch("evch.train.train_rl.configure_torch_runtime", return_value={}),
                mock.patch("evch.train.train_rl.init_wandb", return_value=run),
                mock.patch("evch.train.train_rl.log_artifact"),
                mock.patch("evch.train.train_rl.make_env", return_value=_FakeEnv()),
                mock.patch(
                    "evch.train.train_rl._train_with_torch_dqn",
                    return_value=("torch_dql", str(checkpoint_path), [{"episode": 0.0, "reward": 1.0, "loss": 0.5}]),
                ),
                mock.patch("evch.train.train_rl._make_rl_policy", return_value=lambda obs, env, deterministic=True: 0),
                mock.patch("evch.train.train_rl.evaluate_policy", side_effect=AssertionError("final validation should be skipped")),
                mock.patch(
                    "evch.train.train_rl.evaluate_policy_suites",
                    side_effect=AssertionError("evaluation suites should be skipped"),
                ),
            ):
                result = run_training(config)

            self.assertEqual(result["backend"], "torch_dql")
            self.assertEqual(result["checkpoint_path"], str(checkpoint_path))
            self.assertIsNone(result["evaluation"])
            self.assertIsNone(result["evaluation_suites"])
            self.assertTrue(run.finished)

            training_summary = Path(result["training_summary_path"]).read_text(encoding="utf-8")
            self.assertIn('"validation": null', training_summary)


class TrainingCurvePlotTest(unittest.TestCase):
    def test_training_curve_includes_sparse_eval_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = Path(tmp_dir) / "training_curve.png"
            history = [
                {
                    "episode": 0.0,
                    "reward": -10.0,
                    "loss": 5.0,
                    "eval_mean_reward": float("nan"),
                    "eval_td_loss": float("nan"),
                },
                {
                    "episode": 1.0,
                    "reward": -8.0,
                    "loss": 4.0,
                    "eval_mean_reward": -7.5,
                    "eval_td_loss": 3.5,
                },
                {
                    "episode": 2.0,
                    "reward": -6.0,
                    "loss": 3.0,
                    "eval_mean_reward": float("nan"),
                    "eval_td_loss": float("nan"),
                },
            ]

            created = _maybe_plot_training_curve(history, plot_path)

            self.assertTrue(created)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


class SimpleDqlRunLoggingTest(unittest.TestCase):
    def test_run_logging_skips_nan_eval_metrics_on_non_eval_episodes(self) -> None:
        from evch.rl.simple_dql import SimpleDQLAgent

        class _TinyEnv:
            def __init__(self) -> None:
                self.observation_space = SimpleNamespace(shape=(4,))
                self.action_space = SimpleNamespace(n=2)
                self._step = 0

            def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, float]]:
                self._step = 0
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
                self._step += 1
                info = {
                    "served_demand": 1.0,
                    "unmet_demand": 0.0,
                    "action_valid": True,
                    "num_active_chargers": 0.0,
                    "num_active_mobile_stations": 0.0,
                    "utilization": 0.0,
                    "idle_capacity": 0.0,
                    "effective_capacity_total": 0.0,
                    "unused_mobile_chargers": 0.0,
                    "unused_mobile_stations_estimate": 0.0,
                    "activated_mobile_stations": 0.0,
                    "adjusted_mobile_stations": 0.0,
                    "queue_length": 0.0,
                    "queue_wait_mean_minutes": 0.0,
                    "queue_wait_mean_minutes_station_ab": 0.0,
                    "queue_wait_mean_minutes_station_bc": 0.0,
                    "queue_wait_excess_minutes": 0.0,
                    "queue_wait_target_breached": 0.0,
                    "disruption_active": 0.0,
                }
                return np.zeros(4, dtype=np.float32), 1.0, True, False, info

            def valid_action_mask(self) -> np.ndarray:
                return np.asarray([True, True], dtype=bool)

        run = _FakeRun()
        agent = SimpleDQLAgent(
            obs_dim=4,
            action_dim=2,
            config={
                "gamma": 0.98,
                "learning_rate": 0.0005,
                "epsilon_start": 0.0,
                "epsilon_end": 0.0,
                "epsilon_decay_steps": 1,
                "wandb_step_log_interval": 1000,
            },
            seed=0,
        )

        history = agent.train(
            env=_TinyEnv(),
            episodes=1,
            max_steps=1,
            run=run,
            eval_env_factory=None,
            eval_interval=0,
            eval_interval_steps=0,
            eval_episodes=1,
        )

        self.assertEqual(len(history), 1)
        rl_logs = [metrics for metrics, _ in run.logs if any(key.startswith("rl/") for key in metrics)]
        self.assertEqual(len(rl_logs), 1)
        self.assertNotIn("rl/eval_mean_reward", rl_logs[0])
        self.assertNotIn("rl/eval_td_loss", rl_logs[0])


class TorchDqnPeriodicEvalConfigTest(unittest.TestCase):
    def test_step_based_eval_disables_default_episode_eval_fallback(self) -> None:
        env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(4,)),
            action_space=SimpleNamespace(n=5),
            max_steps=1728,
            duration_days_range=[2, 6],
        )
        train_calls: list[dict[str, object]] = []

        class _FakeAgent:
            def __init__(self, obs_dim: int, action_dim: int, config: dict[str, object], seed: int) -> None:
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.config = config
                self.seed = seed

            def train(self, **kwargs: object) -> list[dict[str, float]]:
                train_calls.append(kwargs)
                return [{"episode": 0.0, "reward": 1.0, "loss": 0.5}]

            def save(self, path: Path) -> None:
                Path(path).write_text("checkpoint", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            rl_cfg = {
                "episodes": 130,
                "checkpoint_name": "best_model.pt",
                "eval_interval_steps": 15000,
                "eval_during_training_episodes": 10,
                "evaluation_episodes": 12,
            }
            with mock.patch("evch.train.train_rl.SimpleDQNAgent", _FakeAgent):
                _train_with_torch_dqn(
                    env=env,
                    rl_cfg=rl_cfg,
                    seed=7,
                    output_dir=output_dir,
                    run=None,
                    eval_env_factory=lambda eval_seed: env,
                    eval_episode_seeds=[1, 2, 3],
                )

        self.assertEqual(len(train_calls), 1)
        self.assertEqual(train_calls[0]["eval_interval"], 0)
        self.assertEqual(train_calls[0]["eval_interval_steps"], 15000)
        self.assertEqual(train_calls[0]["eval_episodes"], 10)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evch.config.loader import build_config_parser, load_config
from evch.envs.factory import make_env
from evch.rl.evaluation import evaluate_policy
from evch.rl.simple_dql import SimpleDQLAgent as SimpleDQNAgent
from evch.train.eval_suites import build_seed_list, evaluate_policy_suites, resolve_train_seed_range, run_heldout_comparison
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.torch_runtime import configure_torch_runtime

LOGGER = logging.getLogger(__name__)


def _deep_merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _train_with_torch_dqn(
    env: Any,
    rl_cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
    eval_env_factory: Callable[[int], Any] | None = None,
    eval_episode_seeds: list[int] | None = None,
) -> tuple[str, str, list[dict[str, float]]]:
    configured_max_steps = int(rl_cfg.get("max_steps_per_episode", 0))
    env_max_steps = int(getattr(env, "max_steps", 0))
    if getattr(env, "duration_days_range", None) is not None and env_max_steps > 0:
        train_max_steps = max(configured_max_steps, env_max_steps)
    else:
        train_max_steps = configured_max_steps
    agent = SimpleDQNAgent(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.n),
        config=rl_cfg,
        seed=seed,
    )
    eval_interval_steps = int(rl_cfg.get("eval_interval_steps", 0))
    if "eval_interval_episodes" in rl_cfg:
        eval_interval_episodes = int(rl_cfg["eval_interval_episodes"])
    else:
        eval_interval_episodes = 0 if eval_interval_steps > 0 else max(1, int(rl_cfg["episodes"]) // 8)
    history = agent.train(
        env=env,
        episodes=int(rl_cfg["episodes"]),
        max_steps=train_max_steps,
        eval_env_factory=eval_env_factory,
        eval_interval=eval_interval_episodes,
        eval_interval_steps=eval_interval_steps,
        eval_episodes=int(rl_cfg.get("eval_during_training_episodes", max(1, int(rl_cfg.get("evaluation_episodes", 1))))),
        eval_seed=seed + 10_000,
        eval_episode_seeds=eval_episode_seeds,
    )
    checkpoint = output_dir / str(rl_cfg["checkpoint_name"])
    agent.save(checkpoint)
    return "torch_dql", str(checkpoint), history


def _make_rl_policy(backend: str, checkpoint_path: str) -> Callable[[np.ndarray, Any, bool], int]:
    agent = SimpleDQNAgent.load(checkpoint_path)

    def policy(observation: np.ndarray, _env: Any, deterministic: bool = True) -> int:
        return agent.act(observation, deterministic=deterministic, env=_env)

    return policy


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    rl_cfg = config["rl"]
    runtime_info = configure_torch_runtime(rl_cfg)
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "rl")

    split_cfg = dict(config.get("train_val_test", {}))
    training_cfg = dict(split_cfg.get("training", {}))
    val_cfg = dict(split_cfg.get("validation", {}))
    test_id_cfg = dict(split_cfg.get("test_id", {}))
    stress_cfg = dict(split_cfg.get("test_stress", {}))
    val_enabled = bool(val_cfg.get("enabled", True))
    periodic_eval_enabled = bool(val_cfg.get("periodic_enabled", val_enabled))

    val_env_overrides = val_cfg.get("environment_overrides", {})
    val_env_cfg = (
        _deep_merge_dicts(config["environment"], val_env_overrides)
        if isinstance(val_env_overrides, dict) and val_env_overrides
        else copy.deepcopy(config["environment"])
    )
    raw_val_seed_list = build_seed_list(val_cfg.get("seeds", val_cfg))
    val_seed_list = raw_val_seed_list if val_enabled else []
    val_episodes = len(val_seed_list) if val_seed_list else int(val_cfg.get("episodes", rl_cfg["evaluation_episodes"]))
    periodic_eval_seed_count = int(val_cfg.get("periodic_seed_count", min(max(len(raw_val_seed_list), 1), 32)))
    periodic_eval_seeds = raw_val_seed_list[:periodic_eval_seed_count] if (periodic_eval_enabled and raw_val_seed_list) else None

    train_env_cfg = copy.deepcopy(config["environment"])
    train_seed_range = resolve_train_seed_range(training_cfg)
    if train_seed_range is not None:
        train_env_cfg["episode_seed_range"] = [int(train_seed_range[0]), int(train_seed_range[1])]

    env = make_env(train_env_cfg, config["demand"], seed=seed)
    requested_backend = str(rl_cfg.get("backend", "torch_dqn")).strip().lower()
    if requested_backend not in {"torch_dqn", "torch_dql"}:
        raise RuntimeError(f"Unsupported RL backend for exam_submission: {requested_backend}")

    backend, checkpoint_path, history = _train_with_torch_dqn(
        env,
        rl_cfg,
        seed=seed,
        output_dir=output_dir,
        eval_env_factory=(lambda eval_seed: make_env(val_env_cfg, config["demand"], seed=eval_seed))
        if periodic_eval_enabled
        else None,
        eval_episode_seeds=periodic_eval_seeds if periodic_eval_enabled else None,
    )

    policy = _make_rl_policy(backend, checkpoint_path)
    evaluation: dict[str, Any] | None = None
    if val_enabled and val_episodes > 0:
        eval_env = make_env(val_env_cfg, config["demand"], seed=(val_seed_list[0] if val_seed_list else seed + 17))
        evaluation = evaluate_policy(
            env=eval_env,
            policy=policy,
            episodes=val_episodes,
            seed=seed,
            deterministic=bool(rl_cfg.get("deterministic_eval", True)),
            episode_seeds=val_seed_list if val_seed_list else None,
        )

    training_summary_path = output_dir / "training_summary.json"
    write_json(
        training_summary_path,
        {
            "backend": backend,
            "checkpoint_path": checkpoint_path,
            "runtime": runtime_info,
            "validation": (
                {key: value for key, value in evaluation.items() if key not in {"episodes", "episode_seeds"}}
                if evaluation is not None
                else None
            ),
        },
    )
    if history:
        write_json(output_dir / "history.json", {"history": history})

    suite_outputs = None
    if bool(test_id_cfg.get("enabled", False)) or bool(stress_cfg.get("enabled", False)) or (val_enabled and bool(val_seed_list)):
        suite_outputs = evaluate_policy_suites(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=ensure_dir(output_dir / "evaluation_suites"),
        )

    heldout_paths = run_heldout_comparison(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=ensure_dir(output_dir / "heldout_rollout"),
    )

    LOGGER.info("Finished RL training with backend=%s checkpoint=%s", backend, checkpoint_path)
    return {
        "backend": backend,
        "checkpoint_path": checkpoint_path,
        "history": history,
        "evaluation": evaluation,
        "output_dir": str(output_dir),
        "training_summary_path": str(training_summary_path),
        "runtime": runtime_info,
        "evaluation_suites": suite_outputs,
        "heldout_paths": heldout_paths,
    }


def main() -> None:
    parser = build_config_parser("Train the EV charger placement RL baseline.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

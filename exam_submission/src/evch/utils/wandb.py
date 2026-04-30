from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class DummyRun:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def define_metric(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def watch(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def config(self) -> dict[str, Any]:
        return {}

    @property
    def summary(self) -> dict[str, Any]:
        return {}


def _configure_wandb_paths(config: dict[str, Any]) -> None:
    wandb_cfg = (config.get("logging") or {}).get("wandb", {})
    experiment_cfg = config.get("experiment") or {}
    root_dir = Path(
        os.getenv("WANDB_DIR")
        or wandb_cfg.get("dir")
        or (Path(experiment_cfg.get("output_root", "outputs")) / "wandb")
    ).resolve()
    data_dir = Path(os.getenv("WANDB_DATA_DIR") or wandb_cfg.get("data_dir") or (root_dir / "data")).resolve()
    cache_dir = Path(os.getenv("WANDB_CACHE_DIR") or wandb_cfg.get("cache_dir") or (root_dir / "cache")).resolve()
    config_dir = Path(os.getenv("WANDB_CONFIG_DIR") or wandb_cfg.get("config_dir") or (root_dir / "config")).resolve()
    artifact_dir = Path(
        os.getenv("WANDB_ARTIFACT_DIR") or wandb_cfg.get("artifact_dir") or (root_dir / "artifacts")
    ).resolve()

    for path in (root_dir, data_dir, cache_dir, config_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DIR", str(root_dir))
    os.environ.setdefault("WANDB_DATA_DIR", str(data_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("WANDB_ARTIFACT_DIR", str(artifact_dir))


def init_wandb(config: dict[str, Any], job_type: str, run_name: str) -> Any:
    wandb_cfg = (config.get("logging") or {}).get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled or find_spec("wandb") is None:
        return DummyRun()

    _configure_wandb_paths(config)

    import wandb  # type: ignore

    mode = wandb_cfg.get("mode", "offline")

    try:
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "adaptive-ev-charging")),
            entity=os.getenv("WANDB_ENTITY", wandb_cfg.get("entity", "EV-charging")),
            config=config,
            mode=mode,
            job_type=job_type,
            name=run_name,
            group=wandb_cfg.get("group"),
            tags=wandb_cfg.get("tags", []),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to initialize Weights & Biases: %s", exc)
        return DummyRun()


def log_artifact(
    run: Any,
    path: str | Path,
    artifact_name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
) -> None:
    artifact_path = Path(path)
    if isinstance(run, DummyRun) or not artifact_path.exists() or find_spec("wandb") is None:
        return

    try:
        import wandb  # type: ignore

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(str(artifact_path))
        run.log_artifact(artifact, aliases=aliases or [])
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Failed to log W&B artifact for %s: %s", artifact_path, exc)

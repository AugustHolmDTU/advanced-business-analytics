from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from typing import Any

LOGGER = logging.getLogger(__name__)


class DummyRun:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @property
    def config(self) -> dict[str, Any]:
        return {}


def init_wandb(config: dict[str, Any], job_type: str, run_name: str) -> Any:
    wandb_cfg = (config.get("logging") or {}).get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", False))
    if not enabled or find_spec("wandb") is None:
        return DummyRun()

    import wandb  # type: ignore

    mode = wandb_cfg.get("mode", "offline")
    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        LOGGER.warning("WANDB_API_KEY missing, downgrading to offline mode.")
        mode = "offline"

    try:
        return wandb.init(
            project=os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "adaptive-ev-charging")),
            entity=os.getenv("WANDB_ENTITY", wandb_cfg.get("entity")),
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


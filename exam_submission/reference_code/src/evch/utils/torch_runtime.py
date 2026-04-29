from __future__ import annotations

import os
from typing import Any

import torch


def resolve_torch_device(preference: str | None = "auto") -> torch.device:
    requested = str(preference or "auto").lower()
    if requested == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def configure_torch_runtime(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or {}
    num_threads = cfg.get("torch_num_threads")
    num_interop_threads = cfg.get("torch_num_interop_threads")

    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except RuntimeError:
            pass

    return {
        "device": str(resolve_torch_device(cfg.get("device", "auto"))),
        "torch_num_threads": int(torch.get_num_threads()),
        "cpu_count": int(os.cpu_count() or 1),
    }

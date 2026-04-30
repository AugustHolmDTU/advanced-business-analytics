from __future__ import annotations

from typing import Any

from evch.envs.line_corridor_mobile_env import LineCorridorMobileStationEnv


def make_env(environment_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> Any:
    env_type = str(environment_config.get("env_type", "placement")).lower()
    if env_type in {"line_corridor_mobile_mcs", "line_corridor_mobile_station", "corridor_mobile_mcs_abc"}:
        return LineCorridorMobileStationEnv(environment_config, demand_config, seed=seed)
    raise ValueError(f"Unsupported environment env_type for exam_submission: {env_type}")

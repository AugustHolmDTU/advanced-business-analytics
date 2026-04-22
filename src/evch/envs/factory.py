from __future__ import annotations

from typing import Any

from evch.envs.charging_env import ChargingPlacementEnv
from evch.envs.mobile_station_env import MobileStationChargingEnv


def make_env(environment_config: dict[str, Any], demand_config: dict[str, Any], seed: int = 0) -> Any:
    env_type = str(environment_config.get("env_type", "placement")).lower()
    if env_type in {"placement", "charging_placement"}:
        return ChargingPlacementEnv(environment_config, demand_config, seed=seed)
    if env_type in {"mobile_station_capacity", "mobile_mcs"}:
        return MobileStationChargingEnv(environment_config, demand_config, seed=seed)
    raise ValueError(f"Unsupported environment env_type: {env_type}")

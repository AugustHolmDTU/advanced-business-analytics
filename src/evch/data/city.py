from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evch.data.synthetic import SyntheticCity, make_synthetic_city


def _sample_zone_profiles(num_zones: int, demand_config: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    zone_base_demand = rng.uniform(
        float(demand_config["base_rate_min"]),
        float(demand_config["base_rate_max"]),
        size=num_zones,
    ).astype(np.float32)
    zone_scale = rng.normal(0.0, float(demand_config["zone_scale_std"]), size=num_zones).astype(np.float32)
    return zone_base_demand, zone_scale


def load_tomtom_snapshot_city(snapshot_path: str | Path, demand_config: dict[str, Any], seed: int) -> SyntheticCity:
    path = Path(snapshot_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    site_coords = np.asarray(payload["site_coords_km"], dtype=np.float32)
    zone_coords = np.asarray(payload["zone_coords_km"], dtype=np.float32)
    travel_time_matrix = np.asarray(payload["travel_time_matrix_minutes"], dtype=np.float32)
    site_latlon = np.asarray(payload["site_latlon"], dtype=np.float32)
    zone_latlon = np.asarray(payload["zone_latlon"], dtype=np.float32)
    zone_base_demand, zone_scale = _sample_zone_profiles(
        num_zones=int(zone_coords.shape[0]),
        demand_config=demand_config,
        seed=seed,
    )

    return SyntheticCity(
        site_coords=site_coords,
        zone_coords=zone_coords,
        zone_base_demand=zone_base_demand,
        zone_scale=zone_scale,
        travel_time_matrix=travel_time_matrix,
        site_names=list(payload.get("site_names", [])) or None,
        site_addresses=list(payload.get("site_addresses", [])) or None,
        site_latlon=site_latlon,
        zone_latlon=zone_latlon,
        center_latlon=(float(payload["center"]["lat"]), float(payload["center"]["lon"])),
    )


def build_city(env_config: dict[str, Any], demand_config: dict[str, Any], seed: int) -> SyntheticCity:
    city_cfg = env_config.get("city_data", {})
    source = str(city_cfg.get("source", "synthetic"))
    if source == "synthetic":
        return make_synthetic_city(
            num_sites=int(env_config["num_candidate_sites"]),
            num_zones=int(env_config["num_demand_zones"]),
            city_extent_km=float(env_config["city_extent_km"]),
            seed=seed,
            base_rate_min=float(demand_config["base_rate_min"]),
            base_rate_max=float(demand_config["base_rate_max"]),
            zone_scale_std=float(demand_config["zone_scale_std"]),
        )
    if source == "tomtom_snapshot":
        snapshot_path = city_cfg.get("snapshot_path")
        if not snapshot_path:
            raise ValueError("environment.city_data.snapshot_path is required for source=tomtom_snapshot")
        return load_tomtom_snapshot_city(snapshot_path=snapshot_path, demand_config=demand_config, seed=seed)
    raise ValueError(f"Unknown city data source: {source}")

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evch.data.synthetic import SyntheticCity, make_synthetic_layout


def _sample_zone_profiles_from_snapshot(
    payload: dict[str, Any],
    demand_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    stations = list(payload.get("stations", []))
    num_zones = len(payload["zone_coords_km"])
    base_low = float(demand_config["base_rate_min"])
    base_high = float(demand_config["base_rate_max"])
    sampled_base = rng.uniform(base_low, base_high, size=num_zones)
    sampled_scale = rng.normal(0.0, float(demand_config["zone_scale_std"]), size=num_zones)

    if not stations:
        return sampled_base.astype(np.float32), sampled_scale.astype(np.float32)

    travel_time_matrix = np.asarray(payload["travel_time_matrix_minutes"], dtype=np.float32)
    connector_counts = np.asarray([float(station.get("total_connectors", 0.0)) for station in stations], dtype=np.float32)
    available_counts = np.asarray([float(station.get("available_connectors", 0.0)) for station in stations], dtype=np.float32)
    power_levels = np.asarray([float(station.get("max_power_kw") or 50.0) for station in stations], dtype=np.float32)
    supply_score = connector_counts + 0.6 * available_counts + 0.015 * power_levels
    if float(np.max(supply_score)) <= 0.0:
        return sampled_base.astype(np.float32), sampled_scale.astype(np.float32)

    weights = np.exp(-0.06 * travel_time_matrix)
    accessibility = weights @ supply_score
    if float(np.max(accessibility)) > float(np.min(accessibility)):
        normalized_access = (accessibility - float(np.min(accessibility))) / (
            float(np.max(accessibility)) - float(np.min(accessibility))
        )
    else:
        normalized_access = np.zeros_like(accessibility)

    grounded_base = sampled_base * (0.7 + 0.7 * normalized_access)
    grounded_scale = sampled_scale + 0.15 * (normalized_access - 0.5)
    return grounded_base.astype(np.float32), grounded_scale.astype(np.float32)


def _classify_station_types(payload: dict[str, Any]) -> tuple[str, ...]:
    stations = list(payload.get("stations", []))
    if not stations:
        return tuple(["tomtom_station"] * len(payload.get("site_coords_km", [])))
    connectors = np.asarray([float(station.get("total_connectors", 0.0)) for station in stations], dtype=np.float32)
    power_levels = np.asarray([float(station.get("max_power_kw") or 50.0) for station in stations], dtype=np.float32)
    connector_threshold = float(np.quantile(connectors, 0.7)) if len(connectors) > 1 else float(connectors[0])
    types: list[str] = []
    for connector_count, power_kw in zip(connectors, power_levels, strict=True):
        if connector_count >= connector_threshold or power_kw >= 150.0:
            types.append("major_hub")
        else:
            types.append("standard_station")
    return tuple(types)


def load_tomtom_snapshot_city(snapshot_path: str | Path, demand_config: dict[str, Any], seed: int) -> SyntheticCity:
    path = Path(snapshot_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    site_coords = np.asarray(payload["site_coords_km"], dtype=np.float32)
    zone_coords = np.asarray(payload["zone_coords_km"], dtype=np.float32)
    travel_time_matrix = np.asarray(payload["travel_time_matrix_minutes"], dtype=np.float32)
    site_latlon = np.asarray(payload.get("site_latlon", []), dtype=np.float32) if payload.get("site_latlon") else None
    zone_latlon = np.asarray(payload.get("zone_latlon", []), dtype=np.float32) if payload.get("zone_latlon") else None
    zone_base_demand, zone_scale = _sample_zone_profiles_from_snapshot(
        payload=payload,
        demand_config=demand_config,
        seed=seed,
    )

    station_names = tuple(
        station.get("name") or f"TomTom Station {idx + 1}"
        for idx, station in enumerate(payload.get("stations", []))
    )
    station_types = _classify_station_types(payload)
    center = payload.get("center") or {}
    return SyntheticCity(
        site_coords=site_coords,
        zone_coords=zone_coords,
        zone_base_demand=zone_base_demand,
        zone_scale=zone_scale,
        travel_time_matrix=travel_time_matrix,
        site_latlon=site_latlon,
        zone_latlon=zone_latlon,
        center_latlon=(float(center["lat"]), float(center["lon"])) if center else None,
        layout_type="tomtom_snapshot",
        site_labels=station_names,
        site_types=station_types,
        zone_labels=tuple(f"Grounded Zone {idx + 1}" for idx in range(len(zone_coords))),
        zone_types=tuple(["grounded_zone"] * len(zone_coords)),
        metadata={
            "location_query": payload.get("location_query"),
            "search_radius_m": payload.get("search_radius_m"),
            "stations": payload.get("stations", []),
            "clustering": payload.get("clustering"),
        },
    )


def build_city(env_config: dict[str, Any], demand_config: dict[str, Any], seed: int) -> SyntheticCity:
    city_cfg = env_config.get("city_data", {})
    source = str(city_cfg.get("source", "synthetic")).lower()
    if source == "synthetic":
        return make_synthetic_layout(env_config=env_config, demand_config=demand_config, seed=seed)
    if source == "tomtom_snapshot":
        snapshot_path = city_cfg.get("snapshot_path")
        if not snapshot_path:
            raise ValueError("environment.city_data.snapshot_path is required for source=tomtom_snapshot")
        return load_tomtom_snapshot_city(snapshot_path=snapshot_path, demand_config=demand_config, seed=seed)
    raise ValueError(f"Unknown city data source: {source}")

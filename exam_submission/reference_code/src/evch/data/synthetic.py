from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class SyntheticCity:
    site_coords: np.ndarray
    zone_coords: np.ndarray
    zone_base_demand: np.ndarray
    zone_scale: np.ndarray
    travel_time_matrix: np.ndarray
    site_latlon: np.ndarray | None = None
    zone_latlon: np.ndarray | None = None
    center_latlon: tuple[float, float] | None = None
    layout_type: str = "urban_grid"
    site_labels: tuple[str, ...] = field(default_factory=tuple)
    site_types: tuple[str, ...] = field(default_factory=tuple)
    zone_labels: tuple[str, ...] = field(default_factory=tuple)
    zone_types: tuple[str, ...] = field(default_factory=tuple)
    corridor_polyline: np.ndarray | None = None
    site_positions_km: np.ndarray | None = None
    zone_positions_km: np.ndarray | None = None
    site_travel_time_matrix: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def make_synthetic_city(
    num_sites: int,
    num_zones: int,
    city_extent_km: float,
    seed: int,
    base_rate_min: float,
    base_rate_max: float,
    zone_scale_std: float,
) -> SyntheticCity:
    rng = np.random.default_rng(seed)
    zone_coords = rng.uniform(0.0, city_extent_km, size=(num_zones, 2))
    site_coords = rng.uniform(0.0, city_extent_km, size=(num_sites, 2))
    zone_base_demand = rng.uniform(base_rate_min, base_rate_max, size=num_zones)
    zone_scale = rng.normal(0.0, zone_scale_std, size=num_zones)

    # TODO: replace this Euclidean proxy with road-network travel times from OSM/TomTom.
    distances = np.linalg.norm(zone_coords[:, None, :] - site_coords[None, :, :], axis=2)
    travel_time_matrix = 3.0 + distances * 2.5
    return SyntheticCity(
        site_coords=site_coords.astype(np.float32),
        zone_coords=zone_coords.astype(np.float32),
        zone_base_demand=zone_base_demand.astype(np.float32),
        zone_scale=zone_scale.astype(np.float32),
        travel_time_matrix=travel_time_matrix.astype(np.float32),
        layout_type="urban_grid",
    )


def _allocate_counts(total: int, weights: dict[str, float], minimum_if_possible: int = 1) -> dict[str, int]:
    keys = list(weights.keys())
    if total <= 0:
        return {key: 0 for key in keys}

    counts = {key: 0 for key in keys}
    remaining = total
    if total >= len(keys) * minimum_if_possible:
        for key in keys:
            counts[key] = minimum_if_possible
        remaining -= len(keys) * minimum_if_possible

    weight_values = np.asarray([max(float(weights[key]), 0.0) for key in keys], dtype=np.float64)
    if weight_values.sum() <= 0.0:
        weight_values[:] = 1.0
    raw = remaining * weight_values / weight_values.sum()
    floor = np.floor(raw).astype(int)
    for key, value in zip(keys, floor):
        counts[key] += int(value)
    leftovers = remaining - int(floor.sum())
    if leftovers > 0:
        order = np.argsort(raw - floor)[::-1]
        for idx in order[:leftovers]:
            counts[keys[int(idx)]] += 1
    return counts


def _build_segment_speeds(
    corridor_length_km: float,
    city_positions_km: np.ndarray,
    mainline_speed_kmh: float,
    urban_speed_kmh: float,
    urban_influence_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    breaks = [0.0, corridor_length_km]
    for position in city_positions_km:
        breaks.append(max(0.0, float(position) - urban_influence_km))
        breaks.append(min(corridor_length_km, float(position) + urban_influence_km))
    segment_breaks = np.unique(np.round(np.asarray(breaks, dtype=np.float32), 3))
    segment_speeds: list[float] = []
    for start, end in zip(segment_breaks[:-1], segment_breaks[1:]):
        midpoint = 0.5 * float(start + end)
        in_urban_window = np.any(np.abs(city_positions_km - midpoint) <= urban_influence_km)
        segment_speeds.append(urban_speed_kmh if in_urban_window else mainline_speed_kmh)
    return segment_breaks.astype(np.float32), np.asarray(segment_speeds, dtype=np.float32)


def _segment_travel_minutes(start_km: float, end_km: float, segment_breaks: np.ndarray, segment_speeds: np.ndarray) -> float:
    lo = min(start_km, end_km)
    hi = max(start_km, end_km)
    total = 0.0
    for seg_start, seg_end, speed in zip(segment_breaks[:-1], segment_breaks[1:], segment_speeds):
        overlap = max(0.0, min(hi, float(seg_end)) - max(lo, float(seg_start)))
        if overlap > 0.0:
            total += overlap / float(speed) * 60.0
    return total


def _sample_service_positions(
    rng: np.random.Generator,
    count: int,
    corridor_length_km: float,
    avoid_positions: np.ndarray,
    min_separation_km: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    base = np.linspace(0.12 * corridor_length_km, 0.88 * corridor_length_km, count + 2, dtype=np.float32)[1:-1]
    jitter = rng.normal(0.0, 0.03 * corridor_length_km, size=count)
    positions = np.clip(base + jitter, 0.05 * corridor_length_km, 0.95 * corridor_length_km)
    if avoid_positions.size == 0:
        return np.sort(positions.astype(np.float32))
    adjusted = []
    for position in positions:
        candidate = float(position)
        for _ in range(12):
            if np.all(np.abs(avoid_positions - candidate) >= min_separation_km):
                break
            candidate = float(np.clip(candidate + rng.normal(0.0, min_separation_km * 0.6), 0.05 * corridor_length_km, 0.95 * corridor_length_km))
        adjusted.append(candidate)
    return np.sort(np.asarray(adjusted, dtype=np.float32))


def _site_label(site_type: str, index: int) -> str:
    prefix = {
        "city_hub": "City Hub",
        "service_area": "Service Area",
        "highway_exit": "Exit",
    }.get(site_type, "Site")
    return f"{prefix} {index + 1}"


def _zone_label(zone_type: str, index: int) -> str:
    prefix = {
        "city": "Metro Demand",
        "suburb": "Suburban Demand",
        "corridor": "Corridor Demand",
        "logistics": "Logistics Demand",
    }.get(zone_type, "Demand")
    return f"{prefix} {index + 1}"


def _load_calibration_priors(priors_path: str | None) -> dict[str, Any]:
    if not priors_path:
        return {}
    path = Path(priors_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration priors file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _prior_values(payload: dict[str, Any], key: str) -> np.ndarray:
    values = (((payload.get("distributions") or {}).get(key) or {}).get("values") or [])
    if not values:
        return np.empty(0, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _sample_prior_values(
    rng: np.random.Generator,
    values: np.ndarray,
    count: int,
    fallback_low: float,
    fallback_high: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    if values.size > 0:
        return rng.choice(values, size=count, replace=True).astype(np.float32)
    return rng.uniform(fallback_low, fallback_high, size=count).astype(np.float32)


def _coerce_positions(raw_values: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray:
    if raw_values is None:
        return np.empty(0, dtype=np.float32)
    values = np.asarray(raw_values, dtype=np.float32)
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Explicit corridor positions must be a 1D list.")
    return np.sort(values.astype(np.float32))


def _sample_site_profile_value(
    rng: np.random.Generator,
    profile: dict[str, Any],
    field: str,
    fallback_sampler: Callable[[], float],
) -> float:
    if profile:
        raw_range = profile.get(f"{field}_range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            low = float(raw_range[0])
            high = float(raw_range[1])
            if high < low:
                low, high = high, low
            if field == "connector":
                return float(rng.integers(max(int(round(low)), 1), max(int(round(high)), 1) + 1))
            return float(rng.uniform(low, high))
        if field == "availability":
            mean = profile.get("availability_mean")
            std = profile.get("availability_std")
            if mean is not None:
                sampled = float(rng.normal(float(mean), float(std or 0.08)))
                return float(np.clip(sampled, 0.15, 1.0))
    return float(fallback_sampler())


def make_synthetic_corridor(
    env_config: dict[str, Any],
    demand_config: dict[str, Any],
    seed: int,
) -> SyntheticCity:
    rng = np.random.default_rng(seed)
    num_sites = int(env_config["num_candidate_sites"])
    num_zones = int(env_config["num_demand_zones"])
    corridor_cfg = env_config.get("corridor", {})
    priors = _load_calibration_priors(corridor_cfg.get("calibration_priors_path"))
    recommended = priors.get("recommended_corridor_defaults") or {}
    corridor_length_km = float(env_config.get("corridor_length_km", env_config.get("city_extent_km", 220.0)))
    explicit_city_positions = _coerce_positions(corridor_cfg.get("city_anchor_positions_km"))
    explicit_service_positions = _coerce_positions(corridor_cfg.get("service_area_positions_km"))
    explicit_exit_positions = _coerce_positions(corridor_cfg.get("exit_positions_km"))
    num_cities = int(corridor_cfg.get("num_cities", len(explicit_city_positions) or min(max(2, num_sites // 4), 4)))
    num_cities = max(1, min(num_cities, num_sites))
    mainline_speed_kmh = float(corridor_cfg.get("mainline_speed_kmh", recommended.get("mainline_speed_kmh", 105.0)))
    urban_speed_kmh = float(corridor_cfg.get("urban_speed_kmh", recommended.get("urban_speed_kmh", 72.0)))
    urban_influence_km = float(corridor_cfg.get("urban_influence_km", 14.0))

    if explicit_city_positions.size > 0:
        city_positions = np.clip(explicit_city_positions, 0.06 * corridor_length_km, 0.94 * corridor_length_km).astype(np.float32)
        num_cities = int(len(city_positions))
    else:
        city_anchor_positions = np.linspace(0.08, 0.92, num_cities, dtype=np.float32) * corridor_length_km
        city_jitter = rng.normal(0.0, 0.03 * corridor_length_km, size=num_cities)
        city_positions = np.sort(
            np.clip(city_anchor_positions + city_jitter, 0.06 * corridor_length_km, 0.94 * corridor_length_km)
        ).astype(np.float32)
    city_strength = rng.uniform(0.9, 1.4, size=num_cities).astype(np.float32)

    site_mix = corridor_cfg.get(
        "site_mix",
        {
            "city_hub": 0.28,
            "service_area": 0.34,
            "highway_exit": 0.38,
        },
    )
    explicit_counts = {
        "city_hub": int(corridor_cfg.get("site_type_counts", {}).get("city_hub", len(explicit_city_positions))),
        "service_area": int(corridor_cfg.get("site_type_counts", {}).get("service_area", len(explicit_service_positions))),
        "highway_exit": int(corridor_cfg.get("site_type_counts", {}).get("highway_exit", len(explicit_exit_positions))),
    }
    required_sites = sum(max(value, 0) for value in explicit_counts.values())
    if required_sites > num_sites:
        raise ValueError("Configured explicit corridor site counts exceed num_candidate_sites.")
    site_counts = _allocate_counts(num_sites - required_sites, site_mix, minimum_if_possible=0)
    for key, value in explicit_counts.items():
        site_counts[key] = site_counts.get(key, 0) + max(value, 0)
    site_counts["city_hub"] = max(site_counts.get("city_hub", 0), min(num_cities, num_sites))
    overflow = sum(site_counts.values()) - num_sites
    if overflow > 0:
        for key in ("highway_exit", "service_area", "city_hub"):
            reducible = min(overflow, max(site_counts[key] - (1 if key == "city_hub" else 0), 0))
            site_counts[key] -= reducible
            overflow -= reducible
            if overflow <= 0:
                break

    city_site_positions_list: list[float] = list(explicit_city_positions.tolist())
    for idx in range(len(city_site_positions_list), site_counts["city_hub"]):
        anchor = float(city_positions[idx % len(city_positions)])
        jitter = 0.0 if idx < len(city_positions) and explicit_city_positions.size == 0 else float(rng.normal(0.0, 4.0))
        position = float(np.clip(anchor + jitter, 0.03 * corridor_length_km, 0.97 * corridor_length_km))
        city_site_positions_list.append(position)
    city_site_positions = np.sort(np.asarray(city_site_positions_list, dtype=np.float32))
    service_positions = _sample_service_positions(
        rng=rng,
        count=max(site_counts["service_area"] - len(explicit_service_positions), 0),
        corridor_length_km=corridor_length_km,
        avoid_positions=city_positions,
        min_separation_km=float(corridor_cfg.get("service_area_min_gap_km", recommended.get("service_area_min_gap_km", 18.0))),
    )
    service_positions = np.sort(np.concatenate([explicit_service_positions, service_positions]).astype(np.float32))
    exit_positions = np.sort(
        np.concatenate(
            [
                explicit_exit_positions,
                rng.uniform(
                    0.04 * corridor_length_km,
                    0.96 * corridor_length_km,
                    size=max(site_counts["highway_exit"] - len(explicit_exit_positions), 0),
                ).astype(np.float32),
            ]
        )
    ).astype(np.float32)

    site_positions = np.concatenate([city_site_positions, service_positions, exit_positions]).astype(np.float32)
    site_types = (
        ["city_hub"] * len(city_site_positions)
        + ["service_area"] * len(service_positions)
        + ["highway_exit"] * len(exit_positions)
    )
    connector_prior_values = _prior_values(priors, "connector_count")
    power_prior_values = _prior_values(priors, "max_power_kw")
    availability_prior_values = _prior_values(priors, "availability_ratio")
    site_type_profiles = corridor_cfg.get("site_type_profiles", {})
    site_connector_proxy = np.zeros(len(site_positions), dtype=np.float32)
    site_power_proxy = np.zeros(len(site_positions), dtype=np.float32)
    site_availability_proxy = np.zeros(len(site_positions), dtype=np.float32)
    site_coords = np.zeros((num_sites, 2), dtype=np.float32)
    site_detours = np.zeros(num_sites, dtype=np.float32)
    site_labels: list[str] = []
    type_counts_seen = {"city_hub": 0, "service_area": 0, "highway_exit": 0}
    for index, site_type in enumerate(site_types):
        profile = dict(site_type_profiles.get(site_type, {}))
        site_connector_proxy[index] = float(
            _sample_site_profile_value(
                rng,
                profile,
                "connector",
                lambda: _sample_prior_values(rng, connector_prior_values, 1, 2.0, 10.0)[0],
            )
        )
        site_power_proxy[index] = float(
            _sample_site_profile_value(
                rng,
                profile,
                "power_kw",
                lambda: _sample_prior_values(rng, power_prior_values, 1, 22.0, 150.0)[0],
            )
        )
        site_availability_proxy[index] = float(
            _sample_site_profile_value(
                rng,
                profile,
                "availability",
                lambda: _sample_prior_values(rng, availability_prior_values, 1, 0.35, 0.85)[0],
            )
        )
    connector_median = float(np.median(site_connector_proxy)) if site_connector_proxy.size else 1.0
    site_capacity_scale = np.clip(site_connector_proxy / max(connector_median, 1.0), 0.5, 2.5).astype(np.float32)
    for index, (position, site_type) in enumerate(zip(site_positions, site_types, strict=True)):
        connector_scale = float(site_connector_proxy[index] / max(connector_median, 1.0))
        if site_type == "city_hub":
            y = rng.choice((-1.0, 1.0)) * rng.uniform(8.0, 15.0)
            detour = rng.uniform(7.0, 12.0)
            city_strength[index % len(city_strength)] *= float(np.clip(0.85 + 0.25 * connector_scale, 0.75, 1.45))
        elif site_type == "service_area":
            y = rng.choice((-1.0, 1.0)) * rng.uniform(0.4, 1.5)
            detour = rng.uniform(1.5, 3.5)
        else:
            y = rng.choice((-1.0, 1.0)) * rng.uniform(2.5, 6.0)
            detour = rng.uniform(4.0, 7.5)
        site_coords[index] = (position, y)
        site_detours[index] = detour
        current_count = type_counts_seen[site_type]
        site_labels.append(_site_label(site_type, current_count))
        type_counts_seen[site_type] = current_count + 1

    zone_mix = corridor_cfg.get(
        "zone_mix",
        {
            "city": 0.42,
            "suburb": 0.24,
            "corridor": 0.22,
            "logistics": 0.12,
        },
    )
    zone_counts = _allocate_counts(num_zones, zone_mix, minimum_if_possible=1 if num_zones >= 4 else 0)
    zone_positions: list[float] = []
    zone_types: list[str] = []
    zone_base_demand: list[float] = []
    zone_scale: list[float] = []
    zone_coords: list[tuple[float, float]] = []
    zone_labels: list[str] = []
    zone_detours: list[float] = []
    zone_type_counts_seen = {key: 0 for key in zone_counts}

    def closest_city_strength(position: float) -> float:
        city_index = int(np.argmin(np.abs(city_positions - position)))
        return float(city_strength[city_index])

    for zone_type, count in zone_counts.items():
        for _ in range(count):
            if zone_type == "city":
                anchor = float(rng.choice(city_positions))
                position = float(np.clip(anchor + rng.normal(0.0, 10.0), 0.02 * corridor_length_km, 0.98 * corridor_length_km))
                y = rng.choice((-1.0, 1.0)) * rng.uniform(10.0, 18.0)
                base = rng.uniform(6.5, 10.5) * closest_city_strength(position)
                scale = rng.normal(0.08, float(demand_config["zone_scale_std"]))
                detour = rng.uniform(6.0, 10.0)
            elif zone_type == "suburb":
                anchor = float(rng.choice(city_positions))
                position = float(np.clip(anchor + rng.normal(0.0, 18.0), 0.02 * corridor_length_km, 0.98 * corridor_length_km))
                y = rng.choice((-1.0, 1.0)) * rng.uniform(5.0, 11.0)
                base = rng.uniform(3.5, 6.5) * closest_city_strength(position)
                scale = rng.normal(0.0, float(demand_config["zone_scale_std"]))
                detour = rng.uniform(4.0, 7.0)
            elif zone_type == "logistics":
                position = float(rng.uniform(0.2 * corridor_length_km, 0.85 * corridor_length_km))
                y = rng.choice((-1.0, 1.0)) * rng.uniform(1.0, 3.5)
                base = rng.uniform(4.0, 7.0)
                scale = rng.normal(0.12, float(demand_config["zone_scale_std"]))
                detour = rng.uniform(2.0, 4.5)
            else:
                position = float(rng.uniform(0.03 * corridor_length_km, 0.97 * corridor_length_km))
                y = rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 2.0)
                base = rng.uniform(float(demand_config["base_rate_min"]), float(demand_config["base_rate_max"]) * 0.6)
                scale = rng.normal(-0.05, float(demand_config["zone_scale_std"]))
                detour = rng.uniform(1.0, 2.5)

            zone_positions.append(position)
            zone_types.append(zone_type)
            zone_base_demand.append(base)
            zone_scale.append(scale)
            zone_coords.append((position, y))
            zone_detours.append(detour)
            current_count = zone_type_counts_seen[zone_type]
            zone_labels.append(_zone_label(zone_type, current_count))
            zone_type_counts_seen[zone_type] = current_count + 1

    zone_positions_arr = np.asarray(zone_positions, dtype=np.float32)
    zone_coords_arr = np.asarray(zone_coords, dtype=np.float32)
    zone_base_demand_arr = np.asarray(zone_base_demand, dtype=np.float32)
    zone_scale_arr = np.asarray(zone_scale, dtype=np.float32)
    zone_detours_arr = np.asarray(zone_detours, dtype=np.float32)

    segment_breaks, segment_speeds = _build_segment_speeds(
        corridor_length_km=corridor_length_km,
        city_positions_km=city_positions,
        mainline_speed_kmh=mainline_speed_kmh,
        urban_speed_kmh=urban_speed_kmh,
        urban_influence_km=urban_influence_km,
    )

    travel_time_matrix = np.zeros((num_zones, num_sites), dtype=np.float32)
    for zone_idx, zone_position in enumerate(zone_positions_arr):
        for site_idx, site_position in enumerate(site_positions):
            mainline_minutes = _segment_travel_minutes(
                start_km=float(zone_position),
                end_km=float(site_position),
                segment_breaks=segment_breaks,
                segment_speeds=segment_speeds,
            )
            travel_time_matrix[zone_idx, site_idx] = (
                mainline_minutes + float(zone_detours_arr[zone_idx]) + float(site_detours[site_idx])
            )

    site_to_site_travel = np.zeros((num_sites, num_sites), dtype=np.float32)
    for start_idx, start_position in enumerate(site_positions):
        for end_idx, end_position in enumerate(site_positions):
            mainline_minutes = _segment_travel_minutes(
                start_km=float(start_position),
                end_km=float(end_position),
                segment_breaks=segment_breaks,
                segment_speeds=segment_speeds,
            )
            site_to_site_travel[start_idx, end_idx] = (
                mainline_minutes + float(site_detours[start_idx]) + float(site_detours[end_idx])
            )

    return SyntheticCity(
        site_coords=site_coords.astype(np.float32),
        zone_coords=zone_coords_arr.astype(np.float32),
        zone_base_demand=zone_base_demand_arr.astype(np.float32),
        zone_scale=zone_scale_arr.astype(np.float32),
        travel_time_matrix=travel_time_matrix.astype(np.float32),
        layout_type="corridor",
        site_labels=tuple(site_labels),
        site_types=tuple(site_types),
        zone_labels=tuple(zone_labels),
        zone_types=tuple(zone_types),
        corridor_polyline=np.asarray([[0.0, 0.0], [corridor_length_km, 0.0]], dtype=np.float32),
        site_positions_km=site_positions.astype(np.float32),
        zone_positions_km=zone_positions_arr.astype(np.float32),
        site_travel_time_matrix=site_to_site_travel.astype(np.float32),
        metadata={
            "corridor_length_km": corridor_length_km,
            "city_positions_km": city_positions.astype(np.float32).tolist(),
            "city_strength": city_strength.astype(np.float32).tolist(),
            "segment_breaks_km": segment_breaks.astype(np.float32).tolist(),
            "segment_speeds_kmh": segment_speeds.astype(np.float32).tolist(),
            "calibration_source": priors.get("source"),
            "calibration_location_count": priors.get("location_count"),
            "site_connector_proxy": site_connector_proxy.astype(np.float32).tolist(),
            "site_power_proxy_kw": site_power_proxy.astype(np.float32).tolist(),
            "site_availability_proxy": np.clip(site_availability_proxy.astype(np.float32), 0.15, 1.0).tolist(),
            "site_capacity_scale": site_capacity_scale.tolist(),
        },
    )


def make_synthetic_layout(env_config: dict[str, Any], demand_config: dict[str, Any], seed: int) -> SyntheticCity:
    layout_type = str(env_config.get("layout", "urban_grid")).lower()
    if layout_type == "corridor":
        return make_synthetic_corridor(env_config=env_config, demand_config=demand_config, seed=seed)
    return make_synthetic_city(
        num_sites=int(env_config["num_candidate_sites"]),
        num_zones=int(env_config["num_demand_zones"]),
        city_extent_km=float(env_config["city_extent_km"]),
        seed=seed,
        base_rate_min=float(demand_config["base_rate_min"]),
        base_rate_max=float(demand_config["base_rate_max"]),
        zone_scale_std=float(demand_config["zone_scale_std"]),
    )


def static_site_accessibility_scores(city: SyntheticCity, decay: float) -> np.ndarray:
    weights = np.exp(-decay * city.travel_time_matrix)
    return weights.T @ city.zone_base_demand

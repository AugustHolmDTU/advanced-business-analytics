from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from evch.config.loader import build_config_parser, load_config
from evch.data.city import build_city
from evch.data.tomtom import (
    TomTomApiError,
    build_station_records,
    fetch_charging_availability,
    geocode_location,
    route_summary_between_points,
    search_nearby_ev_stations,
    summarize_availability_payload,
    resolve_api_key,
)
from evch.train.generate_synthetic_data import render_synthetic_layout_svg
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CorridorAnchor:
    label: str
    query: str
    site_type: str
    search_radius_m: int
    result_limit: int
    route_position_km: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    freeform_address: str = ""


def _default_corridor_definition() -> dict[str, Any]:
    return {
        "name": "denmark_hybrid_corridor",
        "timezone": "Europe/Copenhagen",
        "country_set": "DK",
        "language": "en-GB",
        "candidate_site_count": 18,
        "demand_zone_count": 10,
        "snapshot_rounds": 3,
        "snapshot_interval_seconds": 0,
        "append_history": True,
        "use_cached_calibration_on_failure": True,
        "cached_calibration_root": "outputs/tomtom_calibration/calibration",
        "routing": {
            "travel_mode": "car",
            "route_type": "fastest",
            "traffic": "historical",
            "depart_at": "any",
        },
        "site_mix": {
            "city_hub": 0.22,
            "service_area": 0.40,
            "highway_exit": 0.38,
        },
        "anchors": [
            {"label": "Copenhagen", "query": "Copenhagen, Denmark", "site_type": "city_hub", "search_radius_m": 7000, "result_limit": 20},
            {"label": "Karlslunde", "query": "Karlslunde, Denmark", "site_type": "service_area", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Køge", "query": "Køge, Denmark", "site_type": "highway_exit", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Sorø", "query": "Sorø, Denmark", "site_type": "service_area", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Slagelse", "query": "Slagelse, Denmark", "site_type": "highway_exit", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Korsør", "query": "Korsør, Denmark", "site_type": "service_area", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Nyborg", "query": "Nyborg, Denmark", "site_type": "highway_exit", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Odense", "query": "Odense, Denmark", "site_type": "city_hub", "search_radius_m": 7000, "result_limit": 20},
            {"label": "Ejby, Funen", "query": "Ejby, Denmark", "site_type": "service_area", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Middelfart", "query": "Middelfart, Denmark", "site_type": "highway_exit", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Fredericia", "query": "Fredericia, Denmark", "site_type": "highway_exit", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Hylkedal", "query": "Hylkedal, Denmark", "site_type": "service_area", "search_radius_m": 5000, "result_limit": 16},
            {"label": "Kolding", "query": "Kolding, Denmark", "site_type": "city_hub", "search_radius_m": 7000, "result_limit": 20},
        ],
    }


def _make_anchors(config: dict[str, Any]) -> list[CorridorAnchor]:
    anchors: list[CorridorAnchor] = []
    for raw in config["anchors"]:
        anchors.append(
            CorridorAnchor(
                label=str(raw["label"]),
                query=str(raw["query"]),
                site_type=str(raw["site_type"]),
                search_radius_m=int(raw["search_radius_m"]),
                result_limit=int(raw["result_limit"]),
            )
        )
    return anchors


def _distribution(values: list[float], *, quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, Any]:
    if not values:
        return {"count": 0, "values": []}
    array = np.asarray(values, dtype=np.float32)
    payload = {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "values": [float(value) for value in array.tolist()],
    }
    for quantile in quantiles:
        label = f"p{int(round(quantile * 100)):02d}"
        payload[label] = float(np.quantile(array, quantile))
    return payload


def _nearest_anchor_type(lat: float, lon: float, anchors: list[CorridorAnchor]) -> str:
    best_type = anchors[0].site_type
    best_distance = float("inf")
    for anchor in anchors:
        distance = math.hypot(lat - anchor.lat, lon - anchor.lon)
        if distance < best_distance:
            best_distance = distance
            best_type = anchor.site_type
    return best_type


def _anchor_positions_by_type(anchors: list[CorridorAnchor]) -> dict[str, list[float]]:
    grouped = {"city_hub": [], "service_area": [], "highway_exit": []}
    for anchor in anchors:
        grouped[anchor.site_type].append(float(anchor.route_position_km))
    return grouped


def _build_site_type_profiles(history: pd.DataFrame) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if history.empty:
        return profiles
    grouped = history.groupby("site_type")
    for site_type, frame in grouped:
        connectors = frame["total_connectors"].dropna().tolist()
        power = frame["max_power_kw"].dropna().tolist()
        availability = frame["availability_ratio"].dropna().tolist()
        connector_dist = _distribution([float(value) for value in connectors], quantiles=(0.25, 0.5, 0.9))
        power_dist = _distribution([float(value) for value in power], quantiles=(0.25, 0.5, 0.9))
        availability_dist = _distribution([float(value) for value in availability], quantiles=(0.2, 0.5, 0.8))
        if connector_dist["count"] == 0:
            continue
        profiles[str(site_type)] = {
            "connector_range": [
                int(max(1, round(connector_dist.get("p25", connector_dist["min"])))),
                int(max(1, round(connector_dist.get("p90", connector_dist["max"])))),
            ],
            "power_kw_range": [
                float(power_dist.get("p25", power_dist.get("min", 50.0))),
                float(power_dist.get("p90", power_dist.get("max", 150.0))),
            ],
            "availability_mean": float(availability_dist.get("p50", availability_dist.get("mean", 0.55))),
            "availability_std": float(
                max(0.04, availability_dist.get("p80", availability_dist.get("max", 0.65)) - availability_dist.get("p20", availability_dist.get("min", 0.45)))
                / 2.0
            ),
        }
    return profiles


def _build_demand_profile(history: pd.DataFrame) -> dict[str, float]:
    defaults = {
        "morning_peak_hour": 8.0,
        "evening_peak_hour": 17.0,
        "peak_width": 2.4,
        "morning_peak_weight": 0.75,
        "evening_peak_weight": 1.0,
        "background_intensity": 0.35,
        "weekday_multiplier": 1.0,
        "weekend_multiplier": 0.82,
    }
    if history.empty or history["snapshot_hour"].nunique() < 3:
        return defaults

    hourly = history.groupby("snapshot_hour")["utilization_ratio"].mean().reindex(range(24), fill_value=np.nan)
    observed = hourly.dropna()
    if observed.empty:
        return defaults

    min_value = float(observed.min())
    max_value = float(observed.max())
    span = max(max_value - min_value, 1e-6)
    normalized = ((hourly - min_value) / span).fillna(0.0)

    morning_window = normalized.loc[5:11]
    evening_window = normalized.loc[14:21]
    morning_peak_hour = float(morning_window.idxmax()) if not morning_window.empty else defaults["morning_peak_hour"]
    evening_peak_hour = float(evening_window.idxmax()) if not evening_window.empty else defaults["evening_peak_hour"]
    background = float(np.clip(observed.quantile(0.15), 0.2, 0.7))
    morning_peak_weight = float(np.clip(normalized.loc[int(morning_peak_hour)] - background, 0.2, 1.4))
    evening_peak_weight = float(np.clip(normalized.loc[int(evening_peak_hour)] - background, 0.2, 1.6))

    weekday = history.loc[history["is_weekend"] == 0, "utilization_ratio"]
    weekend = history.loc[history["is_weekend"] == 1, "utilization_ratio"]
    weekday_multiplier = 1.0
    weekend_multiplier = defaults["weekend_multiplier"]
    if not weekday.empty and not weekend.empty and float(weekday.mean()) > 0.0:
        weekend_multiplier = float(np.clip(float(weekend.mean()) / float(weekday.mean()), 0.65, 1.05))

    return {
        "morning_peak_hour": morning_peak_hour,
        "evening_peak_hour": evening_peak_hour,
        "peak_width": defaults["peak_width"],
        "morning_peak_weight": morning_peak_weight,
        "evening_peak_weight": evening_peak_weight,
        "background_intensity": background,
        "weekday_multiplier": weekday_multiplier,
        "weekend_multiplier": weekend_multiplier,
    }


def _collect_station_history(
    stations: pd.DataFrame,
    *,
    api_key: str,
    timezone_name: str,
    snapshot_rounds: int,
    snapshot_interval_seconds: int,
) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame()
    timezone = ZoneInfo(timezone_name)
    records: list[dict[str, Any]] = []
    unique_station_rows = stations.drop_duplicates(subset=["charging_availability_id", "lat", "lon"])
    for round_idx in range(snapshot_rounds):
        now = datetime.now(timezone)
        for station in unique_station_rows.itertuples(index=False):
            availability_id = getattr(station, "charging_availability_id")
            if not availability_id:
                continue
            try:
                payload = fetch_charging_availability(str(availability_id), api_key=api_key)
                summary = summarize_availability_payload(payload)
            except TomTomApiError as exc:
                LOGGER.warning("Skipping availability refresh for %s because TomTom failed: %s", getattr(station, "name"), exc)
                continue
            total_connectors = int(summary["total_connectors"])
            available_connectors = int(summary["available_connectors"])
            utilization = 0.0 if total_connectors <= 0 else 1.0 - (available_connectors / float(total_connectors))
            records.append(
                {
                    "snapshot_time": now.isoformat(),
                    "snapshot_hour": int(now.hour),
                    "day_of_week": int(now.weekday()),
                    "is_weekend": int(now.weekday() >= 5),
                    "round_index": round_idx,
                    "site_type": getattr(station, "site_type"),
                    "station_name": getattr(station, "name"),
                    "charging_availability_id": availability_id,
                    "lat": float(getattr(station, "lat")),
                    "lon": float(getattr(station, "lon")),
                    "total_connectors": total_connectors,
                    "available_connectors": available_connectors,
                    "occupied_connectors": int(summary["occupied_connectors"]),
                    "max_power_kw": summary["max_power_kw"],
                    "availability_ratio": 0.0 if total_connectors <= 0 else available_connectors / float(total_connectors),
                    "utilization_ratio": utilization,
                }
            )
        if round_idx + 1 < snapshot_rounds and snapshot_interval_seconds > 0:
            time.sleep(snapshot_interval_seconds)
    return pd.DataFrame.from_records(records)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _apply_cached_anchor_defaults(anchors: list[CorridorAnchor]) -> tuple[list[dict[str, Any]], float]:
    cached_positions = {
        "Copenhagen": 0.0,
        "Karlslunde": 22.0,
        "Køge": 36.0,
        "Sorø": 73.0,
        "Slagelse": 97.0,
        "Korsør": 118.0,
        "Nyborg": 144.0,
        "Odense": 177.0,
        "Ejby, Funen": 205.0,
        "Middelfart": 224.0,
        "Fredericia": 233.0,
        "Hylkedal": 255.0,
        "Kolding": 275.0,
    }
    route_segments: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor.route_position_km = float(cached_positions.get(anchor.label, 0.0))
    for start, end in zip(anchors[:-1], anchors[1:]):
        route_segments.append(
            {
                "origin": start.label,
                "destination": end.label,
                "route_length_km": float(max(end.route_position_km - start.route_position_km, 0.0)),
                "travel_time_minutes": float(max(end.route_position_km - start.route_position_km, 0.0) / 88.0 * 60.0),
            }
        )
    return route_segments, float(max(anchor.route_position_km for anchor in anchors))


def _load_cached_station_pool(cached_root: str | Path) -> pd.DataFrame:
    source_map = {
        "roskilde_denmark": "city_hub",
        "randers_denmark": "city_hub",
        "k_ge_denmark": "highway_exit",
        "vejle_denmark": "service_area",
    }
    frames: list[pd.DataFrame] = []
    for folder_name, site_type in source_map.items():
        csv_path = Path(cached_root) / folder_name / "stations.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame["site_type"] = site_type
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No cached station calibration files found under {cached_root}")
    station_frame = pd.concat(frames, ignore_index=True)
    station_frame = station_frame.drop_duplicates(subset=["charging_availability_id", "lat", "lon"]).reset_index(drop=True)
    return station_frame


def _build_generated_environment(
    corridor_cfg: dict[str, Any],
    route_length_km: float,
    positions_by_type: dict[str, list[float]],
    site_type_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    city_hubs = positions_by_type["city_hub"]
    return {
        "seed": 17,
        "environment": {
            "layout": "corridor",
            "num_candidate_sites": int(corridor_cfg["candidate_site_count"]),
            "num_demand_zones": int(corridor_cfg["demand_zone_count"]),
            "max_chargers": 4,
            "charger_capacity": 10.0,
            "horizon": 24,
            "max_steps": 24,
            "service_decay": 0.055,
            "deployment_cost": 0.12,
            "relocation_cost": 1.25,
            "unmet_penalty": 1.65,
            "outage_penalty": 0.22,
            "reward_scale": 0.08,
            "utilization_bonus": 1.3,
            "coverage_bonus": 0.10,
            "invalid_action_penalty": 1.0,
            "city_extent_km": float(route_length_km),
            "corridor_length_km": float(route_length_km),
            "randomize_on_reset": True,
            "reset_seed_stride": 101,
            "start_filled": True,
            "corridor": {
                "num_cities": len(city_hubs),
                "urban_influence_km": 16.0,
                "city_anchor_positions_km": [float(value) for value in city_hubs],
                "service_area_positions_km": [float(value) for value in positions_by_type["service_area"]],
                "exit_positions_km": [float(value) for value in positions_by_type["highway_exit"]],
                "site_type_counts": {
                    "city_hub": len(city_hubs),
                    "service_area": len(positions_by_type["service_area"]),
                    "highway_exit": len(positions_by_type["highway_exit"]),
                },
                "site_mix": dict(corridor_cfg["site_mix"]),
                "site_type_profiles": site_type_profiles,
                "zone_mix": {
                    "city": 0.42,
                    "suburb": 0.24,
                    "corridor": 0.22,
                    "logistics": 0.12,
                },
            },
            "disruption": {
                "demand_spike_probability": 0.14,
                "demand_spike_multiplier": 1.7,
                "outage_probability": 0.06,
                "observation_noise_scale": 0.30,
                "travel_slowdown_probability": 0.18,
                "travel_slowdown_multiplier": 1.45,
                "travel_slowdown_fraction": 0.18,
                "road_closure_probability": 0.05,
                "road_closure_fraction": 0.08,
                "road_closure_penalty_minutes": 20.0,
            },
        },
    }


def _build_generated_demand(profile: dict[str, float]) -> dict[str, Any]:
    return {
        "demand": {
            "base_rate_min": 3.2,
            "base_rate_max": 8.5,
            "zone_scale_std": 0.22,
            "morning_peak_hour": float(profile["morning_peak_hour"]),
            "evening_peak_hour": float(profile["evening_peak_hour"]),
            "peak_width": float(profile["peak_width"]),
            "morning_peak_weight": float(profile["morning_peak_weight"]),
            "evening_peak_weight": float(profile["evening_peak_weight"]),
            "weekday_multiplier": float(profile["weekday_multiplier"]),
            "weekend_multiplier": float(profile["weekend_multiplier"]),
            "background_intensity": float(profile["background_intensity"]),
            "poisson_clip": 60.0,
            "observation_noise_std": 0.5,
        },
        "data": {
            "num_days": 45,
            "val_split": 0.2,
            "include_prev_observation": True,
        },
    }


def main() -> None:
    parser = build_config_parser("Build a Denmark-based hybrid corridor simulation calibrated with TomTom data.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    experiment_cfg = config["experiment"]
    corridor_cfg = {**_default_corridor_definition(), **config.get("denmark_hybrid_corridor", {})}
    api_key = resolve_api_key(corridor_cfg.get("api_key"))
    anchors = _make_anchors(corridor_cfg)

    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"])
    generated_dir = ensure_dir(output_dir / "generated")
    raw_dir = ensure_dir(output_dir / "raw")
    calibration_mode = "live_tomtom"
    fallback_reason = ""
    try:
        for anchor in anchors:
            geocode_result = geocode_location(
                query=anchor.query,
                api_key=api_key,
                limit=1,
                country_set=corridor_cfg.get("country_set"),
                language=corridor_cfg.get("language"),
            )
            position = geocode_result.get("position") or {}
            anchor.lat = float(position["lat"])
            anchor.lon = float(position["lon"])
            anchor.freeform_address = str((geocode_result.get("address") or {}).get("freeformAddress") or "")

        route_segments = []
        cumulative_km = 0.0
        anchors[0].route_position_km = 0.0
        for start, end in zip(anchors[:-1], anchors[1:]):
            summary = route_summary_between_points(
                start.lat,
                start.lon,
                end.lat,
                end.lon,
                api_key=api_key,
                travel_mode=str(corridor_cfg["routing"]["travel_mode"]),
                route_type=str(corridor_cfg["routing"]["route_type"]),
                traffic=str(corridor_cfg["routing"]["traffic"]),
                depart_at=str(corridor_cfg["routing"]["depart_at"]),
            )
            cumulative_km += float(summary["route_length_km"])
            end.route_position_km = cumulative_km
            route_segments.append(
                {
                    "origin": start.label,
                    "destination": end.label,
                    "route_length_km": float(summary["route_length_km"]),
                    "travel_time_minutes": float(summary["travel_time_minutes"]),
                }
            )

        deduped: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            search_results = search_nearby_ev_stations(
                lat=anchor.lat,
                lon=anchor.lon,
                radius_m=anchor.search_radius_m,
                api_key=api_key,
                limit=anchor.result_limit,
                language=corridor_cfg.get("language"),
            )
            for station in build_station_records(search_results, api_key=api_key):
                dedupe_key = str(station.charging_availability_id or f"{station.lat:.6f}:{station.lon:.6f}")
                site_type = _nearest_anchor_type(station.lat, station.lon, anchors)
                deduped[dedupe_key] = {
                    "name": station.name,
                    "address": station.address,
                    "lat": station.lat,
                    "lon": station.lon,
                    "charging_availability_id": station.charging_availability_id,
                    "total_connectors": station.total_connectors,
                    "available_connectors": station.available_connectors,
                    "occupied_connectors": station.occupied_connectors,
                    "reserved_connectors": station.reserved_connectors,
                    "unknown_connectors": station.unknown_connectors,
                    "out_of_service_connectors": station.out_of_service_connectors,
                    "max_power_kw": station.max_power_kw,
                    "site_type": site_type,
                }

        if not deduped:
            raise TomTomApiError("No EV charging stations were found around the configured Denmark corridor anchors.")
        station_frame = pd.DataFrame.from_records(deduped.values()).sort_values(["site_type", "name"]).reset_index(drop=True)
        history_frame = _collect_station_history(
            station_frame,
            api_key=api_key,
            timezone_name=str(corridor_cfg["timezone"]),
            snapshot_rounds=int(corridor_cfg["snapshot_rounds"]),
            snapshot_interval_seconds=int(corridor_cfg["snapshot_interval_seconds"]),
        )
    except (TomTomApiError, FileNotFoundError) as exc:
        if not bool(corridor_cfg.get("use_cached_calibration_on_failure", True)):
            raise
        calibration_mode = "cached_denmark_calibration"
        fallback_reason = str(exc)
        LOGGER.warning("Falling back to cached Denmark calibration because live TomTom fetch failed: %s", exc)
        route_segments, cumulative_km = _apply_cached_anchor_defaults(anchors)
        station_frame = _load_cached_station_pool(corridor_cfg["cached_calibration_root"])
        history_frame = pd.DataFrame()

    station_frame.to_csv(raw_dir / "station_pool.csv", index=False)
    history_path = raw_dir / "snapshot_history.csv"
    if bool(corridor_cfg.get("append_history", True)) and history_path.exists():
        existing_history = pd.read_csv(history_path)
        history_frame = pd.concat([existing_history, history_frame], ignore_index=True)
    if not history_frame.empty:
        history_frame = history_frame.sort_values("snapshot_time").reset_index(drop=True)
        history_frame.to_csv(history_path, index=False)

    positions_by_type = _anchor_positions_by_type(anchors)
    site_type_profiles = _build_site_type_profiles(history_frame if not history_frame.empty else station_frame.assign(
        availability_ratio=lambda frame: frame["available_connectors"] / np.maximum(frame["total_connectors"], 1),
        utilization_ratio=lambda frame: 1.0 - (frame["available_connectors"] / np.maximum(frame["total_connectors"], 1)),
        snapshot_hour=12,
        is_weekend=0,
    ))
    demand_profile = _build_demand_profile(history_frame)

    generated_env = _build_generated_environment(corridor_cfg, cumulative_km, positions_by_type, site_type_profiles)
    generated_demand = _build_generated_demand(demand_profile)
    env_path = generated_dir / "environment.yaml"
    demand_path = generated_dir / "demand.yaml"
    _write_yaml(env_path, generated_env)
    _write_yaml(demand_path, generated_demand)

    preview_city = build_city(generated_env["environment"], generated_demand["demand"], seed=int(generated_env["seed"]))
    render_synthetic_layout_svg(preview_city, generated_dir / "synthetic_layout.svg")
    pd.DataFrame(
        {
            "site_id": list(range(len(preview_city.site_coords))),
            "label": list(preview_city.site_labels),
            "type": list(preview_city.site_types),
            "x_km": preview_city.site_coords[:, 0],
            "y_km": preview_city.site_coords[:, 1],
            "connector_proxy": preview_city.metadata.get("site_connector_proxy", []),
            "capacity_scale": preview_city.metadata.get("site_capacity_scale", []),
            "availability_proxy": preview_city.metadata.get("site_availability_proxy", []),
        }
    ).to_csv(generated_dir / "synthetic_sites.csv", index=False)
    pd.DataFrame(
        {
            "zone_id": list(range(len(preview_city.zone_coords))),
            "label": list(preview_city.zone_labels),
            "type": list(preview_city.zone_types),
            "x_km": preview_city.zone_coords[:, 0],
            "y_km": preview_city.zone_coords[:, 1],
            "base_demand": preview_city.zone_base_demand,
        }
    ).to_csv(generated_dir / "synthetic_zones.csv", index=False)

    calibration_summary = {
        "corridor_length_km": float(cumulative_km),
        "anchor_count": len(anchors),
        "candidate_site_count": int(corridor_cfg["candidate_site_count"]),
        "demand_zone_count": int(corridor_cfg["demand_zone_count"]),
        "calibration_mode": calibration_mode,
        "fallback_reason": fallback_reason,
        "anchors": [
            {
                "label": anchor.label,
                "query": anchor.query,
                "site_type": anchor.site_type,
                "route_position_km": float(anchor.route_position_km),
                "lat": float(anchor.lat),
                "lon": float(anchor.lon),
                "address": anchor.freeform_address,
            }
            for anchor in anchors
        ],
        "route_segments": route_segments,
        "site_type_profiles": site_type_profiles,
        "demand_profile": demand_profile,
        "report_language": (
            "Because a fully realistic reconstruction of a specific charging corridor would require detailed and mostly "
            "unavailable demand, behavior, and disruption data, we adopt a stylized corridor simulation. The environment "
            "is synthetic in layout, but key structural parameters, including charger capacity ranges, travel times, and "
            "baseline utilization patterns, are calibrated using real-world API data. This produces a hybrid digital twin "
            "that is simple enough for controlled experimentation while still grounded in realistic infrastructure behavior."
        ),
        "artifacts": {
            "environment_config": str(env_path),
            "demand_config": str(demand_path),
            "layout_svg": str(generated_dir / "synthetic_layout.svg"),
            "station_pool_csv": str(raw_dir / "station_pool.csv"),
            "snapshot_history_csv": str(history_path),
        },
    }
    write_json(output_dir / "calibration_summary.json", calibration_summary)
    LOGGER.info("Saved Denmark hybrid corridor package to %s", output_dir)


if __name__ == "__main__":
    main()

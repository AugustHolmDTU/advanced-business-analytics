from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from evch.config.loader import build_config_parser, load_config
from evch.data.tomtom import (
    _fallback_travel_minutes,
    TomTomApiError,
    build_snapshot_payload,
    build_station_records,
    build_zone_grid,
    geocode_location,
    latlon_to_local_xy,
    request_matrix_travel_times,
    resolve_api_key,
    search_nearby_ev_stations,
    snapshot_to_frame,
    write_station_snapshot_svg,
)
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "location"


def _nearest_neighbor_distances(points_xy: np.ndarray) -> list[float]:
    if len(points_xy) < 2:
        return []
    distances: list[float] = []
    for idx, point in enumerate(points_xy):
        deltas = points_xy - point[None, :]
        norms = np.sqrt(np.sum(deltas * deltas, axis=1))
        norms[idx] = np.inf
        distances.append(float(np.min(norms)))
    return distances


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "values": []}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "values": [float(value) for value in array.tolist()],
    }


def _recommended_defaults(distributions: dict[str, dict[str, Any]]) -> dict[str, float]:
    speed = distributions.get("implied_speed_kmh", {})
    availability = distributions.get("availability_ratio", {})
    neighbor = distributions.get("nearest_neighbor_km", {})
    connector = distributions.get("connector_count", {})
    mainline_speed = float(np.clip(speed.get("p75", 105.0), 75.0, 125.0))
    urban_speed = float(np.clip(speed.get("median", 72.0) * 0.72, 35.0, 90.0))
    service_gap = float(np.clip(neighbor.get("p90", 18.0) * 4.0, 14.0, 30.0))
    city_hub_scale = float(np.clip(1.0 + connector.get("p90", 10.0) / max(connector.get("median", 6.0), 1.0) * 0.08, 1.0, 1.5))
    expected_availability = float(np.clip(availability.get("mean", 0.55), 0.2, 0.95))
    return {
        "mainline_speed_kmh": mainline_speed,
        "urban_speed_kmh": urban_speed,
        "service_area_min_gap_km": service_gap,
        "city_hub_strength_scale": city_hub_scale,
        "expected_availability_ratio": expected_availability,
    }


def main() -> None:
    parser = build_config_parser("Build TomTom-grounded calibration priors for the synthetic corridor.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    calibration_cfg = config["tomtom_calibration"]
    api_key = resolve_api_key(calibration_cfg.get("api_key"))
    locations = list(calibration_cfg["locations"])
    radius_m = int(calibration_cfg.get("radius_m", 6000))
    result_limit = int(calibration_cfg.get("result_limit", 30))
    zone_probe_count = int(calibration_cfg.get("zone_probe_count", 12))
    language = calibration_cfg.get("language")
    country_set = calibration_cfg.get("country_set")
    experiment_cfg = config["experiment"]

    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "calibration")

    connector_values: list[float] = []
    power_values: list[float] = []
    availability_values: list[float] = []
    nearest_neighbor_values: list[float] = []
    speed_values: list[float] = []
    location_summaries: list[dict[str, Any]] = []

    for location_query in locations:
        geocode_result = geocode_location(
            query=str(location_query),
            api_key=api_key,
            limit=1,
            country_set=country_set,
            language=language,
        )
        center_lat = float((geocode_result.get("position") or {})["lat"])
        center_lon = float((geocode_result.get("position") or {})["lon"])
        nearby_results = search_nearby_ev_stations(
            lat=center_lat,
            lon=center_lon,
            radius_m=radius_m,
            api_key=api_key,
            limit=result_limit,
            language=language,
        )
        stations = build_station_records(nearby_results, api_key=api_key)
        if not stations:
            LOGGER.warning("Skipping %s because no charging stations were returned.", location_query)
            continue

        zone_coords_km, zone_latlon = build_zone_grid(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=radius_m,
            num_zones=zone_probe_count,
        )
        station_latlon = np.asarray([[station.lat, station.lon] for station in stations], dtype=np.float32)
        try:
            travel_time_matrix = request_matrix_travel_times(
                origins=zone_latlon,
                destinations=station_latlon,
                api_key=api_key,
                travel_mode=str(calibration_cfg.get("travel_mode", "car")),
                route_type=str(calibration_cfg.get("route_type", "fastest")),
                traffic=str(calibration_cfg.get("traffic", "historical")),
                depart_at=str(calibration_cfg.get("depart_at", "any")),
            )
            if np.isnan(travel_time_matrix).any():
                fallback = _fallback_travel_minutes(zone_latlon, station_latlon)
                travel_time_matrix = np.where(np.isnan(travel_time_matrix), fallback, travel_time_matrix).astype(np.float32)
        except TomTomApiError as exc:
            LOGGER.warning("Routing failed for %s, using geometric fallback: %s", location_query, exc)
            travel_time_matrix = _fallback_travel_minutes(zone_latlon, station_latlon)

        snapshot_payload = build_snapshot_payload(
            location_query=str(location_query),
            geocode_result=geocode_result,
            stations=stations,
            zone_coords_km=zone_coords_km,
            zone_latlon=zone_latlon,
            travel_time_matrix_minutes=travel_time_matrix,
            radius_m=radius_m,
        )
        location_slug = _slugify(str(location_query))
        location_dir = ensure_dir(output_dir / location_slug)
        write_json(location_dir / "snapshot.json", snapshot_payload)
        snapshot_to_frame(snapshot_payload).to_csv(location_dir / "stations.csv", index=False)
        write_station_snapshot_svg(snapshot_payload, location_dir / "station_map.svg", title=str(location_query))

        site_coords_xy = np.asarray(
            [latlon_to_local_xy(station.lat, station.lon, center_lat=center_lat, center_lon=center_lon) for station in stations],
            dtype=np.float32,
        )
        connector_values.extend(float(station.total_connectors) for station in stations if station.total_connectors > 0)
        power_values.extend(float(station.max_power_kw) for station in stations if station.max_power_kw is not None)
        availability_values.extend(
            float(station.available_connectors / max(station.total_connectors, 1))
            for station in stations
            if station.total_connectors > 0
        )
        nearest_neighbor_values.extend(_nearest_neighbor_distances(site_coords_xy))

        for zone_idx, zone_latlon_row in enumerate(zone_latlon):
            for station_idx, station in enumerate(stations):
                dx, dy = latlon_to_local_xy(
                    station.lat,
                    station.lon,
                    center_lat=float(zone_latlon_row[0]),
                    center_lon=float(zone_latlon_row[1]),
                )
                distance_km = float(np.sqrt(dx * dx + dy * dy))
                travel_minutes = float(travel_time_matrix[zone_idx, station_idx])
                if travel_minutes > 0.0 and distance_km > 0.1:
                    speed_values.append(distance_km / travel_minutes * 60.0)

        station_frame = snapshot_to_frame(snapshot_payload)
        location_summaries.append(
            {
                "location_query": str(location_query),
                "snapshot_path": str(location_dir / "snapshot.json"),
                "num_stations": int(len(stations)),
                "total_connectors": int(station_frame["total_connectors"].sum()) if not station_frame.empty else 0,
                "available_connectors": int(station_frame["available_connectors"].sum()) if not station_frame.empty else 0,
                "mean_travel_time_minutes": float(np.mean(travel_time_matrix)),
            }
        )

    distributions = {
        "connector_count": _distribution(connector_values),
        "max_power_kw": _distribution(power_values),
        "availability_ratio": _distribution(availability_values),
        "nearest_neighbor_km": _distribution(nearest_neighbor_values),
        "implied_speed_kmh": _distribution(speed_values),
    }
    priors_payload = {
        "source": "tomtom_multi_snapshot",
        "location_count": int(len(location_summaries)),
        "locations": location_summaries,
        "radius_m": radius_m,
        "distributions": distributions,
        "recommended_corridor_defaults": _recommended_defaults(distributions),
    }
    write_json(output_dir / "priors.json", priors_payload)
    LOGGER.info("Saved TomTom calibration priors to %s", output_dir / "priors.json")


if __name__ == "__main__":
    main()

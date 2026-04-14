from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from evch.config.loader import build_config_parser, load_config
from evch.data.tomtom import (
    _fallback_travel_minutes,
    TomTomApiError,
    build_snapshot_payload,
    build_station_records,
    build_zone_grid,
    geocode_location,
    request_matrix_travel_times,
    resolve_api_key,
    search_nearby_ev_stations,
    snapshot_to_frame,
    write_station_snapshot_svg,
)
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = build_config_parser("Fetch a TomTom-backed EV charging snapshot.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    env_cfg = config["environment"]
    experiment_cfg = config["experiment"]
    tomtom_cfg = config.get("tomtom", {})

    api_key = resolve_api_key(tomtom_cfg.get("api_key"))
    location_query = str(tomtom_cfg["location_query"])
    radius_m = int(tomtom_cfg.get("radius_m", 25000))
    result_limit = int(tomtom_cfg.get("result_limit", env_cfg["num_candidate_sites"]))
    language = tomtom_cfg.get("language")

    geocode_result = geocode_location(
        query=location_query,
        api_key=api_key,
        limit=1,
        country_set=tomtom_cfg.get("country_set"),
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
        raise TomTomApiError(f"No EV charging stations found for {location_query} within {radius_m} meters.")

    zone_coords_km, zone_latlon = build_zone_grid(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m,
        num_zones=int(env_cfg["num_demand_zones"]),
    )
    station_latlon = np.asarray([[station.lat, station.lon] for station in stations], dtype=np.float32)
    try:
        travel_time_matrix_minutes = request_matrix_travel_times(
            origins=zone_latlon,
            destinations=station_latlon,
            api_key=api_key,
            travel_mode=str(tomtom_cfg.get("travel_mode", "car")),
            route_type=str(tomtom_cfg.get("route_type", "fastest")),
            traffic=str(tomtom_cfg.get("traffic", "historical")),
            depart_at=str(tomtom_cfg.get("depart_at", "any")),
        )
        if np.isnan(travel_time_matrix_minutes).any():
            fallback_minutes = _fallback_travel_minutes(zone_latlon, station_latlon)
            travel_time_matrix_minutes = np.where(
                np.isnan(travel_time_matrix_minutes),
                fallback_minutes,
                travel_time_matrix_minutes,
            ).astype(np.float32)
    except TomTomApiError as exc:
        LOGGER.warning("Falling back to geometric travel times because TomTom routing failed: %s", exc)
        travel_time_matrix_minutes = _fallback_travel_minutes(zone_latlon, station_latlon)

    snapshot_payload = build_snapshot_payload(
        location_query=location_query,
        geocode_result=geocode_result,
        stations=stations,
        zone_coords_km=zone_coords_km,
        zone_latlon=zone_latlon,
        travel_time_matrix_minutes=travel_time_matrix_minutes,
        radius_m=radius_m,
    )

    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "tomtom")
    snapshot_path = output_dir / "snapshot.json"
    write_json(snapshot_path, snapshot_payload)
    snapshot_to_frame(snapshot_payload).to_csv(output_dir / "stations.csv", index=False)
    write_station_snapshot_svg(snapshot_payload, output_dir / "station_map.svg", title=location_query)

    station_frame = snapshot_to_frame(snapshot_payload)
    total_connectors = int(station_frame["total_connectors"].sum()) if not station_frame.empty else 0
    available_connectors = int(station_frame["available_connectors"].sum()) if not station_frame.empty else 0
    write_json(
        output_dir / "metadata.json",
        {
            "location_query": location_query,
            "snapshot_path": str(snapshot_path),
            "num_stations": int(len(stations)),
            "num_zones": int(len(zone_coords_km)),
            "mean_travel_time_minutes": float(np.mean(travel_time_matrix_minutes)),
            "max_travel_time_minutes": float(np.max(travel_time_matrix_minutes)),
            "total_connectors": total_connectors,
            "available_connectors": available_connectors,
            "availability_ratio": float(available_connectors / max(total_connectors, 1)),
        },
    )
    LOGGER.info("Saved TomTom snapshot for %s to %s", location_query, output_dir)


if __name__ == "__main__":
    main()

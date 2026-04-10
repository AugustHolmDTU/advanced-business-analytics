from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.data.tomtom import (
    _fallback_travel_minutes,
    TomTomApiError,
    build_snapshot_payload,
    build_station_records,
    build_zone_grid,
    cluster_zones_and_stations,
    fetch_osm_boundary,
    geocode_location,
    latlon_to_local_xy,
    plot_station_snapshot,
    render_boundary_tiled_fixed_chargers_map,
    render_fixed_station_map,
    request_matrix_travel_times,
    resolve_api_key,
    search_ev_stations_in_boundary,
    snapshot_to_frame,
)
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = build_config_parser("Fetch a TomTom-backed EV charging snapshot for Phase 1 integration.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    env_cfg = config["environment"]
    experiment_cfg = config["experiment"]
    tomtom_cfg = config.get("tomtom", {})

    api_key = resolve_api_key(tomtom_cfg.get("api_key"))
    location_query = str(tomtom_cfg["location_query"])
    radius_m = int(tomtom_cfg.get("radius_m", 3000))
    result_limit = int(tomtom_cfg.get("result_limit", env_cfg["num_candidate_sites"]))
    cluster_count = int(tomtom_cfg.get("cluster_count", 12))
    fixed_cluster_count = int(tomtom_cfg.get("fixed_cluster_count", 5))
    language = tomtom_cfg.get("language")
    boundary_record = fetch_osm_boundary(str(tomtom_cfg.get("boundary_query", "Frederiksberg Kommune, Denmark")))
    boundary_geojson = boundary_record["geojson"]

    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "tomtom")
    geocode_result = geocode_location(
        query=location_query,
        api_key=api_key,
        limit=1,
        country_set=tomtom_cfg.get("country_set"),
        language=language,
    )
    center_lat = float((geocode_result.get("position") or {})["lat"])
    center_lon = float((geocode_result.get("position") or {})["lon"])
    nearby_results = search_ev_stations_in_boundary(
        boundary_geojson=boundary_geojson,
        api_key=api_key,
        limit_per_tile=result_limit,
        language=language,
        grid_size=int(tomtom_cfg.get("search_grid_size", 4)),
    )
    stations = build_station_records(nearby_results, api_key=api_key)
    if not stations:
        raise TomTomApiError(f"No EV charging stations found for {location_query} within {radius_m} meters.")
    site_coords_km = np.asarray(
        [latlon_to_local_xy(station.lat, station.lon, center_lat=center_lat, center_lon=center_lon) for station in stations],
        dtype=np.float32,
    )

    zone_coords_km, zone_latlon = build_zone_grid(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m,
        num_zones=int(env_cfg["num_demand_zones"]),
    )
    clustering = cluster_zones_and_stations(
        zone_coords_km=zone_coords_km,
        stations=stations,
        site_coords_km=site_coords_km,
        num_clusters=cluster_count,
        seed=int(config.get("seed", 0)),
        num_fixed_clusters=fixed_cluster_count,
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
            LOGGER.warning("TomTom matrix response had unresolved routes. Filling missing cells with geometric fallback estimates.")
            fallback_minutes = _fallback_travel_minutes(zone_latlon, station_latlon)
            travel_time_matrix_minutes = np.where(
                np.isnan(travel_time_matrix_minutes),
                fallback_minutes,
                travel_time_matrix_minutes,
            ).astype(np.float32)
    except TomTomApiError as exc:
        LOGGER.warning("Falling back to geometric travel-time estimates because TomTom routing failed: %s", exc)
        travel_time_matrix_minutes = _fallback_travel_minutes(zone_latlon, station_latlon)

    snapshot_payload = build_snapshot_payload(
        location_query=location_query,
        geocode_result=geocode_result,
        stations=stations,
        zone_coords_km=zone_coords_km,
        zone_latlon=zone_latlon,
        travel_time_matrix_minutes=travel_time_matrix_minutes,
        radius_m=radius_m,
        clustering=clustering,
    )
    snapshot_path = output_dir / "snapshot.json"
    write_json(snapshot_path, snapshot_payload)

    station_frame = snapshot_to_frame(snapshot_payload)
    station_frame.to_csv(output_dir / "stations.csv", index=False)
    cluster_frame = pd.DataFrame.from_records(snapshot_payload["clustering"]["clusters"])
    cluster_frame.to_csv(output_dir / "clusters.csv", index=False)
    plot_station_snapshot(
        snapshot_payload,
        path=output_dir / "station_map.svg",
        title="Frederiksberg Municipality",
    )
    render_fixed_station_map(
        snapshot_payload=snapshot_payload,
        api_key=api_key,
        path=output_dir / "fixed_chargers_map.png",
    )
    write_json(output_dir / "frederiksberg_boundary.json", boundary_geojson)
    render_boundary_tiled_fixed_chargers_map(
        snapshot_payload=snapshot_payload,
        api_key=api_key,
        boundary_geojson=boundary_geojson,
        path=output_dir / "frederiksberg_boundary_fixed_chargers.png",
        language=str(language or "en-GB"),
    )

    metadata = {
        "location_query": location_query,
        "center": snapshot_payload["center"],
        "num_stations": int(len(stations)),
        "num_zones": int(len(zone_coords_km)),
        "num_clusters": int(cluster_count),
        "snapshot_path": str(snapshot_path),
        "mean_travel_time_minutes": float(np.mean(travel_time_matrix_minutes)),
        "max_travel_time_minutes": float(np.max(travel_time_matrix_minutes)),
        "total_connectors": int(station_frame["total_connectors"].sum()),
        "available_connectors": int(station_frame["available_connectors"].sum()),
    }
    write_json(output_dir / "metadata.json", metadata)
    LOGGER.info("Saved TomTom snapshot for %s to %s", location_query, output_dir)


if __name__ == "__main__":
    main()

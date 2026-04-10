from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape


class TomTomApiError(RuntimeError):
    pass


@dataclass(slots=True)
class TomTomStation:
    name: str
    address: str
    lat: float
    lon: float
    charging_availability_id: str | None
    total_connectors: int
    available_connectors: int
    occupied_connectors: int
    reserved_connectors: int
    unknown_connectors: int
    out_of_service_connectors: int
    max_power_kw: float | None
    raw_result: dict[str, Any]
    availability_payload: dict[str, Any] | None


def resolve_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        raise TomTomApiError(
            "Missing TomTom API key. Set TOMTOM_API_KEY or pass an explicit key to the TomTom client."
        )
    return api_key


def _fetch_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TomTomApiError(f"TomTom API request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise TomTomApiError(f"TomTom API request failed: {exc}") from exc
    return json.loads(body)


def _fetch_bytes(url: str, *, timeout: int = 30) -> bytes:
    request = Request(url, headers={"Accept": "*/*"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TomTomApiError(f"TomTom binary request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise TomTomApiError(f"TomTom binary request failed: {exc}") from exc


def geocode_location(
    query: str,
    api_key: str,
    *,
    limit: int = 1,
    country_set: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    encoded_query = quote(query, safe="")
    params: dict[str, Any] = {
        "key": api_key,
        "limit": limit,
    }
    if country_set:
        params["countrySet"] = country_set
    if language:
        params["language"] = language
    url = f"https://api.tomtom.com/search/2/geocode/{encoded_query}.json?{urlencode(params)}"
    payload = _fetch_json(url)
    results = payload.get("results", [])
    if not results:
        raise TomTomApiError(f"No geocoding result found for query: {query}")
    return results[0]


def fetch_osm_boundary(query: str) -> dict[str, Any]:
    encoded_query = quote(query, safe="")
    url = f"https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&polygon_geojson=1&q={encoded_query}"
    headers = {"User-Agent": "codex-evch/1.0"}
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TomTomApiError(f"OSM boundary request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise TomTomApiError(f"OSM boundary request failed: {exc}") from exc
    if not data:
        raise TomTomApiError(f"No OSM boundary geometry returned for query: {query}")
    return data[0]


def fetch_osm_roads(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> list[dict[str, Any]]:
    headers = {"User-Agent": "codex-evch/1.0"}
    endpoints = [
        "https://overpass.kumi.systems/api/interpreter?data=",
        "https://overpass-api.de/api/interpreter?data=",
    ]
    highway_filter = "^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified|living_street)$"
    lat_steps = np.linspace(min_lat, max_lat, 4)
    lon_steps = np.linspace(min_lon, max_lon, 4)
    merged: dict[int, dict[str, Any]] = {}

    for lat_idx in range(len(lat_steps) - 1):
        for lon_idx in range(len(lon_steps) - 1):
            tile_min_lat = float(lat_steps[lat_idx])
            tile_max_lat = float(lat_steps[lat_idx + 1])
            tile_min_lon = float(lon_steps[lon_idx])
            tile_max_lon = float(lon_steps[lon_idx + 1])
            bbox = f"{tile_min_lat},{tile_min_lon},{tile_max_lat},{tile_max_lon}"
            query = f'[out:json][timeout:60];way["highway"~"{highway_filter}"]({bbox});out geom;'
            last_error: Exception | None = None
            for endpoint in endpoints:
                url = endpoint + quote(query, safe="")
                request = Request(url, headers=headers, method="GET")
                try:
                    with urlopen(request, timeout=120) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    for element in data.get("elements", []):
                        if "id" in element:
                            merged[int(element["id"])] = element
                    last_error = None
                    break
                except HTTPError as exc:
                    last_error = exc
                    continue
                except URLError as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise TomTomApiError(f"OSM roads request failed for tile {bbox}: {last_error}")
    return list(merged.values())


def search_nearby_ev_stations(
    lat: float,
    lon: float,
    radius_m: int,
    api_key: str,
    *,
    limit: int,
    language: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "key": api_key,
        "lat": lat,
        "lon": lon,
        "radius": radius_m,
        "categorySet": 7309,
        "limit": limit,
    }
    if language:
        params["language"] = language
    url = f"https://api.tomtom.com/search/2/nearbySearch/.json?{urlencode(params)}"
    payload = _fetch_json(url)
    return list(payload.get("results", []))


def search_ev_stations_in_boundary(
    *,
    boundary_geojson: dict[str, Any],
    api_key: str,
    limit_per_tile: int,
    language: str | None = None,
    grid_size: int = 4,
) -> list[dict[str, Any]]:
    geometry = _geometry_from_geojson(boundary_geojson)
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    lat_values = np.linspace(min_lat, max_lat, grid_size + 1)
    lon_values = np.linspace(min_lon, max_lon, grid_size + 1)
    deduped: dict[str, dict[str, Any]] = {}

    for lat_idx in range(grid_size):
        for lon_idx in range(grid_size):
            tile_min_lat = float(lat_values[lat_idx])
            tile_max_lat = float(lat_values[lat_idx + 1])
            tile_min_lon = float(lon_values[lon_idx])
            tile_max_lon = float(lon_values[lon_idx + 1])
            center_lat = 0.5 * (tile_min_lat + tile_max_lat)
            center_lon = 0.5 * (tile_min_lon + tile_max_lon)
            tile_height_m = 111_320.0 * (tile_max_lat - tile_min_lat)
            tile_width_m = 111_320.0 * math.cos(math.radians(center_lat)) * (tile_max_lon - tile_min_lon)
            radius_m = int(max(math.sqrt(tile_height_m**2 + tile_width_m**2) * 0.7, 400.0))
            results = search_nearby_ev_stations(
                lat=center_lat,
                lon=center_lon,
                radius_m=radius_m,
                api_key=api_key,
                limit=limit_per_tile,
                language=language,
            )
            for result in results:
                position = result.get("position") or {}
                point = Point(float(position.get("lon")), float(position.get("lat")))
                if not geometry.covers(point):
                    continue
                availability_id = ((result.get("dataSources") or {}).get("chargingAvailability") or {}).get("id")
                key = str(availability_id or f"{position.get('lat')}::{position.get('lon')}::{(result.get('poi') or {}).get('name')}")
                deduped[key] = result
    return list(deduped.values())


def fetch_charging_availability(charging_availability_id: str, api_key: str) -> dict[str, Any]:
    params = {
        "key": api_key,
        "chargingAvailability": charging_availability_id,
    }
    url = f"https://api.tomtom.com/search/2/chargingAvailability.json?{urlencode(params)}"
    return _fetch_json(url)


def _summarize_availability(payload: dict[str, Any] | None) -> dict[str, Any]:
    summary = {
        "total_connectors": 0,
        "available_connectors": 0,
        "occupied_connectors": 0,
        "reserved_connectors": 0,
        "unknown_connectors": 0,
        "out_of_service_connectors": 0,
        "max_power_kw": None,
    }
    if payload is None:
        return summary

    max_power_kw = None
    for connector in payload.get("connectors", []):
        summary["total_connectors"] += int(connector.get("total") or 0)
        current = (connector.get("availability") or {}).get("current") or {}
        summary["available_connectors"] += int(current.get("available") or 0)
        summary["occupied_connectors"] += int(current.get("occupied") or 0)
        summary["reserved_connectors"] += int(current.get("reserved") or 0)
        summary["unknown_connectors"] += int(current.get("unknown") or 0)
        summary["out_of_service_connectors"] += int(current.get("outOfService") or 0)

        for power_level in (connector.get("availability") or {}).get("perPowerLevel") or []:
            power_kw = power_level.get("powerKW")
            if power_kw is not None:
                max_power_kw = float(power_kw) if max_power_kw is None else max(max_power_kw, float(power_kw))
    summary["max_power_kw"] = max_power_kw
    return summary


def build_station_records(results: list[dict[str, Any]], api_key: str) -> list[TomTomStation]:
    stations: list[TomTomStation] = []
    for result in results:
        position = result.get("position") or {}
        data_sources = result.get("dataSources") or {}
        availability_id = ((data_sources.get("chargingAvailability") or {}).get("id"))
        availability_payload = fetch_charging_availability(availability_id, api_key) if availability_id else None
        availability_summary = _summarize_availability(availability_payload)
        stations.append(
            TomTomStation(
                name=str((result.get("poi") or {}).get("name") or (result.get("address") or {}).get("freeformAddress") or "Unknown station"),
                address=str((result.get("address") or {}).get("freeformAddress") or ""),
                lat=float(position.get("lat")),
                lon=float(position.get("lon")),
                charging_availability_id=str(availability_id) if availability_id else None,
                total_connectors=int(availability_summary["total_connectors"]),
                available_connectors=int(availability_summary["available_connectors"]),
                occupied_connectors=int(availability_summary["occupied_connectors"]),
                reserved_connectors=int(availability_summary["reserved_connectors"]),
                unknown_connectors=int(availability_summary["unknown_connectors"]),
                out_of_service_connectors=int(availability_summary["out_of_service_connectors"]),
                max_power_kw=availability_summary["max_power_kw"],
                raw_result=result,
                availability_payload=availability_payload,
            )
        )
    return stations


def latlon_to_local_xy(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    earth_radius_km = 6371.0088
    lat_rad = math.radians(lat)
    center_lat_rad = math.radians(center_lat)
    delta_lat = math.radians(lat - center_lat)
    delta_lon = math.radians(lon - center_lon)
    x = earth_radius_km * delta_lon * math.cos(0.5 * (lat_rad + center_lat_rad))
    y = earth_radius_km * delta_lat
    return x, y


def _mercator_xy(lat: float, lon: float) -> tuple[float, float]:
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    x = (lon + 180.0) / 360.0
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0
    return x, y


def local_xy_to_latlon(x_km: float, y_km: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    earth_radius_km = 6371.0088
    lat = center_lat + math.degrees(y_km / earth_radius_km)
    lon = center_lon + math.degrees(x_km / (earth_radius_km * math.cos(math.radians(center_lat))))
    return lat, lon


def build_zone_grid(center_lat: float, center_lon: float, radius_m: int, num_zones: int) -> tuple[np.ndarray, np.ndarray]:
    radius_km = float(radius_m) / 1000.0
    density = 0
    filtered_points: list[tuple[float, float]] = []
    while len(filtered_points) < num_zones:
        density += 1
        cols = int(math.ceil(math.sqrt(num_zones) * (1.4 + 0.15 * density)))
        rows = int(math.ceil(math.sqrt(num_zones) * (1.2 + 0.15 * density)))
        x_values = np.linspace(-radius_km * 1.08, radius_km * 1.08, cols, dtype=np.float32)
        y_values = np.linspace(-radius_km * 0.94, radius_km * 0.94, rows, dtype=np.float32)
        candidate_points: list[tuple[float, float]] = []
        for y_km in y_values:
            for x_km in x_values:
                ellipse = (float(x_km) / (radius_km * 1.02)) ** 2 + (float(y_km) / (radius_km * 0.82)) ** 2
                top_left_cut = float(y_km) > radius_km * 0.45 and float(x_km) < -radius_km * 0.45
                if ellipse <= 1.0 and not top_left_cut:
                    candidate_points.append((float(x_km), float(y_km)))
        filtered_points = candidate_points

    filtered_points.sort(key=lambda item: (-item[1], item[0]))
    if len(filtered_points) > num_zones:
        chosen_indices = np.linspace(0, len(filtered_points) - 1, num=num_zones, dtype=int)
        filtered_points = [filtered_points[int(index)] for index in chosen_indices]

    zone_coords: list[list[float]] = []
    zone_latlon: list[list[float]] = []
    for x_km, y_km in filtered_points:
        lat, lon = local_xy_to_latlon(x_km, y_km, center_lat=center_lat, center_lon=center_lon)
        zone_coords.append([x_km, y_km])
        zone_latlon.append([lat, lon])
    return np.asarray(zone_coords, dtype=np.float32), np.asarray(zone_latlon, dtype=np.float32)


def _kmeans(points: np.ndarray, num_clusters: int, seed: int, iterations: int = 40) -> tuple[np.ndarray, np.ndarray]:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if len(points) < num_clusters:
        raise ValueError("Number of points must be at least num_clusters")

    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=num_clusters, replace=False)].astype(np.float32)
    labels = np.zeros(len(points), dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for cluster_idx in range(num_clusters):
            members = points[labels == cluster_idx]
            if len(members) == 0:
                new_centers[cluster_idx] = points[int(rng.integers(len(points)))]
            else:
                new_centers[cluster_idx] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def _grid_cell_size(zone_coords_km: np.ndarray) -> tuple[float, float]:
    unique_x = np.unique(np.round(zone_coords_km[:, 0], 4))
    unique_y = np.unique(np.round(zone_coords_km[:, 1], 4))
    dx = float(np.min(np.diff(np.sort(unique_x)))) if len(unique_x) > 1 else 0.35
    dy = float(np.min(np.diff(np.sort(unique_y)))) if len(unique_y) > 1 else 0.35
    return dx, dy


def cluster_zones_and_stations(
    zone_coords_km: np.ndarray,
    stations: list[TomTomStation],
    site_coords_km: np.ndarray,
    num_clusters: int,
    seed: int,
    num_fixed_clusters: int = 5,
) -> dict[str, Any]:
    fixed_cluster_count = min(max(int(num_fixed_clusters), 0), int(min(len(site_coords_km), num_clusters)))
    mobile_cluster_count = max(0, int(num_clusters - fixed_cluster_count))

    centers_list: list[np.ndarray] = []
    center_types: list[str] = []
    station_assignments = np.zeros(len(site_coords_km), dtype=np.int32)

    if fixed_cluster_count > 0:
        _, fixed_centers = _kmeans(site_coords_km, num_clusters=fixed_cluster_count, seed=seed + 11)
        centers_list.append(fixed_centers)
        center_types.extend(["fixed"] * fixed_cluster_count)
        station_distances = np.sum((site_coords_km[:, None, :] - fixed_centers[None, :, :]) ** 2, axis=2)
        station_assignments = np.argmin(station_distances, axis=1).astype(np.int32)

    if mobile_cluster_count > 0:
        mobile_points = zone_coords_km
        if fixed_cluster_count > 0:
            fixed_distances = np.min(np.sum((zone_coords_km[:, None, :] - fixed_centers[None, :, :]) ** 2, axis=2), axis=1)
            threshold = float(np.quantile(fixed_distances, 0.38))
            filtered = zone_coords_km[fixed_distances >= threshold]
            if len(filtered) >= mobile_cluster_count:
                mobile_points = filtered
        _, mobile_centers = _kmeans(mobile_points, num_clusters=mobile_cluster_count, seed=seed + 29)
        centers_list.append(mobile_centers)
        center_types.extend(["mobile"] * mobile_cluster_count)

    centers = np.concatenate(centers_list, axis=0).astype(np.float32)
    distances = np.sum((zone_coords_km[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int32)

    ordering = np.lexsort((centers[:, 0], -centers[:, 1]))
    ordered_labels = labels.copy()
    ordered_station_assignments = station_assignments.copy()
    reordered_centers = centers.copy()
    reordered_types = list(center_types)
    for new_idx, old_idx in enumerate(ordering):
        ordered_labels[labels == int(old_idx)] = new_idx
        if fixed_cluster_count > 0:
            ordered_station_assignments[station_assignments == int(old_idx)] = new_idx
        reordered_centers[new_idx] = centers[int(old_idx)]
        reordered_types[new_idx] = center_types[int(old_idx)]

    zone_cell_width_km, zone_cell_height_km = _grid_cell_size(zone_coords_km)
    clusters: list[dict[str, Any]] = []
    for cluster_idx in range(num_clusters):
        zone_indices = np.flatnonzero(ordered_labels == cluster_idx)
        station_indices = np.flatnonzero(ordered_station_assignments == cluster_idx)
        total_connectors = sum(stations[int(index)].total_connectors for index in station_indices)
        available_connectors = sum(stations[int(index)].available_connectors for index in station_indices)
        cluster_type = reordered_types[cluster_idx]
        clusters.append(
            {
                "cluster_id": int(cluster_idx + 1),
                "type": cluster_type,
                "center_km": [float(reordered_centers[cluster_idx, 0]), float(reordered_centers[cluster_idx, 1])],
                "zone_indices": zone_indices.astype(int).tolist(),
                "station_indices": station_indices.astype(int).tolist(),
                "num_zones": int(len(zone_indices)),
                "num_stations": int(len(station_indices)),
                "total_connectors": int(total_connectors),
                "available_connectors": int(available_connectors),
            }
        )
    return {
        "num_clusters": int(num_clusters),
        "zone_assignments": ordered_labels.astype(int).tolist(),
        "station_assignments": ordered_station_assignments.astype(int).tolist(),
        "zone_cell_width_km": float(zone_cell_width_km),
        "zone_cell_height_km": float(zone_cell_height_km),
        "clusters": clusters,
    }


def _fallback_travel_minutes(origins: np.ndarray, destinations: np.ndarray, speed_kmh: float = 28.0) -> np.ndarray:
    minutes = np.zeros((len(origins), len(destinations)), dtype=np.float32)
    for row_idx, origin in enumerate(origins):
        for col_idx, destination in enumerate(destinations):
            origin_x, origin_y = latlon_to_local_xy(float(origin[0]), float(origin[1]), float(origin[0]), float(origin[1]))
            del origin_x, origin_y
            dx, dy = latlon_to_local_xy(float(destination[0]), float(destination[1]), float(origin[0]), float(origin[1]))
            distance_km = math.sqrt(dx * dx + dy * dy)
            minutes[row_idx, col_idx] = float((distance_km / speed_kmh) * 60.0 + 2.0)
    return minutes


def request_matrix_travel_times(
    origins: np.ndarray,
    destinations: np.ndarray,
    api_key: str,
    *,
    travel_mode: str = "car",
    route_type: str = "fastest",
    traffic: str = "historical",
    depart_at: str = "any",
    max_cells_per_request: int = 25,
) -> np.ndarray:
    matrix = np.full((len(origins), len(destinations)), np.nan, dtype=np.float32)
    if len(origins) == 0 or len(destinations) == 0:
        return matrix

    url = f"https://api.tomtom.com/routing/matrix/2?{urlencode({'key': api_key})}"
    max_origins = max(1, int(math.sqrt(max_cells_per_request)))
    max_destinations = max(1, int(max_cells_per_request // max_origins))
    for origin_start in range(0, len(origins), max_origins):
        origin_end = min(len(origins), origin_start + max_origins)
        origin_batch = origins[origin_start:origin_end]
        for dest_start in range(0, len(destinations), max_destinations):
            dest_end = min(len(destinations), dest_start + max_destinations)
            destination_batch = destinations[dest_start:dest_end]
            body = {
                "origins": [{"point": {"latitude": float(origin[0]), "longitude": float(origin[1])}} for origin in origin_batch],
                "destinations": [{"point": {"latitude": float(destination[0]), "longitude": float(destination[1])}} for destination in destination_batch],
                "options": {
                    "departAt": depart_at,
                    "routeType": route_type,
                    "traffic": traffic,
                    "travelMode": travel_mode,
                },
            }
            payload = _fetch_json(url, method="POST", payload=body, timeout=60)
            for cell in payload.get("data", []):
                if not isinstance(cell, dict):
                    continue
                row_idx = int(cell.get("originIndex", -1))
                col_idx = int(cell.get("destinationIndex", -1))
                global_row_idx = origin_start + row_idx
                global_col_idx = dest_start + col_idx
                if not (0 <= global_row_idx < len(origins) and 0 <= global_col_idx < len(destinations)):
                    continue
                route_summary = cell.get("routeSummary") or {}
                travel_seconds = route_summary.get("travelTimeInSeconds")
                if travel_seconds is not None:
                    matrix[global_row_idx, global_col_idx] = float(travel_seconds) / 60.0
    return matrix


def build_snapshot_payload(
    *,
    location_query: str,
    geocode_result: dict[str, Any],
    stations: list[TomTomStation],
    zone_coords_km: np.ndarray,
    zone_latlon: np.ndarray,
    travel_time_matrix_minutes: np.ndarray,
    radius_m: int,
    clustering: dict[str, Any],
) -> dict[str, Any]:
    center_lat = float((geocode_result.get("position") or {})["lat"])
    center_lon = float((geocode_result.get("position") or {})["lon"])
    site_coords_km = np.asarray(
        [latlon_to_local_xy(station.lat, station.lon, center_lat=center_lat, center_lon=center_lon) for station in stations],
        dtype=np.float32,
    )
    site_latlon = np.asarray([[station.lat, station.lon] for station in stations], dtype=np.float32)

    return {
        "location_query": location_query,
        "center": {
            "lat": center_lat,
            "lon": center_lon,
            "freeform_address": (geocode_result.get("address") or {}).get("freeformAddress"),
        },
        "search_radius_m": int(radius_m),
        "site_coords_km": site_coords_km.tolist(),
        "zone_coords_km": zone_coords_km.tolist(),
        "site_latlon": site_latlon.tolist(),
        "zone_latlon": zone_latlon.tolist(),
        "travel_time_matrix_minutes": travel_time_matrix_minutes.tolist(),
        "clustering": clustering,
        "site_names": [station.name for station in stations],
        "site_addresses": [station.address for station in stations],
        "stations": [
            {
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
            }
            for station in stations
        ],
    }


def snapshot_to_frame(snapshot_payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_records(snapshot_payload["stations"])


def _geometry_from_geojson(geojson_payload: dict[str, Any]) -> Polygon | MultiPolygon:
    geometry = shape(geojson_payload)
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"Expected polygonal geometry, got {geometry.geom_type}")
    return geometry


def _iter_polygon_rings(geometry: Polygon | MultiPolygon) -> list[list[tuple[float, float]]]:
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    rings: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        rings.append([(float(x), float(y)) for x, y in polygon.exterior.coords])
    return rings


def _clip_roads_to_boundary(roads: list[dict[str, Any]], boundary_geometry: Polygon | MultiPolygon) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for road in roads:
        coords = road.get("geometry") or []
        if len(coords) < 2:
            continue
        line = LineString([(float(point["lon"]), float(point["lat"])) for point in coords])
        intersection = line.intersection(boundary_geometry)
        if intersection.is_empty:
            continue
        geometries = [intersection] if intersection.geom_type == "LineString" else list(getattr(intersection, "geoms", []))
        for geom in geometries:
            if geom.geom_type != "LineString":
                continue
            line_coords = [(float(x), float(y)) for x, y in geom.coords]
            if len(line_coords) >= 2:
                clipped.append(
                    {
                        "highway": (road.get("tags") or {}).get("highway", "road"),
                        "name": (road.get("tags") or {}).get("name"),
                        "coordinates": line_coords,
                    }
                )
    return clipped


def _compute_station_bbox(snapshot_payload: dict[str, Any], padding_ratio: float = 0.15) -> tuple[float, float, float, float]:
    stations = snapshot_payload["stations"]
    center = snapshot_payload["center"]
    lats = [float(station["lat"]) for station in stations] + [float(center["lat"])]
    lons = [float(station["lon"]) for station in stations] + [float(center["lon"])]
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)
    lat_pad = max((max_lat - min_lat) * padding_ratio, 0.0035)
    lon_pad = max((max_lon - min_lon) * padding_ratio, 0.0050)
    return min_lon - lon_pad, min_lat - lat_pad, max_lon + lon_pad, max_lat + lat_pad


def _pick_static_map_zoom(bbox: tuple[float, float, float, float], width: int, height: int, padding: int = 60) -> int:
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y1 = _mercator_xy(min_lat, min_lon)
    x1, y0 = _mercator_xy(max_lat, max_lon)
    dx = max(abs(x1 - x0), 1e-8)
    dy = max(abs(y1 - y0), 1e-8)
    usable_width = max(width - 2 * padding, 64)
    usable_height = max(height - 2 * padding, 64)
    zoom_x = math.log2(usable_width / (256.0 * dx))
    zoom_y = math.log2(usable_height / (256.0 * dy))
    zoom = int(math.floor(min(zoom_x, zoom_y, 20.0)))
    return max(0, min(20, zoom))


def fetch_static_map_image(
    *,
    api_key: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    layer: str = "basic",
    style: str = "main",
    fmt: str = "png",
) -> tuple[Image.Image, dict[str, float]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    zoom = _pick_static_map_zoom(bbox, width=width, height=height)
    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)
    params = {
        "key": api_key,
        "center": f"{center_lon},{center_lat}",
        "zoom": zoom,
        "width": width,
        "height": height,
        "format": fmt,
        "layer": layer,
        "style": style,
        "view": "Unified",
    }
    url = f"https://api.tomtom.com/map/1/staticimage?{urlencode(params)}"
    raw = _fetch_bytes(url, timeout=60)
    return Image.open(io.BytesIO(raw)).convert("RGBA"), {
        "zoom": float(zoom),
        "center_lat": float(center_lat),
        "center_lon": float(center_lon),
    }


def _lonlat_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    x, y = _mercator_xy(lat, lon)
    scale = float(2**zoom)
    return x * scale, y * scale


def _world_pixel(lat: float, lon: float, zoom: int, tile_size: int) -> tuple[float, float]:
    tile_x, tile_y = _lonlat_to_tile_xy(lat, lon, zoom)
    return tile_x * tile_size, tile_y * tile_size


def _pick_tile_zoom(bbox: tuple[float, float, float, float], width: int, height: int, tile_size: int, padding: int = 60) -> int:
    min_lon, min_lat, max_lon, max_lat = bbox
    usable_width = max(width - 2 * padding, 128)
    usable_height = max(height - 2 * padding, 128)
    best_zoom = 10
    for zoom in range(20, -1, -1):
        left_px, top_px = _world_pixel(max_lat, min_lon, zoom, tile_size)
        right_px, bottom_px = _world_pixel(min_lat, max_lon, zoom, tile_size)
        span_x = abs(right_px - left_px)
        span_y = abs(bottom_px - top_px)
        if span_x <= usable_width and span_y <= usable_height:
            best_zoom = zoom
            break
    return best_zoom


def fetch_tomtom_tile(api_key: str, zoom: int, x: int, y: int, tile_size: int = 512, language: str = "en-GB") -> Image.Image:
    params = {
        "key": api_key,
        "tileSize": tile_size,
        "view": "Unified",
        "language": language,
    }
    url = f"https://api.tomtom.com/map/1/tile/basic/main/{zoom}/{x}/{y}.png?{urlencode(params)}"
    raw = _fetch_bytes(url, timeout=60)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def render_boundary_tiled_fixed_chargers_map(
    *,
    snapshot_payload: dict[str, Any],
    api_key: str,
    boundary_geojson: dict[str, Any],
    path: str | Path,
    language: str = "en-GB",
) -> None:
    geometry = _geometry_from_geojson(boundary_geojson)
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    pad_lon = max((max_lon - min_lon) * 0.06, 0.004)
    pad_lat = max((max_lat - min_lat) * 0.06, 0.003)
    bbox = (min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat)

    width = 1400
    height = 1120
    header_h = 68
    footer_h = 36
    margin = 18
    map_width = width - 2 * margin
    map_height = height - header_h - footer_h - margin
    tile_size = 512
    zoom = _pick_tile_zoom(bbox, width=map_width, height=map_height, tile_size=tile_size)

    min_tile_x_f, min_tile_y_f = _lonlat_to_tile_xy(bbox[3], bbox[0], zoom)
    max_tile_x_f, max_tile_y_f = _lonlat_to_tile_xy(bbox[1], bbox[2], zoom)
    min_tile_x = int(math.floor(min(min_tile_x_f, max_tile_x_f)))
    max_tile_x = int(math.floor(max(min_tile_x_f, max_tile_x_f)))
    min_tile_y = int(math.floor(min(min_tile_y_f, max_tile_y_f)))
    max_tile_y = int(math.floor(max(min_tile_y_f, max_tile_y_f)))

    stitched = Image.new("RGBA", ((max_tile_x - min_tile_x + 1) * tile_size, (max_tile_y - min_tile_y + 1) * tile_size))
    for tile_x in range(min_tile_x, max_tile_x + 1):
        for tile_y in range(min_tile_y, max_tile_y + 1):
            tile = fetch_tomtom_tile(api_key=api_key, zoom=zoom, x=tile_x, y=tile_y, tile_size=tile_size, language=language)
            stitched.alpha_composite(tile, ((tile_x - min_tile_x) * tile_size, (tile_y - min_tile_y) * tile_size))

    left_px, top_px = _world_pixel(bbox[3], bbox[0], zoom, tile_size)
    right_px, bottom_px = _world_pixel(bbox[1], bbox[2], zoom, tile_size)
    crop_left = int(round(min(left_px, right_px) - min_tile_x * tile_size))
    crop_right = int(round(max(left_px, right_px) - min_tile_x * tile_size))
    crop_top = int(round(min(top_px, bottom_px) - min_tile_y * tile_size))
    crop_bottom = int(round(max(top_px, bottom_px) - min_tile_y * tile_size))
    cropped = stitched.crop((crop_left, crop_top, crop_right, crop_bottom))
    map_image = cropped.resize((map_width, map_height), Image.Resampling.LANCZOS)

    bbox_left, bbox_top = _world_pixel(bbox[3], bbox[0], zoom, tile_size)
    bbox_right, bbox_bottom = _world_pixel(bbox[1], bbox[2], zoom, tile_size)
    bbox_width_px = max(abs(bbox_right - bbox_left), 1e-9)
    bbox_height_px = max(abs(bbox_bottom - bbox_top), 1e-9)

    def project(lon: float, lat: float) -> tuple[float, float]:
        px, py = _world_pixel(lat, lon, zoom, tile_size)
        rel_x = (px - min(bbox_left, bbox_right)) / bbox_width_px
        rel_y = (py - min(bbox_top, bbox_bottom)) / bbox_height_px
        return margin + rel_x * map_width, header_h + rel_y * map_height

    canvas = Image.new("RGBA", (width, height), (246, 243, 238, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=10, fill=(250, 248, 243, 255), outline=(184, 177, 166, 255), width=2)
    canvas.alpha_composite(map_image, (margin, header_h))

    boundary_outline = (28, 38, 54, 255)
    for ring in _iter_polygon_rings(geometry):
        projected = [project(lon, lat) for lon, lat in ring]
        draw.line(projected + [projected[0]], fill=boundary_outline, width=4)

    fixed_color = (160, 0, 125, 255)
    halo_color = (255, 255, 255, 245)
    outline_color = (64, 32, 57, 255)
    for station in snapshot_payload["stations"]:
        x, y = project(float(station["lon"]), float(station["lat"]))
        connectors = max(int(station["total_connectors"]), 1)
        radius = 5 + min(connectors, 12)
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=halo_color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fixed_color, outline=outline_color, width=2)

    font = ImageFont.load_default()
    draw.text((width / 2 - 98, 24), "Frederiksberg Geography And Fixed EV Chargers", fill=(34, 34, 34, 255), font=font)
    legend_x = 28
    legend_y = 84
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 280, legend_y + 72), radius=6, fill=(255, 255, 255, 228), outline=(209, 203, 193, 255), width=1)
    draw.line((legend_x + 12, legend_y + 20, legend_x + 34, legend_y + 20), fill=boundary_outline, width=4)
    draw.text((legend_x + 44, legend_y + 14), "Frederiksberg boundary", fill=(34, 34, 34, 255), font=font)
    draw.ellipse((legend_x + 12, legend_y + 40, legend_x + 32, legend_y + 60), fill=fixed_color, outline=outline_color, width=2)
    draw.text((legend_x + 44, legend_y + 44), "Fixed EV chargers", fill=(34, 34, 34, 255), font=font)
    draw.text((24, height - 28), "Map background: TomTom raster tiles. Boundary geometry: OpenStreetMap.", fill=(90, 90, 90, 255), font=font)
    canvas.convert("RGB").save(path)


def render_fixed_station_map(snapshot_payload: dict[str, Any], api_key: str, path: str | Path) -> None:
    import io

    stations = snapshot_payload["stations"]
    if not stations:
        raise ValueError("No stations available to render.")

    width = 1200
    height = 900
    header_h = 68
    footer_h = 34
    margin = 18
    map_height = height - header_h - footer_h - margin
    bbox = _compute_station_bbox(snapshot_payload)
    map_image, viewport = fetch_static_map_image(
        api_key=api_key,
        bbox=bbox,
        width=width - 2 * margin,
        height=map_height,
    )

    canvas = Image.new("RGBA", (width, height), (245, 243, 238, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=10, fill=(250, 248, 243, 255), outline=(184, 177, 166, 255), width=2)
    canvas.alpha_composite(map_image, (margin, header_h))

    world_scale = 256.0 * (2.0 ** viewport["zoom"])
    center_x, center_y = _mercator_xy(viewport["center_lat"], viewport["center_lon"])
    center_x *= world_scale
    center_y *= world_scale
    top_left_x = center_x - map_image.width / 2.0
    top_left_y = center_y - map_image.height / 2.0

    def project(lat: float, lon: float) -> tuple[float, float]:
        x, y = _mercator_xy(lat, lon)
        world_x = x * world_scale
        world_y = y * world_scale
        px = margin + (world_x - top_left_x)
        py = header_h + (world_y - top_left_y)
        return px, py

    fixed_color = (160, 0, 125, 255)
    halo_color = (255, 255, 255, 235)
    marker_outline = (64, 32, 57, 255)
    for station in stations:
        lat = float(station["lat"])
        lon = float(station["lon"])
        connectors = max(int(station["total_connectors"]), 1)
        radius = 5 + min(connectors, 16)
        x, y = project(lat, lon)
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=halo_color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fixed_color, outline=marker_outline, width=2)

    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    draw.text((width / 2 - 85, 24), "Frederiksberg Fixed EV Chargers", fill=(34, 34, 34, 255), font=title_font)

    legend_x = 28
    legend_y = 84
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 230, legend_y + 50), radius=6, fill=(255, 255, 255, 228), outline=(209, 203, 193, 255), width=1)
    draw.ellipse((legend_x + 12, legend_y + 13, legend_x + 32, legend_y + 33), fill=fixed_color, outline=marker_outline, width=2)
    draw.text((legend_x + 44, legend_y + 16), "Fixed EV chargers", fill=(34, 34, 34, 255), font=body_font)

    attribution = "Map background: TomTom Map Display API"
    draw.text((24, height - 28), attribution, fill=(90, 90, 90, 255), font=body_font)
    canvas.convert("RGB").save(path)


def render_boundary_static_fixed_chargers_map(
    *,
    snapshot_payload: dict[str, Any],
    api_key: str,
    boundary_geojson: dict[str, Any],
    path: str | Path,
) -> None:
    geometry = _geometry_from_geojson(boundary_geojson)
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    pad_lon = max((max_lon - min_lon) * 0.06, 0.004)
    pad_lat = max((max_lat - min_lat) * 0.06, 0.003)
    bbox = (min_lon - pad_lon, min_lat - pad_lat, max_lon + pad_lon, max_lat + pad_lat)

    width = 1400
    height = 1120
    header_h = 68
    footer_h = 36
    margin = 18
    map_height = height - header_h - footer_h - margin
    map_image, viewport = fetch_static_map_image(
        api_key=api_key,
        bbox=bbox,
        width=width - 2 * margin,
        height=map_height,
    )

    canvas = Image.new("RGBA", (width, height), (246, 243, 238, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=10, fill=(250, 248, 243, 255), outline=(184, 177, 166, 255), width=2)
    canvas.alpha_composite(map_image, (margin, header_h))

    world_scale = 256.0 * (2.0 ** viewport["zoom"])
    center_x, center_y = _mercator_xy(viewport["center_lat"], viewport["center_lon"])
    center_x *= world_scale
    center_y *= world_scale
    top_left_x = center_x - map_image.width / 2.0
    top_left_y = center_y - map_image.height / 2.0

    def project(lon: float, lat: float) -> tuple[float, float]:
        x, y = _mercator_xy(lat, lon)
        world_x = x * world_scale
        world_y = y * world_scale
        px = margin + (world_x - top_left_x)
        py = header_h + (world_y - top_left_y)
        return px, py

    boundary_outline = (28, 38, 54, 255)
    for ring in _iter_polygon_rings(geometry):
        projected = [project(lon, lat) for lon, lat in ring]
        draw.line(projected + [projected[0]], fill=boundary_outline, width=4)

    fixed_color = (160, 0, 125, 255)
    halo_color = (255, 255, 255, 245)
    outline_color = (64, 32, 57, 255)
    for station in snapshot_payload["stations"]:
        x, y = project(float(station["lon"]), float(station["lat"]))
        connectors = max(int(station["total_connectors"]), 1)
        radius = 5 + min(connectors, 12)
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=halo_color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fixed_color, outline=outline_color, width=2)

    font = ImageFont.load_default()
    draw.text((width / 2 - 98, 24), "Frederiksberg Geography And Fixed EV Chargers", fill=(34, 34, 34, 255), font=font)
    legend_x = 28
    legend_y = 84
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 280, legend_y + 72), radius=6, fill=(255, 255, 255, 228), outline=(209, 203, 193, 255), width=1)
    draw.line((legend_x + 12, legend_y + 20, legend_x + 34, legend_y + 20), fill=boundary_outline, width=4)
    draw.text((legend_x + 44, legend_y + 14), "Frederiksberg boundary", fill=(34, 34, 34, 255), font=font)
    draw.ellipse((legend_x + 12, legend_y + 40, legend_x + 32, legend_y + 60), fill=fixed_color, outline=outline_color, width=2)
    draw.text((legend_x + 44, legend_y + 44), "Fixed EV chargers", fill=(34, 34, 34, 255), font=font)
    draw.text((24, height - 28), "Map background: TomTom. Boundary geometry: OpenStreetMap.", fill=(90, 90, 90, 255), font=font)
    canvas.convert("RGB").save(path)


def render_boundary_roads_fixed_chargers_map(
    *,
    snapshot_payload: dict[str, Any],
    boundary_geojson: dict[str, Any],
    roads: list[dict[str, Any]],
    path: str | Path,
) -> None:
    geometry = _geometry_from_geojson(boundary_geojson)
    clipped_roads = _clip_roads_to_boundary(roads, geometry)
    min_lon, min_lat, max_lon, max_lat = geometry.bounds

    width = 1400
    height = 1200
    margin = 50
    header_h = 70
    footer_h = 40
    plot_left = margin
    plot_top = header_h
    plot_width = width - 2 * margin
    plot_height = height - header_h - footer_h - margin

    lon_span = max(max_lon - min_lon, 1e-8)
    lat_span = max(max_lat - min_lat, 1e-8)
    scale = min(plot_width / lon_span, plot_height / lat_span)
    x_offset = plot_left + 0.5 * (plot_width - lon_span * scale)
    y_offset = plot_top + 0.5 * (plot_height - lat_span * scale)

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = x_offset + (lon - min_lon) * scale
        y = y_offset + (max_lat - lat) * scale
        return x, y

    canvas = Image.new("RGBA", (width, height), (248, 246, 241, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((16, 16, width - 16, height - 16), radius=10, fill=(252, 250, 246, 255), outline=(184, 177, 166, 255), width=2)

    boundary_fill = (244, 240, 233, 255)
    boundary_outline = (92, 88, 82, 255)
    for ring in _iter_polygon_rings(geometry):
        projected = [project(lon, lat) for lon, lat in ring]
        draw.polygon(projected, fill=boundary_fill, outline=boundary_outline)

    road_styles = {
        "motorway": ((203, 158, 101, 255), 7),
        "trunk": ((214, 169, 112, 255), 6),
        "primary": ((220, 178, 122, 255), 5),
        "secondary": ((206, 190, 158, 255), 4),
        "tertiary": ((196, 196, 196, 255), 3),
        "residential": ((170, 170, 170, 255), 2),
        "service": ((184, 184, 184, 255), 2),
        "unclassified": ((180, 180, 180, 255), 2),
        "footway": ((205, 205, 205, 255), 1),
        "cycleway": ((190, 210, 210, 255), 1),
        "path": ((205, 205, 205, 255), 1),
    }
    for road in clipped_roads:
        coords = [project(lon, lat) for lon, lat in road["coordinates"]]
        if len(coords) < 2:
            continue
        color, width_px = road_styles.get(road["highway"], ((182, 182, 182, 255), 2))
        draw.line(coords, fill=color, width=width_px, joint="curve")

    fixed_color = (160, 0, 125, 255)
    halo_color = (255, 255, 255, 245)
    outline_color = (64, 32, 57, 255)
    for station in snapshot_payload["stations"]:
        x, y = project(float(station["lon"]), float(station["lat"]))
        connectors = max(int(station["total_connectors"]), 1)
        radius = 5 + min(connectors, 12)
        draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), fill=halo_color)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fixed_color, outline=outline_color, width=2)

    font = ImageFont.load_default()
    draw.text((width / 2 - 92, 26), "Frederiksberg Fixed EV Chargers", fill=(34, 34, 34, 255), font=font)
    legend_x = 58
    legend_y = 88
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 220, legend_y + 46), radius=6, fill=(255, 255, 255, 228), outline=(209, 203, 193, 255), width=1)
    draw.ellipse((legend_x + 12, legend_y + 13, legend_x + 32, legend_y + 33), fill=fixed_color, outline=outline_color, width=2)
    draw.text((legend_x + 44, legend_y + 17), "Fixed EV chargers", fill=(34, 34, 34, 255), font=font)
    draw.text((24, height - 28), "Boundary and roads: OpenStreetMap data", fill=(90, 90, 90, 255), font=font)
    canvas.convert("RGB").save(path)


def _ratio_color(ratio: float) -> str:
    clipped = min(max(ratio, 0.0), 1.0)
    red = int(round((1.0 - clipped) * 214 + clipped * 40))
    green = int(round((1.0 - clipped) * 68 + clipped * 167))
    blue = int(round((1.0 - clipped) * 68 + clipped * 69))
    return f"#{red:02x}{green:02x}{blue:02x}"


def plot_station_snapshot(snapshot_payload: dict[str, Any], path: str | Path, *, title: str) -> None:
    stations = snapshot_payload["stations"]
    center = snapshot_payload["center"]
    zone_coords = np.asarray(snapshot_payload["zone_coords_km"], dtype=np.float32)
    site_coords = np.asarray(snapshot_payload["site_coords_km"], dtype=np.float32)
    clustering = snapshot_payload.get("clustering") or {}
    clusters = clustering.get("clusters") or []
    if len(stations) == 0:
        raise ValueError("Cannot plot a TomTom snapshot without stations.")

    radius_km = float(snapshot_payload["search_radius_m"]) / 1000.0
    max_extent = max(radius_km * 1.15, float(np.max(np.abs(zone_coords))) if len(zone_coords) else 0.0, float(np.max(np.abs(site_coords))) if len(site_coords) else 0.0, 1.0)
    zone_labels = np.asarray(clustering.get("zone_assignments") or [0] * len(zone_coords), dtype=np.int32)
    zone_cell_width = float(clustering.get("zone_cell_width_km") or 0.35)
    zone_cell_height = float(clustering.get("zone_cell_height_km") or 0.35)
    station_assignments = np.asarray(clustering.get("station_assignments") or [0] * len(site_coords), dtype=np.int32)

    width = 1180
    height = 920
    top_pad = 120.0
    side_pad = 90.0
    bottom_pad = 60.0
    plot_width = width - 2.0 * side_pad
    plot_height = height - top_pad - bottom_pad
    plot_size = min(plot_width, plot_height)
    origin_x = side_pad + 0.5 * plot_width
    origin_y = top_pad + 0.5 * plot_height
    scale = plot_size / (2.0 * max_extent)
    palette = [
        "#f7b2ad",
        "#fcd6a4",
        "#f8e38f",
        "#d7ec9f",
        "#c4e3d4",
        "#a8d8de",
        "#abc9ea",
        "#c6c2e8",
        "#e1c0e8",
        "#ecc7d8",
        "#d9d9d9",
        "#b9d7ea",
    ]

    def project(point: np.ndarray) -> tuple[float, float]:
        return origin_x + float(point[0]) * scale, origin_y - float(point[1]) * scale

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f3ef" />',
        '<rect x="36" y="34" width="1108" height="852" rx="8" fill="#f9f7f2" stroke="#b8b1a6" stroke-width="2" />',
        f'<text x="{width / 2:.1f}" y="62" text-anchor="middle" font-size="30" font-family="Helvetica, Arial, sans-serif" fill="#222222">Frederiksberg Municipality</text>',
    ]

    center_px, center_py = project(np.asarray([0.0, 0.0], dtype=np.float32))
    for idx in range(12):
        x_offset = (-1.0 + idx / 5.5) * max_extent
        x1, y1 = project(np.asarray([x_offset, -max_extent], dtype=np.float32))
        x2, y2 = project(np.asarray([x_offset + max_extent * 0.5, max_extent], dtype=np.float32))
        color = "#d7d2c9" if idx % 3 else "#efc28f"
        width_px = 2.0 if idx % 3 == 0 else 1.0
        opacity = 0.6 if idx % 3 == 0 else 0.28
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width_px}" opacity="{opacity}" />')
    for idx in range(11):
        y_offset = (-0.95 + idx / 5.3) * max_extent
        x1, y1 = project(np.asarray([-max_extent, y_offset], dtype=np.float32))
        x2, y2 = project(np.asarray([max_extent, y_offset + max_extent * 0.14], dtype=np.float32))
        lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d8d4cc" stroke-width="1" opacity="0.35" />')

    for zone_index, point in enumerate(zone_coords):
        x_px, y_px = project(point)
        fill = palette[int(zone_labels[zone_index]) % len(palette)]
        lines.append(
            f'<rect x="{x_px - 0.5 * zone_cell_width * scale:.1f}" y="{y_px - 0.5 * zone_cell_height * scale:.1f}" width="{zone_cell_width * scale:.1f}" height="{zone_cell_height * scale:.1f}" fill="{fill}" fill-opacity="0.46" stroke="#5e6570" stroke-width="0.85" stroke-opacity="0.35" />'
        )

    for point in zone_coords:
        x_px, y_px = project(point)
        lines.append(f'<text x="{x_px:.1f}" y="{y_px + 3:.1f}" text-anchor="middle" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#0070c9">★</text>')

    for idx, point in enumerate(site_coords):
        x_px, y_px = project(point)
        lines.append(
            f'<circle cx="{x_px:.1f}" cy="{y_px:.1f}" r="3.2" fill="#1f2937" fill-opacity="0.55" stroke="#ffffff" stroke-width="0.4" />'
        )

    for cluster in clusters:
        point = np.asarray(cluster["center_km"], dtype=np.float32)
        x_px, y_px = project(point)
        fill = "#a0007d" if cluster["type"] == "fixed" else "#0d2bd7"
        radius_px = 15.5 if cluster["type"] == "fixed" else 16.5
        lines.append(
            f'<circle cx="{x_px:.1f}" cy="{y_px:.1f}" r="{radius_px:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2" opacity="0.96" />'
        )
        lines.append(
            f'<text x="{x_px:.1f}" y="{y_px + 5:.1f}" text-anchor="middle" font-size="15" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#ffffff">{cluster["cluster_id"]}</text>'
        )

    legend_x = 54
    legend_y = 86
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="420" height="118" rx="6" fill="#ffffff" fill-opacity="0.85" stroke="#d1cbc1" stroke-width="1.3" />',
            f'<text x="{legend_x + 22}" y="{legend_y + 34}" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#0070c9">★</text>',
            f'<text x="{legend_x + 62}" y="{legend_y + 34}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#222222">Zone Center</text>',
            f'<circle cx="{legend_x + 32}" cy="{legend_y + 64}" r="14" fill="#0d2bd7" stroke="#ffffff" stroke-width="2" />',
            f'<text x="{legend_x + 62}" y="{legend_y + 69}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#222222">Mobile charging station potential locations</text>',
            f'<circle cx="{legend_x + 32}" cy="{legend_y + 96}" r="14" fill="#a0007d" stroke="#ffffff" stroke-width="2" />',
            f'<text x="{legend_x + 62}" y="{legend_y + 101}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#222222">Fixed charging stations cluster</text>',
        ]
    )

    fixed_clusters = sum(1 for cluster in clusters if cluster["type"] == "fixed")
    mobile_clusters = sum(1 for cluster in clusters if cluster["type"] == "mobile")
    lines.append(f'<text x="52" y="{height - 28:.1f}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">TomTom-backed Frederiksberg snapshot. {len(zone_coords)} zone centers, {len(clusters)} k-means clusters, {fixed_clusters} fixed clusters, {mobile_clusters} mobile-potential clusters.</text>')
    lines.append("</svg>")

    target = Path(path)
    target.write_text("\n".join(lines), encoding="utf-8")

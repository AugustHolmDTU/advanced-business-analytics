from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


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


def resolve_api_key(explicit_api_key: str | None = None) -> str:
    api_key = explicit_api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        raise TomTomApiError("Missing TomTom API key. Set TOMTOM_API_KEY or pass an explicit key.")
    return api_key


def _fetch_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TomTomApiError(f"TomTom API request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise TomTomApiError(f"TomTom API request failed: {exc}") from exc


def geocode_location(
    query: str,
    api_key: str,
    *,
    limit: int = 1,
    country_set: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"key": api_key, "limit": limit}
    if country_set:
        params["countrySet"] = country_set
    if language:
        params["language"] = language
    encoded_query = quote(query, safe="")
    url = f"https://api.tomtom.com/search/2/geocode/{encoded_query}.json?{urlencode(params)}"
    payload = _fetch_json(url)
    results = payload.get("results", [])
    if not results:
        raise TomTomApiError(f"No geocoding result found for query: {query}")
    return results[0]


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


def fetch_charging_availability(charging_availability_id: str, api_key: str) -> dict[str, Any]:
    params = {
        "key": api_key,
        "chargingAvailability": charging_availability_id,
    }
    url = f"https://api.tomtom.com/search/2/chargingAvailability.json?{urlencode(params)}"
    return _fetch_json(url)


def summarize_availability_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
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
    deduped: dict[str, TomTomStation] = {}
    for result in results:
        position = result.get("position") or {}
        availability_id = ((result.get("dataSources") or {}).get("chargingAvailability") or {}).get("id")
        availability_payload = fetch_charging_availability(availability_id, api_key) if availability_id else None
        availability_summary = summarize_availability_payload(availability_payload)
        dedupe_key = str(availability_id or f"{position.get('lat')}::{position.get('lon')}")
        deduped[dedupe_key] = TomTomStation(
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
        )
    return sorted(deduped.values(), key=lambda station: (station.lon, station.lat))


def route_summary_between_points(
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    api_key: str,
    *,
    travel_mode: str = "car",
    route_type: str = "fastest",
    traffic: str = "historical",
    depart_at: str = "any",
) -> dict[str, float]:
    origin = f"{origin_lat:.6f},{origin_lon:.6f}"
    destination = f"{destination_lat:.6f},{destination_lon:.6f}"
    params = {
        "key": api_key,
        "travelMode": travel_mode,
        "routeType": route_type,
        "traffic": traffic,
        "departAt": depart_at,
    }
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin}:{destination}/json?{urlencode(params)}"
    payload = _fetch_json(url, timeout=60)
    routes = payload.get("routes") or []
    if not routes:
        raise TomTomApiError("TomTom routing returned no routes.")
    summary = (routes[0] or {}).get("summary") or {}
    travel_seconds = summary.get("travelTimeInSeconds")
    route_length = summary.get("lengthInMeters")
    if travel_seconds is None or route_length is None:
        raise TomTomApiError("TomTom routing summary was missing travel time or route length.")
    return {
        "travel_time_minutes": float(travel_seconds) / 60.0,
        "route_length_km": float(route_length) / 1000.0,
    }


def latlon_to_local_xy(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    earth_radius_km = 6371.0088
    lat_rad = math.radians(lat)
    center_lat_rad = math.radians(center_lat)
    delta_lat = math.radians(lat - center_lat)
    delta_lon = math.radians(lon - center_lon)
    x = earth_radius_km * delta_lon * math.cos(0.5 * (lat_rad + center_lat_rad))
    y = earth_radius_km * delta_lat
    return x, y


def local_xy_to_latlon(x_km: float, y_km: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    earth_radius_km = 6371.0088
    lat = center_lat + math.degrees(y_km / earth_radius_km)
    lon = center_lon + math.degrees(x_km / (earth_radius_km * math.cos(math.radians(center_lat))))
    return lat, lon


def build_zone_grid(center_lat: float, center_lon: float, radius_m: int, num_zones: int) -> tuple[np.ndarray, np.ndarray]:
    radius_km = float(radius_m) / 1000.0
    cols = int(math.ceil(math.sqrt(num_zones)))
    rows = int(math.ceil(num_zones / cols))
    x_values = np.linspace(-radius_km, radius_km, cols, dtype=np.float32)
    y_values = np.linspace(-radius_km, radius_km, rows, dtype=np.float32)
    points: list[tuple[float, float]] = []
    for y_km in y_values:
        for x_km in x_values:
            if (float(x_km) / radius_km) ** 2 + (float(y_km) / radius_km) ** 2 <= 1.0:
                points.append((float(x_km), float(y_km)))
    if len(points) < num_zones:
        extra_angles = np.linspace(0.0, 2.0 * math.pi, num_zones - len(points), endpoint=False)
        for angle in extra_angles:
            points.append((0.82 * radius_km * math.cos(float(angle)), 0.82 * radius_km * math.sin(float(angle))))
    points = points[:num_zones]

    zone_coords: list[list[float]] = []
    zone_latlon: list[list[float]] = []
    for x_km, y_km in points:
        lat, lon = local_xy_to_latlon(x_km, y_km, center_lat=center_lat, center_lon=center_lon)
        zone_coords.append([x_km, y_km])
        zone_latlon.append([lat, lon])
    return np.asarray(zone_coords, dtype=np.float32), np.asarray(zone_latlon, dtype=np.float32)


def _fallback_travel_minutes(origins: np.ndarray, destinations: np.ndarray, speed_kmh: float = 70.0) -> np.ndarray:
    minutes = np.zeros((len(origins), len(destinations)), dtype=np.float32)
    for row_idx, origin in enumerate(origins):
        for col_idx, destination in enumerate(destinations):
            dx, dy = latlon_to_local_xy(float(destination[0]), float(destination[1]), float(origin[0]), float(origin[1]))
            distance_km = math.sqrt(dx * dx + dy * dy)
            minutes[row_idx, col_idx] = float((distance_km / speed_kmh) * 60.0 + 3.0)
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
        origin_batch = origins[origin_start : origin_start + max_origins]
        for dest_start in range(0, len(destinations), max_destinations):
            destination_batch = destinations[dest_start : dest_start + max_destinations]
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
    return pd.DataFrame.from_records(snapshot_payload.get("stations", []))


def write_station_snapshot_svg(snapshot_payload: dict[str, Any], path: str | Path, title: str) -> None:
    site_coords = np.asarray(snapshot_payload["site_coords_km"], dtype=np.float32)
    zone_coords = np.asarray(snapshot_payload["zone_coords_km"], dtype=np.float32)
    width = 960
    height = 720
    padding = 56.0
    all_coords = np.vstack([site_coords, zone_coords]) if len(zone_coords) and len(site_coords) else site_coords
    min_x = float(np.min(all_coords[:, 0]))
    max_x = float(np.max(all_coords[:, 0]))
    min_y = float(np.min(all_coords[:, 1]))
    max_y = float(np.max(all_coords[:, 1]))
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    def project(x: float, y: float) -> tuple[float, float]:
        px = padding + (x - min_x) / span_x * (width - 2.0 * padding)
        py = height - padding - (y - min_y) / span_y * (height - 2.0 * padding)
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />',
        f'<text x="52" y="44" font-size="26" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">{title}</text>',
        '<text x="52" y="72" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#475569">TomTom-grounded snapshot with real station positions, connector availability, and route times.</text>',
    ]
    for x_km, y_km in zone_coords:
        px, py = project(float(x_km), float(y_km))
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="#94a3b8" opacity="0.65" />')
    for idx, (station, coord) in enumerate(zip(snapshot_payload.get("stations", []), site_coords, strict=True)):
        px, py = project(float(coord[0]), float(coord[1]))
        radius = 6.0 + min(float(station.get("total_connectors", 0)) * 0.4, 8.0)
        fill = "#0f766e" if float(station.get("available_connectors", 0)) > 0 else "#b91c1c"
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{fill}" stroke="#0f172a" stroke-width="1.2" />')
        lines.append(f'<text x="{px + 9:.1f}" y="{py - 8:.1f}" font-size="11" font-family="Helvetica, Arial, sans-serif" fill="#0f172a">{station.get("name", f"Station {idx + 1}")}</text>')
    lines.append(f'<text x="52" y="{height - 26}" font-size="12" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">Gray circles: synthetic demand zones. Green/red circles: TomTom stations sized by connectors and colored by available connectors.</text>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

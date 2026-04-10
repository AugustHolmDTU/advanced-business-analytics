from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SyntheticCity:
    site_coords: np.ndarray
    zone_coords: np.ndarray
    zone_base_demand: np.ndarray
    zone_scale: np.ndarray
    travel_time_matrix: np.ndarray


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
    )


def static_site_accessibility_scores(city: SyntheticCity, decay: float) -> np.ndarray:
    weights = np.exp(-decay * city.travel_time_matrix)
    return weights.T @ city.zone_base_demand


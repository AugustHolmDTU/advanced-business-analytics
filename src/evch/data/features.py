from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evch.data.synthetic import SyntheticCity

MODEL_FEATURE_COLUMNS = [
    "zone_id",
    "hour",
    "time_sin",
    "time_cos",
    "is_weekend",
    "zone_x",
    "zone_y",
    "base_demand",
    "zone_scale",
    "prev_observed_demand",
]


def build_feature_record(
    city: SyntheticCity,
    zone_id: int,
    hour: int,
    is_weekend: bool,
    prev_observed_demand: float,
    expected_lambda: float,
    observed_demand: float,
    true_demand: float,
) -> dict[str, Any]:
    hour_float = float(hour)
    return {
        "zone_id": zone_id,
        "hour": hour_float,
        "time_sin": np.sin(2.0 * np.pi * hour_float / 24.0),
        "time_cos": np.cos(2.0 * np.pi * hour_float / 24.0),
        "is_weekend": float(is_weekend),
        "zone_x": float(city.zone_coords[zone_id, 0]),
        "zone_y": float(city.zone_coords[zone_id, 1]),
        "base_demand": float(city.zone_base_demand[zone_id]),
        "zone_scale": float(city.zone_scale[zone_id]),
        "prev_observed_demand": float(prev_observed_demand),
        "expected_lambda": float(expected_lambda),
        "observed_demand": float(observed_demand),
        "target_demand": float(true_demand),
    }


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def split_xy(frame: pd.DataFrame, target_column: str = "target_demand") -> tuple[pd.DataFrame, pd.Series]:
    features = frame.drop(columns=[target_column])
    target = frame[target_column]
    return features, target

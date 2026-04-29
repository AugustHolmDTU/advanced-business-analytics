from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def add_derived_metrics(metrics: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    enriched = metrics.copy()
    rolling_window = max(1, int(round(60 / step_minutes)))
    enriched["step_in_day"] = (
        (enriched["hour_of_day"] * 60.0 / float(step_minutes)).round().astype(int)
    )
    enriched["day_progress"] = enriched["hour_of_day"] / 24.0
    for column in (
        "queue_length",
        "utilization",
        "arrivals_total",
        "starts_total",
        "completions_total",
        "queue_wait_mean_minutes",
        "started_service_mean_minutes",
        "started_wait_mean_minutes",
        "effective_num_plugs",
    ):
        enriched[f"{column}_rolling_1h"] = enriched[column].rolling(window=rolling_window, min_periods=1).mean()
    return enriched


def disruption_windows(metrics: pd.DataFrame) -> list[dict[str, float | str]]:
    windows: list[dict[str, float | str]] = []
    active = metrics.loc[metrics["disruption_active"] == 1].copy()
    if active.empty:
        return windows

    block_id = (
        (active["disruption_type"] != active["disruption_type"].shift(1))
        | ((active["global_hour"] - active["global_hour"].shift(1)).fillna(0.0) > 0.51)
    ).cumsum()
    for _, block in active.groupby(block_id):
        first = block.iloc[0]
        last = block.iloc[-1]
        step_hours = float(block["global_hour"].diff().median())
        if not np.isfinite(step_hours) or step_hours <= 0.0:
            step_hours = float(metrics["global_hour"].diff().dropna().median())
        if not np.isfinite(step_hours) or step_hours <= 0.0:
            step_hours = 0.0
        windows.append(
            {
                "start_hour": float(first["global_hour"]),
                "end_hour": float(last["global_hour"] + step_hours),
                "disruption_type": str(first["disruption_type"]),
                "day_index": int(first["day_index"]),
            }
        )
    return windows


def shade_disruptions(axes, windows: list[dict[str, float | str]]) -> None:
    colors = {
        "capacity_drop": "#f39c12",
        "station_outage": "#e74c3c",
        "demand_surge": "#2ecc71",
        "demand_surge_ab": "#2ecc71",
        "demand_surge_ba": "#16a085",
        "service_time_inflation": "#9b59b6",
    }
    for axis in axes:
        for window in windows:
            axis.axvspan(
                float(window["start_hour"]),
                float(window["end_hour"]),
                color=colors.get(str(window["disruption_type"]), "#7f8c8d"),
                alpha=0.16,
            )


def maybe_plot_queue_dynamics(metrics: pd.DataFrame, path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        windows = disruption_windows(metrics)
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        shade_disruptions(axes, windows)

        axes[0].plot(metrics["global_hour"], metrics["queue_length"], color="#c0392b", linewidth=1.5, alpha=0.35)
        axes[0].plot(metrics["global_hour"], metrics["queue_length_rolling_1h"], color="#7f0000", linewidth=2.4)
        axes[0].set_ylabel("Vehicles")
        axes[0].set_title("Queue length with disruption windows")

        axes[1].plot(metrics["global_hour"], metrics["queue_length_rolling_1h"], color="#b03a2e", linewidth=2.2, label="Queue (1h rolling)")
        axes[1].plot(
            metrics["global_hour"],
            metrics["queue_wait_mean_minutes_rolling_1h"],
            color="#34495e",
            linewidth=1.8,
            label="Queue wait mean (1h rolling)",
        )
        axes[1].set_ylabel("Queue / wait")
        axes[1].legend()

        axes[2].plot(metrics["global_hour"], metrics["expected_passing_total"], color="#95a5a6", linewidth=1.4, label="Expected passing")
        axes[2].plot(metrics["global_hour"], metrics["arrivals_total"], color="#27ae60", linewidth=1.0, alpha=0.35, label="Arrivals")
        axes[2].plot(
            metrics["global_hour"],
            metrics["arrivals_total_rolling_1h"],
            color="#1e8449",
            linewidth=2.0,
            label="Arrivals (1h rolling)",
        )
        axes[2].set_ylabel("Vehicles / step")
        axes[2].legend()

        axes[3].plot(metrics["global_hour"], metrics["effective_num_plugs"], color="#1f77b4", linewidth=2.0, label="Effective plugs")
        axes[3].plot(metrics["global_hour"], metrics["active_plugs"], color="#6c5ce7", linewidth=1.4, alpha=0.7, label="Active plugs")
        axes[3].set_xlabel("Global hour")
        axes[3].set_ylabel("Plugs")
        axes[3].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping queue plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def maybe_plot_daily_patterns(metrics: pd.DataFrame, path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

        for day_index, day_frame in metrics.groupby("day_index", sort=True):
            active_types = [str(value) for value in day_frame.loc[day_frame["disruption_active"] == 1, "disruption_type"].unique()]
            label_suffix = active_types[0] if active_types else "normal"
            axes[0].plot(
                day_frame["hour_of_day"],
                day_frame["queue_length_rolling_1h"],
                linewidth=2.0,
                label=f"Day {int(day_index) + 1} ({label_suffix})",
            )
            axes[1].plot(
                day_frame["hour_of_day"],
                day_frame["effective_num_plugs"],
                linewidth=1.8,
                label=f"Day {int(day_index) + 1} ({label_suffix})",
            )

        axes[0].set_xlim(0.0, 24.0)
        axes[0].set_ylabel("Queue (1h rolling)")
        axes[0].set_title("Daily queue profile by hour of day")
        axes[0].legend()

        axes[1].set_xlim(0.0, 24.0)
        axes[1].set_xlabel("Hour of day")
        axes[1].set_ylabel("Effective plugs")
        axes[1].set_title("Per-day available capacity")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Skipping daily pattern plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False

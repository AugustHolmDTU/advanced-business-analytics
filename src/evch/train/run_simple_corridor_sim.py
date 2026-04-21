from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

import pandas as pd

from evch.config.loader import build_config_parser, load_config
from evch.sim.simple_corridor import SimpleCorridorQueueSimulator
from evch.utils.io import ensure_dir, write_json
from evch.utils.logging import configure_logging
from evch.utils.seeding import set_global_seed
from evch.utils.wandb import init_wandb, log_artifact

LOGGER = logging.getLogger(__name__)


def _add_derived_metrics(metrics: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
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
        "started_wait_mean_minutes",
        "queue_wait_mean_minutes",
    ):
        enriched[f"{column}_rolling_1h"] = (
            enriched[column].rolling(window=rolling_window, min_periods=1).mean()
        )
    return enriched


def _maybe_plot_queue_dynamics(metrics, path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(metrics["hour"], metrics["queue_length"], color="#c0392b", linewidth=2.0, label="Queue length")
        axes[0].plot(metrics["hour"], metrics["active_plugs"], color="#1f77b4", linewidth=1.6, label="Active plugs")
        axes[0].set_ylabel("Vehicles")
        axes[0].set_title("Mid-corridor charging queue over the day")
        axes[0].legend()

        axes[1].plot(metrics["hour"], metrics["arrivals_total"], color="#2e7d32", linewidth=1.8, label="Charging arrivals")
        axes[1].plot(metrics["hour"], metrics["completions_total"], color="#8e44ad", linewidth=1.6, label="Charging completions")
        axes[1].set_xlabel("Hour of day")
        axes[1].set_ylabel("Vehicles / step")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping queue plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def _maybe_plot_daily_patterns(metrics: pd.DataFrame, path: Path) -> bool:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
        day_boundaries = sorted(metrics["global_hour"].loc[metrics["hour_of_day"] == 0.0].tolist())
        for boundary in day_boundaries[1:]:
            axes[0].axvline(boundary, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
            axes[1].axvline(boundary, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)

        axes[0].plot(metrics["global_hour"], metrics["queue_length"], color="#d62728", alpha=0.35, label="Queue")
        axes[0].plot(
            metrics["global_hour"],
            metrics["queue_length_rolling_1h"],
            color="#7f0000",
            linewidth=2.2,
            label="Queue (1h rolling)",
        )
        axes[0].set_ylabel("Vehicles")
        axes[0].set_title("Queue length across the full 3-day run")
        axes[0].legend()

        axes[1].plot(
            metrics["global_hour"],
            metrics["arrivals_total_rolling_1h"],
            color="#2ca02c",
            linewidth=1.8,
            label="Arrivals (1h rolling)",
        )
        axes[1].plot(
            metrics["global_hour"],
            metrics["starts_total_rolling_1h"],
            color="#1f77b4",
            linewidth=1.8,
            label="Charging starts (1h rolling)",
        )
        axes[1].plot(
            metrics["global_hour"],
            metrics["completions_total_rolling_1h"],
            color="#9467bd",
            linewidth=1.8,
            label="Completions (1h rolling)",
        )
        axes[1].set_ylabel("Vehicles / 5 min")
        axes[1].set_title("Smoothed charging flow with day boundaries")
        axes[1].legend()

        for day_index, day_frame in metrics.groupby("day_index", sort=True):
            axes[2].plot(
                day_frame["hour_of_day"],
                day_frame["queue_length_rolling_1h"],
                linewidth=2.0,
                label=f"Day {int(day_index) + 1}",
            )
        axes[2].set_xlabel("Hour of day")
        axes[2].set_ylabel("Queue (1h rolling)")
        axes[2].set_xlim(0.0, 24.0)
        axes[2].set_title("Daily queue profile by hour of day")
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True
    except Exception as exc:  # pragma: no cover - depends on local plotting stack
        LOGGER.warning("Skipping daily pattern plot because matplotlib is unavailable: %s", exc)
        if path.exists():
            path.unlink()
        return False


def main() -> None:
    parser = build_config_parser("Run a simple two-city corridor charging queue simulation.")
    args = parser.parse_args()
    config = load_config(args.config)

    configure_logging(config.get("logging", {}).get("level", "INFO"))
    seed = int(config.get("seed", 0))
    set_global_seed(seed)

    experiment_cfg = config["experiment"]
    output_dir = ensure_dir(Path(experiment_cfg["output_root"]) / experiment_cfg["name"] / "corridor_queue")
    run = init_wandb(config=config, job_type="simple_corridor_sim", run_name=f"{experiment_cfg['name']}_corridor_queue")

    simulator = SimpleCorridorQueueSimulator(config=config["simulation"], seed=seed)
    result = simulator.run()
    result.metrics = _add_derived_metrics(result.metrics, step_minutes=int(config["simulation"]["step_minutes"]))

    metrics_path = output_dir / "timestep_metrics.csv"
    summary_path = output_dir / "summary.json"
    plot_path = output_dir / "queue_dynamics.png"
    daily_plot_path = output_dir / "daily_patterns.png"

    result.metrics.to_csv(metrics_path, index=False)
    write_json(summary_path, result.summary)
    _maybe_plot_queue_dynamics(result.metrics, plot_path)
    _maybe_plot_daily_patterns(result.metrics, daily_plot_path)

    for column in result.metrics.columns:
        metric_name = f"sim/{column}"
        if column != "global_hour":
            run.define_metric(metric_name, step_metric="sim/global_hour")
    run.define_metric("summary/*")

    for row in result.metrics.to_dict(orient="records"):
        run.log({f"sim/{key}": value for key, value in row.items() if key != "time_label"}, step=int(row["step"]))
    run.log({f"summary/{key}": value for key, value in result.summary.items()})

    log_artifact(
        run=run,
        path=metrics_path,
        artifact_name=f"{experiment_cfg['name']}-corridor-queue-metrics",
        artifact_type="metrics",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=summary_path,
        artifact_name=f"{experiment_cfg['name']}-corridor-queue-summary",
        artifact_type="metrics",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=plot_path,
        artifact_name=f"{experiment_cfg['name']}-corridor-queue-plot",
        artifact_type="plot",
        aliases=["latest"],
    )
    log_artifact(
        run=run,
        path=daily_plot_path,
        artifact_name=f"{experiment_cfg['name']}-corridor-daily-patterns",
        artifact_type="plot",
        aliases=["latest"],
    )

    LOGGER.info("Saved queue simulation outputs to %s", output_dir)
    run.finish()


if __name__ == "__main__":
    main()

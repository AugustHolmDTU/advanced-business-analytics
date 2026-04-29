from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_LABELS = {
    "mobile_noop": "Fixed / no-agent",
    "mobile_threshold": "Threshold",
    "mobile_reactive": "Reactive",
}

POLICY_COLORS = {
    "mobile_noop": "#4C566A",
    "mobile_threshold": "#2A9D8F",
    "mobile_reactive": "#E76F51",
}


def set_report_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_demand_profile(frame: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame["hour_of_day"], frame["expected_charging_total"], color="#264653", linewidth=2.2, label="Total expected charging demand")
    axes[0].plot(frame["hour_of_day"], frame["expected_station_arrivals_ab"], color="#2A9D8F", linewidth=1.8, label="AB charging demand")
    axes[0].plot(frame["hour_of_day"], frame["expected_station_arrivals_bc"], color="#E76F51", linewidth=1.8, label="BC charging demand")
    axes[0].set_title("Simulation building block: expected daily charging-demand profile")
    axes[0].set_ylabel("Expected charging arrivals per 5-minute step")
    axes[0].legend(ncol=3, loc="upper right")

    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_ab"], linewidth=1.4, label="OD A→B")
    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_bc"], linewidth=1.4, label="OD B→C")
    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_ba"], linewidth=1.4, label="OD B→A")
    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_cb"], linewidth=1.4, label="OD C→B")
    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_ac"], linewidth=1.4, label="OD A→C")
    axes[1].plot(frame["hour_of_day"], frame["expected_passing_od_ca"], linewidth=1.4, label="OD C→A")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Expected passing vehicles per 5-minute step")
    axes[1].set_title("Underlying OD flow pattern that drives charging demand")
    axes[1].legend(ncol=3, loc="upper right")
    fig.tight_layout()
    return fig, axes


def plot_corridor_schematic() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.plot([0, 100, 200], [0, 0, 0], color="#264653", linewidth=3)
    for xpos, label, color in [
        (0, "City A", "#457B9D"),
        (50, "Station AB", "#2A9D8F"),
        (100, "City B / Depot", "#F4A261"),
        (150, "Station BC", "#E76F51"),
        (200, "City C", "#457B9D"),
    ]:
        ax.scatter([xpos], [0], s=220, color=color, zorder=3)
        ax.text(xpos, 0.18 if xpos in {0, 100, 200} else -0.24, label, ha="center", va="center", fontsize=11)
    ax.annotate("30 min depot-to-station travel", xy=(100, 0.04), xytext=(50, 0.55), arrowprops={"arrowstyle": "<->", "color": "#555555"}, ha="center")
    ax.annotate("30 min depot-to-station travel", xy=(100, 0.04), xytext=(150, 0.55), arrowprops={"arrowstyle": "<->", "color": "#555555"}, ha="center")
    ax.text(100, -0.55, "Mobile charging stations can be deployed to AB or BC,\nthen must return to the middle depot for recharge.", ha="center", fontsize=10)
    ax.set_title("A-B-C corridor layout and the role of mobile charging stations")
    ax.set_xlim(-15, 215)
    ax.set_ylim(-0.8, 0.8)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def _shade_disruptions(ax: plt.Axes, frame: pd.DataFrame) -> None:
    active = frame["disruption_active"].fillna(0).astype(int).to_numpy()
    hours = frame["global_hour"].to_numpy()
    start = None
    for idx, flag in enumerate(active):
        if flag == 1 and start is None:
            start = hours[idx]
        if flag == 0 and start is not None:
            ax.axvspan(start, hours[idx], color="#F4A261", alpha=0.18)
            start = None
    if start is not None:
        ax.axvspan(start, hours[-1], color="#F4A261", alpha=0.18)


def plot_policy_rollout_panel(run_frames: dict[str, pd.DataFrame]) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    baseline_frame = next(iter(run_frames.values()))
    for axis in axes:
        _shade_disruptions(axis, baseline_frame)
    for policy_name, frame in run_frames.items():
        color = POLICY_COLORS[policy_name]
        label = POLICY_LABELS[policy_name]
        axes[0].plot(frame["global_hour"], frame["queue_length"], color=color, linewidth=2.0, label=label)
        axes[1].plot(frame["global_hour"], frame["queue_wait_mean_minutes"], color=color, linewidth=2.0, label=label)
        axes[2].plot(frame["global_hour"], frame["num_active_mobile_stations"], color=color, linewidth=2.0, label=label)
    axes[0].set_title("Held-out rollout comparison on one unseen 5-day `test_id` scenario")
    axes[0].set_ylabel("Total queue length")
    axes[1].set_ylabel("Mean queue wait (minutes)")
    axes[2].set_ylabel("Active MCS")
    axes[2].set_xlabel("Global hour")
    axes[0].legend(ncol=3, loc="upper right")
    fig.tight_layout()
    return fig, axes


def plot_station_rollout_panel(run_frames: dict[str, pd.DataFrame]) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    baseline_frame = next(iter(run_frames.values()))
    for axis in axes.ravel():
        _shade_disruptions(axis, baseline_frame)
    for policy_name, frame in run_frames.items():
        color = POLICY_COLORS[policy_name]
        label = POLICY_LABELS[policy_name]
        axes[0, 0].plot(frame["global_hour"], frame["queue_length_station_ab"], color=color, linewidth=1.8, label=label)
        axes[0, 1].plot(frame["global_hour"], frame["queue_length_station_bc"], color=color, linewidth=1.8, label=label)
        axes[1, 0].plot(frame["global_hour"], frame["num_active_mobile_stations_station_ab"], color=color, linewidth=1.8, label=label)
        axes[1, 1].plot(frame["global_hour"], frame["num_active_mobile_stations_station_bc"], color=color, linewidth=1.8, label=label)
    axes[0, 0].set_title("Queue at station AB")
    axes[0, 1].set_title("Queue at station BC")
    axes[1, 0].set_title("Active MCS at AB")
    axes[1, 1].set_title("Active MCS at BC")
    for axis in axes[1, :]:
        axis.set_xlabel("Global hour")
    for axis in axes[:, 0]:
        axis.set_ylabel("Level")
    axes[0, 0].legend(ncol=3, loc="upper right")
    fig.tight_layout()
    return fig, axes


def plot_spatial_awareness(frame: pd.DataFrame, policy_name: str) -> tuple[plt.Figure, np.ndarray]:
    color = POLICY_COLORS[policy_name]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _shade_disruptions(axes[0, 0], frame)
    _shade_disruptions(axes[0, 1], frame)
    _shade_disruptions(axes[1, 0], frame)
    axes[0, 0].plot(frame["global_hour"], frame["num_active_mobile_stations_station_ab"], color="#2A9D8F", linewidth=2, label="MCS at AB")
    axes[0, 0].plot(frame["global_hour"], frame["num_active_mobile_stations_station_bc"], color="#E76F51", linewidth=2, label="MCS at BC")
    axes[0, 0].set_title(f"{POLICY_LABELS[policy_name]}: MCS allocation by station")
    axes[0, 0].legend()

    axes[0, 1].plot(frame["global_hour"], frame["queue_length_station_ab"], color="#2A9D8F", linewidth=2, label="Queue AB")
    axes[0, 1].plot(frame["global_hour"], frame["queue_length_station_bc"], color="#E76F51", linewidth=2, label="Queue BC")
    axes[0, 1].set_title("Local queue pressure by station")
    axes[0, 1].legend()

    axes[1, 0].plot(frame["global_hour"], frame["allocation_bias_ab_minus_bc"], color=color, linewidth=2, label="Allocation bias AB-BC")
    axes[1, 0].plot(frame["global_hour"], frame["local_deficit_bias_ab_minus_bc"], color="#264653", linewidth=1.6, alpha=0.85, label="Local deficit bias AB-BC")
    axes[1, 0].set_title("Allocation bias versus local deficit bias")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        frame["local_deficit_bias_ab_minus_bc"],
        frame["allocation_bias_ab_minus_bc"],
        s=12,
        alpha=0.35,
        color=color,
        edgecolors="none",
    )
    axes[1, 1].axhline(0, color="#999999", linewidth=1)
    axes[1, 1].axvline(0, color="#999999", linewidth=1)
    axes[1, 1].set_title("Scatter: local deficit bias vs allocation bias")
    axes[1, 1].set_xlabel("Local deficit bias (AB - BC)")
    axes[1, 1].set_ylabel("Allocation bias (AB - BC)")

    for axis in axes[1, :]:
        if axis is not axes[1, 1]:
            axis.set_xlabel("Global hour")
    for axis in axes[:, 0]:
        axis.set_ylabel("Level")
    fig.tight_layout()
    return fig, axes


def plot_reward_sweep(summary: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    scatter = axes[0].scatter(
        summary["final_mean_active_mobile_stations"],
        summary["final_mean_queue_length"],
        c=summary["active_mcs_cost"],
        s=90,
        cmap="viridis",
    )
    for _, row in summary.iterrows():
        axes[0].text(row["final_mean_active_mobile_stations"] + 0.03, row["final_mean_queue_length"] + 0.03, row["run"], fontsize=8)
    axes[0].set_title("Historical reward sweep: queue versus MCS usage")
    axes[0].set_xlabel("Final mean active mobile stations")
    axes[0].set_ylabel("Final mean queue length")
    fig.colorbar(scatter, ax=axes[0], label="Active MCS cost")

    axes[1].plot(summary["active_mcs_cost"], summary["final_mean_queue_length"], marker="o", linewidth=2, label="Mean queue length")
    axes[1].plot(summary["active_mcs_cost"], summary["final_mean_queue_wait_minutes"], marker="s", linewidth=2, label="Mean queue wait")
    axes[1].set_title("Higher mobile-station cost changes the learned operating point")
    axes[1].set_xlabel("Active MCS cost weight")
    axes[1].set_ylabel("Final episode metric")
    axes[1].legend()
    fig.tight_layout()
    return fig, axes


def plot_training_history(history: list[dict[str, Any]], source: str = "historical") -> tuple[plt.Figure, np.ndarray]:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(frame["episode"], frame["reward"], color="#264653", linewidth=2)
    axes[0, 0].set_title("Train reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 1].plot(frame["episode"], frame["loss"], color="#E76F51", linewidth=2)
    axes[0, 1].set_title("Train TD loss")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")

    has_eval_reward = "eval_mean_reward" in frame and frame["eval_mean_reward"].notna().any()
    has_eval_loss = "eval_td_loss" in frame and frame["eval_td_loss"].notna().any()

    if has_eval_reward:
        reward_frame = frame.loc[frame["eval_mean_reward"].notna(), ["episode", "eval_mean_reward"]]
        axes[1, 0].plot(reward_frame["episode"], reward_frame["eval_mean_reward"], color="#2A9D8F", linewidth=2, marker="o", markersize=4)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Reward")
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(
            0.5,
            0.55,
            "No evaluation reward series was available\nin the selected history artifact.",
            ha="center",
            va="center",
            fontsize=11,
        )
    axes[1, 0].set_title("Evaluation reward")

    if has_eval_loss:
        loss_frame = frame.loc[frame["eval_td_loss"].notna(), ["episode", "eval_td_loss"]]
        axes[1, 1].plot(loss_frame["episode"], loss_frame["eval_td_loss"], color="#F4A261", linewidth=2, marker="o", markersize=4)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Loss")
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.5,
            0.55,
            "No evaluation TD-loss series was available\nin the selected history artifact.",
            ha="center",
            va="center",
            fontsize=11,
        )
    axes[1, 1].set_title("Evaluation TD loss")

    if source == "submission_copy":
        title = "Current training-time RL diagnostics from exam_submission/data/generated/training/history_current.json"
    elif source == "current_outputs":
        title = "Current training-time RL diagnostics from outputs/mobile_mcs_line_abc/rl/history.json"
    else:
        title = "Historical RL diagnostics from a superseded absolute-allocation variant"
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_historical_training(history: list[dict[str, Any]]) -> tuple[plt.Figure, np.ndarray]:
    return plot_training_history(history, source="historical")

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_LABELS = {
    "mobile_noop": "No-agent baseline",
    "mobile_threshold": "RL agent",
    "mobile_reactive": "Reactive",
    "fixed": "No-agent baseline",
    "threshold": "RL agent",
    "reactive": "Reactive",
    "rl": "RL agent",
}

POLICY_COLORS = {
    "mobile_noop": "#CC79A7",
    "mobile_threshold": "#0072B2",
    "mobile_reactive": "#E76F51",
    "fixed": "#CC79A7",
    "threshold": "#0072B2",
    "reactive": "#E76F51",
    "rl": "#0072B2",
}

POLICY_LINESTYLES = {
    "mobile_noop": "--",
    "mobile_threshold": "-",
    "mobile_reactive": "-.",
    "fixed": "--",
    "threshold": "-",
    "reactive": "-.",
    "rl": "-",
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


def plot_policy_rollout_panel(run_frames: dict[str, pd.DataFrame], title: str | None = None) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    baseline_frame = next(iter(run_frames.values()))
    # all policies use the same seeded scenario here, so one disruption mask is enough
    for axis in axes:
        _shade_disruptions(axis, baseline_frame)
    for policy_name, frame in run_frames.items():
        color = POLICY_COLORS[policy_name]
        label = POLICY_LABELS[policy_name]
        linestyle = POLICY_LINESTYLES.get(policy_name, "-")
        axes[0].plot(frame["global_hour"], frame["queue_length"], color=color, linestyle=linestyle, linewidth=2.4, label=label)
        axes[1].plot(frame["global_hour"], frame["queue_wait_mean_minutes"], color=color, linestyle=linestyle, linewidth=2.4, label=label)
        axes[2].plot(frame["global_hour"], frame["num_active_mobile_stations"], color=color, linestyle=linestyle, linewidth=2.4, label=label)
    axes[0].set_ylabel("Total queue length")
    axes[1].set_ylabel("Mean queue wait (minutes)")
    axes[2].set_ylabel("Active MCS")
    axes[2].set_xlabel("Global hour")
    axes[0].legend(ncol=max(1, min(3, len(run_frames))), loc="upper right")
    if title is not None:
        axes[0].set_title(title)
    elif len(run_frames) == 1:
        policy_name = next(iter(run_frames))
        axes[0].set_title(f"Held-out rollout for {POLICY_LABELS.get(policy_name, policy_name)}")
    else:
        axes[0].set_title("Held-out rollout comparison on one unseen 5-day `test_id` scenario")
    fig.tight_layout()
    return fig, axes


def plot_training_history(history: list[dict[str, Any]], source: str = "historical") -> tuple[plt.Figure, np.ndarray]:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True)
    axes[0].plot(frame["episode"], frame["reward"], color="#264653", linewidth=2.1, label="Train reward")
    axes[0].set_title("Train and evaluation reward")
    axes[0].set_ylabel("Reward")

    # some older histories do not have periodic eval metrics, so this should fail soft
    has_eval_reward = "eval_mean_reward" in frame and frame["eval_mean_reward"].notna().any()
    has_eval_loss = "eval_td_loss" in frame and frame["eval_td_loss"].notna().any()

    if has_eval_reward:
        reward_frame = frame.loc[frame["eval_mean_reward"].notna(), ["episode", "eval_mean_reward"]]
        axes[0].plot(
            reward_frame["episode"],
            reward_frame["eval_mean_reward"],
            color="#2A9D8F",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Eval reward",
        )
    else:
        axes[0].text(
            0.5,
            0.12,
            "No evaluation reward series was available in the selected history artifact.",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[0].transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb"},
        )
    axes[0].legend(loc="upper right")

    axes[1].plot(frame["episode"], frame["loss"], color="#E76F51", linewidth=2.1, label="Train TD loss")
    axes[1].set_title("Train and evaluation TD loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    if has_eval_loss:
        loss_frame = frame.loc[frame["eval_td_loss"].notna(), ["episode", "eval_td_loss"]]
        axes[1].plot(
            loss_frame["episode"],
            loss_frame["eval_td_loss"],
            color="#F4A261",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Eval TD loss",
        )
    else:
        axes[1].text(
            0.5,
            0.12,
            "No evaluation TD-loss series was available in the selected history artifact.",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[1].transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb"},
        )
    axes[1].legend(loc="upper right")

    if source == "submission_copy":
        title = "Current training-time RL diagnostics from exam_submission/data/generated/training/training_history.json"
    elif source == "current_outputs":
        title = "Current training-time RL diagnostics from outputs/mobile_mcs_line_abc/rl/history.json"
    elif source == "appendix_demo":
        title = "Demo training reward and loss"
    else:
        title = "Historical RL diagnostics from a superseded absolute-allocation variant"
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_disruption_scenario(
    frame: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    rolling_steps: int = 3,
) -> tuple[plt.Figure, np.ndarray]:
    hours = frame["global_hour"].to_numpy()
    demand = frame["expected_station_arrivals_ab"].to_numpy() + frame["expected_station_arrivals_bc"].to_numpy()
    queue = pd.Series(frame["queue_length"]).rolling(rolling_steps, min_periods=1).mean().to_numpy()
    wait = pd.Series(frame["queue_wait_mean_minutes"]).rolling(rolling_steps, min_periods=1).mean().to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.6), sharex=True)
    for axis in axes:
        _shade_disruptions(axis, frame)

    queue_ax = axes[0]
    queue_ax.plot(hours, demand, color="#264653", linewidth=2.2, label="Expected charging demand")
    queue_ax.set_ylabel("Expected demand", color="#264653")
    queue_ax.tick_params(axis="y", labelcolor="#264653")

    queue_ax_right = queue_ax.twinx()
    queue_ax_right.plot(hours, queue, color="#E76F51", linewidth=2.0, label="Queue length")
    queue_ax_right.set_ylabel("Queue length", color="#E76F51")
    queue_ax_right.tick_params(axis="y", labelcolor="#E76F51")

    wait_ax = axes[1]
    wait_ax.plot(hours, demand, color="#264653", linewidth=2.2, label="Expected charging demand")
    wait_ax.set_ylabel("Expected demand", color="#264653")
    wait_ax.tick_params(axis="y", labelcolor="#264653")

    wait_ax_right = wait_ax.twinx()
    wait_ax_right.plot(hours, wait, color="#2A9D8F", linewidth=2.0, label="Queue wait")
    wait_ax_right.set_ylabel("Mean queue wait (min)", color="#2A9D8F")
    wait_ax_right.tick_params(axis="y", labelcolor="#2A9D8F")
    wait_ax.set_xlabel("Hour of day")

    handles = [
        plt.Line2D([0], [0], color="#264653", linewidth=2.2, label="Expected charging demand"),
        plt.Line2D([0], [0], color="#E76F51", linewidth=2.0, label="Queue length"),
        plt.Line2D([0], [0], color="#2A9D8F", linewidth=2.0, label="Queue wait"),
        plt.Rectangle((0, 0), 1, 1, fc="#F4A261", alpha=0.18, label="Disruption window"),
    ]
    fig.suptitle(title, y=0.965, fontsize=14)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=True,
        borderaxespad=0.4,
    )
    if subtitle:
        axes[1].text(0.0, -0.42, subtitle, transform=axes[1].transAxes, fontsize=10, color="#444444")
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.10, right=0.90, hspace=0.16)
    return fig, axes

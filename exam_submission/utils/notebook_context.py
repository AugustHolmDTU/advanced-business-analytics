from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

from exam_submission.utils.demo_runner import run_appendix_demo
from exam_submission.utils.load_artifacts import (
    load_core_configs,
    load_generated_reward_sweeps,
    load_heldout_runs,
    load_training_history,
)
from exam_submission.utils.plotting import (
    plot_corridor_schematic,
    plot_demand_profile,
    plot_disruption_scenario,
    plot_training_history,
    set_report_style,
)
from exam_submission.utils.project_summary import (
    action_table,
    environment_table,
    expected_daily_profile,
    observation_table,
    reward_table,
)


def _repo_root() -> Path:
    return next(
        path.parent if path.name == "exam_submission" else path
        for path in [Path.cwd(), *Path.cwd().parents]
        if path.name == "exam_submission" or (path / "exam_submission").exists()
    )


def _activate_submission_src(repo_root: Path) -> None:
    submission_src = repo_root / "exam_submission" / "src"
    submission_src_str = str(submission_src)
    if submission_src_str not in sys.path:
        sys.path.insert(0, submission_src_str)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace(chr(10), "<br>")


def _markdown_table(frame: pd.DataFrame, index: bool = False) -> str:
    columns = list(frame.columns)
    if index:
        columns = ["index", *columns]
    header = "| " + " | ".join(_markdown_escape(col) for col in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        values = row.tolist()
        if index:
            values = [frame.index[idx], *values]
        rows.append("| " + " | ".join(_markdown_escape(value) for value in values) + " |")
    return chr(10).join([header, divider, *rows])


def show_table(df: pd.DataFrame, precision: int | None = 2, index: bool = False) -> None:
    frame = df.copy()
    numeric_cols = frame.select_dtypes(include="number").columns
    if precision is not None and len(numeric_cols) > 0:
        frame.loc[:, numeric_cols] = frame.loc[:, numeric_cols].round(precision)
    display(Markdown(_markdown_table(frame, index=index)))


def rolling_mean(series: Any, window_steps: int = 12) -> pd.Series:
    return pd.Series(series).rolling(window_steps, min_periods=1).mean()


def shade_disruptions(ax: plt.Axes, frame: pd.DataFrame) -> None:
    active = frame["disruption_active"].fillna(0).astype(int).to_numpy()
    hours = frame["global_hour"].to_numpy()
    start = None
    for idx, flag in enumerate(active):
        if flag == 1 and start is None:
            start = hours[idx]
        if flag == 0 and start is not None:
            ax.axvspan(start, hours[idx], color="#F4A261", alpha=0.16)
            start = None
    if start is not None:
        ax.axvspan(start, hours[-1], color="#F4A261", alpha=0.16)


def bootstrap_notebook(namespace: dict[str, Any] | None = None) -> dict[str, Any]:
    repo_root = _repo_root()
    _activate_submission_src(repo_root)

    from evch.sim.line_corridor import LineCorridorQueueSimulator

    set_report_style()
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 160)

    configs = load_core_configs()
    env_cfg = configs["environment"]
    rl_simple_cfg = configs["rl_simple"]
    rl_agent_cfg = configs["rl_agent"]
    experiment_cfg = configs["experiment"]
    heldout_cfg = configs["heldout"]
    generated_reward_sweeps = load_generated_reward_sweeps()
    training_history, training_history_source = load_training_history()
    daily_profile = expected_daily_profile(env_cfg)
    heldout_frames = load_heldout_runs()

    disruption_case_specs = {
        "capacity_drop": {
            "title": "Capacity drop",
            "disruption_type": "capacity_drop",
            "target": "station_ab",
            "start_hour": 8.0,
            "duration_hours": 2.0,
            "severity": 4.0,
            "description": "Representative service-capacity shock: AB base capacity falls from 12 plugs to 4 during the morning peak.",
        },
        "station_outage": {
            "title": "Station outage",
            "disruption_type": "station_outage",
            "target": "station_ab",
            "start_hour": 8.5,
            "duration_hours": 1.5,
            "severity": 0.0,
            "description": "Representative hard failure: AB is forced fully offline for 90 minutes during the peak build-up.",
        },
        "service_time_inflation": {
            "title": "Service-time inflation",
            "disruption_type": "service_time_inflation",
            "target": "station_ab",
            "start_hour": 8.0,
            "duration_hours": 2.5,
            "severity": 1.8,
            "description": "Representative process slowdown: vehicles at AB occupy plugs 1.8x longer than normal.",
        },
        "demand_surge": {
            "title": "Demand surge",
            "disruption_type": "demand_surge",
            "target": "eastbound",
            "start_hour": 7.5,
            "duration_hours": 2.0,
            "severity": 2.2,
            "description": "Representative demand shock: eastbound OD flows are multiplied by 2.2 during the morning rush.",
        },
    }

    def run_scripted_disruption_case(case_key: str, seed: int = 7) -> pd.DataFrame:
        spec = disruption_case_specs[case_key]
        sim_cfg = copy.deepcopy(env_cfg["environment"]["simulation"])
        sim_cfg["duration_hours"] = 24.0
        sim_cfg["disruption"] = {
            "enabled": True,
            "mode": "scripted",
            "event_types": [spec["disruption_type"]],
            "scripted_events": [
                {
                    "disruption_type": spec["disruption_type"],
                    "target": spec["target"],
                    "day_index": 0,
                    "start_hour": spec["start_hour"],
                    "duration_hours": spec["duration_hours"],
                    "severity": spec["severity"],
                }
            ],
        }
        simulator = LineCorridorQueueSimulator(sim_cfg, seed=seed)
        return simulator.run().metrics

    disruption_case_runs = {
        case_key: run_scripted_disruption_case(case_key, seed=7 + idx)
        for idx, case_key in enumerate(disruption_case_specs)
    }

    main_policies = ["mobile_noop", "mobile_threshold"]
    main_policy_labels = {"mobile_noop": "No-agent baseline", "mobile_threshold": "RL agent"}
    main_colors = {"mobile_noop": "#4C566A", "mobile_threshold": "#2A9D8F"}
    all_policy_labels = {
        "mobile_noop": "No-agent baseline",
        "mobile_threshold": "RL agent",
        "mobile_reactive": "Reactive",
    }
    all_colors = {
        "mobile_noop": "#4C566A",
        "mobile_threshold": "#2A9D8F",
        "mobile_reactive": "#E76F51",
    }

    def build_environment_overview_table() -> pd.DataFrame:
        base = environment_table(env_cfg).copy()
        sim = env_cfg["environment"]["simulation"]
        chargers = int(env_cfg["environment"]["mobile_station_chargers"])
        mean_service = float(sim["service_time"]["mean_minutes"])
        vehicles_per_hour = chargers * 60.0 / mean_service
        extra_rows = pd.DataFrame(
            [
                ("Mean MCS throughput", f"{vehicles_per_hour:.1f} vehicles/hour per MCS (~{vehicles_per_hour / 12:.2f} vehicles per 5-minute step)"),
                ("Depot-to-station travel", f"{float(env_cfg['environment'].get('mcs_middle_travel_minutes', 30.0)):.0f} minutes"),
                ("Full MCS recharge time", f"{float(env_cfg['environment'].get('mcs_charge_full_minutes', 60.0)):.0f} minutes"),
            ],
            columns=["Setting", "Value"],
        )
        return pd.concat([base, extra_rows], ignore_index=True)

    def build_state_group_table() -> pd.DataFrame:
        rows = [
            ("Queue state", 4, "Current queue length and current mean wait at AB and BC."),
            ("Recent flow", 4, "Latest arrivals and latest service starts at each station."),
            ("Capacity state", 4, "Effective plugs and active mobile capacity at AB and BC."),
            ("Spatial pressure", 4, "Committed MCS bias and local deficit estimates by side of corridor."),
            ("Disruption state", 4, "Whether a disruption is active, what type it is, and which side it affects."),
            ("MCS logistics", 5, "Units available in depot, charging, or in transit to AB, BC, or middle."),
            ("Time-of-day", 2, "Sine and cosine encodings of time within the day."),
        ]
        return pd.DataFrame(rows, columns=["Feature group", "Count", "Why it is in the state"])

    def build_reward_term_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        positive = reward_table(env_cfg, positive=True).rename(
            columns={"Term": "Positive term", "Weight": "Weight", "Interpretation": "Operational role"}
        )
        negative = reward_table(env_cfg, positive=False).rename(
            columns={"Term": "Negative term", "Weight": "Weight", "Interpretation": "Operational role"}
        )
        return positive, negative

    def plot_generated_reward_sweep(frame: pd.DataFrame, x_label: str, title: str, color: str) -> tuple[plt.Figure, Any]:
        prepared = frame.copy() if frame is not None else pd.DataFrame()
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
        if prepared.empty:
            for ax in axes:
                ax.axis("off")
                ax.text(0.5, 0.5, "Sweep CSV missing or empty.", ha="center", va="center", fontsize=11)
            return fig, axes

        prepared = prepared.sort_values("reward_value").reset_index(drop=True)
        x = prepared["reward_value"]
        axes[0].errorbar(
            x,
            prepared["mean_queue_length_mean"],
            yerr=prepared["mean_queue_length_std"],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            capsize=4,
        )
        axes[0].set_title(f"{title}: mean queue length")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("Mean queue length")

        axes[1].errorbar(
            x,
            prepared["mean_queue_wait_minutes_mean"],
            yerr=prepared["mean_queue_wait_minutes_std"],
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
            capsize=4,
        )
        axes[1].set_title(f"{title}: mean queue wait")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Mean queue wait (min)")

        for ax in axes:
            ax.set_xticks(x.tolist())
            ax.grid(True, alpha=0.35)

        if "num_rollouts" in prepared.columns and prepared["num_rollouts"].nunique() == 1:
            fig.suptitle(f"{title} ({int(prepared['num_rollouts'].iloc[0])} held-out seeds per point)", y=1.02, fontsize=13)
        fig.tight_layout()
        return fig, axes

    def build_training_setup_table() -> pd.DataFrame:
        simple = rl_simple_cfg["rl"]
        rows = [
            ("Training episodes", "Randomized 2-6 day episodes sampled from training seeds 0-9999."),
            ("Periodic evaluation", f"{simple['eval_during_training_episodes']} held-out episodes every {simple['eval_interval_steps']} environment steps without exploration or learning updates."),
            ("Validation split", "Configured held-out validation seeds starting at 10000; useful for policy checks even without hyperparameter tuning."),
            ("Held-out rollout", "A fixed unseen `test_id` scenario used for the main time-series comparison in this report."),
            ("Test stress suite", "Named one-day scripted disruptions used for robustness checks; kept in the appendix to avoid clutter."),
        ]
        return pd.DataFrame(rows, columns=["Component", "How it is used in this report"])

    def build_main_results_table() -> pd.DataFrame:
        fixed = heldout_frames["mobile_noop"]
        adaptive = heldout_frames["mobile_threshold"]
        rows = []
        for key, frame in [("mobile_noop", fixed), ("mobile_threshold", adaptive)]:
            rows.append(
                {
                    "Policy": main_policy_labels[key],
                    "Mean queue": frame["queue_length"].mean(),
                    "Peak queue": frame["queue_length"].max(),
                    "Mean wait (min)": frame["queue_wait_mean_minutes"].mean(),
                    "Peak wait (min)": frame["queue_wait_mean_minutes"].max(),
                    "Breach rate": frame["queue_wait_target_breached"].mean(),
                    "Started sessions": frame["starts_total"].sum(),
                    "Mean active MCS": frame["num_active_mobile_stations"].mean(),
                    "MCS hours": frame["num_active_mobile_stations"].sum() * 5 / 60,
                    "Reward": frame["reward"].sum(),
                }
            )
        table = pd.DataFrame(rows)
        fixed_row = table.iloc[0]
        table["Queue improvement vs fixed (%)"] = [
            0.0,
            100.0 * (fixed_row["Mean queue"] - table.iloc[1]["Mean queue"]) / fixed_row["Mean queue"],
        ]
        table["Wait improvement vs fixed (%)"] = [
            0.0,
            100.0 * (fixed_row["Mean wait (min)"] - table.iloc[1]["Mean wait (min)"]) / fixed_row["Mean wait (min)"],
        ]
        table["Reward change vs fixed"] = [0.0, table.iloc[1]["Reward"] - fixed_row["Reward"]]
        return table

    def build_observation_appendix_table() -> pd.DataFrame:
        return observation_table().rename(columns={"Index": "Idx"})

    def plot_main_rollout_comparison(run_frames: dict[str, pd.DataFrame], policy_names: list[str] | None = None) -> tuple[plt.Figure, Any]:
        active_policies = main_policies if policy_names is None else policy_names
        fig, axes = plt.subplots(3, 1, figsize=(12, 8.6), sharex=True)
        base_frame = run_frames[active_policies[0]]
        for ax in axes:
            shade_disruptions(ax, base_frame)
        metrics = [
            ("queue_length", "Total queue length"),
            ("queue_wait_mean_minutes", "Mean queue wait (minutes)"),
            ("num_active_mobile_stations", "Active mobile stations"),
        ]
        for policy_name in active_policies:
            frame = run_frames[policy_name]
            for ax, (col, ylabel) in zip(axes, metrics):
                ax.plot(frame["global_hour"], frame[col], color=main_colors[policy_name], linewidth=2.0, label=main_policy_labels[policy_name])
                ax.set_ylabel(ylabel)
        axes[0].set_title("Reproducible held-out rollout: no-agent baseline versus RL agent")
        axes[2].set_xlabel("Global hour")
        axes[0].legend(loc="upper right", ncol=2)
        fig.tight_layout()
        return fig, axes

    def plot_spatial_station_overlays(frame: pd.DataFrame, rolling_steps: int = 12) -> tuple[plt.Figure, Any]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), sharex=True)
        station_specs = [
            ("ab", "AB station", "#2A9D8F"),
            ("bc", "BC station", "#E76F51"),
        ]
        for ax, (suffix, title, color) in zip(axes, station_specs):
            shade_disruptions(ax, frame)
            queue = rolling_mean(frame[f"queue_length_station_{suffix}"], rolling_steps)
            mcs = rolling_mean(frame[f"num_active_mobile_stations_station_{suffix}"], rolling_steps)
            ax.plot(frame["global_hour"], queue, color=color, linewidth=2.0, label="Queue length (1h avg)")
            ax.set_ylabel("Queue length (1h avg)")
            ax.set_title(title)
            twin = ax.twinx()
            twin.plot(frame["global_hour"], mcs, color="#8D5524", linewidth=2.0, label="MCS allocation (1h avg)")
            twin.set_ylabel("Allocated MCS")
            lines, labels = ax.get_legend_handles_labels()
            twin_lines, twin_labels = twin.get_legend_handles_labels()
            ax.legend(lines + twin_lines, labels + twin_labels, loc="upper right")
        axes[-1].set_xlabel("Global hour")
        fig.suptitle("Spatial-awareness check for the RL agent", y=1.02, fontsize=14)
        fig.tight_layout()
        return fig, axes

    context = {
        "plt": plt,
        "display": display,
        "run_appendix_demo": run_appendix_demo,
        "action_table": action_table,
        "plot_corridor_schematic": plot_corridor_schematic,
        "plot_demand_profile": plot_demand_profile,
        "plot_disruption_scenario": plot_disruption_scenario,
        "plot_training_history": plot_training_history,
        "daily_profile": daily_profile,
        "disruption_case_runs": disruption_case_runs,
        "generated_reward_sweeps": generated_reward_sweeps,
        "training_history": training_history,
        "training_history_source": training_history_source,
        "heldout_frames": heldout_frames,
        "MAIN_POLICIES": main_policies,
        "show_table": show_table,
        "build_environment_overview_table": build_environment_overview_table,
        "build_state_group_table": build_state_group_table,
        "build_reward_term_tables": build_reward_term_tables,
        "plot_generated_reward_sweep": plot_generated_reward_sweep,
        "build_training_setup_table": build_training_setup_table,
        "build_main_results_table": build_main_results_table,
        "plot_main_rollout_comparison": plot_main_rollout_comparison,
        "plot_spatial_station_overlays": plot_spatial_station_overlays,
        "build_observation_appendix_table": build_observation_appendix_table,
    }

    if namespace is not None:
        namespace.update(context)
    return context

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


OBSERVATION_FEATURES = [
    ("Queue state", 1, "queue_length_ab", "Current queue length at station AB."),
    ("Queue state", 2, "queue_length_bc", "Current queue length at station BC."),
    ("Queue state", 3, "mean_queue_wait_ab", "Mean current waiting time in the AB queue."),
    ("Queue state", 4, "mean_queue_wait_bc", "Mean current waiting time in the BC queue."),
    ("Recent flow", 5, "last_arrivals_ab", "Vehicles that arrived to AB in the last step."),
    ("Recent flow", 6, "last_arrivals_bc", "Vehicles that arrived to BC in the last step."),
    ("Recent flow", 7, "last_starts_ab", "Charging sessions started at AB in the last step."),
    ("Recent flow", 8, "last_starts_bc", "Charging sessions started at BC in the last step."),
    ("Capacity", 9, "effective_plugs_ab", "Effective plugs at AB after disruption and MCS effects."),
    ("Capacity", 10, "effective_plugs_bc", "Effective plugs at BC after disruption and MCS effects."),
    ("Capacity", 11, "active_mobile_capacity_ab", "Currently active mobile-station capacity at AB."),
    ("Capacity", 12, "active_mobile_capacity_bc", "Currently active mobile-station capacity at BC."),
    ("Spatial", 13, "committed_mcs_bias_ab_minus_bc", "Committed MCS bias toward AB versus BC."),
    ("Spatial", 14, "local_deficit_ab", "Local deficit estimate at AB."),
    ("Spatial", 15, "local_deficit_bc", "Local deficit estimate at BC."),
    ("Spatial", 16, "local_deficit_bias_ab_minus_bc", "Local deficit bias toward AB versus BC."),
    ("Disruption", 17, "disruption_active", "Whether a disruption is currently active."),
    ("Disruption", 18, "disruption_type_code", "Encoded disruption type."),
    ("Disruption", 19, "target_affects_ab", "Whether the disruption affects AB."),
    ("Disruption", 20, "target_affects_bc", "Whether the disruption affects BC."),
    ("MCS logistics", 21, "middle_available", "Mobile stations available at the middle depot."),
    ("MCS logistics", 22, "middle_charging", "Mobile stations currently recharging at the middle depot."),
    ("MCS logistics", 23, "transit_to_ab", "Mobile stations moving toward AB."),
    ("MCS logistics", 24, "transit_to_bc", "Mobile stations moving toward BC."),
    ("MCS logistics", 25, "transit_to_middle", "Mobile stations returning to the middle depot."),
    ("Time", 26, "sin_time_of_day", "Sine encoding of time of day."),
    ("Time", 27, "cos_time_of_day", "Cosine encoding of time of day."),
]

ACTION_SPACE = [
    ("hold", "Keep the committed mobile-station allocation unchanged."),
    ("toward_ab", "Move one unit of commitment toward AB, either from depot or from BC."),
    ("toward_bc", "Move one unit of commitment toward BC, either from depot or from AB."),
    ("recall_ab", "Recall one committed unit away from AB toward the middle depot."),
    ("recall_bc", "Recall one committed unit away from BC toward the middle depot."),
]

POSITIVE_REWARD_TERMS = [
    ("served_reward_weight", "Rewards charging sessions that start service."),
    ("utilization_bonus", "Rewards productive use of available charging capacity."),
    ("spatial_deficit_alignment_bonus", "Rewards mobile capacity that covers estimated local deficits."),
    ("spatial_deficit_direction_bonus", "Rewards sending MCS in the correct corridor direction."),
]

NEGATIVE_REWARD_TERMS = [
    ("unmet_penalty", "Penalizes vehicles left in queue after the step."),
    ("queue_length_penalty", "Adds extra pressure against persistent queue accumulation."),
    ("queue_wait_penalty", "Penalizes queue-wait burden, not just queue count."),
    ("local_queue_peak_penalty", "Penalizes the worst station queue in the step."),
    ("local_wait_peak_penalty", "Penalizes the worst station waiting time in the step."),
    ("active_mobile_station_cost", "Charges for keeping MCS active in the field."),
    ("activation_cost", "Charges for newly activating MCS."),
    ("adjustment_cost", "Charges for reallocating MCS commitment."),
    ("idle_capacity_penalty", "Penalizes carrying unused capacity."),
]

WHAT_MODEL_DOES_NOT_KNOW = [
    "No exact future arrivals or exact future queue realizations.",
    "No oracle action labels or optimal allocation target.",
    "No foreknowledge of future random disruptions before they begin.",
    "No exact per-MCS battery state-of-charge representation.",
    "No direct remaining-disruption-time feature in the current observation.",
    "No explicit demand forecast beyond current local state and disruption indicators.",
]


def observation_table() -> pd.DataFrame:
    return pd.DataFrame(OBSERVATION_FEATURES, columns=["Group", "Index", "Feature", "Meaning"])


def action_table() -> pd.DataFrame:
    return pd.DataFrame(ACTION_SPACE, columns=["Action", "Interpretation"])


def reward_table(env_cfg: dict[str, Any], positive: bool = True) -> pd.DataFrame:
    reward_cfg = env_cfg["environment"]["reward"]
    source = POSITIVE_REWARD_TERMS if positive else NEGATIVE_REWARD_TERMS
    return pd.DataFrame(
        [
            {"Term": key, "Weight": reward_cfg.get(key, np.nan), "Interpretation": description}
            for key, description in source
        ]
    )


def environment_table(env_cfg: dict[str, Any]) -> pd.DataFrame:
    environment = env_cfg["environment"]
    simulation = environment["simulation"]
    traffic = simulation["traffic"]
    disruption = simulation["disruption"]
    rows = [
        ("Cities", "A-B-C synthetic corridor"),
        ("Inter-city distance", f"{simulation['inter_city_distance_km']} km"),
        ("Fixed stations", "2 (one between A-B and one between B-C)"),
        ("Fixed plugs", f"{simulation['num_plugs'][0]} at AB, {simulation['num_plugs'][1]} at BC"),
        ("Mobile stations", f"Up to {environment['max_mobile_stations']} units"),
        ("Chargers per MCS", environment["mobile_station_chargers"]),
        ("Decision step", f"{simulation['step_minutes']} minutes"),
        ("Episode duration", f"{simulation['duration_days_range'][0]}-{simulation['duration_days_range'][1]} days during training"),
        ("Mean service time", f"{simulation['service_time']['mean_minutes']} minutes"),
        ("Morning peak", f"{traffic['morning_peak_hour']}:00"),
        ("Evening peak", f"{traffic['evening_peak_hour']}:00"),
        ("Disruption mode", disruption["mode"]),
        ("Supported disruption types", ", ".join(disruption["event_types"])),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def rl_table(configs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    simple = configs["rl_simple"]["rl"]
    agent = configs.get("rl_agent", {}).get("rl", {})
    rows = [
        ("Reported model name", "RL agent"),
        ("Available training backend", simple["backend"]),
        ("Training episodes", simple["episodes"]),
        ("Discount factor gamma", simple["gamma"]),
        ("Learning rate", simple["learning_rate"]),
        ("Batch size", simple["batch_size"]),
        ("Replay capacity", simple["replay_capacity"]),
        ("Learning starts", simple["learning_starts"]),
        ("Train frequency", simple["train_frequency"]),
        ("Gradient steps", simple["gradient_steps"]),
        ("Target update interval", simple["target_update_interval"]),
        ("Hidden dimensions", str(simple["hidden_dims"])),
        ("Exploration start", simple["epsilon_start"]),
        ("Exploration end", simple["epsilon_end"]),
        ("Exploration decay steps", simple["epsilon_decay_steps"]),
        ("Periodic evaluation cadence", f"Every {simple['eval_interval_steps']} training steps"),
        ("Periodic evaluation episodes", simple["eval_during_training_episodes"]),
        ("Final evaluation episodes", simple["evaluation_episodes"]),
    ]
    if agent:
        agent_inner = agent.get("agent", {})
        rows.extend(
            [
                ("RL agent total timesteps", agent_inner.get("total_timesteps", np.nan)),
                ("RL agent replay buffer", agent_inner.get("buffer_size", np.nan)),
                ("RL agent exploration fraction", agent_inner.get("exploration_fraction", np.nan)),
            ]
        )
    return pd.DataFrame(rows, columns=["Hyperparameter", "Value"])


def split_table(experiment_cfg: dict[str, Any]) -> pd.DataFrame:
    split = experiment_cfg["train_val_test"]
    rows = [
        ("Training", "Randomized 2-6 day episodes from seeds 0-9999"),
        ("Validation", "Held-out same-distribution seeds starting at 10000"),
        ("Test ID", "Unseen deployment-style seeds starting at 20000 with 3-5 day overrides"),
        ("Test stress", "Named scripted disruption scenarios"),
        ("Held-out rollout", "One fixed unseen test_id seed for direct time-series comparison"),
    ]
    return pd.DataFrame(rows, columns=["Split", "Purpose"])


def expected_daily_profile(env_cfg: dict[str, Any], steps_per_day: int = 288) -> pd.DataFrame:
    simulation = env_cfg["environment"]["simulation"]
    traffic = simulation["traffic"]
    stop_probability = float(simulation["charging_stop_probability"])
    long_trip_share = float(traffic["long_trip_share"])
    middle_city_share = float(traffic["middle_city_share"])
    directional_bias = float(traffic["directional_bias_amplitude"])
    middle_dest_bias = float(traffic["middle_destination_bias_amplitude"])
    peak_width = float(traffic["peak_width_hours"])
    baseline = float(traffic["baseline_cars_per_step"])
    morning_hour = float(traffic["morning_peak_hour"])
    evening_hour = float(traffic["evening_peak_hour"])
    midday_hour = float(traffic["midday_bump_hour"])
    morning_amp = float(traffic["morning_peak_cars_per_step"])
    evening_amp = float(traffic["evening_peak_cars_per_step"])
    midday_amp = float(traffic["midday_bump_cars_per_step"])
    step_minutes = float(simulation["step_minutes"])

    def gaussian(hour: float, center: float, amplitude: float) -> float:
        return amplitude * math.exp(-((hour - center) ** 2) / (2.0 * peak_width**2))

    def origin_shares(hour_of_day: float) -> np.ndarray:
        morning_signal = math.exp(-((hour_of_day - morning_hour) ** 2) / (2.0 * peak_width**2))
        evening_signal = math.exp(-((hour_of_day - evening_hour) ** 2) / (2.0 * peak_width**2))
        midday_signal = math.exp(-((hour_of_day - midday_hour) ** 2) / (2.0 * peak_width**2))
        edge_share = max(1.0 - middle_city_share, 1e-6)
        scores = np.asarray(
            [
                edge_share * (1.0 + directional_bias * (morning_signal - evening_signal)),
                middle_city_share * (1.0 + 0.35 * midday_signal),
                edge_share * (1.0 + directional_bias * (evening_signal - morning_signal)),
            ],
            dtype=float,
        )
        scores = np.clip(scores, 1e-6, None)
        return scores / scores.sum()

    rows: list[dict[str, float]] = []
    for step in range(steps_per_day):
        hour_of_day = step * step_minutes / 60.0
        total_flow = max(
            baseline
            + gaussian(hour_of_day, morning_hour, morning_amp)
            + gaussian(hour_of_day, evening_hour, evening_amp)
            + gaussian(hour_of_day, midday_hour, midday_amp),
            0.0,
        )
        shares = origin_shares(hour_of_day)
        middle_to_east = float(
            np.clip(
                0.5
                + middle_dest_bias
                * (
                    math.exp(-((hour_of_day - morning_hour) ** 2) / (2.0 * peak_width**2))
                    - math.exp(-((hour_of_day - evening_hour) ** 2) / (2.0 * peak_width**2))
                ),
                0.15,
                0.85,
            )
        )
        expected_by_trip = {
            "od_ab": total_flow * shares[0] * (1.0 - long_trip_share),
            "od_ac": total_flow * shares[0] * long_trip_share,
            "od_ba": total_flow * shares[1] * (1.0 - middle_to_east),
            "od_bc": total_flow * shares[1] * middle_to_east,
            "od_ca": total_flow * shares[2] * long_trip_share,
            "od_cb": total_flow * shares[2] * (1.0 - long_trip_share),
        }
        station_ab_passing = expected_by_trip["od_ab"] + expected_by_trip["od_ba"] + 0.5 * (expected_by_trip["od_ac"] + expected_by_trip["od_ca"])
        station_bc_passing = expected_by_trip["od_bc"] + expected_by_trip["od_cb"] + 0.5 * (expected_by_trip["od_ac"] + expected_by_trip["od_ca"])
        rows.append(
            {
                "hour_of_day": hour_of_day,
                "expected_total_passing": float(sum(expected_by_trip.values())),
                "expected_station_arrivals_ab": float(stop_probability * station_ab_passing),
                "expected_station_arrivals_bc": float(stop_probability * station_bc_passing),
                "expected_passing_od_ab": expected_by_trip["od_ab"],
                "expected_passing_od_ba": expected_by_trip["od_ba"],
                "expected_passing_od_bc": expected_by_trip["od_bc"],
                "expected_passing_od_cb": expected_by_trip["od_cb"],
                "expected_passing_od_ac": expected_by_trip["od_ac"],
                "expected_passing_od_ca": expected_by_trip["od_ca"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["expected_charging_total"] = frame["expected_station_arrivals_ab"] + frame["expected_station_arrivals_bc"]
    frame["expected_demand_bias_ab_minus_bc"] = frame["expected_station_arrivals_ab"] - frame["expected_station_arrivals_bc"]
    return frame


def metric_summary_table(run_summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    fixed = run_summaries["mobile_noop"]
    fixed_mean_queue = float(fixed["mean_queue_length"])
    fixed_mean_wait = float(fixed["mean_queue_wait_minutes"])
    fixed_reward = float(fixed["mean_utilization"])  # placeholder for type consistency below
    del fixed_reward
    for policy_name, summary in run_summaries.items():
        label = {
            "mobile_noop": "Fixed / no-agent baseline",
            "mobile_threshold": "RL agent",
            "mobile_reactive": "Reactive baseline",
        }[policy_name]
        mean_queue = float(summary["mean_queue_length"])
        mean_wait = float(summary["mean_queue_wait_minutes"])
        reward = float(summary.get("mean_allocation_vs_demand_alignment", np.nan))
        del reward
        rows.append(
            {
                "Policy": label,
                "Mean queue": mean_queue,
                "Peak queue": float(summary["peak_queue_length"]),
                "Mean wait (min)": mean_wait,
                "Peak wait (min)": float(summary["peak_queue_wait_minutes"]),
                "Mean active MCS": float(summary["mean_active_mobile_stations"]),
                "Mean utilization": float(summary["mean_utilization"]),
                "Queue improvement vs fixed (%)": 100.0 * (fixed_mean_queue - mean_queue) / fixed_mean_queue if fixed_mean_queue > 0 else np.nan,
                "Wait improvement vs fixed (%)": 100.0 * (fixed_mean_wait - mean_wait) / fixed_mean_wait if fixed_mean_wait > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)

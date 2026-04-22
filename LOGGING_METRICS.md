# Logging Metrics Reference

This document explains the metrics that the project logs to Weights & Biases (W&B) and writes to local output files. It is based on the current implementations in:

- `src/evch/train/run_simple_corridor_sim.py`
- `src/evch/sim/simple_corridor.py`
- `src/evch/train/train_uncertainty.py`
- `src/evch/train/train_rl.py`
- `src/evch/rl/simple_dqn.py`
- `src/evch/train/evaluate_policies.py`
- `src/evch/rl/evaluation.py`
- `src/evch/utils/torch_runtime.py`

## Overview

The code logs metrics under these namespaces:

- `sim/*`: timestep-level metrics from the simple corridor queue simulation
- `summary/*`: aggregate simulation summary metrics
- `uncertainty/*`: uncertainty model training and validation metrics
- `epoch`: training epoch index for uncertainty training
- `rl/*`: episode-level metrics from the local Torch DQN trainer
- `runtime/*`: PyTorch runtime metadata logged at RL startup
- `rl_eval/*`: evaluation summary of the trained RL policy
- `eval/<policy_name>/*`: evaluation summary for each policy in policy comparison runs
- `sb3/*`: numeric Stable-Baselines3 logger values forwarded during SB3 training

## 1. Simple Corridor Simulation Metrics

These are logged by `src/evch/train/run_simple_corridor_sim.py`. Each row in the simulation dataframe becomes a W&B record under `sim/*`.

### Step and time tracking

| Metric | Meaning |
| --- | --- |
| `sim/step` | Integer timestep index in the simulation. |
| `sim/hour` | Elapsed simulation time in hours from the start. This is numerically the same as `sim/global_hour`. |
| `sim/global_hour` | Absolute hour count from the start of the run, used as the W&B x-axis for all other `sim/*` metrics. |
| `sim/day_index` | Zero-based day number in the simulation horizon. |
| `sim/hour_of_day` | Hour within the current day, from `0.0` to `<24.0`. |
| `sim/step_in_day` | Timestep position within the day after converting `hour_of_day` to the configured step size. |
| `sim/day_progress` | Fraction of the current day completed, computed as `hour_of_day / 24`. |

### Traffic expectation metrics

| Metric | Meaning |
| --- | --- |
| `sim/expected_passing_total` | Expected number of total vehicles passing the corridor charger in the timestep before random sampling. |
| `sim/expected_passing_ab` | Expected passing vehicles in the `City A -> City B` direction. |
| `sim/expected_passing_ba` | Expected passing vehicles in the `City B -> City A` direction. |

These are deterministic expectations from the traffic model. They are not the realized charging arrivals.

### Realized arrival, start, and completion metrics

| Metric | Meaning |
| --- | --- |
| `sim/arrivals_total` | Number of vehicles that decided to stop for charging in the timestep. |
| `sim/arrivals_ab` | Charging arrivals in the `City A -> City B` direction. |
| `sim/arrivals_ba` | Charging arrivals in the `City B -> City A` direction. |
| `sim/starts_total` | Number of charging sessions that started in the timestep. |
| `sim/starts_ab` | Charging session starts in the `City A -> City B` direction. |
| `sim/starts_ba` | Charging session starts in the `City B -> City A` direction. |
| `sim/completions_total` | Number of charging sessions that finished in the timestep. |
| `sim/completions_ab` | Charging session completions in the `City A -> City B` direction. |
| `sim/completions_ba` | Charging session completions in the `City B -> City A` direction. |

### Queue and utilization metrics

| Metric | Meaning |
| --- | --- |
| `sim/queue_length` | Number of vehicles waiting in queue after the timestep state update. |
| `sim/queue_wait_mean_minutes` | Mean waiting time, in minutes, of vehicles still in the queue at that timestep. |
| `sim/queue_wait_max_minutes` | Maximum waiting time, in minutes, among vehicles still in the queue at that timestep. |
| `sim/active_plugs` | Number of charger plugs occupied in the timestep. |
| `sim/utilization` | Fraction of plugs in use, computed as `active_plugs / num_plugs`. |
| `sim/started_wait_mean_minutes` | Mean wait time, in minutes, for sessions that started in the timestep. |
| `sim/completed_wait_mean_minutes` | Mean wait time, in minutes, for sessions that completed in the timestep. |

### Rolling 1-hour metrics

These are derived in `run_simple_corridor_sim.py` with a rolling mean over roughly one hour of timesteps.

| Metric | Meaning |
| --- | --- |
| `sim/queue_length_rolling_1h` | Smoothed queue length. |
| `sim/utilization_rolling_1h` | Smoothed charger utilization. |
| `sim/arrivals_total_rolling_1h` | Smoothed charging arrivals. |
| `sim/starts_total_rolling_1h` | Smoothed session starts. |
| `sim/completions_total_rolling_1h` | Smoothed session completions. |
| `sim/started_wait_mean_minutes_rolling_1h` | Smoothed mean wait of newly started sessions. |
| `sim/queue_wait_mean_minutes_rolling_1h` | Smoothed mean wait among vehicles still queued. |

### Simulation summary metrics

These are logged once per run under `summary/*`.

| Metric | Meaning |
| --- | --- |
| `summary/road_length_km` | Total corridor length in kilometers. |
| `summary/station_position_km` | Charger position measured from the start of the corridor. |
| `summary/num_plugs` | Number of available charger plugs. |
| `summary/step_minutes` | Length of each simulation timestep in minutes. |
| `summary/duration_hours` | Total simulated duration in hours. |
| `summary/charging_stop_probability` | Probability that a passing vehicle decides to stop and charge. |
| `summary/total_arrivals` | Sum of all charging arrivals across the run. |
| `summary/total_starts` | Sum of all charging session starts across the run. |
| `summary/total_completions` | Sum of all charging session completions across the run. |
| `summary/final_queue_length` | Queue length at the final timestep. |
| `summary/peak_queue_length` | Maximum queue length observed during the run. |
| `summary/peak_queue_time` | Clock-style label for when peak queue length occurred. |
| `summary/mean_queue_length` | Average queue length over all timesteps. |
| `summary/mean_utilization` | Average charger utilization over all timesteps. |
| `summary/max_utilization` | Maximum charger utilization observed. |
| `summary/mean_wait_started_minutes` | Average wait, in minutes, for all sessions that started during the run. |
| `summary/max_wait_started_minutes` | Maximum wait, in minutes, among sessions that started during the run. |
| `summary/mean_wait_completed_minutes` | Average wait, in minutes, for sessions that completed during the run. |
| `summary/max_wait_completed_minutes` | Maximum wait, in minutes, among sessions that completed during the run. |
| `summary/mean_service_time_started_minutes` | Mean sampled service time for sessions that started. |
| `summary/mean_service_time_completed_minutes` | Mean realized service time for sessions that completed. |

## 2. Uncertainty Model Metrics

These are logged by `src/evch/train/train_uncertainty.py`.

### Per-epoch training metrics

| Metric | Meaning |
| --- | --- |
| `epoch` | Zero-based epoch counter. Logged as a standalone field, not under a namespace. |
| `uncertainty/train_loss` | Mean training loss for the epoch. This is Gaussian NLL for the Gaussian model and quantile loss for the quantile model. |
| `uncertainty/val_loss` | Mean validation loss for the epoch using the same loss definition as training. |

### Final evaluation metrics

These are logged once at the end under `uncertainty/*`.

| Metric | Meaning |
| --- | --- |
| `uncertainty/val_loss` | Best validation loss achieved during training. |
| `uncertainty/mae` | Mean absolute error on the validation set using the predictive center (`mu` or `q50`). |
| `uncertainty/rmse` | Root mean squared error on the validation set using the predictive center. |
| `uncertainty/interval_coverage_90` | Fraction of validation targets that fall inside the nominal 90% predictive interval. |
| `uncertainty/interval_width_90` | Mean width of the nominal 90% predictive interval. |
| `uncertainty/crossing_rate` | Quantile-model-only metric measuring how often predicted quantiles violate monotonic ordering. This is only logged for the quantile model. |

### Interpretation notes

- Lower `uncertainty/train_loss` and `uncertainty/val_loss` are better.
- Higher `uncertainty/interval_coverage_90` is better only up to calibration targets. Much higher than `0.90` can indicate overly wide intervals.
- Lower `uncertainty/interval_width_90` means tighter uncertainty intervals, but only meaningful together with coverage.
- `uncertainty/crossing_rate` should be as close to `0.0` as possible.

## 3. RL Training Metrics

These are logged by `src/evch/rl/simple_dqn.py` for the local Torch DQN backend and by `src/evch/train/train_rl.py` for setup and evaluation.

### Torch DQN episode metrics

Logged once per episode under `rl/*`.

| Metric | Meaning |
| --- | --- |
| `rl/episode` | Zero-based training episode index. |
| `rl/reward` | Total reward accumulated across the episode. |
| `rl/served_demand` | Total served demand accumulated across the episode. |
| `rl/unmet_demand` | Total unmet demand accumulated across the episode. |
| `rl/invalid_actions` | Count of invalid actions taken during the episode. |
| `rl/epsilon` | Exploration rate after the episode, based on epsilon-greedy decay. |
| `rl/loss` | Mean TD loss across optimization updates that happened during the episode. |

### Runtime metadata

Logged once at RL startup under `runtime/*`.

| Metric | Meaning |
| --- | --- |
| `runtime/device` | Selected PyTorch execution device, such as `cpu`, `cuda`, or `mps`. |
| `runtime/torch_num_threads` | Number of intra-op PyTorch CPU threads in use. |
| `runtime/cpu_count` | Number of CPU cores visible to the process. |

### RL evaluation summary

Logged after training under `rl_eval/*`.

| Metric | Meaning |
| --- | --- |
| `rl_eval/mean_reward` | Mean cumulative reward across evaluation episodes. |
| `rl_eval/mean_served_demand` | Mean served demand across evaluation episodes. |
| `rl_eval/mean_unmet_demand` | Mean unmet demand across evaluation episodes. |
| `rl_eval/mean_true_demand_total` | Mean total underlying demand across evaluation episodes. |

## 4. Policy Comparison Metrics

These are logged by `src/evch/train/evaluate_policies.py` when comparing the RL agent against heuristic baselines.

Each policy gets its own namespace:

- `eval/rl/*`
- `eval/random/*`
- `eval/greedy/*`
- `eval/coverage/*`
- any other policy names listed in `evaluation.policies`

Within each policy namespace, the logged metrics are:

| Metric | Meaning |
| --- | --- |
| `eval/<policy_name>/mean_reward` | Mean cumulative reward over evaluation episodes. |
| `eval/<policy_name>/mean_served_demand` | Mean served demand over evaluation episodes. |
| `eval/<policy_name>/mean_unmet_demand` | Mean unmet demand over evaluation episodes. |
| `eval/<policy_name>/mean_true_demand_total` | Mean total underlying demand over evaluation episodes. |

## 5. Stable-Baselines3 Metrics

When RL training uses the SB3 backend, `src/evch/train/train_rl.py` forwards numeric entries from the SB3 logger under `sb3/*`.

### Explicit SB3 metrics added by this project

| Metric | Meaning |
| --- | --- |
| `rl/timesteps` | Current total environment timesteps seen by the SB3 agent. |
| `rl/exploration_rate` | Current epsilon-style exploration rate reported by the SB3 DQN model. |
| `rl/replay_buffer_size` | Current number of transitions stored in the replay buffer. |

### Dynamic forwarded SB3 metrics

| Metric pattern | Meaning |
| --- | --- |
| `sb3/<key>` | Any finite numeric metric present in `callback.model.logger.name_to_value` at the logging interval. The exact keys depend on the SB3 backend and version. |

Examples of what may appear here include training-loss, timing, or rollout statistics, but the project does not define those names itself. It only forwards what SB3 exposes at runtime.

## 6. Metrics vs. Artifacts

The following files are also uploaded to W&B as artifacts, but they are not metrics:

- model checkpoints
- `metrics.json`
- `history.json`
- simulation CSV/JSON outputs
- generated plots such as training curves and queue visualizations

## 7. Important Nuances

- `sim/time_label` exists in the local simulation dataframe and CSV, but it is intentionally not sent to W&B.
- In the corridor simulation, both `sim/hour` and `sim/global_hour` are logged and currently carry the same numeric value.
- In uncertainty training, `uncertainty/val_loss` appears both during per-epoch logging and again at the end as the best validation loss. The final metric is therefore the best value, not necessarily the last epoch's value.
- `rl_eval/*` logs only aggregate evaluation metrics. Per-episode evaluation records are kept in local JSON output, not as W&B scalar metrics.

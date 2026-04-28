# Adaptive EV Charging Under Uncertainty

Research codebase for a DTU course project on resilient EV charging operations under disruption.

## Active Benchmark

The main benchmark is a synthetic `A-B-C` line corridor:
- `100 km` between neighboring cities
- one fixed charging station between `A-B`
- one fixed charging station between `B-C`
- up to `10` mobile charging stations (`2` plugs each)
- `6` OD flows:
  - `od_ab`, `od_ba`, `od_bc`, `od_cb`, `od_ac`, `od_ca`

The active research question is not only whether RL improves reward. It is whether the controller reallocates MCS to the correct side of the corridor under localized stress.

## Current Environment Design

The main environment is [line_corridor_mobile_env.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/envs/line_corridor_mobile_env.py).

Important current mechanics:
- MCS do not teleport
- travel from middle to a station takes `30` minutes
- relocation from `AB` to `BC` or back takes `60` minutes
- returning to middle triggers charging before redeployment
- MCS state includes:
  - `middle_available`
  - `middle_charging`
  - `transit_to_ab`
  - `transit_to_bc`
  - `transit_to_middle`
  - `station_ab`
  - `station_bc`

The line-ABC control problem now uses a small directional action space:
- `hold`
- `toward_ab`
- `toward_bc`
- `recall_ab`
- `recall_bc`

The observation is station-aware and includes:
- station queue length and mean wait
- realized arrivals and starts by station
- expected charging demand by station
- expected demand bias `AB - BC`
- active mobile capacity by station
- committed MCS bias including in-transit units
- local deficit by station and deficit bias
- disruption activity, type, remaining time
- `target_affects_ab` / `target_affects_bc`
- MCS logistics counts including `transit_to_middle`
- time-of-day sin/cos

## Reward Design

The current line-ABC reward in [mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/env/mobile_mcs_line_abc.yaml) combines:
- served demand reward
- unmet-demand and total queue penalties
- queue-wait burden penalty
- local peak queue penalty
- local peak wait penalty
- active / activation / adjustment MCS costs
- idle-capacity penalty
- utilization bonus
- spatial deficit coverage bonus
- spatial deficit direction bonus

This is meant to punish one-sided station failures more directly than a pure system-average objective.

## Main Configs

- environment: [configs/env/mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/env/mobile_mcs_line_abc.yaml)
- experiment split: [configs/experiment/mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/experiment/mobile_mcs_line_abc.yaml)
- held-out comparison override: [configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml)
- simple learner config: [configs/rl/dqn_mobile_simple.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_simple.yaml)
- SB3 config: [configs/rl/dqn_mobile_sb3.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_sb3.yaml)

## Train / Validation / Test Split

The active split is seed-driven:
- training:
  - randomized `2-6` day episodes
  - `0-3` disruptions per day
  - broader severity ranges than deployment-style evaluation
- validation:
  - fixed held-out seeds from the same generator
- `test_id`:
  - fixed unseen seeds from the deployment-style generator
  - `3-5` day episodes
  - random combinations, not a fixed script
  - guaranteed at least one AB-side demand surge and one BC-side demand surge
- `test_stress`:
  - separate explicit edge-case scenarios

Interpretation:
- `validation` checks learning on unseen but same-distribution episodes
- `test_id` is the main unseen deployment-style benchmark
- `test_stress` is for robustness, not the primary in-distribution result

## Current RL Paths

Two RL paths exist:

### `torch_dqn`

Implemented in [simple_dql.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/rl/simple_dql.py).

Current status:
- replay buffer
- minibatch updates
- delayed learning start
- multiple gradient steps
- full-horizon multi-day training support
- no target network

This is a baseline learner, not the strongest path.

### `sb3_dqn`

Configured through [dqn_mobile_sb3.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_sb3.yaml).

This is the main serious RL path because it has a proper DQN implementation with target-network stabilization.

## Current Training Budgets

At the time of writing:
- simple learner:
  - `episodes: 300`
  - `max_steps_per_episode: 1728`
- SB3 learner:
  - `total_timesteps: 250000`

If these change, treat the config files as authoritative.

## Local Commands

### Train Simple

```bash
PYTHONPATH=src python3 -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_train_only.yaml
```

### Train SB3

```bash
PYTHONPATH=src python3 -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/rl/dqn_mobile_sb3.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_sb3.yaml \
  --config configs/experiment/mobile_mcs_line_abc_train_only.yaml
```

### Run A Single Held-Out Comparison Rollout

Simple checkpoint:

```bash
PYTHONPATH=src python3 -m evch.train.run_mobile_rl_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc/rl/best_model.pt
```

SB3 checkpoint:

```bash
PYTHONPATH=src python3 -m evch.train.run_mobile_rl_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/rl/dqn_mobile_sb3.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_sb3.yaml \
  --config configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc_sb3/rl/best_model.zip
```

Noop baseline:

```bash
PYTHONPATH=src python3 -m evch.train.run_mobile_noop_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml
```

### Evaluate Full Validation / Test / Stress Suites

```bash
PYTHONPATH=src python3 -m evch.train.evaluate_policies \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc/rl/best_model.pt
```

## HPC Commands

Training:

```bash
bsub < bsub/train_mobile_line_abc.bsub
bsub < bsub/train_mobile_line_abc_sb3.bsub
```

Held-out comparison runs:

```bash
bsub < bsub/run_mobile_rl_line_abc.bsub
bsub < bsub/run_mobile_rl_line_abc_sb3.bsub
bsub < bsub/run_mobile_noop_line_abc.bsub
```

## What To Inspect

The most important held-out rollout artifacts are:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`
- `comparison_mcs_allocation_by_station.png`

For spatial awareness, the highest-signal columns and plots are:
- `disruption_target`
- `expected_demand_bias_ab_minus_bc`
- `allocation_bias_ab_minus_bc`
- `committed_mobile_stations_bias_ab_minus_bc`
- `queue_wait_mean_minutes_station_ab`
- `queue_wait_mean_minutes_station_bc`
- `comparison_mcs_allocation_by_station.png`
- `comparison_station_demand_vs_mcs.png`
- W&B `sim_overlay/allocation_bias_vs_expected_bias`
- W&B `sim_overlay/mcs_allocation_by_station`

## Practical Notes

- old checkpoints are not guaranteed to load after observation or action-space changes
- the simple learner is useful as a baseline, but SB3 should be treated as the main RL candidate
- the held-out comparison jobs use one fixed unseen `test_id` seed so RL and baselines see the same scenario
- the full suite evaluation is the correct place to report validation, `test_id`, and stress results

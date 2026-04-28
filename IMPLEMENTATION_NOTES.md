# Implementation Notes

## Current Main Path

The active main benchmark is the synthetic `A-B-C` line corridor with:
- two fixed charging stations
- six OD flows
- targeted disruptions
- mobile charging stations reallocated between `AB` and `BC`

The most relevant current files are:
- [src/evch/sim/line_corridor.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/sim/line_corridor.py)
- [src/evch/envs/line_corridor_mobile_env.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/envs/line_corridor_mobile_env.py)
- [src/evch/train/train_rl.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/train/train_rl.py)

## Current Environment Design

The line mobile env currently has:
- directional delta-style actions
- travel lag
- return-to-middle state
- recharge delay
- station-aware observations
- local and spatial reward terms

Observation-side signals now include:
- per-station queue and wait
- expected station arrivals
- expected demand bias
- active MCS counts
- committed MCS bias
- target-affects flags
- `transit_to_middle`

## Current Reward Design

The line benchmark currently uses:
- total unmet-demand penalty
- total queue penalty
- queue-wait burden penalty
- local peak queue penalty
- local peak wait penalty
- MCS activation / adjustment / active-use costs
- utilization bonus
- spatial deficit alignment bonus
- spatial deficit direction bonus

The intent is to combine system-level efficiency with local station resilience.

## Current Evaluation Design

There are two important evaluation modes:

### Full seeded suite evaluation

This is the main evaluation story:
- validation
- `test_id`
- `test_stress`

### Single held-out comparison rollout

This is a diagnostic mode:
- one fixed unseen `test_id` seed
- full time-series export
- direct visual comparison across RL and baselines

## Important Produced Artifacts

Held-out comparison runs currently export:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`
- `comparison_mcs_allocation_by_station.png`

These are the main artifacts for diagnosing spatial awareness.

## Current RL Paths

### Simple learner

Implemented in [simple_dql.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/rl/simple_dql.py).

Current properties:
- replay
- minibatches
- delayed learning start
- full-horizon episode support
- no target network

### SB3 DQN

Configured in [configs/rl/dqn_mobile_sb3.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_sb3.yaml).

This is the preferred RL path for stronger results.

## Practical Constraint

When the observation size or action space changes, older checkpoints are not expected to remain compatible. Retraining is the normal outcome after environment changes in this branch.

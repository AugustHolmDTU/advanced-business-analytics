# Implementation Notes

## Implemented

- Synthetic queue simulators for:
  - a simple corridor setup
  - a three-city `A-B-C` line corridor
- `A-B-C` RL environment with:
  - two fixed stations
  - six OD flows
  - discrete allocation of mobile charging stations across the two stations
- Fixed 3-day comparison rollouts for:
  - noop
  - trained RL
- DTU HPC `bsub` runners for:
  - noop comparison
  - RL training
  - trained RL comparison
  - 24-run reward sweep
  - no-disruption sanity run
- W&B logging for:
  - training metrics
  - comparison rollout metrics
  - station-specific queue and wait
  - station-specific MCS allocation
  - station demand versus MCS overlays
- Uncertainty-aware supervised models:
  - Gaussian NLL regression
  - quantile regression
- Test coverage for the line-corridor simulator, env, noop comparison, and RL comparison.

## Current Active Benchmark

The main benchmark is now:
- `A -> B -> C`
- `100 km` between neighboring cities
- one fixed charging station between `A-B`
- one fixed charging station between `B-C`

Current default capacity:
- `12` base plugs at `AB`
- `12` base plugs at `BC`
- up to `10` mobile charging stations total
- `2` plugs per mobile charging station

The active question is whether the policy allocates MCS to the correct station during localized disruptions.

## Current Logging Design

The current comparison outputs intentionally expose station-level behavior.

Important rollout metrics include:
- `queue_length_station_ab`
- `queue_length_station_bc`
- `queue_wait_mean_minutes_station_ab`
- `queue_wait_mean_minutes_station_bc`
- `num_active_mobile_stations_station_ab`
- `num_active_mobile_stations_station_bc`
- `expected_station_arrivals_ab`
- `expected_station_arrivals_bc`
- `disruption_target`

Current comparison artifacts include:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`

## Simplifying Assumptions

- Geography is synthetic and linear.
- There are exactly two fixed charging stations in the active `A-B-C` benchmark.
- Long trips `A<->C` are currently split across the two stations with a simple rule rather than a behavioral charging-choice model.
- MCS deployment is modeled as station-level allocation, not explicit parking layout or power-grid constraints.
- The benchmark focuses on operational disruption response, not full infrastructure design.

## Current Disruption Style

The line-corridor benchmark currently supports:
- `capacity_drop`
- `station_outage`
- `demand_surge`
- `service_time_inflation`

Disruptions are targeted, for example:
- `station_ab`
- `station_bc`
- `od_ab`
- `od_bc`
- `eastbound`
- `westbound`

The main comparison rollout is fixed and scripted so noop and RL are directly comparable.

## Recommended Next Extensions

- Add stronger station-aware heuristic baselines.
- Make OD-to-station charging choice more behaviorally realistic.
- Tighten reward tuning around localized disruption response.
- Expand comparison exports that directly score alignment between disruption target and MCS placement.

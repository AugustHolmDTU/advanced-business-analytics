# RL Change Log

## Current Active Setup

The active RL benchmark is no longer the original toy placement environment.

The current focus is:
- `A-B-C` line corridor
- two fixed charging stations
- mobile charging stations allocated across `AB` and `BC`
- fixed 3-day scripted comparison rollouts
- DTU HPC execution through `bsub`

## Key Changes

### 1. Added a dedicated `A-B-C` line-corridor simulator

This introduced:
- `6` OD flows:
  - `od_ab`, `od_ba`, `od_bc`, `od_cb`, `od_ac`, `od_ca`
- two fixed stations
- station-targeted disruptions
- station-specific queue and wait tracking

Reason:
- the previous single-corridor setup could not show whether a controller reacted at the correct location.

### 2. Added a two-station mobile-allocation RL environment

The control problem changed from:
- one station with action `0..10`

to:
- allocate up to `10` total MCS across `AB` and `BC`

Reason:
- the new benchmark should test location-aware allocation, not only total deployment level.

### 3. Standardized fixed 3-day comparison rollouts

The comparison workflow now uses:
- the same scripted 3-day scenario
- noop comparison
- trained RL comparison

Reason:
- this makes runs directly comparable in W&B and in exported CSV files.

### 4. Added station-specific diagnostics

The comparison outputs now include:
- station-specific queue length
- station-specific queue wait
- station-specific active MCS
- station-specific expected charging demand
- disruption target

Reason:
- the main project question is whether the controller sends MCS to the correct station.

### 5. Added demand-versus-MCS visual checks

The current comparison workflow now produces:
- `comparison_station_demand_vs_mcs.png`
- W&B overlays:
  - `sim_overlay/station_ab_demand_vs_mcs`
  - `sim_overlay/station_bc_demand_vs_mcs`

Reason:
- the team needs a direct visual check of whether allocation matches station demand.

### 6. Added DTU HPC scripts for the new workflow

Main scripts:
- `bsub/run_mobile_noop_line_abc.bsub`
- `bsub/train_mobile_line_abc.bsub`
- `bsub/run_mobile_rl_line_abc.bsub`
- `bsub/sweep_mobile_line_abc.bsub`
- `bsub/run_mobile_noop_line_abc_no_disruptions.bsub`

Reason:
- the `A-B-C` benchmark is now the main runnable experiment path.

## Current Interpretation

The important success criterion is now:
- not just whether reward improves,
- but whether the controller allocates MCS to `AB` when `AB` is stressed and to `BC` when `BC` is stressed.

That interpretability requirement is part of the benchmark itself.

## Open RL Questions

- Does trained RL beat `noop` consistently?
- Does it beat the current threshold baseline?
- Does it allocate MCS to the correct station during targeted disruptions?
- Is a stronger station-aware heuristic a better benchmark than the current simple one?

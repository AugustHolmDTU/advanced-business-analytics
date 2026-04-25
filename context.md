# Project Context

This file is the lightweight orientation note for coding agents working in this repository.

It should answer:
- what kind of project this is,
- what the repository is currently being used for,
- what constraints matter,
- what is still open and should not be treated as fixed.

It should be updated as the project evolves.

## Project Theme

This is a DTU Advanced Business Analytics course project on resilience-oriented EV charging operations.

The current project framing is:
- simulate corridor charging demand under disruptions,
- add mobile charging support as an operational intervention,
- compare simple baselines and RL policies,
- evaluate whether policies respond correctly to localized disruptions rather than only improving aggregate reward.

The main benchmark is no longer the original toy urban placement setting. The active direction is now a corridor control problem with explicit station-level stress and mobile support allocation.

## What This Repo Is For

This repository is a research sandbox, not a production platform.

It is used to:
- build synthetic charging-system simulations,
- train and compare RL and heuristic policies,
- run reproducible experiment batches locally and on DTU HPC,
- log comparisons in Weights & Biases,
- iterate quickly on modeling assumptions.

The code should stay modular and config-driven so the team can change the benchmark without rebuilding the whole stack.

## Current Direction

The main active setup is a line corridor with three cities:
- `A -> B -> C`,
- `100 km` between neighboring cities,
- one charging station halfway between `A-B`,
- one charging station halfway between `B-C`.

This setup currently includes:
- `6` OD flows:
  - `od_ab`, `od_ba`, `od_bc`, `od_cb`, `od_ac`, `od_ca`
- `12` base plugs at each fixed station
- up to `10` mobile charging stations total
- `2` plugs per mobile charging station
- a total possible corridor capacity of `44` plugs when all MCS are deployed
- scripted 3-day comparison rollouts with fixed disruption windows
- station-specific logging so it is visible whether the controller sends MCS to the correct station

The current research question is not just "does RL improve reward?" It is also:
- does the controller react at the correct location,
- does it allocate MCS to `AB` versus `BC` in a way that matches the disruption,
- and does it beat meaningful simple baselines.

## Current Repo Capabilities

The repository currently contains working components for:
- synthetic demand generation,
- two-city and three-city corridor queue simulation,
- Gym-compatible RL environments,
- corridor-specific disruption schedules,
- uncertainty-aware supervised models,
- RL training and evaluation,
- fixed 3-day comparison rollouts,
- W&B logging and artifact export,
- DTU-style `bsub` job scripts,
- tests for the line-corridor setup and comparison runners.

The most relevant current pieces for the active benchmark are:
- `src/evch/sim/line_corridor.py`
- `src/evch/envs/line_corridor_mobile_env.py`
- `configs/env/mobile_mcs_line_abc.yaml`
- `configs/experiment/mobile_mcs_line_abc.yaml`
- `bsub/run_mobile_noop_line_abc.bsub`
- `bsub/train_mobile_line_abc.bsub`
- `bsub/run_mobile_rl_line_abc.bsub`
- `bsub/sweep_mobile_line_abc.bsub`

## What Is Important

When making decisions in this repo, prioritize:
- resilience framing over generic optimization,
- station-level interpretability over abstract aggregate gains,
- reproducibility over convenience,
- config-driven experimentation over hard-coded assumptions,
- small explainable experiments over large opaque ones.

The project should remain easy to explain in a course setting.

## What Is Fixed Right Now

These should currently be treated as fixed unless explicitly changed:
- the benchmark is synthetic rather than a direct replay of one real place,
- the active main benchmark is corridor-based,
- the current main corridor benchmark is the `A-B-C` line corridor,
- there are exactly `2` fixed stations in that benchmark,
- the control problem is allocation of mobile charging stations across those `2` stations,
- the main comparison window is a fixed `3`-day scripted rollout,
- W&B logging should expose both aggregate performance and station-specific behavior.

## What Is Not Fixed

These should still be treated as open design choices:
- the exact disruption set and severity,
- the reward weights,
- whether RL is ultimately better than heuristics,
- whether the final story should emphasize learning or heuristic robustness,
- whether OD-to-station assignment should stay simple or become more behaviorally realistic,
- whether the three-city line corridor is the final benchmark or an intermediate step.

Do not hard-code assumptions that make these difficult to change later.

## Scope Boundaries

This project is intentionally not trying to build:
- a full transport simulator,
- a power-grid simulator,
- a city-scale digital twin,
- a multi-agent control system,
- a production optimization service.

A smaller, sharper benchmark is preferable.

## What Good Progress Looks Like

Good progress in this repo usually means:
- the benchmark becomes easier to interpret,
- the logged outputs make policy behavior clearer,
- the disruption-response story becomes more convincing,
- baseline comparisons become stronger,
- the implementation becomes easier for the team to reuse.

For the current `A-B-C` work, especially valuable progress includes:
- clearer station-level diagnostics,
- stronger non-RL baselines,
- better evidence that the agent allocates MCS to the correct station,
- cleaner W&B comparisons over the same fixed 3-day scenario.

## Guidance For Coding Agents

When working in this repository:
- understand the active benchmark before extending it,
- preserve comparability between noop, heuristic, and RL runs,
- keep changes modular and config-driven,
- avoid mixing unrelated benchmark ideas together,
- expose assumptions in config and logging,
- prefer simple station-level explanations over black-box complexity.

If a task is ambiguous, prefer the solution that makes comparison easier.

Also note:
- the fixed 3-day comparison rollout is important and should remain stable unless intentionally changed,
- station-specific outputs are first-class, not optional,
- if RL underperforms a simple heuristic, that is a valid project result.

## Current Findings

The current implementation supports:
- noop comparison runs,
- trained RL comparison runs,
- 24-run reward sweeps for the `A-B-C` setup,
- station-specific MCS allocation tracking,
- station-specific queue and wait tracking,
- custom W&B overlays for station demand versus allocated MCS.

The key behavioral question is now visible in the outputs:
- when a disruption affects `station_bc`, does the controller allocate MCS to `BC`,
- and when a disruption affects `station_ab` or `od_ab`, does it allocate to `AB`.

That interpretability requirement is central to the current benchmark.

## Practical Reading Order

For fast orientation, start with:
- [README.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/README.md)
- [IMPLEMENTATION_NOTES.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/IMPLEMENTATION_NOTES.md)
- [context.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/context.md)
- [RL_CHANGELOG.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/RL_CHANGELOG.md)

Then inspect:
- `src/evch/sim/`
- `src/evch/envs/`
- `src/evch/rl/`
- `src/evch/train/`
- `configs/env/`
- `configs/experiment/`
- `bsub/`

## How To Update This File

Update this file when:
- the main benchmark changes,
- the corridor structure changes,
- the control problem changes,
- the disruption philosophy changes,
- the comparison workflow changes,
- or the main evaluation story changes.

Keep it high-level.

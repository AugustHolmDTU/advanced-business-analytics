# Project Context

This is the lightweight orientation note for coding agents working in this repository.

## Project Theme

This is a DTU Advanced Business Analytics course project on resilient EV charging operations under uncertainty and disruption.

The active framing is:
- simulate corridor EV charging demand under disruptions
- allocate mobile charging stations as an operational intervention
- compare heuristics and RL policies
- evaluate whether policies react at the correct station, not only whether they improve aggregate reward

## What This Repo Is For

This repository is a research sandbox. It is used to:
- build synthetic charging-system simulations
- train and compare RL and heuristic policies
- run reproducible local and HPC experiment batches
- log diagnostics in Weights & Biases
- iterate quickly on modeling assumptions

Keep the code modular and config-driven.

## Current Main Benchmark

The active main benchmark is the synthetic `A-B-C` line corridor:
- `100 km` between neighboring cities
- one fixed station between `A-B`
- one fixed station between `B-C`
- `6` OD flows
- up to `10` mobile charging stations total
- `2` plugs per MCS

The main environment is [line_corridor_mobile_env.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/envs/line_corridor_mobile_env.py).

## Current Control Problem

The controller does not choose a full absolute station allocation every step any more.

The active action space is directional and small:
- `hold`
- `toward_ab`
- `toward_bc`
- `recall_ab`
- `recall_bc`

This is intentional. The benchmark is about spatial reallocation under lag, not about learning a large combinatorial action table.

## Current MCS Logistics Assumptions

The line benchmark currently models simple deployment frictions:
- middle to station travel: `30` minutes
- station to station relocation: `60` minutes
- return to middle triggers recharge time

The MCS state machine includes:
- `middle_available`
- `middle_charging`
- `transit_to_ab`
- `transit_to_bc`
- `transit_to_middle`
- `station_ab`
- `station_bc`

The policy observation includes active and committed spatial information, including:
- expected demand bias `AB - BC`
- committed MCS bias `AB - BC`
- `target_affects_ab` / `target_affects_bc`
- `transit_to_middle`

## Current Evaluation Philosophy

The active split is:
- training: broader randomized `2-6` day episodes
- validation: held-out same-distribution seeds
- `test_id`: unseen deployment-style `3-5` day seeds
- `test_stress`: separate robustness scenarios

The main `test_id` story is:
- unseen
- representative
- not necessarily harder than training

Stress scenarios are reported separately.

## Current Reward Philosophy

The benchmark no longer optimizes only aggregate queue pressure.

Important current reward ingredients include:
- total unmet demand and queue penalties
- queue-wait burden penalty
- local peak queue penalty
- local peak wait penalty
- MCS activation / adjustment / active-use costs
- spatial deficit coverage and direction bonuses

This means one-sided station blowups are meant to be costly even if the system-wide average looks acceptable.

## Current RL Status

There are two RL paths:

### `torch_dqn`

Implemented in [simple_dql.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/rl/simple_dql.py).

Current status:
- replay buffer
- minibatches
- delayed learning start
- multiple gradient steps
- still no target network

Treat this as a baseline learner.

### `sb3_dqn`

Configured through [dqn_mobile_sb3.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_sb3.yaml).

This is the main serious RL candidate.

## What Is Fixed Right Now

These should be treated as fixed unless explicitly changed:
- the active main benchmark is the synthetic `A-B-C` line corridor
- there are exactly `2` fixed stations in that benchmark
- MCS allocation is a station-level operational control problem
- the primary generalization benchmark is seed-driven `test_id`
- station-level interpretability is a first-class requirement
- held-out comparison rollouts should remain directly comparable between RL and baselines

## What Is Still Open

These are still open design choices:
- exact disruption severity mix
- exact reward weights
- whether SB3 DQN is sufficient or another RL algorithm is needed
- how strong the best heuristic baseline should become
- how behaviorally realistic OD-to-station charging choice should be
- whether the final project story emphasizes RL or robust heuristics

Do not hard-code assumptions that make these hard to revisit.

## What Good Progress Looks Like

Good progress usually means:
- clearer spatial diagnostics
- stronger same-scenario comparisons between RL and baselines
- better evidence that the controller sends MCS to the correct side
- cleaner and more reproducible train / validation / test workflows
- more interpretable logs and artifacts

## Practical Reading Order

Start with:
- [README.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/README.md)
- [context.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/context.md)
- [IMPLEMENTATION_NOTES.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/IMPLEMENTATION_NOTES.md)
- [RL_CHANGELOG.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/RL_CHANGELOG.md)

Then inspect:
- `src/evch/sim/`
- `src/evch/envs/`
- `src/evch/rl/`
- `src/evch/train/`
- `configs/env/`
- `configs/experiment/`
- `bsub/`

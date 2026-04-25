# Adaptive EV Charging Under Uncertainty

Research codebase for a DTU course project on resilient EV charging operations under disruption.

The current main benchmark is a synthetic `A-B-C` line corridor:
- cities `A -> B -> C`
- `100 km` between neighboring cities
- one fixed charging station between `A-B`
- one fixed charging station between `B-C`
- mobile charging stations allocated across the two fixed stations

## Current Focus

The active question is:
- can a controller allocate mobile charging stations to the correct station when disruptions happen,
- and does it do better than simple baselines on the same fixed 3-day scenario.

This is not just a generic RL benchmark. The important outputs are station-specific:
- queue length and wait at `AB` and `BC`
- number of MCS allocated to `AB` and `BC`
- whether allocation follows the disruption location

## Repo Structure

```text
.
├── bsub/
├── configs/
├── src/evch/
│   ├── baselines/
│   ├── config/
│   ├── data/
│   ├── envs/
│   ├── models/
│   ├── rl/
│   ├── sim/
│   ├── train/
│   └── utils/
├── tests/
├── context.md
├── IMPLEMENTATION_NOTES.md
├── RL_CHANGELOG.md
└── README.md
```

## Main Benchmark

The active `A-B-C` environment is defined in:
- [mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/env/mobile_mcs_line_abc.yaml)
- [mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/experiment/mobile_mcs_line_abc.yaml)

It currently uses:
- `12` base plugs at `station_ab`
- `12` base plugs at `station_bc`
- up to `10` mobile charging stations total
- `2` plugs per mobile charging station
- maximum corridor capacity of `44` plugs when all MCS are deployed

The fixed comparison rollout is:
- `3` days
- scripted disruptions
- the same scenario for noop and trained RL comparisons

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Main Local Commands

Train the current `A-B-C` RL setup:

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

Run the fixed 3-day noop comparison:

```bash
PYTHONPATH=src python -m evch.train.run_mobile_noop_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

Run the fixed 3-day trained RL comparison:

```bash
PYTHONPATH=src python -m evch.train.run_mobile_rl_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc/rl/best_model.pt
```

Run the no-disruption sanity check:

```bash
PYTHONPATH=src python -m evch.train.run_mobile_noop_comparison \
  --config configs/env/mobile_mcs_line_abc_no_disruptions.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc_no_disruptions.yaml
```

## DTU HPC

The current `A-B-C` `bsub` scripts are:
- [run_mobile_noop_line_abc.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/bsub/run_mobile_noop_line_abc.bsub)
- [train_mobile_line_abc.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/bsub/train_mobile_line_abc.bsub)
- [run_mobile_rl_line_abc.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/bsub/run_mobile_rl_line_abc.bsub)
- [sweep_mobile_line_abc.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/bsub/sweep_mobile_line_abc.bsub)
- [run_mobile_noop_line_abc_no_disruptions.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/bsub/run_mobile_noop_line_abc_no_disruptions.bsub)

Typical order:

```bash
bsub < bsub/run_mobile_noop_line_abc.bsub
bsub < bsub/train_mobile_line_abc.bsub
bsub < bsub/run_mobile_rl_line_abc.bsub
```

The RL comparison script loads:

```text
outputs/mobile_mcs_line_abc/rl/best_model.pt
```

The current sweep is a 24-run full grid:

```bash
bsub < bsub/sweep_mobile_line_abc.bsub
```

## W&B Logging

The current comparison runs log:
- aggregate queue and wait metrics
- station-specific queue and wait metrics
- station-specific MCS allocation
- expected passing demand by OD
- expected charging demand at `AB` and `BC`
- disruption type and disruption target

Important metrics include:
- `sim/queue_wait_mean_minutes_station_ab`
- `sim/queue_wait_mean_minutes_station_bc`
- `sim/num_active_mobile_stations_station_ab`
- `sim/num_active_mobile_stations_station_bc`
- `sim/expected_station_arrivals_ab`
- `sim/expected_station_arrivals_bc`
- `sim/disruption_target`

Comparison runs also produce:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`

There are also custom W&B overlay panels for:
- `sim_overlay/station_ab_demand_vs_mcs`
- `sim_overlay/station_bc_demand_vs_mcs`

Those are intended to show whether MCS are allocated to the correct station when a disruption occurs.

## Baselines

The most relevant baselines for the current setup are:
- `mobile_noop`
- `mobile_threshold`
- trained RL

`mobile_noop` is the clean "do nothing" baseline.
`mobile_threshold` is the main simple rule-based baseline.
An untrained RL policy is only a sanity check, not a main benchmark.

## Tests

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## What Is Implemented

- two-city and three-city corridor queue simulators
- mobile charging support environments
- uncertainty-aware supervised models
- RL training and evaluation
- fixed 3-day comparison rollouts
- DTU HPC `bsub` templates
- W&B logging and artifact export
- station-level diagnostics for the `A-B-C` setup

## What Is Still Open

- stronger heuristic baselines for the `A-B-C` environment
- richer OD-to-station charging choice behavior
- more systematic hyperparameter sweeps
- clearer final evaluation story around when RL is justified

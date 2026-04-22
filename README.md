# Adaptive EV Charging Under Uncertainty

Minimal research codebase for a DTU course project on adaptive EV charging placement with a stochastic digital twin and uncertainty-aware demand prediction.

## Goal

The repository provides a small but extensible setup to:

- simulate EV charging demand in a toy urban area,
- train uncertainty-aware demand models on simulator-generated data,
- train a single-agent RL policy to place or relocate a limited number of chargers,
- compare RL against simple heuristic baselines,
- track experiments with Weights & Biases,
- run locally first and scale later via Slurm.

The first version intentionally stays small. It is designed to run end-to-end on synthetic data in minutes, while keeping clean seams for future real-data ingestion and richer digital-twin logic.

## Repo Structure

```text
.
├── configs/
├── scripts/
├── slurm/
├── src/evch/
│   ├── baselines/
│   ├── config/
│   ├── data/
│   ├── envs/
│   ├── models/
│   ├── rl/
│   ├── train/
│   └── utils/
├── tests/
├── IMPLEMENTATION_NOTES.md
└── README.md
```

## Digital Twin

The environment models:

- demand zones with coordinates and base demand rates,
- candidate charging sites with coordinates,
- either a toy urban layout or a long-distance corridor with city hubs, service areas, and exits,
- or a TomTom-grounded snapshot with real station coordinates, connector availability, and routing times,
- a travel-time matrix derived from synthetic road structure,
- stochastic zone demand with morning and evening peaks,
- optional disruptions:
  - demand spikes,
  - temporary station outages,
  - corridor slowdowns and road-closure penalties,
  - noisy observations.

The first RL-friendly action space is discrete:

- select one candidate site,
- if chargers remain, deploy a charger there,
- otherwise relocate one charger from the currently weakest occupied site to the chosen site.

Reward combines:

- served demand,
- unmet demand penalty,
- deployment or relocation costs,
- optional outage penalty.

## Uncertainty Modeling

Two uncertainty-aware demand model families are included:

1. Gaussian probabilistic regression
   - predicts `mu(x)` and `log_sigma(x)`
   - uses Gaussian negative log-likelihood
   - supports heteroscedastic uncertainty through `sigma(x)`
2. Quantile regression
   - predicts `q05`, `q50`, `q95`
   - uses tilted loss
   - exposes predictive intervals and warns on quantile crossing

The default local baseline uses the Gaussian model. Quantile regression is implemented and ready to compare later.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you want the exact requested stack, keep `gymnasium`, `stable-baselines3`, and `wandb` installed. The code also contains local fallbacks so the toy pipeline can still run when some optional packages are unavailable.

## Quickstart

Generate synthetic supervised data:

```bash
PYTHONPATH=src python -m evch.train.generate_synthetic_data \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/experiment/local_demo.yaml
```

For the corridor-focused resilience setup, swap `configs/env/base.yaml` for `configs/env/corridor.yaml`.

For a TomTom-grounded hybrid setup, first fetch a cached snapshot and then point the training scripts at `configs/env/tomtom_frederiksberg.yaml`.

```bash
export TOMTOM_API_KEY=...
PYTHONPATH=src python3 -m evch.train.fetch_tomtom_snapshot \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml \
  --config configs/tomtom/frederiksberg.yaml
```

Train an uncertainty model:

```bash
PYTHONPATH=src python -m evch.train.train_uncertainty \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/model/gaussian.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/local_demo.yaml
```

Train the RL agent:

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/local_demo.yaml
```

Train the simple fixed-location mobile charging support agent:

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_simple.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_simple.yaml
```

This setup keeps a fixed station with 12 plugs, lets the DQN choose `0..10` mobile charging stations at each step, and uses a reward that trades off served demand, unmet demand, active MCS cost, change cost, and idle overcapacity.

Build the Denmark hybrid corridor example from TomTom and generate ready-to-train configs:

```bash
export TOMTOM_API_KEY=...
PYTHONPATH=src python -m evch.train.build_denmark_hybrid_corridor \
  --config configs/logging/base.yaml \
  --config configs/experiment/denmark_hybrid_corridor.yaml \
  --config configs/tomtom/denmark_hybrid_corridor.yaml
```

Train the local RL agent on Apple Silicon using the generated corridor package:

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config outputs/denmark_hybrid_corridor/generated/environment.yaml \
  --config outputs/denmark_hybrid_corridor/generated/demand.yaml \
  --config configs/rl/dqn_apple_silicon.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/denmark_hybrid_corridor.yaml
```

Evaluate RL against heuristics:

```bash
PYTHONPATH=src python -m evch.train.evaluate_policies \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/local_demo.yaml \
  --agent-checkpoint outputs/latest/rl/best_model.pt
```

Or run the small local pipeline:

```bash
bash scripts/run_local_pipeline.sh
```

## Weights & Biases

W&B is controlled through config:

- `logging.wandb.enabled`
- `logging.wandb.mode` set to `offline` or `online`
- `rl.wandb_log_interval` for SB3 training metric upload cadence

Environment variables:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY` (optional)

If W&B is disabled or not installed, runs continue without crashing.

Ready-made overlays are included:

- [configs/logging/wandb_offline.yaml](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/configs/logging/wandb_offline.yaml)
- [configs/logging/wandb_online.yaml](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/configs/logging/wandb_online.yaml)

Online RL training example:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=adaptive-ev-charging
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/local_demo.yaml
```

Offline logging example:

```bash
PYTHONPATH=src python -m evch.train.train_uncertainty \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/model/gaussian.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_offline.yaml \
  --config configs/experiment/local_demo.yaml

wandb sync wandb/offline-run-*
```

RL and uncertainty training now upload the generated checkpoint, metrics/history JSON, and plots as W&B artifacts when W&B is enabled.

## Slurm

Templates are in [slurm/train_uncertainty.slurm](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/slurm/train_uncertainty.slurm), [slurm/train_rl.slurm](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/slurm/train_rl.slurm), and [slurm/eval.slurm](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/slurm/eval.slurm).

For DTU-style LSF clusters, simple `bsub` templates for the mobile-agent setup are included in [bsub/train_mobile_rl.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2. Sem/42578/advanced-business-analytics/bsub/train_mobile_rl.bsub) and [bsub/eval_mobile_rl.bsub](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2. Sem/42578/advanced-business-analytics/bsub/eval_mobile_rl.bsub). They follow the same header style as the standard DTU HPC example you submit with `bsub < bsub/train_mobile_rl.bsub`.

Typical submission:

```bash
sbatch slurm/train_uncertainty.slurm
sbatch slurm/train_rl.slurm
sbatch slurm/eval.slurm
```

The templates use environment variables and placeholders rather than site-specific paths.

## Tests

Run the minimal test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## What Is Implemented

- stochastic Gymnasium-compatible charging placement environment,
- synthetic city and demand generation,
- Gaussian NLL and quantile regressors in PyTorch,
- local DQN implementation with optional Stable-Baselines3 DQN backend,
- heuristic baselines,
- evaluation scripts and plots,
- YAML config loading,
- optional W&B logging,
- Slurm templates,
- minimal tests.

## What Is Still Placeholder

- real OSM or TomTom ingestion,
- richer traffic assignment and queueing,
- explicit uncertainty integration inside the RL state or reward,
- multi-step relocation planning and constrained deployment budgets,
- stronger robustness evaluation across many disruption regimes.

# Adaptive EV Charging Under Uncertainty

Research codebase for a DTU course project on resilient EV charging operations under disruption.

The main benchmark is a synthetic A-B-C corridor:
- cities A -> B -> C
- 100 km between neighboring cities
- one fixed charging station between A-B
- one fixed charging station between B-C
- mobile charging stations (MCS) allocated between the two fixed stations

## Current Branch State

This branch uses a simple online DQL agent for the corridor benchmark:
- file: `src/evch/rl/simple_dql.py`
- architecture: plain MLP mapping observation -> Q-values
- exploration: epsilon-greedy
- update rule: 1-step TD, online update on each transition
- intentionally removed from this agent: replay buffer, target network, dueling head, heuristic priors

`src/evch/rl/simple_dqn.py` is kept as a compatibility shim that imports this new implementation.

## Main Configs

- environment: `configs/env/mobile_mcs_line_abc.yaml`
- experiment setup: `configs/experiment/mobile_mcs_line_abc.yaml`
- default corridor RL config: `configs/rl/dqn_mobile_simple.yaml`
- stronger SB3 DQN override: `configs/rl/dqn_mobile_sb3.yaml`
- W&B online logging config: `configs/logging/wandb_online.yaml`

W&B project used by this branch:
- `A-B-C`

## Train / Validation / Test Design

The A-B-C benchmark now uses a seed-driven scenario split.

- training: randomized 1-3 day episodes drawn from the environment generator
- validation: fixed held-out seeds from the same generator (`train_val_test.validation`)
- main test: fixed held-out seeds from the same generator (`train_val_test.test_id`)
- stress/OOD: separate fixed stress scenarios (`train_val_test.test_stress`)

For the current line-ABC setup, the main held-out `test_id` split is heavier than training:
- 5-10 day episodes
- at least one disruption per day
- some days with two disruptions

An unseen seed is treated as an unseen simulated day. The seed determines the stochastic demand realization, charging-stop decisions, service times, disruption count, disruption type, disruption target, disruption timing, and disruption severity. This makes held-out seeds a valid in-distribution generalization test for the simulator.

The primary claim is therefore:

`The policy generalizes to unseen simulated days drawn from the same randomized scenario distribution.`

Stress scenarios are reported separately and should be interpreted as robustness tests, not the main in-distribution test.

## Local Commands

If your default interpreter is missing project dependencies, run with the known working environment interpreter on Windows:

```powershell
$env:PYTHONPATH='src'
& "C:\Users\augus\miniconda3\envs\vae\python.exe" -m evch.train.train_rl ...
```

### Train

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

This is the current simple online DQL path, even though the compatibility config name still says `dqn_mobile_simple`.

### Train Stronger SB3 DQN

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/rl/dqn_mobile_sb3.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_sb3.yaml
```

### Evaluate Paired Held-Out Seeds and Stress Scenarios

```bash
PYTHONPATH=src python -m evch.train.evaluate_policies \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc/rl/best_model.pt
```

This evaluates all configured policies on the exact same validation seeds, main test seeds, and stress scenarios:
- `mobile_noop` (exported as `fixed`)
- `mobile_threshold` (exported as `threshold`)
- RL policy (`rl`)

### Legacy Scripted Rollout Diagnostics

The old one-off scripted rollout helpers still exist:
- `evch.train.run_mobile_noop_comparison`
- `evch.train.run_mobile_rl_comparison`

These are useful for manual diagnostics, but they are not the main experiment any more.

For line-ABC, the `bsub/run_mobile_rl_line_abc.bsub` and `bsub/run_mobile_noop_line_abc.bsub` jobs use
`configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml`, which picks one fixed seed from the held-out
`test_id` split and logs the full `sim/*` time series. This keeps the W&B overlays comparable while still using an
unseen scenario from the main test distribution.

## W&B Logs: What They Mean

The project logs are grouped into namespaces.

### 1) Training Episode Logs (`rl/*`)

Logged once per episode during training, including:
- reward and loss (`rl/reward`, `rl/loss`)
- exploration (`rl/epsilon`)
- demand served vs unmet (`rl/served_demand`, `rl/unmet_demand`)
- queue and utilization aggregates
- periodic validation metrics (`rl/eval_mean_reward`, `rl/eval_td_loss`, etc.)

Use these to evaluate learning progress and generalization on validation episodes.

### 2) Training Step Logs (`rl_step/*`)

Logged every `wandb_step_log_interval` simulation steps:
- action and reward
- served and unmet demand
- station-specific MCS allocation (`..._station_ab`, `..._station_bc`)
- station-specific queue and wait
- utilization and unused mobile capacity
- disruption indicators (`rl_step/disruption_active`, `rl_step/disruption_type_code`)

Use these to diagnose behavior at fine timescale, especially around disruptions.

### 3) Validation / Test / Stress Summaries

The main experiment logs split-specific summaries:
- `validation/*`
- `test_id/*`
- `test_stress/*`
- `paired_test/*`

`validation/*` and `test_id/*` report per-policy operational summaries on fixed held-out seeds.

`paired_test/*` reports paired RL improvements against the fixed and threshold baselines on the exact same test seeds, including:
- queue reduction
- wait reduction
- breach-rate reduction
- reward difference
- extra MCS usage

### 4) Legacy Comparison Rollout Time-Series (`sim/*`)

If you run the legacy scripted rollout helpers, they still log `sim/*`, including:
- queue and wait metrics (global and station-specific)
- active plugs, effective plugs, utilization
- active/unused mobile station estimates
- expected station arrivals and OD flow signals
- disruption state (`sim/disruption_active`, `sim/disruption_type_code`, `sim/disruption_target`)

These series are logged against `sim/global_hour` for consistent timeline plots.

### 5) Legacy Comparison Summary Metrics (`sim_summary/*`)

Aggregated from the rollout and stored in both W&B and JSON summary:
- mean/peak queue and wait
- queue target breach rate and excess wait
- mean utilization
- mean MCS activation in normal vs disrupted periods
- alignment summary metrics (described below)

### 6) Legacy Overlay Panels (`sim_overlay/*`)

Custom line panels designed for decision-quality inspection:
- `sim_overlay/station_ab_demand_vs_mcs`
- `sim_overlay/station_bc_demand_vs_mcs`
- `sim_overlay/allocation_bias_vs_expected_bias`

These directly show if allocation follows expected demand and disruption location.

## Derived Alignment Metrics (Important)

These are computed during comparison rollout logging.

Let:
- `E_ab`, `E_bc`: expected station arrivals at AB and BC
- `M_ab`, `M_bc`: active mobile stations at AB and BC

Then:
- `expected_demand_share_ab = E_ab / (E_ab + E_bc)` (fallback 0.5 when denominator is 0)
- `expected_demand_share_bc = E_bc / (E_ab + E_bc)` (fallback 0.5)
- `allocation_share_ab = M_ab / (M_ab + M_bc)` (fallback 0.5)
- `allocation_share_bc = M_bc / (M_ab + M_bc)` (fallback 0.5)

Alignment score:
- `allocation_demand_gap_ab = |allocation_share_ab - expected_demand_share_ab|`
- `allocation_demand_gap_bc = |allocation_share_bc - expected_demand_share_bc|`
- `allocation_vs_demand_alignment = 1 - 0.5 * (allocation_demand_gap_ab + allocation_demand_gap_bc)`

Interpretation:
- 1.0 = perfect share alignment
- lower values = allocation diverges from expected demand split

Directional bias metrics:
- `allocation_bias_ab_minus_bc = M_ab - M_bc`
- `expected_demand_bias_ab_minus_bc = E_ab - E_bc`

Disruption-conditioned alignment diagnostics:
- `alignment_on_station_ab_disruption`: equals `allocation_bias_ab_minus_bc` during active station_ab disruptions, otherwise NaN
- `alignment_on_station_bc_disruption`: equals `-(allocation_bias_ab_minus_bc)` during active station_bc disruptions, otherwise NaN

Positive values on these two disruption-conditioned metrics indicate movement in the intuitively correct direction.

## Produced Artifacts

Paired evaluation exports:
- `validation_summary.csv`
- `validation_manifest.csv`
- `test_id_summary.csv`
- `paired_test_results.csv`
- `test_stress_summary.csv`
- `scenario_manifest.csv`
- `suite_outputs.json`

Legacy comparison runs export:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`

Training exports include:
- `outputs/<experiment>/rl/best_model.pt`
- `outputs/<experiment>/rl/training_summary.json`
- `outputs/<experiment>/rl/history.json` (for torch_dql backend)

## HPC Commands

Train and run the internal validation/test suites:

```bash
bsub < bsub/train_mobile_line_abc.bsub
```

Re-run the paired held-out seed evaluation and stress suite for an existing checkpoint:

```bash
bsub < bsub/eval_mobile_rl.bsub
```

Train the stronger SB3 DQN variant:

```bash
bsub < bsub/train_mobile_line_abc_sb3.bsub
```

Run the paired held-out seed evaluation for the SB3 DQN checkpoint:

```bash
bsub < bsub/eval_mobile_rl_sb3.bsub
```

Run old-style `sim/*` overlays on one held-out seed for the SB3 DQN variant:

```bash
bsub < bsub/run_mobile_rl_line_abc_sb3.bsub
bsub < bsub/run_mobile_noop_line_abc_sb3.bsub
```

## Baselines

Primary baselines for A-B-C:
- `mobile_noop`
- `mobile_threshold`
- trained RL (simple DQL)

`mobile_noop` is the do-nothing baseline.
`mobile_threshold` is the simple rule-based baseline.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

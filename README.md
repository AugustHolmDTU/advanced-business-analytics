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
- default simple DQL config: `configs/rl/simple_dql.yaml`
- longer training override: `configs/rl/simple_dql_long.yaml`
- W&B online logging config: `configs/logging/wandb_online.yaml`

W&B project used by this branch:
- `A-B-C`

## Train / Validation / Test Design

Training is scenario-based and split-aware:
- train: RL training episodes from the configured environment
- validation: periodic evaluation during and after training (`train_val_test.validation`)
- held-out test rollout: separate rollout recipe (`train_val_test.test`) with 10 days and random station_outage events on station_ab

This is not a timestep split of one trajectory; each split is a separate scenario recipe.

## Local Commands

If your default interpreter is missing project dependencies, run with the known working environment interpreter on Windows:

```powershell
$env:PYTHONPATH='src'
& "C:\Users\augus\miniconda3\envs\vae\python.exe" -m evch.train.train_rl ...
```

### Train (default simple DQL)

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/rl/simple_dql.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

### Train Longer (simple DQL + override)

```bash
PYTHONPATH=src python -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/rl/simple_dql.yaml \
  --config configs/rl/simple_dql_long.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

### No-op Comparison (fixed scripted 3-day rollout)

```bash
PYTHONPATH=src python -m evch.train.run_mobile_noop_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml
```

### RL Comparison (fixed scripted 3-day rollout)

```bash
PYTHONPATH=src python -m evch.train.run_mobile_rl_comparison \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --agent-checkpoint outputs/mobile_mcs_line_abc/rl/best_model.pt
```

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

### 3) Comparison Rollout Time-Series (`sim/*`)

Logged from fixed comparison rollouts (`run_mobile_noop_comparison` or `run_mobile_rl_comparison`), including:
- queue and wait metrics (global and station-specific)
- active plugs, effective plugs, utilization
- active/unused mobile station estimates
- expected station arrivals and OD flow signals
- disruption state (`sim/disruption_active`, `sim/disruption_type_code`, `sim/disruption_target`)

These series are logged against `sim/global_hour` for consistent timeline plots.

### 4) Comparison Summary Metrics (`sim_summary/*`)

Aggregated from the rollout and stored in both W&B and JSON summary:
- mean/peak queue and wait
- queue target breach rate and excess wait
- mean utilization
- mean MCS activation in normal vs disrupted periods
- alignment summary metrics (described below)

### 5) Overlay Panels (`sim_overlay/*`)

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

Comparison runs export:
- `comparison_timestep_metrics.csv`
- `comparison_rollout_summary.json`
- `comparison_queue_dynamics.png`
- `comparison_daily_patterns.png`
- `comparison_station_demand_vs_mcs.png`

Training exports include:
- `outputs/<experiment>/rl/best_model.pt`
- `outputs/<experiment>/rl/training_summary.json`
- `outputs/<experiment>/rl/history.json` (for torch_dql backend)

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

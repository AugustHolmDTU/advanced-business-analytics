# RL Change Log

## Current Active Setup

The active RL benchmark is the synthetic `A-B-C` line corridor with two fixed stations and mobile charging stations allocated across `AB` and `BC`.

The important current criterion is spatial behavior:
- does the controller send capacity to the correct side
- does it do so with realistic travel / recharge lag
- does it reduce local station pain rather than only aggregate averages

## Important Current RL Changes

### 1. Station-aware line-corridor simulator

The current line benchmark introduced:
- `6` OD flows
- two fixed stations
- targeted disruptions
- station-specific queue and wait tracking

Reason:
- the old single-station or aggregate setups could not answer whether control was spatially correct.

### 2. MCS logistics are no longer instantaneous

The active line env now includes:
- travel from middle to station
- station-to-station relocation lag
- return-to-middle state
- recharge delay before reuse

Reason:
- instant allocation encouraged unrealistic jitter and weakened the spatial interpretation.

### 3. Directional delta action space

The line env now uses:
- `hold`
- `toward_ab`
- `toward_bc`
- `recall_ab`
- `recall_bc`

Reason:
- a small directional action space is easier to learn and better aligned with the real control decision than a full combinatorial allocation table.

### 4. Stronger spatial observation

The current observation includes:
- `target_affects_ab`
- `target_affects_bc`
- `expected_demand_bias_ab_minus_bc`
- `committed_mobile_stations_bias_ab_minus_bc`
- `transit_to_middle`

Reason:
- the policy should not have to infer all spatial intent only from delayed queue damage.

### 5. Stronger local reward signal

The reward now includes:
- aggregate queue and wait terms
- local peak queue penalty
- local peak wait penalty
- spatial deficit coverage and direction bonuses

Reason:
- otherwise a policy can improve averages while still leaving one station badly underserved.

### 6. Training split is seed-driven

The current evaluation story is:
- broader randomized training distribution
- held-out same-distribution validation
- unseen deployment-style `test_id`
- separate `test_stress`

Reason:
- the main claim is unseen generalization, not just performance on one scripted diagnostic day.

### 7. Comparison rollouts remain important, but they are diagnostics

The single held-out comparison rollouts still exist and are useful because they export:
- full `sim/*` time series
- station demand vs MCS overlays
- direct allocation-by-station plots

But they are diagnostic artifacts, not the full evaluation story.

## Current Interpretation

The main RL question is now:
- can the controller handle lagged reallocation under disruption
- can it move MCS in the correct direction
- can it avoid one-sided queue and wait blowups

SB3 DQN is the main serious RL path.

The custom `torch_dqn` path is now a stronger baseline than before, but still lacks a target network and should not be treated as the strongest method in the repo.

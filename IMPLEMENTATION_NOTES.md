# Implementation Notes

## Implemented

- Synthetic city generator with candidate charging sites, demand zones, and travel-time matrix.
- TomTom Phase 1 snapshot fetcher with geocoding, nearby EV search, charging availability enrichment, route matrix, and plotted station map output.
- Stochastic digital twin environment with deploy-or-relocate actions, disruption injection, and logging-friendly info outputs.
- Synthetic supervised dataset generator for demand prediction.
- `GaussianNLLRegressor` with heteroscedastic `mu(x)` and `sigma(x)` using Gaussian negative log-likelihood.
- `QuantileRegressor` with multi-quantile outputs and tilted loss.
- RL training with a discrete DQN-friendly action space. The code uses Stable-Baselines3 DQN when available and otherwise falls back to a small internal DQN implementation.
- Heuristic baselines: random, greedy highest-demand, and coverage.
- YAML configs, W&B hooks, Slurm templates, and minimal tests.

## Simplifying Assumptions

- Geography is synthetic and small.
- In TomTom Phase 1, station locations and travel times are real, but zone demand remains synthetic.
- Travel time is based on Euclidean distance rather than a road network.
- Each charger has fixed per-step capacity.
- Demand is generated at zone level and then approximately served through weighted site accessibility.
- The first action design selects a target site only; the environment decides whether that means deployment or relocation.
- The first RL baseline focuses on cumulative reward rather than full operational realism.

## Where Slide-Based Uncertainty Is Included

- Gaussian probabilistic regression:
  - predicts `mu(x)` and `log_sigma(x)`
  - converts `log_sigma` to strictly positive `sigma(x)`
  - trains with Gaussian NLL
  - models heteroscedastic uncertainty
- Quantile regression:
  - predicts `q05`, `q50`, `q95`
  - trains with tilted loss
  - exposes predictive intervals
  - includes a quantile-crossing warning in evaluation

## Recommended Next Extensions

- Replace synthetic zone demand with real or inferred demand labels.
- Feed predictive uncertainty into RL more directly, for example through risk-sensitive rewards or scenario sampling from uncertainty intervals.
- Add budget constraints, charger types, and station-level outages with duration.
- Expand evaluation to repeated seeds, scenario sweeps, and Slurm array jobs.

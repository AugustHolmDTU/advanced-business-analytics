# RL Optimization Log

## Baseline

Configuration:
- Environment: `configs/env/corridor_calibrated.yaml`
- Demand: `configs/demand/base.yaml`
- RL: `configs/rl/dqn.yaml`

Observed baseline behavior before changes:
- RL trained end-to-end but underperformed simple heuristics.
- Invalid actions increased during training instead of decreasing.
- The environment had no explicit `no-op` action.

Baseline evaluation summary:
- RL mean reward: `-184.58`
- Greedy mean reward: `-179.76`
- Coverage mean reward: `-181.81`
- Random mean reward: `-184.04`

Interpretation:
- The agent was likely using invalid actions as a surrogate for “do nothing”.
- This makes the RL problem harder than necessary and distorts the reward signal.

## Change 1

Added an explicit `no-op` action and valid-action masking support.

Reason:
- The agent should be able to keep the current deployment when movement is not useful.
- Invalid actions should represent true mistakes, not a missing control primitive.

Expected effect:
- Lower invalid-action count.
- More stable learning.
- Better comparison between move, relocate, and stay decisions.

Observed result after Change 1:
- Invalid actions dropped from non-zero to `0.0`.
- RL still underperformed greedy and coverage.
- Conclusion: legality was a real bug, but not the only bottleneck.

## Change 2

Introduced a stronger DQN training configuration in `configs/rl/dqn_optimized.yaml`.

Changes:
- more episodes
- larger replay buffer
- larger network
- more gradient steps
- slower epsilon decay
- lower learning rate

Reason:
- The calibrated corridor is more variable and harder than the original toy setup.
- The baseline DQN budget was probably too small for generalization across randomized episodes.

Observed result after Change 2:
- RL improved a little versus random but still did not match greedy.
- Invalid actions stayed at `0.0`.
- Late-training reward drifted downward, suggesting the agent was still not exploiting the strongest site signal reliably.

## Change 3

Added a heuristic action prior inside the DQN policy.

What it does:
- Uses the normalized site-score block already present in the observation.
- Adds a configurable prior to action selection and next-action selection during bootstrapping.
- Treats the `no-op` action prior as the best currently occupied site score.

Reason:
- The current task is close to a structured contextual decision problem.
- The greedy baseline directly exploits the same site-score signal.
- Residual learning over an interpretable prior is more appropriate than forcing the network to rediscover that signal from scratch.

## Change 4

Created a resilience-oriented environment variant:
- `configs/env/corridor_resilience.yaml`
- starts with mobile chargers already filled at strong initial sites
- increases relocation cost
- slightly increases utilization and coverage bonuses

Reason:
- The original benchmark mixed initial deployment and disruption response.
- The project focus is resilience under disruption, not cold-start placement.
- This variant makes the RL decision closer to “reposition or hold” under stress, which is more aligned with the research question.

Observed result after Change 4:
- This was the strongest RL setup tested.
- RL mean reward improved relative to the non-resilience calibrated setup.
- RL served demand increased.
- RL still did not beat the greedy baseline, but the gap narrowed.

## Current Diagnosis

What helped:
- Explicit `no-op` action
- valid-action masking
- larger DQN training budget
- residual learning over a heuristic prior
- resilience-oriented start state

What still limits performance:
- The decision problem is still close to a myopic score-based relocation rule, so greedy remains very hard to beat.
- Reward is dominated by unmet demand, while movement costs are comparatively small.
- Even with `no-op`, the learned policy still relocates frequently.
- The benchmark may still favor simple score-following heuristics over value-based RL.

## Best Result So Far

Best RL configuration:
- Environment: `configs/env/corridor_resilience.yaml`
- RL: `configs/rl/dqn_optimized.yaml`

Best observed evaluation:
- RL mean reward: `-211.62`
- Greedy mean reward: `-207.78`
- Coverage mean reward: `-210.69`
- Random mean reward: `-211.86`

Interpretation:
- RL is now competitive with random and closer to coverage.
- Greedy is still the strongest baseline on the current formulation.

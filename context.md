# Project Context

This file is the short orientation note for work in this repository. It should reflect the **current final-project state**, not older experimental branches.

## Project Theme

This is the DTU 42578 Advanced Business Analytics course project on **resilience in EV charging operations under disruptions**.

The active project question is:

- can mobile charging stations improve resilience in an A-B-C EV corridor
- can adaptive control reduce queues and waiting times under disruptions
- is there evidence of spatially aware behavior rather than only aggregate improvement

## Main Deliverable

The current hand-in is centered on:

- [exam_submission/notebook.ipynb](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/exam_submission/notebook.ipynb)

This notebook is intended to function as the technical report. It is markdown-heavy, report-structured, and designed to run top-to-bottom.

The notebook follows this exact structure:

1. Motivation / problem definition
2. Simulation environment
3. Simulation design
4. RL agent and methods
5. Reward calibration
6. Training setup
7. Training the model
8. Simulation comparison rollout
9. Summary metrics table
10. Spatial awareness
11. Limitations
12. Conclusion
13. Appendix

## Submission Folder

The `exam_submission/` folder is the clean submission area. It contains:

- the main notebook
- copied artifacts used by the notebook
- notebook-only helper utilities
- a minimal `reference_code/` snapshot of the important simulator, training, RL, and config files

Important constraint:

- figures are generated inside notebook outputs
- there is intentionally no submission `figures/` folder

## Active Benchmark

The active benchmark is the synthetic **A-B-C line corridor**:

- three cities on a line
- one fixed station between A and B
- one fixed station between B and C
- six OD flows: `od_ab`, `od_ba`, `od_bc`, `od_cb`, `od_ac`, `od_ca`
- up to `10` mobile charging stations
- `2` chargers per mobile charging station
- `5` minute decision steps

Main active environment:

- [src/evch/envs/line_corridor_mobile_env.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/envs/line_corridor_mobile_env.py)

Main active simulator:

- [src/evch/sim/line_corridor.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/sim/line_corridor.py)

## Current Environment Assumptions

From the active config in [configs/env/mobile_mcs_line_abc.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/env/mobile_mcs_line_abc.yaml):

- inter-city distance: `100 km`
- fixed plugs: `12` at AB and `12` at BC
- service time: mean `30` minutes, std `8`, clipped to `15-50`
- charging-stop probability: `0.055`
- demand profile: morning peak, evening peak, and a midday bump
- disruption mode in the benchmark config: `random`

The current disruption types are:

- `capacity_drop`
- `station_outage`
- `service_time_inflation`
- `demand_surge`

## Current MCS Logistics Model

The corridor benchmark models simple but important deployment frictions:

- depot / middle to station travel
- station-to-station relocation through time, not instantaneous reassignment
- return-to-middle behavior
- recharge / unavailable periods

The observation and environment logic explicitly track logistics state such as:

- `middle_available`
- `middle_charging`
- `transit_to_ab`
- `transit_to_bc`
- `transit_to_middle`

This means the control problem is spatial and delayed. The agent is not simply toggling free capacity on and off.

## Current Control Problem

The active action space is the small directional 5-action design:

- `hold`
- `toward_ab`
- `toward_bc`
- `recall_ab`
- `recall_bc`

This is deliberate. The current project is about learning interpretable spatial reallocation under logistics constraints, not learning a large absolute allocation table.

Operationally, actions change commitment gradually, typically by one unit per step, while the environment resolves where that unit comes from:

- depot if available
- otherwise reallocation from the opposite side when possible

## Current RL Formulation

There are two RL paths in the repo:

### `torch_dqn`

Implemented primarily through:

- [src/evch/rl/simple_dql.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/rl/simple_dql.py)
- [src/evch/train/train_rl.py](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/src/evch/train/train_rl.py)

This path is important because it produces the most notebook-friendly training artifact:

- `history.json` with train reward
- train TD loss
- evaluation reward
- evaluation TD loss

The TD loss is Huber / smooth L1 loss.

### `sb3_dqn`

Configured through:

- [configs/rl/dqn_mobile_sb3.yaml](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/configs/rl/dqn_mobile_sb3.yaml)

This is the stronger candidate final-policy path conceptually, but in the current local snapshot it is not the easiest source for the full train/eval diagnostic curves used by the notebook.

## Current Reward Philosophy

The active reward is not only about aggregate throughput. It combines service incentives, queue penalties, local-pressure penalties, and MCS operating costs.

Important current components include:

- served demand reward
- unmet demand penalty
- queue length penalty
- queue wait penalty
- local queue peak penalty
- local wait peak penalty
- active MCS cost
- activation cost
- adjustment cost
- idle capacity penalty
- spatial deficit alignment and direction bonuses

This means:

- one-sided local service failures are meant to matter
- reward and operational metrics are related but not identical
- reward calibration can change whether the policy prefers aggressive MCS use or more conservative deployment

## Current Evaluation Philosophy

The active project separates:

- training episodes
- periodic held-out evaluation during training
- final held-out comparison rollouts
- separate stress-style robustness scenarios

Important interpretation rule:

- training curves are useful evidence of learning dynamics
- final operational claims should be based on held-out rollout behavior, not on TD loss alone

## Current Submission Evidence Boundary

This is the most important current caveat.

The final notebook is honest about the local artifact state:

- a current-compatible final RL checkpoint is **not** present in the local repository snapshot
- a copied current training-history artifact **is** present
- current held-out baseline rollouts are present for fixed / no-agent, threshold, and reactive baselines

Because of that:

- the notebook can make a reproducible claim about baseline and adaptive-threshold comparisons
- the notebook can discuss RL formulation and RL learning curves
- the notebook does **not** make a final reproducible RL-vs-baseline rollout claim for the active benchmark

That honesty is intentional and should be preserved.

## Current Best Reading Order

Start with:

- [README.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/README.md)
- [context.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/context.md)
- [exam_submission/README.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/exam_submission/README.md)
- [exam_submission/notebook.ipynb](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/exam_submission/notebook.ipynb)

Then inspect:

- `src/evch/sim/`
- `src/evch/envs/`
- `src/evch/rl/`
- `src/evch/train/`
- `configs/env/`
- `configs/rl/`
- `configs/experiment/`
- `bsub/`

## What Good Work Looks Like Now

At the current stage, good work usually means:

- improving clarity, reproducibility, and evidence quality
- making station-level behavior more interpretable
- keeping report claims aligned with actual artifacts
- improving the path from training artifacts to exam-ready outputs
- strengthening the benchmark without hiding uncertainty or missing evidence

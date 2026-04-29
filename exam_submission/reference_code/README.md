This folder contains the minimal project code and configs referenced by the exam notebook.

Purpose
- document the exact simulation and RL implementation used in the project
- make the training and evaluation pipeline inspectable without copying the whole repository
- support the optional appendix demo helper in `exam_submission/exam_storyline.ipynb`

What is included
- `src/evch/train/`: RL training, evaluation-suite, and comparison entrypoints
- `src/evch/envs/`: active line-corridor environment and related environment factory files
- `src/evch/rl/`: simple DQN / DQL implementation and evaluation helper
- `src/evch/sim/`: line-corridor simulator
- `src/evch/utils/`: runtime, logging, seeding, and I/O helpers used by training
- `src/evch/config/`: YAML config loader
- `src/evch/baselines/`: baseline policy definitions
- `src/evch/models/common.py`: shared MLP builder used by the RL agent
- `configs/`: the active benchmark configs plus a short demo override

Important note
- The notebook keeps the main reported results fully reproducible from copied artifacts in `exam_submission/data/`.
- The short appendix demo is separate from the real reported results and is only meant to show how an examiner can launch a small training/evaluation run locally.

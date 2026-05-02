# Exam Submission

`notebook.ipynb` is the main report notebook.

This folder is also trimmed so it can run on its own. It only keeps the code, configs, and copied artifacts that are used in the notebook or in the local train/evaluate path for the line-corridor RL setup.

## What Is In Here

- `notebook.ipynb`
  - the report notebook
- `configs/`
  - the small config set still used
- `data/generated/`
  - copied CSV/JSON artifacts used by the notebook figures and tables
- `demo_outputs/`
  - the kept local checkpoint and demo training outputs
- `src/evch/`
  - the minimal project package needed to simulate, train, and evaluate the model
- `utils/`
  - notebook helper code and the simple appendix demo runner
- `pyproject.toml`
  - local package metadata for this trimmed repo

## Kept Configs

- `configs/env/mobile_mcs_line_abc.yaml`
- `configs/demand/base.yaml`
- `configs/rl/dqn_mobile_simple.yaml`
- `configs/debug/logging_disabled.yaml`
- `configs/experiment/mobile_mcs_line_abc.yaml`
- `configs/experiment/mobile_mcs_line_abc_heldout_comparison.yaml`
- `configs/experiment/mobile_mcs_line_abc_train_only.yaml`
- `configs/experiment/demo_mobile_mcs_line_abc_short.yaml`

## Kept Artifacts

- `data/generated/heldout/heldout_no_agent_baseline_timestep_metrics.csv`
- `data/generated/heldout/heldout_rl_agent_timestep_metrics.csv`
- `data/generated/reward_sweeps/reward_sweep_mcs_cost.csv`
- `data/generated/reward_sweeps/reward_sweep_queue_cost.csv`
- `data/generated/training/training_history.json`
- `demo_outputs/best_model.pt`
- `demo_outputs/history.json`
- `demo_outputs/training_summary.json`

## What Was Removed

- SB3-only configs and code paths
- TomTom, Denmark corridor, and uncertainty-model code
- unused environment variants
- old exported notebook files
- duplicated copied config files under `data/configs/`

## How To Run

From inside `exam_submission/`:

```bash
jupyter notebook notebook.ipynb
```

The notebook setup cell now imports everything from `utils.notebook_context`, so there is no dependency on the parent repo.

## Train A Small Local Run

```bash
PYTHONPATH=src python3 -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/debug/logging_disabled.yaml \
  --config configs/experiment/mobile_mcs_line_abc_train_only.yaml \
  --config configs/experiment/demo_mobile_mcs_line_abc_short.yaml
```

This writes the local run into `demo_outputs/` through the appendix demo helper setup.

## Evaluate The Kept Checkpoint

```bash
PYTHONPATH=src python3 -m evch.train.evaluate_policies \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/debug/logging_disabled.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --agent-checkpoint demo_outputs/best_model.pt
```

## Notes

- the only checkpoint stored in this folder is `demo_outputs/best_model.pt`
- the notebook uses the copied files in `data/generated/`, not live outputs from another folder

# Exam Submission

`exam_storyline.ipynb` is the main hand-in artifact in this folder. The rest of the folder is just the small amount of copied code, configs, and generated artifacts that the notebook still depends on.

## Folder Layout

- `exam_storyline.ipynb`
  - main report notebook
- `data/configs/`
  - copied config files used by the notebook and the optional appendix demo
- `data/generated/heldout/`
  - copied held-out timestep outputs used in the rollout comparison
- `data/generated/reward_sweeps/`
  - copied reward-sweep summary tables
- `data/generated/training/`
  - copied RL training history used for the learning-curve section
- `utils/`
  - notebook-only helper code for loading artifacts, plotting, summary tables, and the optional appendix demo
- `src/evch/sim/`
  - copied simulator code that the notebook imports directly for the scripted disruption examples

## Files Kept In The Submission

Current config files:

- `data/configs/env_mobile_mcs_line_abc_current.yaml`
- `data/configs/rl_dqn_mobile_simple.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_current.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_heldout_comparison.yaml`
- `data/configs/demand_base.yaml`
- `data/configs/logging_disabled.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_train_only.yaml`
- `data/configs/demo_mobile_mcs_line_abc_short.yaml`

Current generated artifacts:

- `data/generated/heldout/heldout_no_agent_baseline_timestep_metrics.csv`
- `data/generated/heldout/heldout_rl_agent_timestep_metrics.csv`
- `data/generated/reward_sweeps/reward_sweep_mcs_cost.csv`
- `data/generated/reward_sweeps/reward_sweep_queue_cost.csv`
- `data/generated/training/training_history.json`

Current helper/code files:

- `utils/demo_runner.py`
- `utils/load_artifacts.py`
- `utils/plotting.py`
- `utils/project_summary.py`
- `src/evch/sim/common.py`
- `src/evch/sim/line_corridor.py`

## What The Notebook Uses

The main notebook flow uses:

- `exam_submission.utils.load_artifacts`
- `exam_submission.utils.plotting`
- `exam_submission.utils.project_summary`
- `evch.sim.line_corridor` from `exam_submission/src`
- the copied files under `data/configs/` and `data/generated/`

The optional appendix demo also uses:

- `exam_submission.utils.demo_runner`
- the main repository `src/` package outside this folder, because a real training run needs more than the slim copied simulator files

## How To Run

From the repository root:

```bash
jupyter notebook exam_submission/exam_storyline.ipynb
```

To execute it non-interactively:

```bash
MPLCONFIGDIR=/private/tmp/codex_mpl_exam jupyter nbconvert --to notebook --execute --inplace exam_submission/exam_storyline.ipynb
```

The notebook is set up to run from the repository root.

## Export Notes

LaTeX export:

```bash
jupyter nbconvert --to latex --no-input exam_submission/exam_storyline.ipynb
```

HTML export:

```bash
jupyter nbconvert --to html --no-input --embed-images exam_submission/exam_storyline.ipynb
```

This cleaned folder does not store exported PDF, HTML, LaTeX, or figure files. Those can be regenerated locally when needed.

## Notes

- `demo_outputs/` is intentionally not stored anymore. If the appendix demo is enabled and rerun, it will be recreated locally.
- The cleaned submission keeps only the files that are still referenced by the notebook or by the optional appendix demo helper.

# Exam Submission Folder

`exam_storyline.ipynb` is the main artifact in this folder. It is written as a polished technical-report notebook for the DTU 42578 final project on resilient EV charging operations under disruption.

## Contents

- `exam_storyline.ipynb`: main submission notebook and technical narrative.
- `data/`: copied configs and selected local artifacts used by the notebook.
- `utils/`: small helper modules used only by the notebook.

## How To Run

From the repository root:

```bash
jupyter notebook exam_submission/exam_storyline.ipynb
```

or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace exam_submission/exam_storyline.ipynb
```

The notebook is designed to run top-to-bottom without importing from `src/`; it only imports from `exam_submission/`.

## Selected Artifacts

Current active benchmark/config artifacts:

- `data/configs/env_mobile_mcs_line_abc_current.yaml`
- `data/configs/rl_dqn_mobile_simple.yaml`
- `data/configs/rl_dqn_mobile_sb3.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_current.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_heldout_comparison.yaml`

Current generated evaluation artifacts copied for the notebook:

- `data/generated/heldout/mobile_noop_comparison_timestep_metrics.csv`
- `data/generated/heldout/mobile_noop_comparison_summary.json`
- `data/generated/heldout/mobile_threshold_comparison_timestep_metrics.csv`
- `data/generated/heldout/mobile_threshold_comparison_summary.json`
- `data/generated/heldout/mobile_reactive_comparison_timestep_metrics.csv`
- `data/generated/heldout/mobile_reactive_comparison_summary.json`
- `data/generated/evaluation_suites_small/test_id_summary.csv`
- `data/generated/evaluation_suites_small/paired_test_results.csv`
- `data/generated/evaluation_suites_small/test_stress_summary.csv`
- `data/generated/evaluation_suites_small/scenario_manifest.csv`

Historical artifacts used only for careful context, not final claims about the active line-corridor setup:

- `data/historical/reward_sweep_summary.csv`
- `data/historical/legacy_training_history_range_c24_q2p5.json`

## Important Assumptions And Caveats

- No current-compatible RL checkpoint was found locally for the active `27`-feature, `5`-action A-B-C corridor environment.
- Because of that, the notebook does **not** make a clean final RL-vs-baseline quantitative claim for the active implementation.
- The notebook uses current baseline rollouts and current reduced baseline evaluation summaries as the main quantitative evidence.
- Older local RL artifacts are treated as historical only because they come from a superseded absolute-allocation action space.
- Queue-wait target breach metrics remain zero in the active line-corridor outputs because the current environment does not expose a nonzero queue-wait target.

## Figure Policy

Figures are generated inside notebook cells and shown as notebook outputs. There is intentionally no `figures/` folder in this submission.

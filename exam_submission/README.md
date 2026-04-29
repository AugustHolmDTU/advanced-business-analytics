# Exam Submission Folder

`exam_storyline.ipynb` is the main hand-in artifact in this folder. It is structured as a technical report for the DTU 42578 final project on resilience in EV charging operations under disruption.

## Contents

- `exam_storyline.ipynb`: main report notebook. This is the primary submission artifact.
- `data/`: copied local artifacts used by the notebook.
- `utils/`: small notebook-only helper modules.
- `reference_code/`: minimal copied training, simulation, environment, and config files needed to inspect the implementation and run the optional short demo training cell.

## Notebook Structure

The notebook follows this exact report outline:

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

Secondary plots and broader artifact summaries are intentionally moved to the appendix to keep the main narrative compact.

## How To Run

From the repository root:

```bash
jupyter notebook exam_submission/exam_storyline.ipynb
```

To execute the notebook non-interactively:

```bash
MPLCONFIGDIR=/private/tmp/codex_mpl_exam jupyter nbconvert --to notebook --execute --inplace exam_submission/exam_storyline.ipynb
```

The notebook runs top-to-bottom from the repository root. It imports only from `exam_submission/` during normal report execution.

## PDF / LaTeX Export

LaTeX export works with:

```bash
jupyter nbconvert --to latex --no-input exam_submission/exam_storyline.ipynb
```

This produces:

```bash
exam_submission/exam_storyline.tex
```

If direct PDF compilation fails, the issue is the local LaTeX installation, not the notebook structure. On this machine, XeLaTeX still requires the missing package `tcolorbox.sty`.

Reliable fallback:

```bash
jupyter nbconvert --to html --no-input --embed-images exam_submission/exam_storyline.ipynb
```

Then print the generated HTML to PDF from the browser.

## Selected Artifacts Used In The Notebook

Current active benchmark/config artifacts:

- `data/configs/env_mobile_mcs_line_abc_current.yaml`
- `data/configs/rl_dqn_mobile_simple.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_current.yaml`
- `data/configs/experiment_mobile_mcs_line_abc_heldout_comparison.yaml`
- `data/configs/rl_agent_current.yaml` if present locally

Current generated artifacts:

- `data/generated/training/history_current.json`
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
- `data/generated/evaluation_suites_small/suite_outputs.json`

Historical context artifacts used only in the reward-calibration / historical-context discussion:

- `data/historical/reward_sweep_summary.csv`
- `data/historical/legacy_training_history_range_c24_q2p5.json`

## Reference Code

`reference_code/` contains the minimal code and configs needed to inspect the active simulator and launch a short training run. It does not duplicate the full repository.

Included categories:

- `src/evch/train/`: training, evaluation, and comparison entry points
- `src/evch/envs/`: active line-corridor environment and factories
- `src/evch/rl/`: RL agent implementation and evaluation helpers
- `src/evch/sim/`: line-corridor simulator logic
- `src/evch/utils/`: logging, I/O, seeding, and runtime helpers
- `src/evch/config/`: YAML config loader
- `src/evch/baselines/`: baseline policy definitions
- `src/evch/models/common.py`: shared MLP builder
- `configs/`: active benchmark configs plus a short demo override

The optional demo cell in section 7 uses these copied configs together with the original repository `src/` package path to run a small local training/evaluation example.

## Important Assumptions And Caveats

- No current-compatible final RL checkpoint was found locally for the active `27`-feature, `5`-action line-corridor environment.
- Because of that, the notebook does not claim a final reproducible RL-vs-baseline held-out rollout result for the active benchmark.
- The train/eval learning-curve section uses `data/generated/training/history_current.json`, copied from the latest RL-agent training run.
- The main rollout comparison therefore emphasizes the strongest current comparable policy evidence available locally: fixed-only versus threshold control.
- The reactive baseline is kept only as secondary appendix material.
- Queue-wait target breach metrics remain zero in the active line-corridor outputs because the current artifacts do not expose a non-zero queue-wait target threshold.
- The short demo training cell is illustrative only and is clearly separated from the reported results.

## Figure Policy

Figures are generated inside notebook cells and shown as notebook outputs. There is intentionally no `figures/` folder in this submission.

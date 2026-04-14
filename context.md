# Project Context

This file is the lightweight orientation note for coding agents working in this repository.

It should answer:
- what kind of project this is,
- what the repository is currently being used for,
- what constraints matter,
- what is still open and should not be treated as fixed.

It should be updated as the project evolves.

## Project Theme

This is a DTU Advanced Business Analytics course project in the broader area of resilience-oriented analytics for EV charging systems.

The central question is not simply how to optimize charging under normal conditions. The more important framing is:

- how EV charging systems behave under stress or disruption,
- how uncertainty should be represented in the modeling pipeline,
- how decision policies can improve robustness or recovery,
- how to evaluate whether a method is actually useful under changing conditions.

The working problem statement is now much narrower than that broad framing:
- model disruption-driven stress on EV charging along a long-distance corridor,
- deploy or reposition a small number of mobile charging units,
- compare RL against simpler baselines under randomized disruption scenarios,
- keep the environment synthetic so the policy can be trained for generalization rather than one fixed geography.

## What This Repo Is For

This repository is a research sandbox for building and comparing methods, not a production platform.

It is meant to support a small but extensible workflow that can include:
- a synthetic or semi-synthetic digital twin,
- demand generation and forecasting,
- uncertainty-aware supervised models,
- decision policies such as heuristics, RL, or optimization-inspired baselines,
- evaluation under disruption scenarios,
- reproducible experiments locally and on Slurm/HPC.

The repo should help the team test ideas quickly and refine the project formulation over time.

## Current Direction

At the moment, the codebase is oriented around resilience-oriented redeployment of mobile EV charging support under disruption.

The current implementation uses:
- a synthetic corridor-based environment with major hubs, service areas, and highway exits,
- stochastic demand with corridor, urban, suburban, and logistics zone profiles,
- disruption scenarios such as station outages, traffic slowdowns, and demand spikes,
- mobile charger deployment / relocation decisions over discrete candidate sites,
- domain randomization across training episodes,
- optional TomTom-grounded calibration priors for infrastructure and travel-time realism,
- uncertainty-aware demand models,
- a single-agent RL baseline,
- heuristic comparison baselines.

This is the current working setup. The main shift that has already happened is:
- away from fixed placement,
- away from small urban abstractions,
- toward a corridor-style resilience benchmark with mobile support and synthetic generalization.

What is still open is whether RL becomes the main decision method or remains one comparator among heuristics and other structured policies.

Agents should preserve flexibility for those changes.

## Current Repo Capabilities

The repository already contains working components for:
- synthetic corridor geography and demand generation,
- a Gymnasium-compatible simulation environment,
- direct SVG plotting of the synthetic corridor layout,
- TomTom snapshot fetching and multi-location calibration of synthetic priors,
- Gaussian probabilistic regression with heteroscedastic uncertainty,
- quantile regression for predictive intervals,
- RL training with a DQN-style baseline,
- heuristic baselines such as greedy, coverage, and random,
- experiment configs, logging hooks, tests, and Slurm templates.

These components are real and runnable, but they still represent an experimental baseline rather than a final scientific result.

The current synthetic environment is not meant to reproduce one real city. Instead, the intended workflow is:
- use real web/API data such as TomTom to calibrate priors,
- generate synthetic corridor instances from those priors,
- train and evaluate decision policies only in the synthetic environment.

## What Is Important

When making decisions in this repo, prioritize:
- resilience framing over generic optimization,
- analytical clarity over engineering complexity,
- modularity over one-off hacks,
- reproducibility over ad hoc experimentation,
- small tractable experiments over unrealistic scope expansion.

The project should remain explainable and defensible in a course setting.

## What Is Not Fixed

The following should be treated as open design choices unless explicitly decided elsewhere:
- the exact corridor graph richness beyond the current linear corridor abstraction,
- whether the controlled asset stays purely mobile charging units or becomes a hybrid with fixed-capacity interventions,
- the exact disruption set,
- the exact resilience metric,
- whether uncertainty is only modeled upstream or also used directly in downstream decisions,
- whether RL remains the main decision method or becomes one of several compared approaches,
- how much real external data must be integrated beyond calibration.

The following should currently be treated as fixed unless explicitly changed:
- the main benchmark is synthetic rather than a direct real-place replay,
- the current testbed is corridor-based,
- the control problem is single-agent mobile charger deployment / repositioning under disruption,
- generalization across randomized scenarios is a core design requirement.

Do not hard-code assumptions that make one of these choices difficult to change later.

## Scope Boundaries

This project is intentionally not trying to build:
- a full city-scale transportation simulator,
- a production-grade digital twin,
- a highly detailed power systems model,
- a giant data engineering stack,
- a large multi-agent system unless that later becomes clearly justified.

A smaller but sharper project is better than a broad but shallow one.

## What Good Progress Looks Like

Good progress in this repo usually means one or more of the following:
- the project question becomes clearer,
- the simulation becomes more aligned with the resilience framing,
- the uncertainty modeling becomes more meaningful,
- the evaluation becomes more rigorous,
- baseline comparisons become more informative,
- the implementation becomes easier for others to understand and extend.

Progress is not just adding more code.

At the current stage, good progress also includes:
- making the synthetic corridor more defensible and interpretable,
- improving the calibration story without turning the project into a one-city case study,
- making baseline comparisons sharper,
- understanding when RL is actually justified instead of assuming it must win.

## Guidance For Coding Agents

When working in this repository:
- first understand the current formulation before expanding it,
- keep changes modular and config-driven,
- avoid locking the project into one narrow formulation unless explicitly requested,
- preserve comparability across baselines,
- prefer simple, well-justified assumptions,
- leave clean seams for later data integration or richer simulation,
- document important assumptions and tradeoffs.

If a task is ambiguous, prefer solutions that improve reuse and future adaptability.

Also note:
- the synthetic-corridor plot should remain easy to interpret and should not depend on a fragile plotting stack,
- the RL benchmark should stay small enough to train locally,
- if RL underperforms heuristics, treat that as a scientific finding to analyze rather than a bug to hide.

## Current Findings

The main empirical findings so far are:
- the synthetic corridor pipeline is working end-to-end,
- TomTom data is currently used to calibrate synthetic priors rather than define the actual RL environment,
- the RL baseline improved after adding an explicit `no-op`, valid-action masking, a larger DQN budget, and a resilience-oriented benchmark,
- the greedy heuristic still outperforms the current RL agent on the strongest benchmark tested so far.

The best current resilience-benchmark comparison is approximately:
- RL mean reward: `-211.62`
- Greedy mean reward: `-207.78`
- Coverage mean reward: `-210.69`
- Random mean reward: `-211.86`

Interpret this carefully:
- the RL pipeline works,
- the current formulation is still hard for RL to beat with a plain DQN,
- this is useful project information, not merely a failed implementation.

## Practical Reading Order

For fast orientation, start with:
- [README.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/README.md)
- [IMPLEMENTATION_NOTES.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/IMPLEMENTATION_NOTES.md)
- [context.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/context.md)
- [RL_CHANGELOG.md](/Users/nicolaigarderhansen/Desktop/DTU/Kandidat/2.%20Sem/42578/advanced-business-analytics/RL_CHANGELOG.md)

Then inspect the main code areas:
- `src/evch/envs/`
- `src/evch/data/`
- `src/evch/models/`
- `src/evch/rl/`
- `src/evch/train/`
- `configs/`

## How To Update This File

This file should evolve with the project.

Update it when:
- the main research framing changes,
- the chosen decision problem changes,
- the simulation assumptions change materially,
- a major data source is added,
- the evaluation philosophy changes,
- the project narrows or broadens in scope.

Keep it high-level.
It should describe the project direction and constraints, not duplicate detailed implementation docs.

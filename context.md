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

The exact final problem statement may still evolve, but it should stay within that resilience-under-uncertainty space.

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

At the moment, the codebase is oriented around EV charging placement or redeployment decisions under uncertain demand.

The current implementation uses:
- a synthetic environment,
- stochastic demand,
- disruption scenarios,
- uncertainty-aware demand models,
- a single-agent RL baseline,
- heuristic comparison baselines.

This is the current working setup, but it should not be treated as the only acceptable formulation. Future work may shift:
- from fixed charger placement toward relocation or mobile support,
- from small urban abstractions toward corridor-based or graph-based settings,
- from pure simulation toward proxy-driven or externally sourced data,
- from RL-centric decision-making toward a stronger mix of heuristics, optimization, and stress testing.

Agents should preserve flexibility for those changes.

## Current Repo Capabilities

The repository already contains working components for:
- synthetic geography and demand generation,
- a Gymnasium-compatible simulation environment,
- Gaussian probabilistic regression with heteroscedastic uncertainty,
- quantile regression for predictive intervals,
- RL training with a DQN-style baseline,
- heuristic baselines such as greedy, coverage, and random,
- experiment configs, logging hooks, tests, and Slurm templates.

These components are real and runnable, but they represent an early-stage baseline rather than the final scientific contribution.

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
- whether the main environment is urban-area based, corridor based, or graph based,
- whether the asset being controlled is fixed chargers, relocatable chargers, mobile charging units, or a hybrid,
- the exact disruption set,
- the exact resilience metric,
- whether uncertainty is only modeled upstream or also used directly in downstream decisions,
- whether RL remains the main decision method or becomes one of several compared approaches,
- how much real external data must be integrated.

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

## Practical Reading Order

For fast orientation, start with:
- [README.md](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/README.md)
- [IMPLEMENTATION_NOTES.md](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/IMPLEMENTATION_NOTES.md)
- [context.md](/Users/Saxe/Desktop/Business Analytics/2. semester/Adv BA/advanced-business-analytics/context.md)

Then inspect the main code areas:
- `src/evch/envs/`
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

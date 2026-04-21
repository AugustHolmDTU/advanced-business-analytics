#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
export WANDB_PROJECT="${WANDB_PROJECT:-adaptive-ev-charging}"
export WANDB_ENTITY="${WANDB_ENTITY:-EV-charging}"
mkdir -p "$MPLCONFIGDIR"

python -m evch.train.generate_synthetic_data \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/experiment/local_demo.yaml

python -m evch.train.train_uncertainty \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/model/gaussian.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/local_demo.yaml

python -m evch.train.train_rl \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/local_demo.yaml

python -m evch.train.evaluate_policies \
  --config configs/env/base.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/local_demo.yaml \
  --agent-checkpoint outputs/local_demo/rl/best_model.pt

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
export WANDB_PROJECT="${WANDB_PROJECT:-A-B-C}"
export WANDB_ENTITY="${WANDB_ENTITY:-EV-charging}"
mkdir -p "$MPLCONFIGDIR"

python3 -m evch.train.train_rl \
  --config configs/env/mobile_mcs_line_abc.yaml \
  --config configs/env/mobile_mcs_line_abc_reward_queue_cost_heavy.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn_mobile_simple.yaml \
  --config configs/logging/base.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/mobile_mcs_line_abc.yaml \
  --config configs/experiment/mobile_mcs_line_abc_reward_queue_cost_heavy.yaml \
  --config configs/experiment/mobile_mcs_line_abc_train_only.yaml

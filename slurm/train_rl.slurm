#!/bin/bash
#SBATCH --job-name=evch-rl
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:0

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/configs}"

# Activate your environment here. Examples:
# source "$PROJECT_ROOT/.venv/bin/activate"
# conda activate my-evch-env

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python -m evch.train.train_rl \
  --config "$CONFIG_DIR/env/base.yaml" \
  --config "$CONFIG_DIR/demand/base.yaml" \
  --config "$CONFIG_DIR/rl/dqn.yaml" \
  --config "$CONFIG_DIR/logging/base.yaml" \
  --config "$CONFIG_DIR/experiment/local_demo.yaml"

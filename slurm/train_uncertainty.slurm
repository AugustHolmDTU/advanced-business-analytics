#!/bin/bash
#SBATCH --job-name=evch-uncertainty
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --array=0-1

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

MODEL_CONFIGS=(
  "$CONFIG_DIR/model/gaussian.yaml"
  "$CONFIG_DIR/model/quantile.yaml"
)

python -m evch.train.train_uncertainty \
  --config "$CONFIG_DIR/env/base.yaml" \
  --config "$CONFIG_DIR/demand/base.yaml" \
  --config "${MODEL_CONFIGS[$SLURM_ARRAY_TASK_ID]}" \
  --config "$CONFIG_DIR/logging/base.yaml" \
  --config "$CONFIG_DIR/experiment/local_demo.yaml"

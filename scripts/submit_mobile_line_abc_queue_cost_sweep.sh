#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALUES=(2.5 3.0 3.5 4.0 4.5)
GROUP_NAME="mobile_mcs_line_abc_queue_cost_sweep"

for value in "${VALUES[@]}"; do
  safe_value="${value/./p}"
  run_name="mobile_mcs_line_abc_sweep_queue_cost_${safe_value}"
  echo "Submitting ${run_name} with queue_length_penalty=${value}"
  (
    export PROJECT_ROOT="$ROOT_DIR"
    export SWEEP_KIND="queue_cost"
    export REWARD_KEY="queue_length_penalty"
    export REWARD_VALUE="$value"
    export RUN_NAME="$run_name"
    export WANDB_GROUP="$GROUP_NAME"
    bsub < "$ROOT_DIR/bsub/train_mobile_line_abc_reward_sweep_variant.bsub"
  )
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALUES=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
GROUP_NAME="mobile_mcs_line_abc_queue_cost_sweep"

for value in "${VALUES[@]}"; do
  safe_value="${value/./p}"
  run_name="mobile_mcs_line_abc_sweep_queue_cost_${safe_value}"
  echo "Submitting ${run_name} with queue_length_penalty=${value}"
  bsub -env "all,PROJECT_ROOT=$ROOT_DIR,SWEEP_KIND=queue_cost,REWARD_KEY=queue_length_penalty,REWARD_VALUE=$value,RUN_NAME=$run_name,WANDB_GROUP=$GROUP_NAME" \
    < "$ROOT_DIR/bsub/train_mobile_line_abc_reward_sweep_variant.bsub"
done

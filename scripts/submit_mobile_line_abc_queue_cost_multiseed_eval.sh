#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALUES=(2.5 3.0 3.5 4.0 4.5)
GROUP_NAME="mobile_mcs_line_abc_queue_cost_multiseed_eval"
SEED_INDEX_START="${SEED_INDEX_START:-0}"
SEED_COUNT="${SEED_COUNT:-20}"

for value in "${VALUES[@]}"; do
  safe_value="${value/./p}"
  run_name="mobile_mcs_line_abc_sweep_queue_cost_${safe_value}"
  checkpoint="$ROOT_DIR/outputs/$run_name/rl/best_model.pt"
  echo "Submitting multiseed eval for ${run_name} with queue_length_penalty=${value}"
  bsub -env "all,PROJECT_ROOT=$ROOT_DIR,REWARD_KEY=queue_length_penalty,REWARD_VALUE=$value,RUN_NAME=$run_name,WANDB_GROUP=$GROUP_NAME,SEED_INDEX_START=$SEED_INDEX_START,SEED_COUNT=$SEED_COUNT,AGENT_CHECKPOINT=$checkpoint" \
    < "$ROOT_DIR/bsub/eval_mobile_line_abc_reward_sweep_multiseed_variant.bsub"
done

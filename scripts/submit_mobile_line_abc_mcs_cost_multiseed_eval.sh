#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VALUES=(18.0 24.0 30.0 36.0 42.0)
GROUP_NAME="mobile_mcs_line_abc_mcs_cost_multiseed_eval"
SEED_INDEX_START="${SEED_INDEX_START:-0}"
SEED_COUNT="${SEED_COUNT:-20}"

for value in "${VALUES[@]}"; do
  safe_value="${value/./p}"
  run_name="mobile_mcs_line_abc_sweep_mcs_cost_${safe_value}"
  checkpoint="$ROOT_DIR/outputs/$run_name/rl/best_model.pt"
  echo "Submitting multiseed eval for ${run_name} with active_mobile_station_cost=${value}"
  bsub -env "all,PROJECT_ROOT=$ROOT_DIR,REWARD_KEY=active_mobile_station_cost,REWARD_VALUE=$value,RUN_NAME=$run_name,WANDB_GROUP=$GROUP_NAME,SEED_INDEX_START=$SEED_INDEX_START,SEED_COUNT=$SEED_COUNT,AGENT_CHECKPOINT=$checkpoint" \
    < "$ROOT_DIR/bsub/eval_mobile_line_abc_reward_sweep_multiseed_variant.bsub"
done

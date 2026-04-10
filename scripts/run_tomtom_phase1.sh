#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

python3 -m evch.train.fetch_tomtom_snapshot \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/demand/base.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml \
  --config configs/tomtom/frederiksberg.yaml

python3 -m evch.train.generate_synthetic_data \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/demand/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml

python3 -m evch.train.train_uncertainty \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/demand/base.yaml \
  --config configs/model/gaussian.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml

python3 -m evch.train.train_rl \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml

python3 -m evch.train.evaluate_policies \
  --config configs/env/tomtom_frederiksberg.yaml \
  --config configs/demand/base.yaml \
  --config configs/rl/dqn.yaml \
  --config configs/logging/base.yaml \
  --config configs/experiment/tomtom_frederiksberg.yaml \
  --agent-checkpoint outputs/tomtom_frederiksberg/rl/best_model.pt

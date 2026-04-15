#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Defaults tuned for stronger baselines (adjust via env: EPOCHS, MAX_SAMPLES, etc.)
EPOCHS="${EPOCHS:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-300000}"
CTX="${CTX:-96}"
HORIZON="${HORIZON:-24}"
SPLITS="${SPLITS:-3}"
BATCH="${BATCH:-256}"
LR="${LR:-3e-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

DATASETS=(electricity weather etth1 etth2 ettm1 ettm2 metr_la)

for ds in "${DATASETS[@]}"; do
  echo "========== ${ds} =========="
  python -u scripts/run_pipeline.py \
    --dataset "${ds}" \
    --horizon "${HORIZON}" \
    --context-length "${CTX}" \
    --splits "${SPLITS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --lr "${LR}" \
    --max-samples "${MAX_SAMPLES}" \
    --num-workers "${NUM_WORKERS}" \
    --html-3d
done

echo "Done. Outputs under outputs/figures and outputs/metrics"

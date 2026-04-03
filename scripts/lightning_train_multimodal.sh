#!/usr/bin/env bash
# Run on Lightning Studio (CUDA) from financial_distress root after pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.train --mode multimodal \
  --epochs 60 \
  --batch-size 128 \
  --lr 2.5e-4 \
  --early-stopping-patience 12 \
  --min-epochs 8 \
  --balance-train \
  --loss focal \
  --lr-scheduler plateau \
  --modality-dropout 0.15 \
  --gate-entropy-coef 0.03 \
  --seed 42

python3 -m src.evaluate --mode multimodal
python3 -m src.evaluate_blend
echo "[DONE] checkpoints: multimodal metrics, blend_metrics_eval.json"

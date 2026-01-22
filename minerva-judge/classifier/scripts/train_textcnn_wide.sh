#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/textcnn_wide"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/textcnn_wide.log"

python "$REPO_ROOT/minerva-judge/classifier/baselines/train_textcnn.py" \
  --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
  --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-mode response_prompt \
  --max-tokens 2048 \
  --max-vocab 120000 \
  --embed-dim 300 \
  --kernel-sizes 3,4,5 \
  --num-filters 384 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 5 \
  --lr 1e-3 \
  | tee "$LOG_PATH"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/textcnn_lr5e-4"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/textcnn_lr5e-4.log"

python "$REPO_ROOT/minerva-judge/classifier/baselines/train_textcnn.py" \
  --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
  --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-mode response_prompt \
  --max-tokens 3072 \
  --max-vocab 100000 \
  --embed-dim 200 \
  --kernel-sizes 3,4,5 \
  --num-filters 256 \
  --dropout 0.3 \
  --batch-size 128 \
  --epochs 6 \
  --lr 5e-4 \
  | tee "$LOG_PATH"

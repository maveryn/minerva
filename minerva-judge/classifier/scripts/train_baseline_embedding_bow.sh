#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/embedding_bow"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/embedding_bow.log"

python "$REPO_ROOT/minerva-judge/classifier/baselines/train_embedding_bow.py" \
  --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
  --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-mode response_prompt \
  --max-tokens 4096 \
  --word-ngrams 1 \
  --max-vocab 200000 \
  --embed-dim 300 \
  --batch-size 256 \
  --epochs 5 \
  --lr 1e-3 \
  | tee "$LOG_PATH"

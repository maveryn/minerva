#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/hashing_sgd"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/hashing_sgd.log"

python "$REPO_ROOT/minerva-judge/classifier/baselines/train_hashing_sgd.py" \
  --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
  --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-mode response_prompt \
  --max-tokens 4096 \
  --n-features 1048576 \
  --ngram-min 1 \
  --ngram-max 2 \
  --alpha 1e-5 \
  --epochs 5 \
  | tee "$LOG_PATH"

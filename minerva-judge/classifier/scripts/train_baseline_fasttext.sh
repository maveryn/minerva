#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/baselines/fasttext"
LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/fasttext.log"

python "$REPO_ROOT/minerva-judge/classifier/baselines/train_fasttext.py" \
  --train-file "$REPO_ROOT/minerva-judge/classifier/data/train.jsonl" \
  --eval-file "$REPO_ROOT/minerva-judge/classifier/data/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --text-mode response_prompt \
  --max-tokens 1024 \
  --min-ngram 3 \
  --max-ngram 6 \
  --hash-buckets 50000 \
  --word-ngrams 2 \
  --max-vocab 50000 \
  --max-indices 20000 \
  --embed-dim 200 \
  --batch-size 512 \
  --epochs 5 \
  --lr 1e-3 \
  | tee "$LOG_PATH"

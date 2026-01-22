#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$CLASSIFIER_DIR/../.." && pwd)"

MODEL_NAME="answerdotai/ModernBERT-base"
MAX_LEN=2048
LR=5e-05
BATCH_SIZE=64
EVAL_BATCH_SIZE=64
GRAD_ACCUM=1
EPOCHS=2
RUN_TAG="modernbert-2k-lr5e-05-focal"
OUTPUT_DIR="$REPO_ROOT/minerva-judge/classifier/outputs/$RUN_TAG"

FOCAL_GAMMA=2.0
FOCAL_ALPHA_BAD=0.75
FOCAL_ALPHA_GOOD=0.25

LOG_DIR="$REPO_ROOT/minerva-judge/classifier/experiments/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/$RUN_TAG.log"

python "$REPO_ROOT/minerva-judge/classifier/train_longformer.py" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "$MAX_LEN" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accumulation "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --epochs "$EPOCHS" \
  --loss-type focal \
  --focal-gamma "$FOCAL_GAMMA" \
  --focal-alpha-bad "$FOCAL_ALPHA_BAD" \
  --focal-alpha-good "$FOCAL_ALPHA_GOOD" \
  --bf16 \
  | tee "$LOG_PATH"

python "$REPO_ROOT/minerva-judge/classifier/scripts/log_results.py" \
  --output-dir "$OUTPUT_DIR" \
  --model-name "$MODEL_NAME" \
  --max-length "$MAX_LEN" \
  --learning-rate "$LR" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --grad-accumulation "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --run-tag "$RUN_TAG"

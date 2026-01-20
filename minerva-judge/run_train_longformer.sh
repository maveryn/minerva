#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

TRAIN_FILE="${MINERVA_JUDGE_CLASSIFIER_TRAIN:-$ROOT_DIR/classifier/data/train.jsonl}"
EVAL_FILE="${MINERVA_JUDGE_CLASSIFIER_EVAL:-$ROOT_DIR/classifier/data/val.jsonl}"
MODEL_NAME="${MINERVA_JUDGE_CLASSIFIER_MODEL:-allenai/longformer-base-4096}"
OUTPUT_DIR="${MINERVA_JUDGE_CLASSIFIER_OUTPUT:-$ROOT_DIR/classifier/outputs/longformer-base-4096}"
MAX_LEN="${MINERVA_JUDGE_CLASSIFIER_MAX_LEN:-4096}"
BATCH_SIZE="${MINERVA_JUDGE_CLASSIFIER_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${MINERVA_JUDGE_CLASSIFIER_EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${MINERVA_JUDGE_CLASSIFIER_GRAD_ACCUM:-8}"
LR="${MINERVA_JUDGE_CLASSIFIER_LR:-2e-5}"
WEIGHT_DECAY="${MINERVA_JUDGE_CLASSIFIER_WEIGHT_DECAY:-0.01}"
EPOCHS="${MINERVA_JUDGE_CLASSIFIER_EPOCHS:-3}"
WARMUP_RATIO="${MINERVA_JUDGE_CLASSIFIER_WARMUP_RATIO:-0.06}"
SEED="${MINERVA_JUDGE_CLASSIFIER_SEED:-1337}"
USE_FP16="${MINERVA_JUDGE_CLASSIFIER_FP16:-}"
USE_BF16="${MINERVA_JUDGE_CLASSIFIER_BF16:-}"

cmd=(python "$ROOT_DIR/classifier/train_longformer.py"
  --train-file "$TRAIN_FILE"
  --eval-file "$EVAL_FILE"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accumulation "$GRAD_ACCUM"
  --learning-rate "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --epochs "$EPOCHS"
  --warmup-ratio "$WARMUP_RATIO"
  --seed "$SEED"
)

if [ -n "$USE_FP16" ]; then
  cmd+=(--fp16)
fi
if [ -n "$USE_BF16" ]; then
  cmd+=(--bf16)
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"

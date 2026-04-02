#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/rlvr${PYTHONPATH:+:$PYTHONPATH}"

export SFT_MODEL_PATH="${SFT_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export SFT_N_GPUS_PER_NODE="${SFT_N_GPUS_PER_NODE:-1}"
export SFT_SAVE_BEST_ONLY="${SFT_SAVE_BEST_ONLY:-true}"
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
export SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-8}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"
export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-10}"
export SFT_TEST_FREQ="${SFT_TEST_FREQ:-10}"

DART_TRAIN_PATH="${DART_SFT_TRAIN_PATH:-$REPO_ROOT/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet}"
DART_MINERVA_VAL_PATH="${DART_SFT_MINERVA_VAL_PATH:-$REPO_ROOT/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet}"
DART_ATHENA_VAL_PATH="${DART_SFT_ATHENA_VAL_PATH:-}"

if [ -z "$DART_ATHENA_VAL_PATH" ]; then
  echo "DART_SFT_ATHENA_VAL_PATH must be set to the full Athena synthetic validation parquet." >&2
  exit 1
fi

for path in "$DART_TRAIN_PATH" "$DART_MINERVA_VAL_PATH" "$DART_ATHENA_VAL_PATH"; do
  if [ ! -f "$path" ]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
done

VAL_FILES="['$DART_MINERVA_VAL_PATH','$DART_ATHENA_VAL_PATH']"

MODEL_NAME="$(basename "$SFT_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-dart_sft_${MODEL_SLUG}}"
export SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$REPO_ROOT/rlvr/checkpoints/dart/$SFT_EXPERIMENT_NAME}"
export SFT_TRAIN_PATH="$DART_TRAIN_PATH"

exec "$REPO_ROOT/rlvr/cti-scripts/train_minerva_sft.sh" \
  data.val_files="$VAL_FILES" \
  data.reward_eval_files='[]' \
  trainer.skip_val_loss=false \
  trainer.save_best_metric='val/loss' \
  trainer.save_best_mode='min' \
  trainer.max_ckpt_to_keep=1 \
  trainer.checkpoint.save_contents='["hf_model"]' \
  trainer.checkpoint.load_contents='[]' \
  model.strategy=fsdp \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  trainer.logger='["console", "wandb"]' \
  "$@"

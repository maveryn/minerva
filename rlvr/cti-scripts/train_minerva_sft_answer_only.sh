#!/usr/bin/env bash
set -euo pipefail

# Pure answer-only SFT training script for Minerva (loss-based validation).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1

MODEL_PATH="${SFT_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-minerva_sft_answer_${MODEL_SLUG}}"

TRAIN_PATH="${SFT_TRAIN_PATH:-$DATA_DIR/minerva_base_sft/minerva_base_sft_answer_train.parquet}"
VAL_PATH="${SFT_VAL_PATH:-$DATA_DIR/minerva_base_sft/minerva_base_sft_answer_val.parquet}"

TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="['$VAL_PATH']"

OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$ROOT_DIR/checkpoints/minerva/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-8}"
MAX_LENGTH="${SFT_MAX_LENGTH:-4048}"
TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-2}"
SAVE_FREQ="${SFT_SAVE_FREQ:-10}"
TEST_FREQ="${SFT_TEST_FREQ:-10}"
N_GPUS="${SFT_N_GPUS_PER_NODE:-1}"

EXTRA_TRAINER_ARGS=()
EXTRA_TRAINER_ARGS+=(trainer.total_epochs="$TOTAL_EPOCHS")
if [ -n "${SFT_TOTAL_STEPS:-}" ]; then
  EXTRA_TRAINER_ARGS+=(trainer.total_training_steps="$SFT_TOTAL_STEPS")
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=left \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  model.partial_pretrain="$MODEL_PATH" \
  model.strategy=fsdp \
  model.fsdp_config.model_dtype=bfloat16 \
  trainer.default_local_dir="$OUTPUT_ROOT" \
  trainer.project_name='minerva' \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.skip_val_loss=false \
  trainer.save_best_only=true \
  trainer.save_best_metric='val/loss' \
  trainer.save_best_mode='min' \
  trainer.save_best_dir='best' \
  trainer.checkpoint.save_contents='["hf_model"]' \
  trainer.checkpoint.load_contents='[]' \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger='["console", "wandb"]' \
  "${EXTRA_TRAINER_ARGS[@]}" \
  "$@"

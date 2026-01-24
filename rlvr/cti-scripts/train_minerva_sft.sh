#!/usr/bin/env bash
set -euo pipefail

# Pure SFT training script for Minerva using judge-selected responses.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

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
EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-minerva_sft_${MODEL_SLUG}}"

TRAIN_PATH="${SFT_TRAIN_PATH:-$DATA_DIR/minerva_base_sft/minerva_base_sft_train.parquet}"

# Reward-eval paths mirror train_minerva_grpo.sh (RL-style parquets).
VAL_PATHS=(
  "${SFT_VAL_PATH_1:-$DATA_DIR/minerva_base/minerva_base_dev.parquet}"
  "${SFT_VAL_PATH_2:-$DATA_DIR/athena/athena_cti_ate.parquet}"
  "${SFT_VAL_PATH_3:-$DATA_DIR/athena/athena_cti_ckt.parquet}"
  "${SFT_VAL_PATH_4:-$DATA_DIR/athena/athena_cti_rcm.parquet}"
  "${SFT_VAL_PATH_5:-$DATA_DIR/athena/athena_cti_rms.parquet}"
  "${SFT_VAL_PATH_6:-$DATA_DIR/athena/athena_cti_taa.parquet}"
  "${SFT_VAL_PATH_7:-$DATA_DIR/athena/athena_cti_vsp.parquet}"
  "${SFT_VAL_PATH_8:-$DATA_DIR/seceval/seceval_mini.parquet}"
)

REWARD_EVAL_PATHS=()
for path in "${VAL_PATHS[@]}"; do
  if [ -n "$path" ] && [ -f "$path" ]; then
    REWARD_EVAL_PATHS+=("$path")
  fi
done
if [ ${#REWARD_EVAL_PATHS[@]} -eq 0 ]; then
  REWARD_EVAL_PATHS=("$DATA_DIR/minerva_base/minerva_base_dev.parquet")
fi

TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="["
for i in "${!REWARD_EVAL_PATHS[@]}"; do
  if [ "$i" -gt 0 ]; then
    VAL_FILES+=","
  fi
  VAL_FILES+="'${REWARD_EVAL_PATHS[$i]}'"
done
VAL_FILES+="]"

OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-$ROOT_DIR/checkpoints/minerva/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-4}"
MAX_LENGTH="${SFT_MAX_LENGTH:-3072}"
REWARD_EVAL_BATCH_SIZE="${SFT_REWARD_EVAL_BATCH_SIZE:-64}"
REWARD_EVAL_MAX_PROMPT_LEN="${SFT_REWARD_EVAL_MAX_PROMPT_LEN:-2048}"
REWARD_EVAL_MAX_RESPONSE_LEN="${SFT_REWARD_EVAL_MAX_RESPONSE_LEN:-1024}"
REWARD_EVAL_TEMPERATURE="${SFT_REWARD_EVAL_TEMPERATURE:-0.0}"
REWARD_EVAL_TOP_P="${SFT_REWARD_EVAL_TOP_P:-1.0}"
TOTAL_STEPS="${SFT_TOTAL_STEPS:-500}"
SAVE_FREQ="${SFT_SAVE_FREQ:-10}"
TEST_FREQ="${SFT_TEST_FREQ:-10}"
N_GPUS="${SFT_N_GPUS_PER_NODE:-2}"

torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$TRAIN_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=error \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.reward_eval_files="$VAL_FILES" \
  data.reward_eval_batch_size="$REWARD_EVAL_BATCH_SIZE" \
  data.reward_eval_max_prompt_length="$REWARD_EVAL_MAX_PROMPT_LEN" \
  data.reward_eval_max_response_length="$REWARD_EVAL_MAX_RESPONSE_LEN" \
  data.reward_eval_temperature="$REWARD_EVAL_TEMPERATURE" \
  data.reward_eval_top_p="$REWARD_EVAL_TOP_P" \
  model.partial_pretrain="$MODEL_PATH" \
  trainer.default_local_dir="$OUTPUT_ROOT" \
  trainer.project_name='minerva' \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.skip_val_loss=true \
  trainer.save_best_only=true \
  trainer.save_best_metric='val/reward_mean' \
  trainer.save_best_mode='max' \
  trainer.save_best_dir='best' \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger='["console", "wandb"]' \
  "$@"

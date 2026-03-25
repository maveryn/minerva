#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/rlvr${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

: "${SFT_MODEL_PATH:?SFT_MODEL_PATH must be set}"
: "${SFT_TRAIN_PATH:?SFT_TRAIN_PATH must be set}"
: "${SFT_OUTPUT_ROOT:?SFT_OUTPUT_ROOT must be set}"
: "${SFT_EXPERIMENT_NAME:?SFT_EXPERIMENT_NAME must be set}"

export SFT_SAVE_BEST_ONLY="${SFT_SAVE_BEST_ONLY:-true}"
export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-1}"
export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-128}"
export SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-8}"
export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-4096}"
export SFT_MODEL_DTYPE="${SFT_MODEL_DTYPE:-bfloat16}"
export SFT_MODEL_STRATEGY="${SFT_MODEL_STRATEGY:-fsdp}"
export SFT_ENABLE_GRADIENT_CHECKPOINTING="${SFT_ENABLE_GRADIENT_CHECKPOINTING:-true}"
export SFT_REWARD_EVAL_BATCH_SIZE="${SFT_REWARD_EVAL_BATCH_SIZE:-64}"
export SFT_REWARD_EVAL_MAX_PROMPT_LEN="${SFT_REWARD_EVAL_MAX_PROMPT_LEN:-2048}"
export SFT_REWARD_EVAL_MAX_RESPONSE_LEN="${SFT_REWARD_EVAL_MAX_RESPONSE_LEN:-1024}"
export SFT_N_GPUS_PER_NODE="${SFT_N_GPUS_PER_NODE:-1}"
export SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-50}"
export SFT_TEST_FREQ="${SFT_TEST_FREQ:-50}"

exec "$ROOT_DIR/rlvr/cti-scripts/train_minerva_sft.sh" \
  data.reward_eval_files='[]' \
  model.strategy="$SFT_MODEL_STRATEGY" \
  model.fsdp_config.model_dtype="$SFT_MODEL_DTYPE" \
  model.enable_gradient_checkpointing="$SFT_ENABLE_GRADIENT_CHECKPOINTING" \
  trainer.logger='["console"]' \
  trainer.max_ckpt_to_keep=1 \
  trainer.checkpoint.save_contents='["hf_model"]' \
  trainer.checkpoint.load_contents='[]'

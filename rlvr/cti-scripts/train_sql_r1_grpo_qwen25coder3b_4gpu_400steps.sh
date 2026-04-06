#!/usr/bin/env bash
set -euo pipefail

# SQL-R1 vanilla GRPO run on 4 GPUs using the Qwen2.5-Coder-3B-Instruct base model.
# This matches the large-batch MinervaRL setup as closely as possible, but without Noctua/ACR,
# and runs for a fixed 400-step budget.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RLVR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export SQLR1_ROOT="${SQLR1_ROOT:-/home/jovyan/work/SQL-R1}"
export SQLR1_MODEL_PATH="${SQLR1_MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
export SQLR1_TOKENIZER_PATH="${SQLR1_TOKENIZER_PATH:-$SQLR1_MODEL_PATH}"

export SQLR1_TRAIN_PATH="${SQLR1_TRAIN_PATH:-$SQLR1_ROOT/example_data/train.parquet}"
export SQLR1_VAL_PATH="${SQLR1_VAL_PATH:-$SQLR1_ROOT/example_data/test.parquet}"
export SQLR1_DB_ROOT="${SQLR1_DB_ROOT:-$SQLR1_ROOT/data/NL2SQL/SynSQL-2.5M/databases}"

export SQLR1_TRAIN_BATCH_SIZE="${SQLR1_TRAIN_BATCH_SIZE:-128}"
export SQLR1_VAL_BATCH_SIZE="${SQLR1_VAL_BATCH_SIZE:-128}"
export SQLR1_MAX_PROMPT_LEN="${SQLR1_MAX_PROMPT_LEN:-4096}"
export SQLR1_MAX_RESPONSE_LEN="${SQLR1_MAX_RESPONSE_LEN:-2048}"
export SQLR1_ROLLOUT_N="${SQLR1_ROLLOUT_N:-8}"
export SQLR1_ROLLOUT_TEMPERATURE="${SQLR1_ROLLOUT_TEMPERATURE:-1.1}"
export SQLR1_ACTOR_LR="${SQLR1_ACTOR_LR:-3e-7}"
export SQLR1_KL_COEF="${SQLR1_KL_COEF:-0.001}"

# 400 steps at batch size 128 needs more than the default 10-epoch budget.
export SQLR1_TOTAL_EPOCHS="${SQLR1_TOTAL_EPOCHS:-16}"

export SQLR1_N_GPUS_PER_NODE="${SQLR1_N_GPUS_PER_NODE:-4}"
export SQLR1_TENSOR_PARALLEL_SIZE="${SQLR1_TENSOR_PARALLEL_SIZE:-1}"
export SQLR1_PPO_MICRO_BATCH_SIZE="${SQLR1_PPO_MICRO_BATCH_SIZE:-4}"
export SQLR1_LOGPROB_MICRO_BATCH_SIZE="${SQLR1_LOGPROB_MICRO_BATCH_SIZE:-16}"
export SQLR1_ROLLOUT_GPU_UTIL="${SQLR1_ROLLOUT_GPU_UTIL:-0.7}"
export SQLR1_FREE_CACHE_ENGINE="${SQLR1_FREE_CACHE_ENGINE:-true}"
export SQLR1_REWARD_TIMEOUT_S="${SQLR1_REWARD_TIMEOUT_S:-10}"

export SQLR1_SAVE_FREQ="${SQLR1_SAVE_FREQ:-20}"
export SQLR1_TEST_FREQ="${SQLR1_TEST_FREQ:-20}"
export SQLR1_EXPERIMENT_NAME="${SQLR1_EXPERIMENT_NAME:-sql_r1_grpo_qwen25coder3b_4gpu_400steps}"
export SQLR1_OUTPUT_ROOT="${SQLR1_OUTPUT_ROOT:-$RLVR_ROOT/checkpoints/sql-r1/$SQLR1_EXPERIMENT_NAME}"

exec bash "$SCRIPT_DIR/train_sql_r1_grpo.sh" \
  trainer.val_before_train=False \
  trainer.total_training_steps=400 \
  "$@"

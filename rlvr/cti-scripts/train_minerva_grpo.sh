#!/usr/bin/env bash
set -euo pipefail

# Base GRPO training script for Minerva (no TARBA/ACRD).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1

MODEL_PATH="${GRPO_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-minerva_grpo_${MODEL_SLUG}}"

TRAIN_PATH="${GRPO_TRAIN_PATH:-$DATA_DIR/minerva_base/minerva_base_train.parquet}"
VAL_PATHS=(
  "${GRPO_VAL_PATH_1:-$DATA_DIR/minerva_base/minerva_base_dev.parquet}"
  "${GRPO_VAL_PATH_2:-$DATA_DIR/athena/athena_cti_ate.parquet}"
  "${GRPO_VAL_PATH_3:-$DATA_DIR/athena/athena_cti_ckt.parquet}"
  "${GRPO_VAL_PATH_4:-$DATA_DIR/athena/athena_cti_rcm.parquet}"
  "${GRPO_VAL_PATH_5:-$DATA_DIR/athena/athena_cti_rms.parquet}"
  "${GRPO_VAL_PATH_6:-$DATA_DIR/athena/athena_cti_taa.parquet}"
  "${GRPO_VAL_PATH_7:-$DATA_DIR/athena/athena_cti_vsp.parquet}"
  "${GRPO_VAL_PATH_8:-$DATA_DIR/ifeval/ifeval_dev.parquet}"
)

TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="['${VAL_PATHS[0]}','${VAL_PATHS[1]}','${VAL_PATHS[2]}','${VAL_PATHS[3]}','${VAL_PATHS[4]}','${VAL_PATHS[5]}','${VAL_PATHS[6]}','${VAL_PATHS[7]}']"

OUTPUT_ROOT="${GRPO_OUTPUT_ROOT:-$ROOT_DIR/checkpoints/minerva/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${GRPO_VAL_BATCH_SIZE:-2500}"
MAX_PROMPT_LEN="${GRPO_MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${GRPO_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_N="${GRPO_ROLLOUT_N:-8}"
ROLLOUT_GPU_UTIL="${GRPO_ROLLOUT_GPU_UTIL:-0.95}"
TOTAL_STEPS="${GRPO_TOTAL_STEPS:-500}"
N_GPUS="${GRPO_N_GPUS_PER_NODE:-2}"
VAL_BEFORE_TRAIN="${GRPO_VAL_BEFORE_TRAIN:-false}"
SAVE_FREQ="${GRPO_SAVE_FREQ:-10}"
TEST_FREQ="${GRPO_TEST_FREQ:-10}"
SAVE_BEST_ONLY="${GRPO_SAVE_BEST_ONLY:-true}"
SAVE_BEST_METRIC="${GRPO_SAVE_BEST_METRIC:-val-core/global-val/reward/mean}"
SAVE_BEST_MODE="${GRPO_SAVE_BEST_MODE:-max}"
SAVE_BEST_DIR="${GRPO_SAVE_BEST_DIR:-best}"

ENTROPY_COEFF="${GRPO_ENTROPY_COEFF:-0.0}"
ACTOR_LR="${GRPO_ACTOR_LR:-1e-6}"
TENSOR_PARALLEL_SIZE="${GRPO_TENSOR_PARALLEL_SIZE:-1}"

REWARD_FN_PATH="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.dataloader_num_workers=0 \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=reward_minerva \
    +custom_reward_function.reward_kwargs.return_dict=true \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    +trainer.save_best_only="$SAVE_BEST_ONLY" \
    +trainer.save_best_metric="$SAVE_BEST_METRIC" \
    +trainer.save_best_mode="$SAVE_BEST_MODE" \
    +trainer.save_best_dir="$SAVE_BEST_DIR" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.default_local_dir="$OUTPUT_ROOT" \
    "$@"

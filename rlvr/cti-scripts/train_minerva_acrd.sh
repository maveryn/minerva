#!/usr/bin/env bash
set -euo pipefail

# ACRD per-batch training: RLVR -> ACR (aux PPO) -> SFT each step.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_VAL_DEBUG="${VERL_VAL_DEBUG:-1}"
export VERL_VAL_LOG_EVERY="${VERL_VAL_LOG_EVERY:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export RAY_DEDUP_LOGS_AGG_WINDOW_S="${RAY_DEDUP_LOGS_AGG_WINDOW_S:-60}"
export RAY_LOG_TO_STDERR="${RAY_LOG_TO_STDERR:-0}"
export ACRD_DEBUG_SAMPLES="${ACRD_DEBUG_SAMPLES:-2}"
export ACRD_DETAILS_DEBUG_SAMPLES="${ACRD_DETAILS_DEBUG_SAMPLES:-2}"
MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi

TRAIN_PATH="${ACRD_TRAIN_PATH:-$DATA_DIR/minerva_base/minerva_base_train.parquet}"
VAL_PATHS=(
  "${ACRD_VAL_PATH_1:-$DATA_DIR/minerva_base/minerva_base_dev.parquet}"
  "${ACRD_VAL_PATH_2:-$DATA_DIR/athena/athena_cti_ate.parquet}"
  "${ACRD_VAL_PATH_3:-$DATA_DIR/athena/athena_cti_ckt.parquet}"
  "${ACRD_VAL_PATH_4:-$DATA_DIR/athena/athena_cti_rcm.parquet}"
  "${ACRD_VAL_PATH_5:-$DATA_DIR/athena/athena_cti_rms.parquet}"
)

TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="['${VAL_PATHS[0]}','${VAL_PATHS[1]}','${VAL_PATHS[2]}','${VAL_PATHS[3]}','${VAL_PATHS[4]}']"

OUTPUT_ROOT="${ACRD_OUTPUT_ROOT:-$REPO_ROOT/outputs/acrd}"
LABEL_DETAILS_DIR="${ACRD_LABEL_DETAILS_DIR:-$REPO_ROOT/dataset/label_details}"

RLVR_MAX_PROMPT_LEN="${ACRD_RLVR_MAX_PROMPT_LEN:-2048}"
ACR_MAX_PROMPT_LEN="${ACRD_ACR_MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${ACRD_MAX_RESPONSE_LEN:-2048}"
ACR_MAX_DETAILS_CHARS="${ACRD_MAX_DETAILS_CHARS:-8096}"
ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-4}"
ACR_RL_WEIGHT="${ACRD_ACR_RL_WEIGHT:-0.3}"

ACR_DISTILL_INTERVAL="${ACRD_ACR_DISTILL_INTERVAL:-1}"
ACR_DISTILL_THRESHOLD="${ACRD_ACR_DISTILL_THRESHOLD:-0.5}"
ACR_DISTILL_MAX_BUFFER="${ACRD_ACR_DISTILL_MAX_BUFFER:-4096}"

TRAIN_BATCH_SIZE="${ACRD_TRAIN_BATCH_SIZE:-64}"
TOTAL_STEPS="${ACRD_TOTAL_STEPS:-500}"
SAVE_FREQ="${ACRD_SAVE_FREQ:-10}"
TEST_FREQ="${ACRD_TEST_FREQ:-10}"
EXPERIMENT_NAME="${ACRD_EXPERIMENT_NAME:-minerva_acrd_grpo_${MODEL_SLUG}}"

REWARD_FN_PATH="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.dataloader_num_workers=0 \
    data.return_raw_chat=true \
    data.max_prompt_length="$RLVR_MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.acr={} \
    +data.acr.distill={} \
    +data.acr.per_batch=true \
    +data.acr.label_details_dir="$LABEL_DETAILS_DIR" \
    +data.acr.max_details_chars="$ACR_MAX_DETAILS_CHARS" \
    +data.acr.max_prompt_length="$ACR_MAX_PROMPT_LEN" \
    +data.acr.rollout_n="$ACR_ROLLOUT_N" \
    +data.acr.rl_weight="$ACR_RL_WEIGHT" \
    +data.acr.enforce_no_id_in_reasoning=true \
    +data.acr.reward_kwargs.r_correct=0.1 \
    +data.acr.reward_kwargs.leak_penalty=0.5 \
    +data.acr.reward_kwargs.multilabel_match=exact \
    +data.acr.reward_kwargs.enforce_no_id_in_reasoning=true \
    +data.acr.distill.enabled=true \
    +data.acr.distill.interval="$ACR_DISTILL_INTERVAL" \
    +data.acr.distill.reward_threshold="$ACR_DISTILL_THRESHOLD" \
    +data.acr.distill.max_buffer="$ACR_DISTILL_MAX_BUFFER" \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=reward_minerva \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.default_local_dir="$OUTPUT_ROOT" \
    "$@"

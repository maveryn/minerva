#!/usr/bin/env bash
set -euo pipefail

# GRPO with matched auxiliary answer-only SFT.
# This ablation keeps the online SFT schedule used by Noctua/ACR distillation,
# but fills the distillation buffer with canonical final-answer targets instead
# of generated answer-conditioned rationales.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

export PYTHONPATH="$ROOT_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1

resolve_hf_snapshot_path() {
  local model_ref="$1"
  local hf_cache_root repo_cache_dir snapshot ref_name ref_path

  if [ -z "$model_ref" ] || [ -d "$model_ref" ] || [[ "$model_ref" != */* ]]; then
    printf '%s' "$model_ref"
    return
  fi

  hf_cache_root="${HF_HOME:-$HOME/.cache/huggingface}"
  repo_cache_dir="$hf_cache_root/hub/models--${model_ref//\//--}"
  snapshot=""

  for ref_name in main master; do
    ref_path="$repo_cache_dir/refs/$ref_name"
    if [ -f "$ref_path" ]; then
      snapshot="$repo_cache_dir/snapshots/$(tr -d '\r\n' < "$ref_path")"
      if [ -d "$snapshot" ]; then
        printf '%s' "$snapshot"
        return
      fi
    fi
  done

  if [ -d "$repo_cache_dir/snapshots" ]; then
    snapshot="$(find "$repo_cache_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [ -n "$snapshot" ]; then
      printf '%s' "$snapshot"
      return
    fi
  fi

  printf '%s' "$model_ref"
}

MODEL_SOURCE="${ANSWER_SFT_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
TOKENIZER_SOURCE="${ANSWER_SFT_TOKENIZER_PATH:-$MODEL_SOURCE}"
MODEL_PATH="$(resolve_hf_snapshot_path "$MODEL_SOURCE")"
TOKENIZER_PATH="$(resolve_hf_snapshot_path "$TOKENIZER_SOURCE")"
MODEL_NAME="$(basename "$MODEL_SOURCE")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
EXPERIMENT_NAME="${ANSWER_SFT_EXPERIMENT_NAME:-minerva_grpo_answer_sft_${MODEL_SLUG}}"

TRAIN_PATH="${ANSWER_SFT_TRAIN_PATH:-$DATA_DIR/minerva_base/minerva_base_train.parquet}"
VAL_PATHS=(
  "${ANSWER_SFT_VAL_PATH_1:-$DATA_DIR/minerva_base/minerva_base_dev.parquet}"
  "${ANSWER_SFT_VAL_PATH_2:-$DATA_DIR/athena/athena_cti_ate.parquet}"
  "${ANSWER_SFT_VAL_PATH_3:-$DATA_DIR/athena/athena_cti_ckt.parquet}"
  "${ANSWER_SFT_VAL_PATH_4:-$DATA_DIR/athena/athena_cti_rcm.parquet}"
  "${ANSWER_SFT_VAL_PATH_5:-$DATA_DIR/athena/athena_cti_rms.parquet}"
  "${ANSWER_SFT_VAL_PATH_6:-$DATA_DIR/athena/athena_cti_taa.parquet}"
  "${ANSWER_SFT_VAL_PATH_7:-$DATA_DIR/athena/athena_cti_vsp.parquet}"
  "${ANSWER_SFT_VAL_PATH_8:-$DATA_DIR/seceval/seceval_mini.parquet}"
)

TRAIN_FILES="['$TRAIN_PATH']"
VAL_FILES="['${VAL_PATHS[0]}','${VAL_PATHS[1]}','${VAL_PATHS[2]}','${VAL_PATHS[3]}','${VAL_PATHS[4]}','${VAL_PATHS[5]}','${VAL_PATHS[6]}','${VAL_PATHS[7]}']"

OUTPUT_ROOT="${ANSWER_SFT_OUTPUT_ROOT:-$ROOT_DIR/checkpoints/minerva/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${ANSWER_SFT_TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${ANSWER_SFT_VAL_BATCH_SIZE:-3000}"
MAX_PROMPT_LEN="${ANSWER_SFT_MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${ANSWER_SFT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_N="${ANSWER_SFT_ROLLOUT_N:-8}"
ROLLOUT_GPU_UTIL="${ANSWER_SFT_ROLLOUT_GPU_UTIL:-0.95}"
TOTAL_STEPS="${ANSWER_SFT_TOTAL_STEPS:-500}"
N_GPUS="${ANSWER_SFT_N_GPUS_PER_NODE:-4}"
VAL_BEFORE_TRAIN="${ANSWER_SFT_VAL_BEFORE_TRAIN:-false}"
SAVE_FREQ="${ANSWER_SFT_SAVE_FREQ:-10}"
TEST_FREQ="${ANSWER_SFT_TEST_FREQ:-10}"
SAVE_BEST_ONLY="${ANSWER_SFT_SAVE_BEST_ONLY:-true}"
SAVE_BEST_METRIC="${ANSWER_SFT_SAVE_BEST_METRIC:-val-core/global-val/reward/mean}"
SAVE_BEST_MODE="${ANSWER_SFT_SAVE_BEST_MODE:-max}"
SAVE_BEST_DIR="${ANSWER_SFT_SAVE_BEST_DIR:-best}"

ENTROPY_COEFF="${ANSWER_SFT_ENTROPY_COEFF:-0.0}"
ACTOR_LR="${ANSWER_SFT_ACTOR_LR:-1e-6}"
TENSOR_PARALLEL_SIZE="${ANSWER_SFT_TENSOR_PARALLEL_SIZE:-1}"

ANSWER_SFT_HARD_ONLY="${ANSWER_SFT_HARD_ONLY:-true}"
ANSWER_SFT_HARD_REWARD_MODE="${ANSWER_SFT_HARD_REWARD_MODE:-max}"
ANSWER_SFT_HARD_REWARD_THRESHOLD="${ANSWER_SFT_HARD_REWARD_THRESHOLD:-1.0}"
ANSWER_SFT_SKIP_CVSS="${ANSWER_SFT_SKIP_CVSS:-true}"
ANSWER_SFT_EXCLUDE_CVSS_TRAIN="${ANSWER_SFT_EXCLUDE_CVSS_TRAIN:-false}"

ANSWER_SFT_DISTILL_INTERVAL="${ANSWER_SFT_DISTILL_INTERVAL:-10}"
ANSWER_SFT_DISTILL_BATCH_SIZE="${ANSWER_SFT_DISTILL_BATCH_SIZE:-256}"
ANSWER_SFT_DISTILL_MAX_BUFFER="${ANSWER_SFT_DISTILL_MAX_BUFFER:-1024}"
ANSWER_SFT_DISTILL_MIN_BUFFER="${ANSWER_SFT_DISTILL_MIN_BUFFER:-0}"
ANSWER_SFT_DISTILL_BUFFER_MODE="${ANSWER_SFT_DISTILL_BUFFER_MODE:-rolling}"
ANSWER_SFT_DISTILL_LR_SCALE="${ANSWER_SFT_DISTILL_LR_SCALE:-1.0}"
ANSWER_SFT_DISTILL_REWARD_THRESHOLD="${ANSWER_SFT_DISTILL_REWARD_THRESHOLD:-1.0}"

REWARD_FN_PATH="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.dataloader_num_workers=0 \
    data.return_raw_chat=true \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=reward_minerva \
    +custom_reward_function.reward_kwargs.return_dict=true \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.exclude_cvss_train="$ANSWER_SFT_EXCLUDE_CVSS_TRAIN" \
    +data.answer_sft={} \
    +data.answer_sft.distill={} \
    +data.answer_sft.enabled=true \
    +data.answer_sft.hard_only="$ANSWER_SFT_HARD_ONLY" \
    +data.answer_sft.hard_reward_mode="$ANSWER_SFT_HARD_REWARD_MODE" \
    +data.answer_sft.hard_reward_threshold="$ANSWER_SFT_HARD_REWARD_THRESHOLD" \
    +data.answer_sft.skip_cvss="$ANSWER_SFT_SKIP_CVSS" \
    +data.answer_sft.distill.enabled=true \
    +data.answer_sft.distill.interval="$ANSWER_SFT_DISTILL_INTERVAL" \
    +data.answer_sft.distill.reward_threshold="$ANSWER_SFT_DISTILL_REWARD_THRESHOLD" \
    +data.answer_sft.distill.batch_size="$ANSWER_SFT_DISTILL_BATCH_SIZE" \
    +data.answer_sft.distill.max_buffer="$ANSWER_SFT_DISTILL_MAX_BUFFER" \
    +data.answer_sft.distill.min_buffer="$ANSWER_SFT_DISTILL_MIN_BUFFER" \
    +data.answer_sft.distill.buffer_mode="$ANSWER_SFT_DISTILL_BUFFER_MODE" \
    +data.answer_sft.distill.lr_scale="$ANSWER_SFT_DISTILL_LR_SCALE" \
    +data.answer_sft.distill.method=sft \
    +data.answer_sft.distill.dedup_by_uid=true \
    +data.answer_sft.distill.disable_filters=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
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

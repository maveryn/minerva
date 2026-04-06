#!/usr/bin/env bash
set -euo pipefail

# Baseline SQL-R1 GRPO training script on the Minerva/VeRL stack.
# Mirrors SQL-R1's public training settings where practical, with 2-GPU defaults.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export PYTHONUNBUFFERED=1
if [ -n "${VLLM_ATTENTION_BACKEND:-}" ]; then
  export VLLM_ATTENTION_BACKEND
fi

resolve_hf_snapshot_path() {
  local model_ref="$1"
  local asset_kind="${2:-model}"
  local hf_cache_root repo_cache_dir snapshot ref_name ref_path

  if [ -z "$model_ref" ] || [ -d "$model_ref" ] || [[ "$model_ref" != */* ]]; then
    printf '%s' "$model_ref"
    return
  fi

  snapshot_is_usable() {
    local snapshot_dir="$1"
    local kind="$2"
    if [ ! -d "$snapshot_dir" ] || [ ! -f "$snapshot_dir/config.json" ]; then
      return 1
    fi
    if [ "$kind" = "tokenizer" ]; then
      [ -f "$snapshot_dir/tokenizer.json" ] || [ -f "$snapshot_dir/vocab.json" ]
      return
    fi
    compgen -G "$snapshot_dir/model*.safetensors" > /dev/null || [ -f "$snapshot_dir/pytorch_model.bin" ]
  }

  hf_cache_root="${HF_HOME:-$HOME/.cache/huggingface}"
  repo_cache_dir="$hf_cache_root/hub/models--${model_ref//\//--}"
  snapshot=""

  for ref_name in main master; do
    ref_path="$repo_cache_dir/refs/$ref_name"
    if [ -f "$ref_path" ]; then
      snapshot="$repo_cache_dir/snapshots/$(tr -d '\r\n' < "$ref_path")"
      if snapshot_is_usable "$snapshot" "$asset_kind"; then
        printf '%s' "$snapshot"
        return
      fi
    fi
  done

  if [ -d "$repo_cache_dir/snapshots" ]; then
    while IFS= read -r snapshot; do
      if snapshot_is_usable "$snapshot" "$asset_kind"; then
        printf '%s' "$snapshot"
        return
      fi
    done < <(find "$repo_cache_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | sort -r)
  fi

  printf '%s' "$model_ref"
}

SQLR1_ROOT="${SQLR1_ROOT:-/home/jovyan/work/SQL-R1}"
MODEL_SOURCE="${SQLR1_MODEL_PATH:-MPX0222forHF/SQL-R1-3B}"
TOKENIZER_SOURCE="${SQLR1_TOKENIZER_PATH:-$MODEL_SOURCE}"
MODEL_PATH="$(resolve_hf_snapshot_path "$MODEL_SOURCE" model)"
TOKENIZER_PATH="$(resolve_hf_snapshot_path "$TOKENIZER_SOURCE" tokenizer)"
MODEL_NAME="$(basename "$MODEL_SOURCE")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi

EXPERIMENT_NAME="${SQLR1_EXPERIMENT_NAME:-sql_r1_grpo_${MODEL_SLUG}}"
TRAIN_PATH="${SQLR1_TRAIN_PATH:-$SQLR1_ROOT/example_data/train.parquet}"
VAL_PATH="${SQLR1_VAL_PATH:-$SQLR1_ROOT/example_data/test.parquet}"
DB_ROOT="${SQLR1_DB_ROOT:-$SQLR1_ROOT/data/NL2SQL/SynSQL-2.5M/databases}"
OUTPUT_ROOT="${SQLR1_OUTPUT_ROOT:-$ROOT_DIR/checkpoints/sql-r1/$EXPERIMENT_NAME}"

TRAIN_BATCH_SIZE="${SQLR1_TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${SQLR1_VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LEN="${SQLR1_MAX_PROMPT_LEN:-4096}"
MAX_RESPONSE_LEN="${SQLR1_MAX_RESPONSE_LEN:-2048}"
ROLLOUT_N="${SQLR1_ROLLOUT_N:-8}"
ROLLOUT_TEMP="${SQLR1_ROLLOUT_TEMPERATURE:-1.1}"
ROLLOUT_GPU_UTIL="${SQLR1_ROLLOUT_GPU_UTIL:-0.35}"
FREE_CACHE_ENGINE="${SQLR1_FREE_CACHE_ENGINE:-false}"
ACTOR_LR="${SQLR1_ACTOR_LR:-3e-7}"
KL_COEF="${SQLR1_KL_COEF:-0.001}"
TOTAL_EPOCHS="${SQLR1_TOTAL_EPOCHS:-10}"
N_GPUS="${SQLR1_N_GPUS_PER_NODE:-2}"
TP_SIZE="${SQLR1_TENSOR_PARALLEL_SIZE:-1}"
PPO_MICRO_BATCH_SIZE="${SQLR1_PPO_MICRO_BATCH_SIZE:-4}"
LOGPROB_MICRO_BATCH_SIZE="${SQLR1_LOGPROB_MICRO_BATCH_SIZE:-16}"
SAVE_FREQ="${SQLR1_SAVE_FREQ:-100}"
TEST_FREQ="${SQLR1_TEST_FREQ:-100}"
REWARD_TIMEOUT_S="${SQLR1_REWARD_TIMEOUT_S:-10}"
SAVE_BEST_ONLY="${SQLR1_SAVE_BEST_ONLY:-true}"
SAVE_BEST_METRIC="${SQLR1_SAVE_BEST_METRIC:-val-core/minerva-dev/reward/mean}"
SAVE_BEST_MODE="${SQLR1_SAVE_BEST_MODE:-max}"
SAVE_BEST_DIR="${SQLR1_SAVE_BEST_DIR:-best}"

DATASET_CLS_PATH="$ROOT_DIR/verl/utils/dataset/sql_r1_dataset.py"
REWARD_FN_PATH="$ROOT_DIR/verl/utils/reward_score/reward_sql_r1.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$VAL_PATH']" \
    data.dataloader_num_workers=0 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=true \
    data.custom_cls.path="$DATASET_CLS_PATH" \
    data.custom_cls.name=SQLR1RLHFDataset \
    custom_reward_function.path="$REWARD_FN_PATH" \
    custom_reward_function.name=reward_sql_r1 \
    +custom_reward_function.reward_kwargs.return_dict=true \
    +custom_reward_function.reward_kwargs.db_root="$DB_ROOT" \
    +custom_reward_function.reward_kwargs.timeout_s="$REWARD_TIMEOUT_S" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_UTIL" \
    actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMP" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef="$KL_COEF" \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='sql-r1' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.default_local_dir="$OUTPUT_ROOT" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    +trainer.save_best_only="$SAVE_BEST_ONLY" \
    +trainer.save_best_metric="$SAVE_BEST_METRIC" \
    +trainer.save_best_mode="$SAVE_BEST_MODE" \
    +trainer.save_best_dir="$SAVE_BEST_DIR" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    "$@"

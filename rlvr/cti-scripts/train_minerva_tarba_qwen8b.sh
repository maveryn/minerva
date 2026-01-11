#!/usr/bin/env bash
set -e

# Training script for Minerva TARBA with Qwen3 8B.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export SGLANG_FORCE_CPU_WEIGHT_SYNC=1
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export VERL_VAL_DEBUG="${VERL_VAL_DEBUG:-1}"
export VERL_VAL_LOG_EVERY="${VERL_VAL_LOG_EVERY:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export RAY_DEDUP_LOGS_AGG_WINDOW_S="${RAY_DEDUP_LOGS_AGG_WINDOW_S:-60}"
export RAY_LOG_TO_STDERR="${RAY_LOG_TO_STDERR:-0}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
TARBA_DEBUG_SAMPLES="${TARBA_DEBUG_SAMPLES:-2}"
export TARBA_DEBUG_SAMPLES
TARBA_HIDE_TOOL_SCHEMA="${TARBA_HIDE_TOOL_SCHEMA:-1}"
export TARBA_HIDE_TOOL_SCHEMA

RETRIEVAL_HOST="${RETRIEVAL_HOST:-127.0.0.1}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_INDEX_DIR="${RETRIEVAL_INDEX_DIR:-$REPO_ROOT/dataset/retrieval/index}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-per_type}"
RETRIEVAL_URL="http://${RETRIEVAL_HOST}:${RETRIEVAL_PORT}/openapi.json"

check_retrieval_server() {
  python3 - "$RETRIEVAL_URL" <<'PY'
import sys
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else ""
if not url:
    sys.exit(1)
try:
    with urllib.request.urlopen(url, timeout=1) as resp:
        sys.exit(0 if resp.status < 400 else 1)
except Exception:
    sys.exit(1)
PY
}

find_retrieval_pids() {
  local pids=""
  if command -v pgrep >/dev/null 2>&1; then
    pids="$(pgrep -f "minerva.retrieval.server" || true)"
  else
    pids="$(ps -ef | awk '/minerva.retrieval.server/ && $0 !~ /awk/ {print $2}')"
  fi
  echo "$pids"
}

stop_retrieval_server() {
  local pids
  pids="$(find_retrieval_pids)"
  if [ -n "$pids" ]; then
    echo "Stopping existing retrieval server..."
    kill $pids >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do
      if ! check_retrieval_server; then
        break
      fi
      sleep 1
    done
  fi
}

RETRIEVAL_STARTED=0
stop_retrieval_server

echo "Starting retrieval server on ${RETRIEVAL_HOST}:${RETRIEVAL_PORT}..."
python3 -m minerva.retrieval.server \
  --index_dir "$RETRIEVAL_INDEX_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --host "$RETRIEVAL_HOST" \
  --port "$RETRIEVAL_PORT" \
  >/tmp/minerva_tarba_retrieval.log 2>&1 &
RETRIEVAL_PID=$!
RETRIEVAL_STARTED=1
for _ in $(seq 1 30); do
  if check_retrieval_server; then
    break
  fi
  sleep 1
done
if ! check_retrieval_server; then
  echo "Retrieval server failed to start. See /tmp/minerva_tarba_retrieval.log" >&2
  if [ -n "${RETRIEVAL_PID:-}" ]; then
    kill "$RETRIEVAL_PID" >/dev/null 2>&1 || true
  fi
  exit 1
fi

cleanup_retrieval() {
  if [ "${RETRIEVAL_STARTED:-0}" -eq 1 ] && [ -n "${RETRIEVAL_PID:-}" ]; then
    kill "$RETRIEVAL_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_retrieval EXIT

train_path="$DATA_DIR/minerva_base/minerva_base_train.parquet"

val_paths=(
  "$DATA_DIR/minerva_base/minerva_base_dev.parquet"
  "$DATA_DIR/athena/athena_cti_ate.parquet"
  "$DATA_DIR/athena/athena_cti_ckt.parquet"
  "$DATA_DIR/athena/athena_cti_rcm.parquet"
  "$DATA_DIR/athena/athena_cti_rms.parquet"
)

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_tarba.py"
custom_dataset_path="$ROOT_DIR/verl/utils/dataset/minerva_tarba_retrieval_dataset.py"
tool_config_path="$ROOT_DIR/cti-scripts/tool_config/cti_retrieval_tool.yaml"
sglang_engine_kwargs="{max_total_tokens: 262144, max_running_requests: 128, disable_cuda_graph: false, disable_radix_cache: false, disable_overlap_schedule: false, attention_backend: fa3, prefill_attention_backend: fa3, log_level: ${SGLANG_LOG_LEVEL}}"

train_files="['$train_path']"
val_files="['${val_paths[0]}','${val_paths[1]}','${val_paths[2]}','${val_paths[3]}','${val_paths[4]}']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.custom_cls.path="$custom_dataset_path" \
    data.custom_cls.name=TarbaRLHFDataset \
    data.dataloader_num_workers=0 \
    data.return_raw_chat=True \
    +data.tarba.enabled=true \
    +data.tarba.p_noret_init=0.10 \
    +data.tarba.p_noret_max=0.90 \
    +data.tarba.p_step=0.05 \
    +data.tarba.default_B_max=8 \
    +data.tarba.ema_beta=0.90 \
    +data.tarba.target_acc_noret=0.60 \
    +data.tarba.tol=0.05 \
    +data.tarba.min_steps_before_anneal=50 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_tarba \
    +custom_reward_function.reward_kwargs.lambda_ret=0.2 \
    +custom_reward_function.reward_kwargs.floor=0.2 \
    +custom_reward_function.reward_kwargs.tool_call_bonus=0.1 \
    data.train_batch_size=64 \
    data.val_batch_size=2048 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.engine_kwargs.sglang="$sglang_engine_kwargs" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name='minerva_tarba_qwen8b' \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_training_steps=500 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

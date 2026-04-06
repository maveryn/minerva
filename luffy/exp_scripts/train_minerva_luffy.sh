#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVE_VENV_PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
CONFIGURED_VENV_PY="${LUFFY_ENV_DIR:+$LUFFY_ENV_DIR/bin/python}"
REPO_VENV_PY_PREFERRED="$ROOT_DIR/../.venv-luffy312/bin/python"
REPO_VENV_PY_LEGACY="$ROOT_DIR/../.venv-luffy/bin/python"

supports_visible_gpu_arch() {
  local py_bin="$1"
  "$py_bin" - <<'PY' >/dev/null 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
major, minor = torch.cuda.get_device_capability(0)
target = f"sm_{major}{minor}"
archs = set(torch.cuda.get_arch_list())
raise SystemExit(0 if target in archs else 1)
PY
}

detect_attention_backend() {
  local py_bin="$1"
  "$py_bin" - <<'PY'
import torch
if not torch.cuda.is_available():
    print("FLASH_ATTN")
    raise SystemExit(0)
major, _ = torch.cuda.get_device_capability(0)
print("XFORMERS" if major < 9 else "FLASH_ATTN")
PY
}

if [ -z "${PYTHON_BIN:-}" ]; then
  for candidate in \
    "${ACTIVE_VENV_PY:-}" \
    "${CONFIGURED_VENV_PY:-}" \
    "$REPO_VENV_PY_PREFERRED" \
    "$REPO_VENV_PY_LEGACY" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if ! supports_visible_gpu_arch "$PYTHON_BIN"; then
  for candidate in \
    "${ACTIVE_VENV_PY:-}" \
    "${CONFIGURED_VENV_PY:-}" \
    "$REPO_VENV_PY_PREFERRED" \
    "$REPO_VENV_PY_LEGACY" \
    /opt/conda/bin/python; do
    if [ -x "$candidate" ] && supports_visible_gpu_arch "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if ! supports_visible_gpu_arch "$PYTHON_BIN"; then
    PYTHON_BIN="/opt/conda/bin/python"
  fi
fi

if ! supports_visible_gpu_arch "$PYTHON_BIN"; then
  echo "No compatible Python/PyTorch runtime was found for the visible GPU architecture." >&2
  exit 1
fi

DATA_DIR="${MINERVA_DATA_DIR:-$ROOT_DIR/data}"

"$PYTHON_BIN" -m ray stop >/dev/null 2>&1 || true

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-$(detect_attention_backend "$PYTHON_BIN")}"
export PYTHONPATH="$ROOT_DIR/luffy:$ROOT_DIR/luffy/verl${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${MINERVA_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi

EXP_NAME="${MINERVA_EXPERIMENT_NAME:-minerva_luffy_${MODEL_SLUG}}"
WANDB_PROJECT="${MINERVA_WANDB_PROJECT:-minerva}"
N_GPUS="${MINERVA_N_GPUS_PER_NODE:-4}"
TRAIN_BATCH_SIZE="${MINERVA_TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MINERVA_PPO_MICRO_BATCH_SIZE:-16}"
MAX_PROMPT_LEN="${MINERVA_MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MINERVA_MAX_RESPONSE_LEN:-1024}"
MAX_PREFIX_LEN="${MINERVA_MAX_PREFIX_LEN:-1024}"
ROLLOUT_N="${MINERVA_ROLLOUT_N:-8}"
ROLLOUT_GPU_UTIL="${MINERVA_ROLLOUT_GPU_UTIL:-0.8}"
ACTOR_LR="${MINERVA_ACTOR_LR:-1e-6}"
ENTROPY_COEFF="${MINERVA_ENTROPY_COEFF:-0.0}"
TEST_FREQ="${MINERVA_TEST_FREQ:-10}"
SAVE_FREQ="${MINERVA_SAVE_FREQ:-10}"
TOTAL_STEPS="${MINERVA_TOTAL_STEPS:-500}"
TENSOR_PARALLEL_SIZE="${MINERVA_TENSOR_PARALLEL_SIZE:-1}"
PREFIX_MIN_RATIO="${MINERVA_PREFIX_MIN_RATIO:-1.0}"
PREFIX_MAX_RATIO="${MINERVA_PREFIX_MAX_RATIO:-1.0}"
OFF_POLICY_RESHAPE="${MINERVA_OFF_POLICY_RESHAPE:-p_div_p_0.1}"
ATTN_IMPLEMENTATION="${MINERVA_ATTN_IMPLEMENTATION:-flash_attention_2}"
VAL_BATCH_SIZE="${MINERVA_VAL_BATCH_SIZE:-3000}"
VAL_BEFORE_TRAIN="${MINERVA_VAL_BEFORE_TRAIN:-false}"
SAVE_BEST_ONLY="${MINERVA_SAVE_BEST_ONLY:-true}"
SAVE_BEST_METRIC="${MINERVA_SAVE_BEST_METRIC:-val-core/global-val/reward/mean}"
SAVE_BEST_MODE="${MINERVA_SAVE_BEST_MODE:-max}"
SAVE_BEST_DIR="${MINERVA_SAVE_BEST_DIR:-best}"
SAVE_LAST_DIR="${MINERVA_SAVE_LAST_DIR:-last}"
MAX_OPTIM_TO_KEEP="${MINERVA_MAX_OPTIM_TO_KEEP:-1}"
ROLLOUT_MICRO_BATCH_SIZE="${MINERVA_ROLLOUT_MICRO_BATCH_SIZE:-}"
ACTOR_PARAM_OFFLOAD="${MINERVA_ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${MINERVA_ACTOR_OPTIMIZER_OFFLOAD:-true}"
REF_PARAM_OFFLOAD="${MINERVA_REF_PARAM_OFFLOAD:-true}"
OUTPUT_DIR="${MINERVA_OUTPUT_DIR:-$ROOT_DIR/checkpoints/minerva/$EXP_NAME}"

if [ -n "${MINERVA_ROLLOUT_NAME:-}" ]; then
  ROLLOUT_NAME="$MINERVA_ROLLOUT_NAME"
else
  if "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
try:
    v = version("vllm")
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if (v in {"0.3.1", "0.4.2", "0.5.4", "0.6.3"} or v.startswith("0.13.")) else 1)
PY
  then
    ROLLOUT_NAME="vllm"
  else
    ROLLOUT_NAME="hf"
  fi
fi

if [ "$ROLLOUT_NAME" = "hf" ] && [ -z "$ROLLOUT_MICRO_BATCH_SIZE" ]; then
  ROLLOUT_MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
fi

TRAIN_FILE="${MINERVA_TRAIN_FILE:-$DATA_DIR/minerva_train_dart32k_offpolicy.parquet}"
VAL_FILES="['${DATA_DIR}/minerva_base_dev.parquet','${DATA_DIR}/athena_cti_ate.parquet','${DATA_DIR}/athena_cti_ckt.parquet','${DATA_DIR}/athena_cti_rcm.parquet','${DATA_DIR}/athena_cti_rms.parquet','${DATA_DIR}/athena_cti_taa.parquet','${DATA_DIR}/athena_cti_vsp.parquet','${DATA_DIR}/seceval_mini.parquet']"

cd "$ROOT_DIR/luffy/verl"

"$PYTHON_BIN" -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="$VAL_FILES" \
    data.dataloader_num_workers=0 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.attn_implementation="$ATTN_IMPLEMENTATION" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.rollout.max_prefix_len="$MAX_PREFIX_LEN" \
    ${ROLLOUT_MICRO_BATCH_SIZE:+actor_rollout_ref.rollout.micro_batch_size="$ROLLOUT_MICRO_BATCH_SIZE"} \
    actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio="$PREFIX_MIN_RATIO" \
    actor_rollout_ref.rollout.max_prefix_ratio="$PREFIX_MAX_RATIO" \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="$OFF_POLICY_RESHAPE" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=3 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_best_only="$SAVE_BEST_ONLY" \
    trainer.save_best_metric="$SAVE_BEST_METRIC" \
    trainer.save_best_mode="$SAVE_BEST_MODE" \
    trainer.save_best_dir="$SAVE_BEST_DIR" \
    trainer.save_last_dir="$SAVE_LAST_DIR" \
    trainer.max_optim_to_keep="$MAX_OPTIM_TO_KEEP" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$OUTPUT_DIR" \
    "$@"

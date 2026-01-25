#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for GPT-OSS 20B (distill lr_scale=0.05, flush buffer, batch size 256,
# TextCNN ML+heuristic filter, deferred ACR generation, EMA teacher alpha 0.995,
# ACR rollout temperature 0.7, top_p 0.9).

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-lmsys/gpt-oss-20b-bf16}"
export ACRD_ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-4}"
export ACRD_ACR_DISTILL_LR_SCALE="${ACRD_ACR_DISTILL_LR_SCALE:-0.05}"
export ACRD_ACR_DISTILL_BUFFER_MODE="${ACRD_ACR_DISTILL_BUFFER_MODE:-flush}"
export ACRD_ACR_DISTILL_BATCH_SIZE="${ACRD_ACR_DISTILL_BATCH_SIZE:-256}"
export ACRD_ACR_DISTILL_DISABLE_FILTERS="${ACRD_ACR_DISTILL_DISABLE_FILTERS:-false}"

export ACRD_ACR_DISTILL_FILTER_MODE="${ACRD_ACR_DISTILL_FILTER_MODE:-ml+heuristic}"
export ACRD_ACR_DISTILL_FILTER_MODEL="${ACRD_ACR_DISTILL_FILTER_MODEL:-xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25}"
export ACRD_ACR_DISTILL_FILTER_MODEL_TYPE="${ACRD_ACR_DISTILL_FILTER_MODEL_TYPE:-textcnn}"
export ACRD_ACR_DISTILL_FILTER_THRESHOLD="${ACRD_ACR_DISTILL_FILTER_THRESHOLD:-0.5}"
export ACRD_ACR_ROLLOUT_TEMPERATURE="${ACRD_ACR_ROLLOUT_TEMPERATURE:-0.7}"
export ACRD_ACR_ROLLOUT_TOP_P="${ACRD_ACR_ROLLOUT_TOP_P:-0.9}"
export ACRD_ACR_DEFER_GENERATION="${ACRD_ACR_DEFER_GENERATION:-true}"
export ACRD_ACR_EMA_TEACHER_ENABLED="${ACRD_ACR_EMA_TEACHER_ENABLED:-true}"
export ACRD_ACR_EMA_TEACHER_ALPHA="${ACRD_ACR_EMA_TEACHER_ALPHA:-0.995}"
export ACRD_ROLLOUT_GPU_UTIL="${ACRD_ROLLOUT_GPU_UTIL:-0.6}"

export ACRD_N_GPUS_PER_NODE="${ACRD_N_GPUS_PER_NODE:-4}"

# Disable Ray OpenTelemetry/OTLP exporters to avoid gRPC segfaults in some envs.
export RAY_enable_open_telemetry="${RAY_enable_open_telemetry:-0}"
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-none}"
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-none}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"

export ACRD_GPTOSS_SOURCE_MODEL="${ACRD_GPTOSS_SOURCE_MODEL:-openai/gpt-oss-20b}"
export ACRD_GPTOSS_BF16_DIR="${ACRD_GPTOSS_BF16_DIR:-$HOME/models/gpt-oss-20b-bf16}"
export ACRD_GPTOSS_PREPARE="${ACRD_GPTOSS_PREPARE:-skip}"

if [ "$ACRD_MODEL_PATH" = "$ACRD_GPTOSS_SOURCE_MODEL" ]; then
  need_prepare="false"
  if [ "$ACRD_GPTOSS_PREPARE" = "force" ]; then
    need_prepare="true"
  elif [ "$ACRD_GPTOSS_PREPARE" != "skip" ] && [ ! -f "$ACRD_GPTOSS_BF16_DIR/config.json" ]; then
    need_prepare="true"
  fi

  if [ "$need_prepare" = "true" ]; then
    python3 - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

model_id = os.environ["ACRD_GPTOSS_SOURCE_MODEL"]
output_dir = os.environ["ACRD_GPTOSS_BF16_DIR"]

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
model.config.attn_implementation = "eager"
model.save_pretrained(output_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)
PY
  fi

  export ACRD_MODEL_PATH="$ACRD_GPTOSS_BF16_DIR"
fi

MODEL_NAME="$(basename "$ACRD_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export ACRD_EXPERIMENT_NAME="${ACRD_EXPERIMENT_NAME:-minerva_noctua_${MODEL_SLUG}_lr0.05_flush_bs256_mlh_t0.7_p0.9_defer_ema}"

export ACRD_ROLLOUT_BACKEND="${ACRD_ROLLOUT_BACKEND:-sglang}"
export ACRD_ROLLOUT_MODE="${ACRD_ROLLOUT_MODE:-async}"
export ACRD_ROLLOUT_TP_SIZE="${ACRD_ROLLOUT_TP_SIZE:-2}"
export ACRD_ROLLOUT_LOAD_FORMAT="${ACRD_ROLLOUT_LOAD_FORMAT:-safetensors}"
export ACRD_SGLANG_ATTENTION_BACKEND="${ACRD_SGLANG_ATTENTION_BACKEND:-triton}"
export ACRD_FSDP_MODEL_DTYPE="${ACRD_FSDP_MODEL_DTYPE:-bfloat16}"
export ACRD_ACTOR_PARAM_OFFLOAD="${ACRD_ACTOR_PARAM_OFFLOAD:-false}"
export ACRD_ACTOR_OPTIMIZER_OFFLOAD="${ACRD_ACTOR_OPTIMIZER_OFFLOAD:-false}"
export ACRD_REF_PARAM_OFFLOAD="${ACRD_REF_PARAM_OFFLOAD:-false}"
export ACRD_PPO_MICRO_BSZ_PER_GPU="${ACRD_PPO_MICRO_BSZ_PER_GPU:-32}"
export ACRD_ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU="${ACRD_ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU:-32}"
export ACRD_ROLLOUT_N="${ACRD_ROLLOUT_N:-5}"

EXTRA_ARGS=(
  "actor_rollout_ref.rollout.name=${ACRD_ROLLOUT_BACKEND}"
  "actor_rollout_ref.rollout.mode=${ACRD_ROLLOUT_MODE}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ACRD_ROLLOUT_TP_SIZE}"
  "actor_rollout_ref.rollout.load_format=${ACRD_ROLLOUT_LOAD_FORMAT}"
  "+actor_rollout_ref.actor.fsdp_config.model_dtype=${ACRD_FSDP_MODEL_DTYPE}"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${ACRD_ACTOR_PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACRD_ACTOR_OPTIMIZER_OFFLOAD}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${ACRD_REF_PARAM_OFFLOAD}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACRD_PPO_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ACRD_ROLLOUT_LOGPROB_MICRO_BSZ_PER_GPU}"
  "actor_rollout_ref.rollout.n=${ACRD_ROLLOUT_N}"
)

if [ "$ACRD_ROLLOUT_BACKEND" = "sglang" ]; then
  EXTRA_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${ACRD_SGLANG_ATTENTION_BACKEND}")
fi

exec "$(dirname "$0")/train_minerva_noctua.sh" "${EXTRA_ARGS[@]}" "$@"

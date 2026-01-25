#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for GPT-OSS 20B.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-lmsys/gpt-oss-20b-bf16}"
export GRPO_ROLLOUT_GPU_UTIL="${GRPO_ROLLOUT_GPU_UTIL:-0.8}"
export GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-5}"
export GRPO_MAX_RESPONSE_LEN="${GRPO_MAX_RESPONSE_LEN:-2048}"
export GRPO_RETURN_RAW_CHAT="${GRPO_RETURN_RAW_CHAT:-true}"

# Disable Ray OpenTelemetry/OTLP exporters to avoid gRPC segfaults in some envs.
export RAY_enable_open_telemetry="${RAY_enable_open_telemetry:-0}"
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-none}"
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-none}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"
export RAY_USAGE_STATS_ENABLED="${RAY_USAGE_STATS_ENABLED:-0}"

export GRPO_ROLLOUT_BACKEND="${GRPO_ROLLOUT_BACKEND:-sglang}"
export GRPO_ROLLOUT_MODE="${GRPO_ROLLOUT_MODE:-async}"
export GRPO_ROLLOUT_TP_SIZE="${GRPO_ROLLOUT_TP_SIZE:-1}"
export GRPO_ROLLOUT_LOAD_FORMAT="${GRPO_ROLLOUT_LOAD_FORMAT:-safetensors}"
export GRPO_SGLANG_ATTENTION_BACKEND="${GRPO_SGLANG_ATTENTION_BACKEND:-triton}"
export GRPO_FSDP_MODEL_DTYPE="${GRPO_FSDP_MODEL_DTYPE:-bfloat16}"

EXTRA_ARGS=(
  "data.return_raw_chat=${GRPO_RETURN_RAW_CHAT}"
  "actor_rollout_ref.rollout.name=${GRPO_ROLLOUT_BACKEND}"
  "actor_rollout_ref.rollout.mode=${GRPO_ROLLOUT_MODE}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${GRPO_ROLLOUT_TP_SIZE}"
  "actor_rollout_ref.rollout.load_format=${GRPO_ROLLOUT_LOAD_FORMAT}"
  "+actor_rollout_ref.actor.fsdp_config.model_dtype=${GRPO_FSDP_MODEL_DTYPE}"
)

if [ "$GRPO_ROLLOUT_BACKEND" = "sglang" ]; then
  EXTRA_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${GRPO_SGLANG_ATTENTION_BACKEND}")
fi

exec "$(dirname "$0")/train_minerva_grpo.sh" "${EXTRA_ARGS[@]}" "$@"

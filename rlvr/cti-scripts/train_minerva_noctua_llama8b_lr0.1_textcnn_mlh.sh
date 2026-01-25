#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 8B (distill lr_scale=0.1, flush buffer, batch size 128, TextCNN ML+heuristic filter).

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ACRD_ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-4}"
export ACRD_ACR_DISTILL_LR_SCALE="${ACRD_ACR_DISTILL_LR_SCALE:-0.02}"
export ACRD_ACR_DISTILL_BUFFER_MODE="${ACRD_ACR_DISTILL_BUFFER_MODE:-flush}"
export ACRD_ACR_DISTILL_BATCH_SIZE="${ACRD_ACR_DISTILL_BATCH_SIZE:-128}"
export ACRD_ACR_DISTILL_DISABLE_FILTERS="${ACRD_ACR_DISTILL_DISABLE_FILTERS:-false}"

export ACRD_ACR_DISTILL_FILTER_MODE="${ACRD_ACR_DISTILL_FILTER_MODE:-ml+heuristic}"
export ACRD_ACR_DISTILL_FILTER_MODEL="${ACRD_ACR_DISTILL_FILTER_MODEL:-xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25}"
export ACRD_ACR_DISTILL_FILTER_MODEL_TYPE="${ACRD_ACR_DISTILL_FILTER_MODEL_TYPE:-textcnn}"
export ACRD_ACR_DISTILL_FILTER_THRESHOLD="${ACRD_ACR_DISTILL_FILTER_THRESHOLD:-0.5}"

export ACRD_N_GPUS_PER_NODE="${ACRD_N_GPUS_PER_NODE:-4}"

MODEL_NAME="$(basename "$ACRD_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export ACRD_EXPERIMENT_NAME="${ACRD_EXPERIMENT_NAME:-minerva_noctua_${MODEL_SLUG}_lr0.02_flush_bs128_mlh}"

exec "$(dirname "$0")/train_minerva_noctua.sh" "$@"

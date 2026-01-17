#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 8B (distill lr_scale=0.1, buffer-triggered SFT).

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ACRD_ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-4}"
export ACRD_ACR_DISTILL_LR_SCALE="${ACRD_ACR_DISTILL_LR_SCALE:-0.1}"
export ACRD_ACR_DISTILL_BUFFER_MODE="${ACRD_ACR_DISTILL_BUFFER_MODE:-buffer}"
export ACRD_ACR_DISTILL_MIN_BUFFER="${ACRD_ACR_DISTILL_MIN_BUFFER:-256}"
MODEL_NAME="$(basename "$ACRD_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export ACRD_EXPERIMENT_NAME="${ACRD_EXPERIMENT_NAME:-minerva_noctua_${MODEL_SLUG}_lr0.1_buffer}"

exec "$(dirname "$0")/train_minerva_noctua.sh" "$@"

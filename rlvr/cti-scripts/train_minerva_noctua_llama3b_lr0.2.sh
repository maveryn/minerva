#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 3B (distill lr_scale=0.2).

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
export ACRD_ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-2}"
export ACRD_ACR_DISTILL_LR_SCALE="${ACRD_ACR_DISTILL_LR_SCALE:-0.2}"
MODEL_NAME="$(basename "$ACRD_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export ACRD_EXPERIMENT_NAME="${ACRD_EXPERIMENT_NAME:-minerva_noctua_${MODEL_SLUG}_lr0.2}"

exec "$(dirname "$0")/train_minerva_noctua.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for Llama 8B with rollout n=12.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-12}"

MODEL_NAME="$(basename "$GRPO_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi

export GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-minerva_grpo_${MODEL_SLUG}_rollout12}"

exec "$(dirname "$0")/train_minerva_grpo.sh" "$@"

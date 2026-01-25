#!/usr/bin/env bash
set -euo pipefail

# Answer-only SFT training script for Llama 8B.

export SFT_MODEL_PATH="${SFT_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export SFT_N_GPUS_PER_NODE="${SFT_N_GPUS_PER_NODE:-1}"
MODEL_NAME="$(basename "$SFT_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-minerva_sft_answer_${MODEL_SLUG}}"

exec "$(dirname "$0")/train_minerva_sft_answer_only.sh" "$@"

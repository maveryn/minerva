#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for Llama 8B initialized from the DART checkpoint.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-xashru/minerva_dart_llama8b}"
export GRPO_TOKENIZER_PATH="${GRPO_TOKENIZER_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-minerva_grpo_dart_llama8b}"
export GRPO_N_GPUS_PER_NODE="${GRPO_N_GPUS_PER_NODE:-8}"

exec "$(dirname "$0")/../train_minerva_grpo.sh" "$@"

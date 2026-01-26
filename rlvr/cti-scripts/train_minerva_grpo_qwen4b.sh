#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for Llama 8B.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-Qwen/Qwen3-4B-Base}"

exec "$(dirname "$0")/train_minerva_grpo.sh" "$@"

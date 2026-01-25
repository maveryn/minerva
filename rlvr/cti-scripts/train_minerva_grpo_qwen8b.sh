#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for Qwen3 8B.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-Qwen/Qwen3-8B}"
export GRPO_MAX_RESPONSE_LEN="${GRPO_MAX_RESPONSE_LEN:-2048}"

exec "$(dirname "$0")/train_minerva_grpo.sh" "$@"

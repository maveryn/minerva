#!/usr/bin/env bash
set -euo pipefail

# GRPO training script for Qwen3 8B.

export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export GRPO_MAX_RESPONSE_LEN="${GRPO_MAX_RESPONSE_LEN:-1024}"

exec "$(dirname "$0")/train_minerva_grpo.sh" "$@"

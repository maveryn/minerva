#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 8B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ACRD_ACR_ROLLOUT_N="${ACRD_ACR_ROLLOUT_N:-4}"
export ACRD_ACR_DISTILL_LR_SCALE="${ACRD_ACR_DISTILL_LR_SCALE:-0.5}"

exec "$(dirname "$0")/train_minerva_noctua.sh" "$@"

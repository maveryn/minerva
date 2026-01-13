#!/usr/bin/env bash
set -euo pipefail

# ACRD DPO training script for Llama 8B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ACRD_ACR_REWARD_MANAGER="${ACRD_ACR_REWARD_MANAGER:-batch}"
export ACRD_DEBUG_SAMPLES="${ACRD_DEBUG_SAMPLES:-0}"
export ACRD_DETAILS_DEBUG_SAMPLES="${ACRD_DETAILS_DEBUG_SAMPLES:-0}"
export ACRD_JUDGE_ENABLED="${ACRD_JUDGE_ENABLED:-true}"
export ACRD_JUDGE_BACKEND="${ACRD_JUDGE_BACKEND:-worker}"
export ACRD_JUDGE_MODEL="${ACRD_JUDGE_MODEL:-Qwen/Qwen3-8B}"
export ACRD_ACR_DISTILL_METHOD="${ACRD_ACR_DISTILL_METHOD:-dpo}"
export ACRD_ACR_DISTILL_BATCH_SIZE="${ACRD_ACR_DISTILL_BATCH_SIZE:-64}"
export ACRD_DPO_BETA="${ACRD_DPO_BETA:-0.1}"

exec "$(dirname "$0")/train_minerva_acrd.sh" "$@"

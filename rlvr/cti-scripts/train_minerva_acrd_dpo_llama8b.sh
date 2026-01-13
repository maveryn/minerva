#!/usr/bin/env bash
set -euo pipefail

# ACRD DPO training script for Llama 8B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ACRD_JUDGE_ENABLED="${ACRD_JUDGE_ENABLED:-true}"
export ACRD_JUDGE_MODEL="${ACRD_JUDGE_MODEL:-$ACRD_MODEL_PATH}"
export ACRD_ACR_DISTILL_METHOD="${ACRD_ACR_DISTILL_METHOD:-dpo}"
export ACRD_DPO_BETA="${ACRD_DPO_BETA:-0.1}"
export ACRD_DPO_REQUIRE_REJECTED_PARSES="${ACRD_DPO_REQUIRE_REJECTED_PARSES:-true}"

exec "$(dirname "$0")/train_minerva_acrd.sh" "$@"

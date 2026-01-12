#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 8B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

exec "$(dirname "$0")/train_minerva_acrd.sh" "$@"

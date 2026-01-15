#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Llama 3B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"

exec "$(dirname "$0")/train_minerva_noctua.sh" "$@"

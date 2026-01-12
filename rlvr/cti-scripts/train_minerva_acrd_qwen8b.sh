#!/usr/bin/env bash
set -euo pipefail

# ACRD training script for Qwen 8B.

export ACRD_MODEL_PATH="${ACRD_MODEL_PATH:-Qwen/Qwen3-8B-Instruct}"

exec "$(dirname "$0")/train_minerva_acrd.sh" "$@"

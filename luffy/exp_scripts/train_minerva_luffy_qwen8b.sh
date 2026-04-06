#!/usr/bin/env bash
set -euo pipefail

export MINERVA_MODEL_PATH="${MINERVA_MODEL_PATH:-Qwen/Qwen3-8B-Base}"
export MINERVA_ATTN_IMPLEMENTATION="${MINERVA_ATTN_IMPLEMENTATION:-flash_attention_2}"

exec bash "$(dirname "$0")/train_minerva_luffy.sh" "$@"

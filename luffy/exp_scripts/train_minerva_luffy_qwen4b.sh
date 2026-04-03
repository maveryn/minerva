#!/usr/bin/env bash
set -euo pipefail

export MINERVA_MODEL_PATH="${MINERVA_MODEL_PATH:-Qwen/Qwen3-4B-Base}"

exec bash "$(dirname "$0")/train_minerva_luffy.sh" "$@"

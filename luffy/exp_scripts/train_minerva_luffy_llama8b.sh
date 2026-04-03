#!/usr/bin/env bash
set -euo pipefail

export MINERVA_MODEL_PATH="${MINERVA_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

exec bash "$(dirname "$0")/train_minerva_luffy.sh" "$@"

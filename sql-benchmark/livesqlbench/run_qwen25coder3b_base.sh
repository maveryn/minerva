#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LIVESQLBENCH_MODEL="${LIVESQLBENCH_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
export LIVESQLBENCH_RUN_NAME="${LIVESQLBENCH_RUN_NAME:-livesqlbench_qwen25coder3b_base}"
exec bash "$SCRIPT_DIR/run_livesqlbench_model.sh" "$@"

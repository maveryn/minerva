#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LIVESQLBENCH_MODEL="${LIVESQLBENCH_MODEL:-${SQLR1_MINERVARL_MODEL_PATH:-}}"
if [ -z "$LIVESQLBENCH_MODEL" ]; then
  echo "Set SQLR1_MINERVARL_MODEL_PATH (or LIVESQLBENCH_MODEL) to a local MinervaRL HF model dir."
  exit 1
fi
export LIVESQLBENCH_RUN_NAME="${LIVESQLBENCH_RUN_NAME:-livesqlbench_qwen25coder3b_minervarl}"
exec bash "$SCRIPT_DIR/run_livesqlbench_model.sh" "$@"

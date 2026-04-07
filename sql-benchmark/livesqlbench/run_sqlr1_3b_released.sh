#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LIVESQLBENCH_MODEL="${LIVESQLBENCH_MODEL:-MPX0222forHF/SQL-R1-3B}"
export LIVESQLBENCH_RUN_NAME="${LIVESQLBENCH_RUN_NAME:-livesqlbench_sqlr1_3b_released}"
exec bash "$SCRIPT_DIR/run_livesqlbench_model.sh" "$@"

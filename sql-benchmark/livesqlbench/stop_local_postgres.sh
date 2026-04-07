#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LIVESQLBENCH_RUNTIME_DIR="${LIVESQLBENCH_RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
export LIVESQLBENCH_PG_DATA_DIR="${LIVESQLBENCH_PG_DATA_DIR:-$LIVESQLBENCH_RUNTIME_DIR/postgres-data}"

if [ -d "$LIVESQLBENCH_PG_DATA_DIR" ] && command -v pg_ctl >/dev/null 2>&1; then
  pg_ctl -D "$LIVESQLBENCH_PG_DATA_DIR" stop || true
fi

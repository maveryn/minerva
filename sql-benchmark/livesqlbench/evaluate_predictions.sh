#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OFFICIAL_EVAL_SRC="$SCRIPT_DIR/official/evaluation/src"

export LIVESQLBENCH_JSONL_FILE="${LIVESQLBENCH_JSONL_FILE:?Set LIVESQLBENCH_JSONL_FILE}"
export LIVESQLBENCH_EVAL_OUTPUT_DIR="${LIVESQLBENCH_EVAL_OUTPUT_DIR:-$(dirname "$LIVESQLBENCH_JSONL_FILE")}"
export LIVESQLBENCH_PG_HOST="${LIVESQLBENCH_PG_HOST:-127.0.0.1}"
export LIVESQLBENCH_PG_PORT="${LIVESQLBENCH_PG_PORT:-15432}"
export LIVESQLBENCH_EVAL_THREADS="${LIVESQLBENCH_EVAL_THREADS:-4}"
export LIVESQLBENCH_EVAL_LOGGING="${LIVESQLBENCH_EVAL_LOGGING:-false}"

if [ ! -f "$OFFICIAL_EVAL_SRC/evaluation.py" ]; then
  echo "Missing official evaluator files under $OFFICIAL_EVAL_SRC"
  echo "Run:"
  echo "  python $SCRIPT_DIR/setup_assets.py"
  exit 1
fi

PYTHONPATH="$OFFICIAL_EVAL_SRC${PYTHONPATH:+:$PYTHONPATH}" \
python "$OFFICIAL_EVAL_SRC/evaluation.py" \
  --jsonl_file "$LIVESQLBENCH_JSONL_FILE" \
  --db_host "$LIVESQLBENCH_PG_HOST" \
  --db_port "$LIVESQLBENCH_PG_PORT" \
  --num_threads "$LIVESQLBENCH_EVAL_THREADS" \
  --logging "$LIVESQLBENCH_EVAL_LOGGING" \
  --output_dir "$LIVESQLBENCH_EVAL_OUTPUT_DIR"

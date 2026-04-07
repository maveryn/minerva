#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINERVA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export LIVESQLBENCH_PG_HOST="${LIVESQLBENCH_PG_HOST:-127.0.0.1}"
export LIVESQLBENCH_PG_PORT="${LIVESQLBENCH_PG_PORT:-15432}"
export LIVESQLBENCH_PG_USER="${LIVESQLBENCH_PG_USER:-root}"
export LIVESQLBENCH_PG_PASSWORD="${LIVESQLBENCH_PG_PASSWORD:-123123}"

export LIVESQLBENCH_RUNTIME_DIR="${LIVESQLBENCH_RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
export LIVESQLBENCH_PG_DATA_DIR="${LIVESQLBENCH_PG_DATA_DIR:-$LIVESQLBENCH_RUNTIME_DIR/postgres-data}"
export LIVESQLBENCH_PG_LOG="${LIVESQLBENCH_PG_LOG:-$LIVESQLBENCH_RUNTIME_DIR/postgres.log}"
export LIVESQLBENCH_INIT_LOG="${LIVESQLBENCH_INIT_LOG:-$LIVESQLBENCH_RUNTIME_DIR/init-databases.log}"
export LIVESQLBENCH_DUMPS_DIR="${LIVESQLBENCH_DUMPS_DIR:-$SCRIPT_DIR/artifacts/postgre_table_dumps_full/bird-interact-full-dumps}"
export LIVESQLBENCH_OFFICIAL_INIT="${LIVESQLBENCH_OFFICIAL_INIT:-$LIVESQLBENCH_DUMPS_DIR/init-databases_postgresql.sh}"

mkdir -p "$LIVESQLBENCH_RUNTIME_DIR"

need_bin() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required PostgreSQL binary: $1"
    echo "Install PostgreSQL so initdb/pg_ctl/psql/createdb/dropdb are on PATH."
    exit 1
  fi
}

for bin in initdb pg_ctl psql createdb dropdb; do
  need_bin "$bin"
done

if [ ! -f "$SCRIPT_DIR/artifacts/livesqlbench_base_full_v1_full.jsonl" ]; then
  echo "Missing integrated dataset. Run:"
  echo "  python $SCRIPT_DIR/setup_assets.py"
  exit 1
fi

if [ ! -f "$LIVESQLBENCH_OFFICIAL_INIT" ]; then
  echo "Missing dump init script: $LIVESQLBENCH_OFFICIAL_INIT"
  echo "Run:"
  echo "  python $SCRIPT_DIR/setup_assets.py"
  exit 1
fi

export PGHOST="$LIVESQLBENCH_PG_HOST"
export PGPORT="$LIVESQLBENCH_PG_PORT"
export PGUSER="$LIVESQLBENCH_PG_USER"
export PGPASSWORD="$LIVESQLBENCH_PG_PASSWORD"
export PGDATABASE="${LIVESQLBENCH_PG_DATABASE:-postgres}"

if [ ! -d "$LIVESQLBENCH_PG_DATA_DIR" ]; then
  initdb -D "$LIVESQLBENCH_PG_DATA_DIR" --username="$LIVESQLBENCH_PG_USER" --auth=trust >/dev/null
  {
    echo "listen_addresses = '$LIVESQLBENCH_PG_HOST'"
    echo "port = $LIVESQLBENCH_PG_PORT"
    echo "max_prepared_transactions = 16"
  } >> "$LIVESQLBENCH_PG_DATA_DIR/postgresql.conf"
fi

if ! pg_ctl -D "$LIVESQLBENCH_PG_DATA_DIR" status >/dev/null 2>&1; then
  pg_ctl -D "$LIVESQLBENCH_PG_DATA_DIR" -l "$LIVESQLBENCH_PG_LOG" start
fi

until psql -h "$LIVESQLBENCH_PG_HOST" -p "$LIVESQLBENCH_PG_PORT" -U "$LIVESQLBENCH_PG_USER" -d postgres -c '\l' >/dev/null 2>&1; do
  >&2 echo "PostgreSQL is unavailable - waiting..."
  sleep 2
done

INIT_MARKER="$LIVESQLBENCH_RUNTIME_DIR/.base_full_initialized"
PATCHED_INIT="$LIVESQLBENCH_RUNTIME_DIR/init-databases_postgresql.local.sh"

if [ ! -f "$INIT_MARKER" ]; then
  python - "$LIVESQLBENCH_OFFICIAL_INIT" "$LIVESQLBENCH_DUMPS_DIR" "$PATCHED_INIT" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dumps_dir = Path(sys.argv[2]).resolve()
dst = Path(sys.argv[3])

text = src.read_text(encoding="utf-8")
text = text.replace("/docker-entrypoint-initdb.d/postgre_table_dumps", str(dumps_dir))
dst.write_text(text, encoding="utf-8")
dst.chmod(0o755)
PY

  bash "$PATCHED_INIT" >"$LIVESQLBENCH_INIT_LOG" 2>&1
  touch "$INIT_MARKER"
fi

echo
echo "LiveSQLBench PostgreSQL is ready."
echo "  host: $LIVESQLBENCH_PG_HOST"
echo "  port: $LIVESQLBENCH_PG_PORT"
echo "  user: $LIVESQLBENCH_PG_USER"
echo "  password: $LIVESQLBENCH_PG_PASSWORD"
if [ -f "$LIVESQLBENCH_INIT_LOG" ]; then
  echo "  init log: $LIVESQLBENCH_INIT_LOG"
fi

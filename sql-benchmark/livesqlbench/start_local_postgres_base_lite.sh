#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export LIVESQLBENCH_PG_HOST="${LIVESQLBENCH_PG_HOST:-127.0.0.1}"
export LIVESQLBENCH_PG_PORT="${LIVESQLBENCH_PG_PORT:-15432}"
export LIVESQLBENCH_PG_USER="${LIVESQLBENCH_PG_USER:-root}"
export LIVESQLBENCH_PG_PASSWORD="${LIVESQLBENCH_PG_PASSWORD:-123123}"

export LIVESQLBENCH_RUNTIME_DIR="${LIVESQLBENCH_RUNTIME_DIR:-$SCRIPT_DIR/runtime}"
export LIVESQLBENCH_PG_DATA_DIR="${LIVESQLBENCH_PG_DATA_DIR:-$LIVESQLBENCH_RUNTIME_DIR/postgres-data}"
export LIVESQLBENCH_PG_LOG="${LIVESQLBENCH_PG_LOG:-$LIVESQLBENCH_RUNTIME_DIR/postgres.log}"
export LIVESQLBENCH_INIT_LOG="${LIVESQLBENCH_INIT_LOG:-$LIVESQLBENCH_RUNTIME_DIR/init-databases-base-lite.log}"
export LIVESQLBENCH_DUMPS_DIR="${LIVESQLBENCH_DUMPS_DIR:-$SCRIPT_DIR/artifacts/postgre_table_dumps_base_lite/livesqlbench-base-lite-dumps}"
export LIVESQLBENCH_PUBLIC_DATASET_ROOT="${LIVESQLBENCH_PUBLIC_DATASET_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/livesqlbench-base-lite}"
export LIVESQLBENCH_INTEGRATED_JSONL="${LIVESQLBENCH_INTEGRATED_JSONL:-$SCRIPT_DIR/artifacts/livesqlbench_base_lite_full.jsonl}"

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

if [ ! -f "$LIVESQLBENCH_INTEGRATED_JSONL" ]; then
  echo "Missing integrated Base-Lite dataset: $LIVESQLBENCH_INTEGRATED_JSONL"
  echo "Run:"
  echo "  python $SCRIPT_DIR/setup_assets_base_lite.py"
  exit 1
fi

if [ ! -d "$LIVESQLBENCH_DUMPS_DIR" ]; then
  echo "Missing Base-Lite dump directory: $LIVESQLBENCH_DUMPS_DIR"
  echo "Run:"
  echo "  python $SCRIPT_DIR/setup_assets_base_lite.py"
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

INIT_MARKER="$LIVESQLBENCH_RUNTIME_DIR/.base_lite_initialized"

if [ ! -f "$INIT_MARKER" ]; then
  python - "$LIVESQLBENCH_INTEGRATED_JSONL" "$LIVESQLBENCH_DUMPS_DIR" "$LIVESQLBENCH_INIT_LOG" <<'PY'
from pathlib import Path
import json
import subprocess
import sys

integrated_jsonl = Path(sys.argv[1])
dumps_dir = Path(sys.argv[2]).resolve()
log_path = Path(sys.argv[3])

rows = [json.loads(line) for line in integrated_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
db_names = sorted({row["selected_database"] for row in rows})

def run_psql(*args: str, sql: str | None = None, log_file) -> None:
    cmd = ["psql", "-U", "root", *args]
    if sql is not None:
        cmd.extend(["-c", sql])
    proc = subprocess.run(cmd, stdout=log_file, stderr=log_file)
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed: {' '.join(cmd)}")

def db_exists(name: str) -> bool:
    result = subprocess.run(
        ["psql", "-U", "root", "-tc", f"SELECT 1 FROM pg_database WHERE datname='{name}'"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "1" in result.stdout

with log_path.open("w", encoding="utf-8") as log:
    if not db_exists("sql_test_template"):
        run_psql(sql="CREATE DATABASE sql_test_template WITH OWNER=root ENCODING='UTF8' TEMPLATE=template0;", log_file=log)
        run_psql("-d", "sql_test_template", sql="CREATE SCHEMA IF NOT EXISTS test_schema;", log_file=log)
        run_psql("-d", "sql_test_template", sql="CREATE SCHEMA IF NOT EXISTS test_schema_2;", log_file=log)
        run_psql("-d", "sql_test_template", sql="CREATE EXTENSION IF NOT EXISTS hstore;", log_file=log)
        run_psql("-d", "sql_test_template", sql="CREATE EXTENSION IF NOT EXISTS citext;", log_file=log)
        run_psql("-d", "sql_test_template", sql="ALTER DATABASE sql_test_template SET default_text_search_config = 'pg_catalog.english';", log_file=log)

    for db_name in db_names:
        template_db = f"{db_name}_template"
        if not db_exists(template_db):
            run_psql(sql=f"CREATE DATABASE {template_db} WITH OWNER=root ENCODING='UTF8' TEMPLATE=template0;", log_file=log)

        folder = dumps_dir / f"{db_name}_template"
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dump folder: {folder}")

        full_sql = next(folder.glob("*_full.sql"), None)
        if full_sql is not None:
            proc = subprocess.run(["psql", "-U", "root", "-d", template_db, "-f", str(full_sql)], stdout=log, stderr=log)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed importing {full_sql} into {template_db}")
        else:
            enum_file = folder / "enum_definitions.sql"
            if enum_file.exists():
                proc = subprocess.run(["psql", "-U", "root", "-d", template_db, "-f", str(enum_file)], stdout=log, stderr=log)
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed importing {enum_file} into {template_db}")

            order_file = dumps_dir / f"{db_name}_table_orders.txt"
            if not order_file.exists():
                raise FileNotFoundError(f"Missing table order file: {order_file}")

            sql_lookup = {}
            for sql_file in folder.glob("*.sql"):
                if sql_file.name == "enum_definitions.sql":
                    continue
                sql_lookup[sql_file.stem.lower()] = sql_file

            ordered_names = order_file.read_text(encoding="utf-8").split()
            for table_name in ordered_names:
                sql_file = sql_lookup.get(table_name.lower())
                if sql_file is None:
                    raise FileNotFoundError(f"No SQL file for table {table_name} in {folder}")
                proc = subprocess.run(["psql", "-U", "root", "-d", template_db, "-f", str(sql_file)], stdout=log, stderr=log)
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed importing {sql_file} into {template_db}")

        run_psql("-d", "postgres", sql=f"UPDATE pg_database SET datistemplate = true WHERE datname = '{template_db}';", log_file=log)

        if not db_exists(db_name):
            run_psql(sql=f"CREATE DATABASE {db_name} WITH OWNER=root TEMPLATE={template_db};", log_file=log)
PY

  touch "$INIT_MARKER"
fi

echo
echo "LiveSQLBench Base-Lite PostgreSQL is ready."
echo "  host: $LIVESQLBENCH_PG_HOST"
echo "  port: $LIVESQLBENCH_PG_PORT"
echo "  user: $LIVESQLBENCH_PG_USER"
echo "  password: $LIVESQLBENCH_PG_PASSWORD"
if [ -f "$LIVESQLBENCH_INIT_LOG" ]; then
  echo "  init log: $LIVESQLBENCH_INIT_LOG"
fi

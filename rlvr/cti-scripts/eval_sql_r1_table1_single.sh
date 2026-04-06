#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINERVA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export SQLR1_ROOT="${SQLR1_ROOT:-/home/jovyan/work/SQL-R1}"
export SQLR1_MERGED_MODEL_DIR="${SQLR1_MERGED_MODEL_DIR:?Set SQLR1_MERGED_MODEL_DIR to a local HF model dir}"
export SQLR1_EVAL_RESULTS_ROOT="${SQLR1_EVAL_RESULTS_ROOT:?Set SQLR1_EVAL_RESULTS_ROOT}"
export SQLR1_EVAL_NUM_GPUS="${SQLR1_EVAL_NUM_GPUS:-1}"
export SQLR1_EVAL_N="${SQLR1_EVAL_N:-8}"
export SQLR1_EVAL_TEMPERATURE="${SQLR1_EVAL_TEMPERATURE:-0.8}"
export SQLR1_EVAL_DATASET="${SQLR1_EVAL_DATASET:?Set SQLR1_EVAL_DATASET to spider-dev, spider-test, or bird-dev}"

mkdir -p "$SQLR1_EVAL_RESULTS_ROOT"

SUMMARY_TSV="$SQLR1_EVAL_RESULTS_ROOT/summary.tsv"
printf "dataset\texec_accuracy_all\tpred_json\teval_log\n" > "$SUMMARY_TSV"

run_spider_dataset() {
  local dataset="$1"
  local input_file=""
  local db_path=""
  local table_value_cache=""
  local table_info_cache=""
  local gold_sql=""
  local table_json=""
  local output_prefix=""

  case "$dataset" in
    spider-dev)
      input_file="$SQLR1_ROOT/data/NL2SQL/Spider/dev.json"
      db_path="$SQLR1_ROOT/data/NL2SQL/Spider/database"
      table_value_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json"
      table_info_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2db_info.json"
      gold_sql="$SQLR1_ROOT/data/NL2SQL/Spider/dev_gold.sql"
      table_json="$SQLR1_ROOT/data/NL2SQL/Spider/tables.json"
      output_prefix="spiderdev"
      ;;
    spider-test)
      input_file="$SQLR1_ROOT/data/NL2SQL/Spider/test.json"
      db_path="$SQLR1_ROOT/data/NL2SQL/Spider/test_database"
      table_value_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spidertest_db_id2sampled_db_values.json"
      table_info_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spidertest_db_id2db_info.json"
      gold_sql="$SQLR1_ROOT/data/NL2SQL/Spider/test_gold.sql"
      table_json="$SQLR1_ROOT/data/NL2SQL/Spider/test_tables.json"
      output_prefix="spidertest"
      ;;
    *)
      echo "Unsupported Spider dataset: $dataset"
      exit 1
      ;;
  esac

  local pred_json="$SQLR1_EVAL_RESULTS_ROOT/${output_prefix}_generated_sql.json"
  local post_log="$SQLR1_EVAL_RESULTS_ROOT/post.log"
  local eval_log="$SQLR1_EVAL_RESULTS_ROOT/eval.log"
  local pred_sql_txt="$SQLR1_EVAL_RESULTS_ROOT/${output_prefix}_generated_sql_pred_major_voting_sqls.txt"

  (
    cd "$SQLR1_ROOT"
    python src/inference.py \
      --nl2sql_ckpt_path "$SQLR1_MERGED_MODEL_DIR" \
      --dataset_name spider \
      --input_file "$input_file" \
      --output_file "$pred_json" \
      --database_path "$db_path" \
      --tensor_parallel_size "$SQLR1_EVAL_NUM_GPUS" \
      --n "$SQLR1_EVAL_N" \
      --temperature "$SQLR1_EVAL_TEMPERATURE" \
      --output_format json \
      --table_value_cache_path "$table_value_cache" \
      --table_info_cache_path "$table_info_cache"

    python src/evaluation_spider_post.py \
      --pred "$pred_json" \
      --gold "$gold_sql" \
      --db_path "$db_path/" \
      --table "$table_json" \
      --mode major_voting \
      --save_dir "$SQLR1_EVAL_RESULTS_ROOT" | tee "$post_log"

    python src/evaluation_spider.py \
      --gold_sql "$gold_sql" \
      --pred_sql "$pred_sql_txt" \
      --db "$db_path" \
      --table "$table_json" \
      --etype exec \
      --save_dir "$SQLR1_EVAL_RESULTS_ROOT" | tee "$eval_log"
  )

  local exec_all
  exec_all="$(python - "$eval_log" <<'PY'
import sys
value = ""
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0] == "execution" and len(parts) >= 6:
            value = parts[-1]
print(value)
PY
)"

  printf "%s\t%s\t%s\t%s\n" "$dataset" "$exec_all" "$pred_json" "$eval_log" >> "$SUMMARY_TSV"
  printf "%s EX(all): %s\n" "$dataset" "$exec_all"
}

run_bird_dataset() {
  local pred_json="$SQLR1_EVAL_RESULTS_ROOT/birddev_generated_sql.json"
  local eval_log="$SQLR1_EVAL_RESULTS_ROOT/eval.log"

  (
    cd "$SQLR1_ROOT"
    python src/inference.py \
      --nl2sql_ckpt_path "$SQLR1_MERGED_MODEL_DIR" \
      --dataset_name bird \
      --input_file "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/dev.json" \
      --output_file "$pred_json" \
      --database_path "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/dev_databases" \
      --tensor_parallel_size "$SQLR1_EVAL_NUM_GPUS" \
      --n "$SQLR1_EVAL_N" \
      --temperature "$SQLR1_EVAL_TEMPERATURE" \
      --output_format json \
      --table_value_cache_path "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/bird_db_id2sampled_db_values.json" \
      --table_info_cache_path "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/bird_db_id2db_info.json"

    python - "$pred_json" "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/dev.json" "$SQLR1_ROOT/data/NL2SQL/BIRD/dev/dev_databases" <<'PY' | tee "$eval_log"
import sys
sys.path.insert(0, "src")
from evaluation_bird_post import run_eval

pred_json, gold_json, db_path = sys.argv[1], sys.argv[2], sys.argv[3]
acc, _ = run_eval(gold_json, pred_json, db_path, "major_voting", True)
print(f"BIRD_EX_ACC={acc}")
PY
  )

  local exec_all
  exec_all="$(python - "$eval_log" <<'PY'
import sys
value = ""
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if line.startswith("BIRD_EX_ACC="):
            value = line.split("=", 1)[1]
print(value)
PY
)"

  printf "%s\t%s\t%s\t%s\n" "bird-dev" "$exec_all" "$pred_json" "$eval_log" >> "$SUMMARY_TSV"
  printf "bird-dev EX(all): %s\n" "$exec_all"
}

case "$SQLR1_EVAL_DATASET" in
  spider-dev|spider-test)
    run_spider_dataset "$SQLR1_EVAL_DATASET"
    ;;
  bird-dev)
    run_bird_dataset
    ;;
  *)
    echo "Unsupported dataset: $SQLR1_EVAL_DATASET"
    exit 1
    ;;
esac

echo
echo "Summary written to $SUMMARY_TSV"

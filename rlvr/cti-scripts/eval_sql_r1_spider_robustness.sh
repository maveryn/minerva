#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained SQL-R1-style checkpoint on the paper's Spider robustness sets:
# Spider-DK, Spider-Syn, and Spider-Realistic.
#
# This script expects a VeRL FSDP checkpoint (best/last actor shards), merges it into a
# normal Hugging Face model if needed, then runs SQL-R1 inference + EX evaluation.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINERVA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RLVR_ROOT="$MINERVA_ROOT/rlvr"

export SQLR1_ROOT="${SQLR1_ROOT:-/home/jovyan/work/SQL-R1}"
export SQLR1_CHECKPOINT_ROOT="${SQLR1_CHECKPOINT_ROOT:-$RLVR_ROOT/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps}"
export SQLR1_CHECKPOINT_KIND="${SQLR1_CHECKPOINT_KIND:-best}"
export SQLR1_ACTOR_CKPT_DIR="${SQLR1_ACTOR_CKPT_DIR:-$SQLR1_CHECKPOINT_ROOT/$SQLR1_CHECKPOINT_KIND/actor}"
export SQLR1_MERGED_MODEL_DIR="${SQLR1_MERGED_MODEL_DIR:-$SQLR1_CHECKPOINT_ROOT/${SQLR1_CHECKPOINT_KIND}_merged_hf}"
export SQLR1_EVAL_RESULTS_ROOT="${SQLR1_EVAL_RESULTS_ROOT:-$SQLR1_CHECKPOINT_ROOT/eval_table2_${SQLR1_CHECKPOINT_KIND}}"

export SQLR1_EVAL_NUM_GPUS="${SQLR1_EVAL_NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SQLR1_EVAL_N="${SQLR1_EVAL_N:-8}"
export SQLR1_EVAL_TEMPERATURE="${SQLR1_EVAL_TEMPERATURE:-0.8}"
export SQLR1_MERGE_IF_MISSING="${SQLR1_MERGE_IF_MISSING:-true}"
export SQLR1_EVAL_DATASETS="${SQLR1_EVAL_DATASETS:-spider-dk spider-syn spider-realistic}"

mkdir -p "$SQLR1_EVAL_RESULTS_ROOT"

if [ ! -f "$SQLR1_MERGED_MODEL_DIR/model.safetensors.index.json" ]; then
  if [ "$SQLR1_MERGE_IF_MISSING" != "true" ]; then
    echo "Merged HF model missing at $SQLR1_MERGED_MODEL_DIR"
    exit 1
  fi
  (
    cd "$RLVR_ROOT"
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$SQLR1_ACTOR_CKPT_DIR" \
      --target_dir "$SQLR1_MERGED_MODEL_DIR"
  )
fi

SUMMARY_TSV="$SQLR1_EVAL_RESULTS_ROOT/summary.tsv"
printf "dataset\texec_accuracy_all\tpred_json\tpred_sql_txt\teval_log\n" > "$SUMMARY_TSV"

prepare_gold_sql() {
  local gold_sql_src="$1"
  local db_root="$2"
  local gold_sql_out="$3"

  python - "$gold_sql_src" "$db_root" "$gold_sql_out" <<'PY'
import sqlite3
import sys
from pathlib import Path

gold_src = Path(sys.argv[1])
db_root = Path(sys.argv[2])
gold_out = Path(sys.argv[3])

lines = gold_src.read_text(encoding="utf-8").splitlines()

def parse_line(raw: str):
    query, db_id = raw.rsplit("\t", 1)
    return query, db_id

def executable(query: str, db_id: str) -> bool:
    db_file = db_root / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return False
    conn = sqlite3.connect(str(db_file))
    conn.text_factory = lambda b: b.decode(errors="ignore")
    try:
        cur = conn.cursor()
        cur.execute(query)
        cur.fetchall()
        return True
    except Exception:
        return False
    finally:
        conn.close()

repaired = []
repair_count = 0

for idx, raw in enumerate(lines):
    query, db_id = parse_line(raw)
    if executable(query, db_id):
        repaired.append(raw)
        continue

    replacement = None
    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(lines):
            n_query, n_db_id = parse_line(lines[neighbor_idx])
            if n_db_id == db_id and executable(n_query, n_db_id):
                replacement = f"{n_query}\t{db_id}"
                break

    if replacement is None:
        raise SystemExit(
            f"Unable to repair invalid gold SQL at line {idx + 1}: {raw[:200]}"
        )

    repaired.append(replacement)
    repair_count += 1

gold_out.write_text("\n".join(repaired) + "\n", encoding="utf-8")
print(f"Sanitized gold SQL: {gold_src} -> {gold_out} (repaired {repair_count} lines)")
PY
}

run_dataset() {
  local dataset="$1"
  local input_file=""
  local db_path=""
  local table_value_cache=""
  local table_info_cache=""
  local gold_sql=""
  local table_json=""
  local output_prefix=""

  case "$dataset" in
    spider-dk)
      input_file="$SQLR1_ROOT/data/NL2SQL/Spider-DK/spiderdk_dev.json"
      db_path="$SQLR1_ROOT/data/NL2SQL/Spider-DK/database"
      table_value_cache="$SQLR1_ROOT/data/NL2SQL/Spider-DK/spiderdkdev_db_id2sampled_db_values.json"
      table_info_cache="$SQLR1_ROOT/data/NL2SQL/Spider-DK/spiderdkdev_db_id2db_info.json"
      gold_sql="$SQLR1_ROOT/data/NL2SQL/Spider-DK/dev_gold.sql"
      table_json="$SQLR1_ROOT/data/NL2SQL/Spider-DK/tables.json"
      output_prefix="spiderdk"
      ;;
    spider-syn)
      input_file="$SQLR1_ROOT/data/NL2SQL/Spider-Syn/spider_syn.json"
      db_path="$SQLR1_ROOT/data/NL2SQL/Spider/database"
      table_value_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json"
      table_info_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2db_info.json"
      gold_sql="$SQLR1_ROOT/data/NL2SQL/Spider-Syn/dev_gold.sql"
      table_json="$SQLR1_ROOT/data/NL2SQL/Spider/tables.json"
      output_prefix="spidersyn"
      ;;
    spider-realistic)
      input_file="$SQLR1_ROOT/data/NL2SQL/Spider-Realistic/spider-realistic.json"
      db_path="$SQLR1_ROOT/data/NL2SQL/Spider/database"
      table_value_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2sampled_db_values.json"
      table_info_cache="$SQLR1_ROOT/data/NL2SQL/Spider/spiderdev_db_id2db_info.json"
      gold_sql="$SQLR1_ROOT/data/NL2SQL/Spider-Realistic/dev_gold.sql"
      table_json="$SQLR1_ROOT/data/NL2SQL/Spider-Realistic/tables.json"
      output_prefix="spiderrealistic"
      ;;
    *)
      echo "Unsupported dataset: $dataset"
      exit 1
      ;;
  esac

  local dataset_results_dir="$SQLR1_EVAL_RESULTS_ROOT/$dataset"
  local pred_json="$dataset_results_dir/${output_prefix}_generated_sql.json"
  local gold_sql_sanitized="$dataset_results_dir/${output_prefix}_gold_sanitized.sql"
  local post_log="$dataset_results_dir/post.log"
  local eval_log="$dataset_results_dir/eval.log"
  local pred_sql_txt="$dataset_results_dir/${output_prefix}_generated_sql_pred_major_voting_sqls.txt"
  mkdir -p "$dataset_results_dir"

  prepare_gold_sql "$gold_sql" "$db_path" "$gold_sql_sanitized"

  (
    cd "$SQLR1_ROOT"
    python src/inference.py \
      --nl2sql_ckpt_path "$SQLR1_MERGED_MODEL_DIR" \
      --dataset_name "$dataset" \
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
      --gold "$gold_sql_sanitized" \
      --db_path "$db_path/" \
      --table "$table_json" \
      --mode major_voting \
      --save_dir "$dataset_results_dir" | tee "$post_log"

    python src/evaluation_spider.py \
      --gold_sql "$gold_sql_sanitized" \
      --pred_sql "$pred_sql_txt" \
      --db "$db_path" \
      --table "$table_json" \
      --etype exec \
      --save_dir "$dataset_results_dir" | tee "$eval_log"
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

  printf "%s\t%s\t%s\t%s\t%s\n" "$dataset" "$exec_all" "$pred_json" "$pred_sql_txt" "$eval_log" >> "$SUMMARY_TSV"
  printf "%s EX(all): %s\n" "$dataset" "$exec_all"
}

for dataset in $SQLR1_EVAL_DATASETS; do
  run_dataset "$dataset"
done

echo
echo "Summary written to $SUMMARY_TSV"

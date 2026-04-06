#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINERVA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export SQLR1_SQL_BENCHMARK_ROOT="${SQLR1_SQL_BENCHMARK_ROOT:-$MINERVA_ROOT/sql-benchmark}"
export SQLR1_GPU_IDS="${SQLR1_GPU_IDS:-0 1 2 3 4 5 6 7}"

QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5}"
SQLR1_RELEASED_MODEL_DIR="${SQLR1_RELEASED_MODEL_DIR:-/home/jovyan/.cache/huggingface/hub/models--MPX0222forHF--SQL-R1-3B/snapshots/49415b2c0e5df9f258a6e3d78affb068a1c23cef}"
NOCTUA_MODEL_DIR="${NOCTUA_MODEL_DIR:-$MINERVA_ROOT/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf}"

mkdir -p "$SQLR1_SQL_BENCHMARK_ROOT"

declare -a MODELS=(
  "qwen25coder3b_instruct_base:$QWEN_MODEL_DIR"
  "sql_r1_3b_released:$SQLR1_RELEASED_MODEL_DIR"
  "noctua_qwen25coder3b_100steps_best:$NOCTUA_MODEL_DIR"
)
declare -a DATASETS=("spider-dev" "spider-test" "bird-dev")

mapfile -t GPU_IDS < <(for gpu in $SQLR1_GPU_IDS; do echo "$gpu"; done)
GPU_COUNT="${#GPU_IDS[@]}"

launch_job() {
  local model_name="$1"
  local model_dir="$2"
  local dataset="$3"
  local gpu="$4"
  local result_dir="$SQLR1_SQL_BENCHMARK_ROOT/${model_name}_table1/${dataset}"
  local log_file="$result_dir/run.log"

  mkdir -p "$result_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SQLR1_EVAL_NUM_GPUS=1
    export SQLR1_MERGED_MODEL_DIR="$model_dir"
    export SQLR1_EVAL_RESULTS_ROOT="$result_dir"
    export SQLR1_EVAL_DATASET="$dataset"
    bash "$SCRIPT_DIR/eval_sql_r1_table1_single.sh"
  ) >"$log_file" 2>&1 &
}

job_idx=0
for model_spec in "${MODELS[@]}"; do
  model_name="${model_spec%%:*}"
  model_dir="${model_spec#*:}"
  for dataset in "${DATASETS[@]}"; do
    while [ "$(jobs -pr | wc -l)" -ge "$GPU_COUNT" ]; do
      wait -n
    done
    gpu="${GPU_IDS[$((job_idx % GPU_COUNT))]}"
    launch_job "$model_name" "$model_dir" "$dataset" "$gpu"
    job_idx=$((job_idx + 1))
  done
done

wait

SUMMARY_TSV="$SQLR1_SQL_BENCHMARK_ROOT/summary_table1.tsv"
printf "model\tdataset\texec_accuracy_all\tpred_json\teval_log\n" > "$SUMMARY_TSV"

for model_spec in "${MODELS[@]}"; do
  model_name="${model_spec%%:*}"
  for dataset in "${DATASETS[@]}"; do
    summary_file="$SQLR1_SQL_BENCHMARK_ROOT/${model_name}_table1/${dataset}/summary.tsv"
    if [ -f "$summary_file" ]; then
      tail -n +2 "$summary_file" | awk -v model="$model_name" 'BEGIN{FS=OFS="\t"} NF {print model, $0}' >> "$SUMMARY_TSV"
    fi
  done
done

echo
echo "Combined summary written to $SUMMARY_TSV"

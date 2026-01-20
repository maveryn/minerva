#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

INPUT_DIR="${MINERVA_JUDGE_RESPONSES_DIR:-$ROOT_DIR/data}"
OUTPUT_PATH="${MINERVA_JUDGE_JUDGED_OUTPUT:-$ROOT_DIR/data}"
JUDGE_MODEL="${MINERVA_JUDGE_JUDGE_MODEL:-openai/gpt-oss-120b}"
BACKEND="${MINERVA_JUDGE_BACKEND:-vllm}"
TEMPERATURE="${MINERVA_JUDGE_TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MINERVA_JUDGE_MAX_NEW_TOKENS:-512}"
VLLM_ARGS="${MINERVA_JUDGE_VLLM_ARGS:-{\"gpu_memory_utilization\":0.95}}"
DEFAULT_BATCH_SIZE="${MINERVA_JUDGE_BATCH_SIZE:-1024}"

mapfile -t INPUT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "responses_*.jsonl" | sort)
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
  echo "No response files found in $INPUT_DIR"
  exit 1
fi

OUTPUT_IS_DIR=true
if [[ "$OUTPUT_PATH" == *.jsonl ]]; then
  OUTPUT_IS_DIR=false
fi

if [ "$OUTPUT_IS_DIR" = true ]; then
  FILTERED_FILES=()
  for file in "${INPUT_FILES[@]}"; do
    base="$(basename "$file")"
    if [[ "$base" == responses_* ]]; then
      judged_name="judged_${base#responses_}"
    else
      judged_name="judged_${base}"
    fi
    judged_path="$OUTPUT_PATH/$judged_name"
    if [ -s "$judged_path" ]; then
      echo "Skipping $file (found $judged_path)"
      continue
    fi
    FILTERED_FILES+=("$file")
  done
  INPUT_FILES=("${FILTERED_FILES[@]}")
  if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
    echo "All response files already have judged outputs in $OUTPUT_PATH"
    exit 0
  fi
fi

cmd=(python "$ROOT_DIR/scripts/score_with_judge.py"
  --output "$OUTPUT_PATH"
  --judge-model "$JUDGE_MODEL"
  --backend "$BACKEND"
  --batch-size "$DEFAULT_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
)

for file in "${INPUT_FILES[@]}"; do
  cmd+=(--input "$file")
done

if [ -n "$VLLM_ARGS" ]; then
  cmd+=(--vllm-args "$VLLM_ARGS")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"

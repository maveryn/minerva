#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT_DIR="${MINERVA_JUDGE_RESPONSES_DIR:-$ROOT_DIR/data}"
OUTPUT_PATH="${MINERVA_JUDGE_JUDGED_OUTPUT:-$ROOT_DIR/data/judged_all.jsonl}"
JUDGE_MODEL="${MINERVA_JUDGE_JUDGE_MODEL:-openai/gpt-oss-120b}"
BACKEND="${MINERVA_JUDGE_BACKEND:-vllm}"
TEMPERATURE="${MINERVA_JUDGE_TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MINERVA_JUDGE_MAX_NEW_TOKENS:-64}"
VLLM_ARGS="${MINERVA_JUDGE_VLLM_ARGS:-}"
DEFAULT_BATCH_SIZE="${MINERVA_JUDGE_BATCH_SIZE:-8}"

mapfile -t INPUT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "responses_*.jsonl" | sort)
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
  echo "No response files found in $INPUT_DIR"
  exit 1
fi

cmd=(python "$ROOT_DIR/scripts/score_with_judge.py"
  --output "$OUTPUT_PATH"
  --judge-model "$JUDGE_MODEL"
  --backend "$BACKEND"
  --batch-size "$DEFAULT_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --include-incorrect
)

for file in "${INPUT_FILES[@]}"; do
  cmd+=(--input "$file")
done

if [ -n "$VLLM_ARGS" ]; then
  cmd+=(--vllm-args "$VLLM_ARGS")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"

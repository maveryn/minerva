#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

INPUT_DIR="${MINERVA_SYNTH_BAD_INPUT_DIR:-$ROOT_DIR/data}"
OUTPUT_PATH="${MINERVA_SYNTH_BAD_OUTPUT:-$ROOT_DIR/data/synthetic_bad.jsonl}"
MODEL="${MINERVA_SYNTH_BAD_MODEL:-openai/gpt-oss-120b}"
BACKEND="${MINERVA_SYNTH_BAD_BACKEND:-vllm}"
TEMPERATURE="${MINERVA_SYNTH_BAD_TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MINERVA_SYNTH_BAD_MAX_NEW_TOKENS:-1024}"
VLLM_ARGS="${MINERVA_SYNTH_BAD_VLLM_ARGS:-{\"gpu_memory_utilization\":0.95}}"
DEFAULT_BATCH_SIZE="${MINERVA_SYNTH_BAD_BATCH_SIZE:-512}"
NUM_SAMPLES="${MINERVA_SYNTH_BAD_NUM_SAMPLES:-5000}"

mapfile -t INPUT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "responses_*.jsonl" | sort)
if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
  echo "No response files found in $INPUT_DIR"
  exit 1
fi

cmd=(python "$ROOT_DIR/scripts/generate_synthetic_bad.py"
  --output "$OUTPUT_PATH"
  --model "$MODEL"
  --backend "$BACKEND"
  --batch-size "$DEFAULT_BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --num-samples "$NUM_SAMPLES"
)

for file in "${INPUT_FILES[@]}"; do
  cmd+=(--input "$file")
done

if [ -n "$VLLM_ARGS" ]; then
  cmd+=(--vllm-args "$VLLM_ARGS")
fi

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"

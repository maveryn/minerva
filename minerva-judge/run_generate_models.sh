#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"

INPUT_JSONL="${MINERVA_JUDGE_INPUT:-$REPO_ROOT/dataset/minerva_base_split/minerva-base-train.jsonl}"
OUTPUT_DIR="${MINERVA_JUDGE_OUTPUT_DIR:-$ROOT_DIR/data}"
BACKEND="${MINERVA_JUDGE_BACKEND:-vllm}"
TEMPERATURE="${MINERVA_JUDGE_TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MINERVA_JUDGE_MAX_NEW_TOKENS:-1024}"
VARIANTS="${MINERVA_JUDGE_VARIANTS:-both}"
START_INDEX="${MINERVA_JUDGE_START_INDEX:-0}"
LIMIT="${MINERVA_JUDGE_LIMIT:-}"

VLLM_ARGS="${MINERVA_JUDGE_VLLM_ARGS:-}"

DEFAULT_BATCH_SIZE="${MINERVA_JUDGE_BATCH_SIZE:-32}"
GPT_OSS_BATCH_SIZE="${MINERVA_JUDGE_BATCH_SIZE_GPT_OSS_20B:-8}"
GPT_OSS_VLLM_ARGS="${MINERVA_JUDGE_VLLM_ARGS_GPT_OSS_20B:-}"

MODELS=(
  "meta-llama/Llama-3.2-3B-Instruct|llama3_2_3b"
  "meta-llama/Llama-3.1-8B-Instruct|llama3_1_8b"
  "Qwen/Qwen3-4B-Instruct-2507|qwen3_4b"
  "Qwen/Qwen3-8B-Instruct|qwen3_8b"
  "openai/gpt-oss-20b|gpt_oss_20b"
)

mkdir -p "$OUTPUT_DIR"

for entry in "${MODELS[@]}"; do
  IFS="|" read -r MODEL_NAME MODEL_SLUG <<<"$entry"
  OUTPUT_PATH="$OUTPUT_DIR/responses_${MODEL_SLUG}.jsonl"

  BATCH_SIZE="$DEFAULT_BATCH_SIZE"
  VLLM_ARGS_USED="$VLLM_ARGS"
  if [ "$MODEL_SLUG" = "gpt_oss_20b" ]; then
    BATCH_SIZE="$GPT_OSS_BATCH_SIZE"
    if [ -n "$GPT_OSS_VLLM_ARGS" ]; then
      VLLM_ARGS_USED="$GPT_OSS_VLLM_ARGS"
    fi
  fi

  cmd=(python "$ROOT_DIR/scripts/generate_answer_guided.py"
    --input "$INPUT_JSONL"
    --output "$OUTPUT_PATH"
    --model "$MODEL_NAME"
    --backend "$BACKEND"
    --batch-size "$BATCH_SIZE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --variants "$VARIANTS"
    --start-index "$START_INDEX"
  )
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [ -n "$VLLM_ARGS_USED" ]; then
    cmd+=(--vllm-args "$VLLM_ARGS_USED")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_PATH="${DART_CONFIG_PATH:-$ROOT_DIR/dart/configs/dart_cti_llama8b.yaml}"
ACCEPTED_V1_PATH="${DART_ACCEPTED_V1_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v1.jsonl}"
GROUP_NAME="${DART_GROUP_NAME:-train}"
OUTPUT_TAG="${DART_OUTPUT_TAG:-llama8b_v1}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec python -m dart.run_dart_v2 \
  --config "$CONFIG_PATH" \
  --accepted-v1 "$ACCEPTED_V1_PATH" \
  --group-name "$GROUP_NAME" \
  --output-tag "$OUTPUT_TAG" \
  "$@"

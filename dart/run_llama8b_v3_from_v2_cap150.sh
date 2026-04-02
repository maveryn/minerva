#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_PATH="${DART_CONFIG_PATH:-$ROOT_DIR/dart/configs/dart_cti_llama8b_v3.yaml}"
GROUP_NAME="${DART_GROUP_NAME:-train}"
TARGET_ACCEPTED="${DART_TARGET_ACCEPTED:-2}"
V2_CAP="${DART_V2_CAP:-150}"

SEED_V1_PATH="${DART_ACCEPTED_V1_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/accepted_v1.jsonl}"
ATTEMPTS_V2_PATH="${DART_ATTEMPTS_V2_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/attempts_v2_from_llama8b_v1.jsonl}"

RECOVER_TAG="${DART_RECOVER_TAG:-llama8b_v2_cap${V2_CAP}_seed}"
RECOVERED_SEED_PATH="${DART_RECOVERED_SEED_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/accepted_${RECOVER_TAG}.jsonl}"
RECOVERED_SUMMARY_PATH="${DART_RECOVERED_SUMMARY_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/summary_${RECOVER_TAG}.json}"

OUTPUT_TAG="${DART_OUTPUT_TAG:-llama8b_v3_from_${RECOVER_TAG}}"

python -m dart.recover_fill_seed \
  --seed-jsonl "$SEED_V1_PATH" \
  --attempts-jsonl "$ATTEMPTS_V2_PATH" \
  --output-jsonl "$RECOVERED_SEED_PATH" \
  --summary-output "$RECOVERED_SUMMARY_PATH" \
  --target-accepted "$TARGET_ACCEPTED" \
  --max-attempts-per-question "$V2_CAP" \
  --accept-mode verifier+filter

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec python -m dart.run_dart_v2 \
  --config "$CONFIG_PATH" \
  --accepted-v1 "$RECOVERED_SEED_PATH" \
  --group-name "$GROUP_NAME" \
  --output-tag "$OUTPUT_TAG" \
  "$@"

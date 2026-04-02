#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG_PATH="${DART_CONFIG_PATH:-$ROOT_DIR/dart/configs/dart_cti_llama8b_v3.yaml}"
GROUP_NAME="${DART_GROUP_NAME:-train}"
TARGET_ACCEPTED="${DART_TARGET_ACCEPTED:-2}"
V3_CAP="${DART_V3_CAP:-100}"

SEED_V2_PATH="${DART_ACCEPTED_V2_CAP150_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/accepted_llama8b_v2_cap150_seed.jsonl}"
ATTEMPTS_V3_PATH="${DART_ATTEMPTS_V3_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/attempts_v2_from_llama8b_v3_from_llama8b_v2_cap150_seed.jsonl}"

RECOVER_TAG="${DART_RECOVER_TAG:-llama8b_v3_cap${V3_CAP}_seed}"
RECOVERED_SEED_PATH="${DART_RECOVERED_SEED_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/accepted_${RECOVER_TAG}.jsonl}"
RECOVERED_SUMMARY_PATH="${DART_RECOVERED_SUMMARY_PATH:-$ROOT_DIR/dart-artifacts/dart_cti_llama8b/datasets/$GROUP_NAME/summary_${RECOVER_TAG}.json}"

OUTPUT_TAG="${DART_OUTPUT_TAG:-llama8b_direct_from_${RECOVER_TAG}}"

python -m dart.recover_fill_seed \
  --seed-jsonl "$SEED_V2_PATH" \
  --attempts-jsonl "$ATTEMPTS_V3_PATH" \
  --output-jsonl "$RECOVERED_SEED_PATH" \
  --summary-output "$RECOVERED_SUMMARY_PATH" \
  --target-accepted "$TARGET_ACCEPTED" \
  --max-attempts-per-question "$V3_CAP" \
  --accept-mode verifier

exec python -m dart.fill_direct_answers \
  --config "$CONFIG_PATH" \
  --accepted-seed "$RECOVERED_SEED_PATH" \
  --group-name "$GROUP_NAME" \
  --output-tag "$OUTPUT_TAG" \
  "$@"

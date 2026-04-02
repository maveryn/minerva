#!/usr/bin/env bash
set -euo pipefail

SEED_FILE="${1:-dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v1.jsonl}"
ATTEMPTS_FILE="${2:-dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/attempts_v2_from_accepted_v1.jsonl}"
TOTAL_QUESTIONS="${3:-1200}"
TARGET_ACCEPTED="${4:-1}"

./dart/count_v2_progress.sh "$SEED_FILE" "$ATTEMPTS_FILE" "$TOTAL_QUESTIONS" "$TARGET_ACCEPTED"

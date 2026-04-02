#!/usr/bin/env bash
set -euo pipefail

MINERVA_ATTEMPTS="${1:-dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/attempts_v1.jsonl}"
ATHENA_ATTEMPTS="${2:-dart-artifacts/dart_cti_llama8b/datasets/athena_bench/attempts_v1.jsonl}"

echo "minerva_dev"
./dart/count_attempts_with_max.sh "$MINERVA_ATTEMPTS"
printf "\n"

echo "athena_bench"
./dart/count_attempts_with_max.sh "$ATHENA_ATTEMPTS"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUNS=(
  "$SCRIPT_DIR/train_modernbert_2k_lr1e-05.sh"
  "$SCRIPT_DIR/train_modernbert_2k_lr5e-05.sh"
  "$SCRIPT_DIR/train_modernbert_4k_lr1e-05.sh"
  "$SCRIPT_DIR/train_modernbert_4k_lr2e-05.sh"
  "$SCRIPT_DIR/train_modernbert_4k_lr5e-05.sh"
)

for run in "${RUNS[@]}"; do
  echo "Running: $run"
  bash "$run"
done

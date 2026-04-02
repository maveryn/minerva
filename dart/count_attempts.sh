#!/usr/bin/env bash
set -euo pipefail

ATTEMPTS_PATH="${1:-dart-artifacts/dart_cti/datasets/train/attempts_v1.jsonl}"

python - "$ATTEMPTS_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
total = 0
accepted = 0

if path.exists():
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("verifier_success"):
                accepted += 1

print(f"attempts_file {path}")
print(f"total_attempts {total}")
print(f"accepted_attempts {accepted}")
PY

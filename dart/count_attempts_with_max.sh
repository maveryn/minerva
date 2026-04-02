#!/usr/bin/env bash
set -euo pipefail

ATTEMPTS_PATH="${1:-dart-artifacts/dart_cti_llama8b/datasets/train/attempts_v1.jsonl}"

python - "$ATTEMPTS_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
total = 0
accepted = 0
seen = {}

if path.exists():
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if row.get("verifier_success"):
                accepted += 1
            uid = str(row.get("uid") or "")
            idx = int(row.get("attempt_index", -1))
            if uid and idx > seen.get(uid, -1):
                seen[uid] = idx

max_attempt_index = max(seen.values()) if seen else -1
max_attempt_count = max_attempt_index + 1 if max_attempt_index >= 0 else 0
num_uids_at_max = sum(1 for v in seen.values() if v == max_attempt_index) if seen else 0

print(f"attempts_file {path}")
print(f"total_attempts {total}")
print(f"accepted_attempts {accepted}")
print(f"max_attempt_index {max_attempt_index}")
print(f"max_attempt_count {max_attempt_count}")
print(f"num_uids_at_max {num_uids_at_max}")
PY

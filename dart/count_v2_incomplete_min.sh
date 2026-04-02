#!/usr/bin/env bash
set -euo pipefail

SEED_PATH="${1:-dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v1.jsonl}"
ATTEMPTS_PATH="${2:-dart-artifacts/dart_cti_llama8b/datasets/train/attempts_v2_from_llama8b_v1.jsonl}"
TARGET_ACCEPTED="${3:-2}"

python - "$SEED_PATH" "$ATTEMPTS_PATH" "$TARGET_ACCEPTED" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

seed_path = Path(sys.argv[1])
attempts_path = Path(sys.argv[2])
target = int(sys.argv[3])

seed_counts = Counter()
if seed_path.exists():
    with seed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("uid") or "")
            if uid:
                seed_counts[uid] += 1

attempt_max = Counter()
accepted_added = Counter()
if attempts_path.exists():
    with attempts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(row.get("uid") or "")
            if not uid:
                continue
            idx = int(row.get("attempt_index", 0) or 0)
            if idx > attempt_max[uid]:
                attempt_max[uid] = idx
            if row.get("verifier_success") and row.get("acr_filter_pass"):
                if seed_counts.get(uid, 0) + accepted_added.get(uid, 0) < target:
                    accepted_added[uid] += 1

incomplete = []
for uid, mx in attempt_max.items():
    total = seed_counts.get(uid, 0) + accepted_added.get(uid, 0)
    if total < target:
        incomplete.append(mx)

print(f"seed_file {seed_path}")
print(f"attempts_file {attempts_path}")
print(f"target_accepted_per_question {target}")
print(f"incomplete_questions {len(incomplete)}")
print(f"incomplete_min_attempt_index {min(incomplete) if incomplete else 0}")
print(f"incomplete_max_attempt_index {max(incomplete) if incomplete else 0}")
if incomplete:
    hist = Counter(incomplete)
    max_idx = max(incomplete)
    for k in range(max(0, max_idx - 3), max_idx + 1):
        print(f"incomplete_uids_at_{k} {hist.get(k, 0)}")
PY

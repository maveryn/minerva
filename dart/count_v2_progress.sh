#!/usr/bin/env bash
set -euo pipefail

SEED_PATH="${1:-dart-artifacts/dart_cti/datasets/train/accepted_v1_30k.jsonl}"
ATTEMPTS_PATH="${2:-dart-artifacts/dart_cti/datasets/train/attempts_v2_from_v1_30k.jsonl}"
TOTAL_QUESTIONS="${3:-32000}"
TARGET_ACCEPTED="${4:-2}"

python - "$SEED_PATH" "$ATTEMPTS_PATH" "$TOTAL_QUESTIONS" "$TARGET_ACCEPTED" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

seed_path = Path(sys.argv[1])
attempts_path = Path(sys.argv[2])
total_questions = int(sys.argv[3])
target_accepted = int(sys.argv[4])

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

attempt_counts = Counter()
accepted_added = Counter()
v2_total_attempts = 0
v2_verifier_correct = 0
v2_filter_pass = 0
max_attempt_index = -1

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
            v2_total_attempts += 1
            attempt_counts[uid] += 1
            attempt_index = int(row.get("attempt_index", -1))
            if attempt_index > max_attempt_index:
                max_attempt_index = attempt_index
            verifier_success = bool(row.get("verifier_success", False))
            filter_pass = bool(row.get("acr_filter_pass", False))
            if verifier_success:
                v2_verifier_correct += 1
            if filter_pass:
                v2_filter_pass += 1
            if verifier_success and filter_pass and seed_counts.get(uid, 0) + accepted_added.get(uid, 0) < target_accepted:
                accepted_added[uid] += 1

all_uids = set(seed_counts) | set(attempt_counts)
completed_questions = 0
one_trace_questions = 0
for uid in all_uids:
    total_for_uid = seed_counts.get(uid, 0) + accepted_added.get(uid, 0)
    if total_for_uid >= target_accepted:
        completed_questions += 1
    elif total_for_uid == 1:
        one_trace_questions += 1

seed_complete = sum(1 for v in seed_counts.values() if v >= target_accepted)
seed_one = sum(1 for v in seed_counts.values() if v == 1)
seed_zero = max(0, total_questions - seed_complete - seed_one)
remaining_questions = max(0, total_questions - completed_questions)
total_traces_now = sum(min(seed_counts.get(uid, 0) + accepted_added.get(uid, 0), target_accepted) for uid in all_uids)
if total_questions > len(all_uids):
    total_traces_now += 0

print(f"seed_file {seed_path}")
print(f"attempts_file {attempts_path}")
print(f"total_questions {total_questions}")
print(f"target_accepted_per_question {target_accepted}")
print(f"seed_plain_traces {sum(seed_counts.values())}")
print(f"seed_complete_questions {seed_complete}")
print(f"seed_one_trace_questions {seed_one}")
print(f"seed_zero_trace_questions {seed_zero}")
print(f"v2_total_attempts {v2_total_attempts}")
print(f"v2_verifier_correct {v2_verifier_correct}")
print(f"v2_filter_pass {v2_filter_pass}")
print(f"v2_added_traces {sum(accepted_added.values())}")
print(f"current_total_traces {sum(seed_counts.values()) + sum(accepted_added.values())}")
print(f"completed_questions {completed_questions}")
print(f"one_trace_questions {one_trace_questions}")
print(f"remaining_questions {remaining_questions}")
print(f"max_fill_attempt_index {max_attempt_index}")
print(f"max_fill_attempt_count {max_attempt_index + 1 if max_attempt_index >= 0 else 0}")
PY

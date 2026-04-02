#!/usr/bin/env bash
set -euo pipefail

SEED_PATH="${1:-dart-artifacts/dart_cti_llama8b/datasets/train/accepted_llama8b_v2_cap150_seed.jsonl}"
ATTEMPTS_PATH="${2:-dart-artifacts/dart_cti_llama8b/datasets/train/attempts_v2_from_llama8b_v3_from_llama8b_v2_cap150_seed.jsonl}"
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
v3_total_attempts = 0
v3_verifier_correct = 0
max_attempt_index = 0

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
            v3_total_attempts += 1
            attempt_counts[uid] += 1
            attempt_index = int(row.get("attempt_index", 0) or 0)
            if attempt_index > max_attempt_index:
                max_attempt_index = attempt_index
            verifier_success = bool(row.get("verifier_success", False))
            if verifier_success:
                v3_verifier_correct += 1
                if seed_counts.get(uid, 0) + accepted_added.get(uid, 0) < target_accepted:
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
current_total_traces = sum(seed_counts.values()) + sum(accepted_added.values())
remaining_questions = max(0, total_questions - completed_questions)
remaining_traces = max(0, total_questions * target_accepted - current_total_traces)

print(f"seed_file {seed_path}")
print(f"attempts_file {attempts_path}")
print(f"total_questions {total_questions}")
print(f"target_accepted_per_question {target_accepted}")
print(f"seed_total_traces {sum(seed_counts.values())}")
print(f"seed_complete_questions {seed_complete}")
print(f"seed_one_trace_questions {seed_one}")
print(f"seed_zero_trace_questions {seed_zero}")
print(f"v3_total_attempts {v3_total_attempts}")
print(f"v3_verifier_correct {v3_verifier_correct}")
print(f"v3_added_traces {sum(accepted_added.values())}")
print(f"current_total_traces {current_total_traces}")
print(f"completed_questions {completed_questions}")
print(f"one_trace_questions {one_trace_questions}")
print(f"remaining_questions {remaining_questions}")
print(f"remaining_traces {remaining_traces}")
print(f"max_fill_attempt_index {max_attempt_index}")
print(f"max_fill_attempt_count {max_attempt_index}")
PY

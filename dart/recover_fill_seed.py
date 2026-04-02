from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from dart.common import ensure_parent, to_jsonable, write_jsonl


def _iter_jsonl_tolerant(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _accept_row(row: Dict[str, Any], *, accept_mode: str) -> bool:
    if not bool(row.get("verifier_success", False)):
        return False
    if accept_mode == "verifier":
        return True
    return bool(row.get("acr_filter_pass", True))


def recover_fill_seed(
    *,
    seed_jsonl: str | Path,
    attempts_jsonl: str | Path,
    output_jsonl: str | Path,
    target_accepted: int = 2,
    max_attempts_per_question: int = 0,
    accept_mode: str = "verifier+filter",
) -> Dict[str, Any]:
    accept_mode = str(accept_mode).strip().lower()
    if accept_mode not in {"verifier", "verifier+filter"}:
        raise ValueError(f"Unsupported accept_mode: {accept_mode}")

    accepted_rows: List[Dict[str, Any]] = []
    accepted_counts = Counter()
    seed_total = 0
    seed_uids = set()
    for row in _iter_jsonl_tolerant(seed_jsonl):
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        accepted_rows.append(dict(row))
        accepted_counts[uid] += 1
        seed_total += 1
        seed_uids.add(uid)

    attempt_rows_seen = 0
    attempt_rows_ignored_due_to_cap = 0
    attempt_rows_parseable = 0
    attempt_uids_seen = set()
    max_attempt_index_seen = 0
    accepted_from_attempts = 0

    for row in _iter_jsonl_tolerant(attempts_jsonl):
        attempt_rows_parseable += 1
        uid = str(row.get("uid") or "")
        if not uid:
            continue
        attempt_uids_seen.add(uid)
        attempt_index = int(row.get("attempt_index", 0) or 0)
        if attempt_index > max_attempt_index_seen:
            max_attempt_index_seen = attempt_index
        attempt_rows_seen += 1

        if max_attempts_per_question > 0 and attempt_index > max_attempts_per_question:
            attempt_rows_ignored_due_to_cap += 1
            continue
        if accepted_counts.get(uid, 0) >= target_accepted:
            continue
        if not _accept_row(row, accept_mode=accept_mode):
            continue

        accepted = dict(row)
        accepted["accepted_rank"] = int(accepted_counts.get(uid, 0)) + 1
        accepted["trace_source"] = str(accepted.get("trace_source") or "guided_fill")
        accepted_rows.append(accepted)
        accepted_counts[uid] += 1
        accepted_from_attempts += 1

    write_jsonl(output_jsonl, accepted_rows)

    hist = Counter(accepted_counts.values())
    summary = {
        "seed_jsonl": str(seed_jsonl),
        "attempts_jsonl": str(attempts_jsonl),
        "output_jsonl": str(output_jsonl),
        "accept_mode": accept_mode,
        "target_accepted": int(target_accepted),
        "max_attempts_per_question": int(max_attempts_per_question),
        "seed_total_traces": int(seed_total),
        "seed_unique_questions": int(len(seed_uids)),
        "accepted_from_attempts": int(accepted_from_attempts),
        "recovered_total_traces": int(len(accepted_rows)),
        "recovered_unique_questions": int(len(accepted_counts)),
        "completed_questions": int(sum(1 for count in accepted_counts.values() if count >= target_accepted)),
        "one_trace_questions": int(hist.get(1, 0)),
        "zero_trace_questions_unobserved": None,
        "accepted_per_question": to_jsonable(dict(sorted(hist.items()))),
        "attempt_rows_seen": int(attempt_rows_seen),
        "attempt_rows_parseable": int(attempt_rows_parseable),
        "attempt_rows_ignored_due_to_cap": int(attempt_rows_ignored_due_to_cap),
        "attempt_uids_seen": int(len(attempt_uids_seen)),
        "max_attempt_index_seen": int(max_attempt_index_seen),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a capped accepted seed from a partial fill attempts log")
    parser.add_argument("--seed-jsonl", required=True)
    parser.add_argument("--attempts-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-output")
    parser.add_argument("--target-accepted", type=int, default=2)
    parser.add_argument("--max-attempts-per-question", type=int, default=0)
    parser.add_argument("--accept-mode", choices=["verifier", "verifier+filter"], default="verifier+filter")
    args = parser.parse_args()

    summary = recover_fill_seed(
        seed_jsonl=args.seed_jsonl,
        attempts_jsonl=args.attempts_jsonl,
        output_jsonl=args.output_jsonl,
        target_accepted=args.target_accepted,
        max_attempts_per_question=args.max_attempts_per_question,
        accept_mode=args.accept_mode,
    )
    if args.summary_output:
        path = ensure_parent(args.summary_output)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

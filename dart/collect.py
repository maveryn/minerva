from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dart.common import (
    append_jsonl,
    iter_jsonl,
    load_parquet_rows,
    make_uid,
    summarize_counts,
    to_jsonable,
    write_jsonl,
)
from dart.generate import DartGenerator
from dart.score import score_rows


def initialize_state(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    state: Dict[str, Dict[str, Any]] = {}
    row_map: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        uid = make_uid(row, fallback_index=idx)
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        data_source = str(row.get("data_source") or row.get("reward_fn") or "")
        state[uid] = {
            "uid": uid,
            "data_source": data_source,
            "source_file": row.get("source_file"),
            "source_index": extra_info.get("index", idx),
            "accepted_count": 0,
            "attempt_count": 0,
            "done_reason": "",
        }
        row_map[uid] = dict(row)
        row_map[uid]["uid"] = uid
    return state, row_map


def _state_rows(state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [to_jsonable(row) for row in state.values()]


def _summarize_uniform_state(
    state: Dict[str, Dict[str, Any]],
    *,
    target_accepted: int,
    accepted_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    accepted_counts = [int(item.get("accepted_count", 0)) for item in state.values()]
    attempt_counts = [int(item.get("attempt_count", 0)) for item in state.values()]
    hist = Counter(accepted_counts)
    return {
        "total_questions": len(state),
        "target_accepted": int(target_accepted),
        "total_generated_samples": int(sum(attempt_counts)),
        "accepted_traces_total": len(accepted_rows),
        "accepted_trace_source_breakdown": {
            "plain": sum(1 for row in accepted_rows if row.get("trace_source") == "plain"),
        },
        "coverage": {
            "zero": int(hist.get(0, 0)),
            "one": int(hist.get(1, 0)),
            "target": int(sum(v for k, v in hist.items() if k >= target_accepted)),
        },
        "accepted_per_question": summarize_counts(accepted_counts),
        "attempts_per_question": summarize_counts(attempt_counts),
    }


def collect_uniform(
    rows: List[Dict[str, Any]],
    generator: DartGenerator,
    *,
    target_accepted: int = 2,
    max_attempts: int = 100,
    success_threshold: float = 1.0,
    attempts_output_path: str | Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    state, row_map = initialize_state(rows)
    active = deque(state.keys())
    accepted_rows: List[Dict[str, Any]] = []

    if attempts_output_path:
        Path(attempts_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(attempts_output_path).write_text("", encoding="utf-8")

    while active:
        batch_uids: List[str] = []
        for _ in range(min(generator.batch_size, len(active))):
            batch_uids.append(active.popleft())
        batch_rows = []
        for uid in batch_uids:
            item = dict(row_map[uid])
            item["attempt_index"] = state[uid]["attempt_count"] + 1
            batch_rows.append(item)

        generated_rows, skipped_rows = generator.generate_rows(batch_rows, mode="plain")
        if skipped_rows:
            raise RuntimeError(f"Plain generation unexpectedly skipped rows: {skipped_rows[:3]}")
        scored_rows = score_rows(generated_rows, success_threshold=success_threshold)
        if attempts_output_path:
            append_jsonl(attempts_output_path, scored_rows)

        scored_by_uid = {str(row.get("uid")): row for row in scored_rows}
        for uid in batch_uids:
            state_row = state[uid]
            scored = scored_by_uid.get(uid)
            if scored is None:
                raise RuntimeError(f"Missing scored row for uid={uid}")
            state_row["attempt_count"] += 1
            if (
                bool(scored.get("verifier_success"))
                and state_row["accepted_count"] < target_accepted
            ):
                state_row["accepted_count"] += 1
                accepted = dict(scored)
                accepted["accepted_rank"] = state_row["accepted_count"]
                accepted["trace_source"] = "plain"
                accepted_rows.append(accepted)

            if state_row["accepted_count"] >= target_accepted:
                state_row["done_reason"] = "accepted_target_reached"
                continue
            if state_row["attempt_count"] >= max_attempts:
                state_row["done_reason"] = "max_attempts_reached"
                continue
            active.append(uid)

    summary = _summarize_uniform_state(state, target_accepted=target_accepted, accepted_rows=accepted_rows)
    return accepted_rows, state, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DART Uniform traces")
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--attempts-output")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-length", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-accepted", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_parquet_rows(args.parquet, limit=args.limit)
    generator = DartGenerator(
        model_path=args.model_path,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    accepted_rows, state, summary = collect_uniform(
        rows,
        generator,
        target_accepted=args.target_accepted,
        max_attempts=args.max_attempts,
        success_threshold=args.success_threshold,
        attempts_output_path=args.attempts_output,
    )
    write_jsonl(args.output, accepted_rows)
    write_jsonl(args.state_output, _state_rows(state))
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

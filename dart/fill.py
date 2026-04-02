from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dart.acr_filter import AcrDistillFilter
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


def initialize_fill_state(
    rows: List[Dict[str, Any]],
    accepted_rows_v1: List[Dict[str, Any]],
    *,
    target_accepted: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
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
            "plain_accepted_count": 0,
            "final_accepted_count": 0,
            "fill_attempt_count": 0,
            "fill_done_reason": "",
        }
        row_map[uid] = dict(row)
        row_map[uid]["uid"] = uid

    accepted_rows_out: List[Dict[str, Any]] = []
    for row in accepted_rows_v1:
        uid = str(row.get("uid") or "")
        if uid not in state:
            continue
        state[uid]["plain_accepted_count"] += 1
        state[uid]["final_accepted_count"] += 1
        accepted_rows_out.append(dict(row))

    for state_row in state.values():
        if state_row["final_accepted_count"] >= target_accepted:
            state_row["fill_done_reason"] = "already_complete_from_plain"

    return state, row_map, accepted_rows_out


def _state_rows(state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [to_jsonable(row) for row in state.values()]


def _summarize_fill_state(
    state: Dict[str, Dict[str, Any]],
    *,
    target_accepted: int,
    accepted_rows: List[Dict[str, Any]],
    filter_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    final_counts = [int(item.get("final_accepted_count", 0)) for item in state.values()]
    plain_counts = [int(item.get("plain_accepted_count", 0)) for item in state.values()]
    fill_attempts = [int(item.get("fill_attempt_count", 0)) for item in state.values()]
    hist = Counter(final_counts)
    plain_hist = Counter(plain_counts)
    summary = {
        "total_questions": len(state),
        "target_accepted": int(target_accepted),
        "accepted_traces_total": len(accepted_rows),
        "accepted_trace_source_breakdown": {
            "plain": sum(1 for row in accepted_rows if row.get("trace_source") == "plain"),
            "guided_fill": sum(1 for row in accepted_rows if row.get("trace_source") == "guided_fill"),
        },
        "seed_coverage": {
            "zero": int(plain_hist.get(0, 0)),
            "one": int(plain_hist.get(1, 0)),
            "target": int(sum(v for k, v in plain_hist.items() if k >= target_accepted)),
        },
        "coverage_after_fill": {
            "zero": int(hist.get(0, 0)),
            "one": int(hist.get(1, 0)),
            "target": int(sum(v for k, v in hist.items() if k >= target_accepted)),
        },
        "fill_attempts_per_question": summarize_counts(fill_attempts),
        "final_accepted_per_question": summarize_counts(final_counts),
        "guided_fill_added": sum(1 for row in accepted_rows if row.get("trace_source") == "guided_fill"),
    }
    if isinstance(filter_metrics, dict):
        summary["acr_filter"] = filter_metrics
    return summary


def fill_missing_traces(
    rows: List[Dict[str, Any]],
    accepted_rows_v1: List[Dict[str, Any]],
    generator: DartGenerator,
    *,
    generation_mode: str = "guided_fill",
    target_accepted: int = 2,
    max_attempts_per_question: int = 0,
    success_threshold: float = 1.0,
    attempts_output_path: str | Path | None = None,
    acr_filter: Optional[AcrDistillFilter] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    state, row_map, accepted_rows = initialize_fill_state(
        rows,
        accepted_rows_v1,
        target_accepted=target_accepted,
    )
    active = deque(
        uid
        for uid, state_row in state.items()
        if int(state_row.get("final_accepted_count", 0)) < target_accepted
    )

    if attempts_output_path:
        Path(attempts_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(attempts_output_path).write_text("", encoding="utf-8")

    aggregate_filter_metrics: Dict[str, Any] = {}

    while active:
        batch_uids: List[str] = []
        for _ in range(min(generator.batch_size, len(active))):
            batch_uids.append(active.popleft())

        batch_rows = []
        for uid in batch_uids:
            item = dict(row_map[uid])
            item["attempt_index"] = state[uid]["fill_attempt_count"] + 1
            batch_rows.append(item)

        generated_rows, skipped_rows = generator.generate_rows(batch_rows, mode=generation_mode)
        scored_rows = score_rows(generated_rows, success_threshold=success_threshold)
        if acr_filter is not None and scored_rows:
            scored_rows, filter_metrics = acr_filter.apply(scored_rows)
            for key, value in filter_metrics.items():
                if key in {"filter_mode"}:
                    aggregate_filter_metrics[key] = value
                elif key in {"filter_threshold", "reward_threshold"}:
                    if key not in aggregate_filter_metrics:
                        aggregate_filter_metrics[key] = float(value)
                elif isinstance(value, (int, float)):
                    aggregate_filter_metrics[key] = float(aggregate_filter_metrics.get(key, 0.0)) + float(value)
                else:
                    aggregate_filter_metrics[key] = value
        if attempts_output_path and scored_rows:
            append_jsonl(attempts_output_path, scored_rows)

        scored_by_uid = {str(row.get("uid")): row for row in scored_rows}
        skipped_by_uid = {str(row.get("uid")): row for row in skipped_rows}

        for uid in batch_uids:
            state_row = state[uid]
            skipped = skipped_by_uid.get(uid)
            if skipped is not None:
                state_row["fill_done_reason"] = skipped.get("generation_skip_reason", "guided_prompt_failed")
                continue

            scored = scored_by_uid.get(uid)
            if scored is None:
                raise RuntimeError(f"Missing scored guided-fill row for uid={uid}")

            state_row["fill_attempt_count"] += 1
            if (
                bool(scored.get("verifier_success"))
                and bool(scored.get("acr_filter_pass", True))
                and state_row["final_accepted_count"] < target_accepted
            ):
                state_row["final_accepted_count"] += 1
                accepted = dict(scored)
                accepted["accepted_rank"] = state_row["final_accepted_count"]
                accepted["trace_source"] = str(generation_mode)
                accepted_rows.append(accepted)

            if state_row["final_accepted_count"] >= target_accepted:
                state_row["fill_done_reason"] = "guided_fill_completed"
                continue
            if max_attempts_per_question > 0 and state_row["fill_attempt_count"] >= max_attempts_per_question:
                state_row["fill_done_reason"] = "guided_fill_max_attempts_reached"
                continue
            active.append(uid)

    summary = _summarize_fill_state(
        state,
        target_accepted=target_accepted,
        accepted_rows=accepted_rows,
        filter_metrics=aggregate_filter_metrics if acr_filter is not None else None,
    )
    return accepted_rows, state, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing DART traces with guided generation")
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--accepted-v1", required=True)
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
    parser.add_argument("--guided-label-details-dir")
    parser.add_argument("--guided-max-details-chars", type=int, default=8096)
    parser.add_argument("--guided-max-prompt-length", type=int, default=4096)
    parser.add_argument("--guided-enforce-no-id", action="store_true")
    parser.add_argument("--target-accepted", type=int, default=2)
    parser.add_argument("--max-attempts-per-question", type=int, default=0)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--generation-mode", choices=["guided_fill", "direct_answer"], default="guided_fill")
    parser.add_argument("--acr-filter-mode", default="off")
    parser.add_argument("--acr-disable-filters", action="store_true")
    parser.add_argument("--acr-reward-threshold", type=float, default=1.0)
    parser.add_argument("--acr-degenerate-filter", action="store_true")
    parser.add_argument("--acr-degenerate-min-tokens", type=int, default=30)
    parser.add_argument("--acr-degenerate-rep-3-max", type=float, default=0.70)
    parser.add_argument("--acr-degenerate-rep-4-max", type=float, default=0.75)
    parser.add_argument("--acr-filter-model", default="xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25")
    parser.add_argument("--acr-filter-model-type", default="textcnn")
    parser.add_argument("--acr-filter-threshold", type=float, default=0.5)
    parser.add_argument("--acr-filter-batch-size", type=int, default=128)
    parser.add_argument("--acr-filter-max-length", type=int, default=1024)
    parser.add_argument("--acr-filter-text-mode", default="response")
    parser.add_argument("--acr-filter-device", default="auto")
    parser.add_argument("--acr-filter-dtype", default="auto")
    parser.add_argument("--acr-r-correct", type=float, default=0.1)
    parser.add_argument("--acr-leak-penalty", type=float, default=0.5)
    parser.add_argument("--acr-multilabel-match", default="exact")
    parser.add_argument("--acr-max-id-mentions", type=int, default=0)
    parser.add_argument("--acr-min-reasoning-chars", type=int, default=100)
    parser.add_argument("--acr-min-overlap-jaccard", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_parquet_rows(args.parquet, limit=args.limit)
    accepted_rows_v1 = list(iter_jsonl(args.accepted_v1))
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
        guided_label_details_dir=args.guided_label_details_dir,
        guided_max_details_chars=args.guided_max_details_chars,
        guided_max_prompt_length=args.guided_max_prompt_length,
        guided_enforce_no_id=args.guided_enforce_no_id,
    )
    acr_filter = None
    if str(args.acr_filter_mode).lower().strip() != "off":
        acr_filter = AcrDistillFilter(
            generator.tokenizer,
            filter_mode=args.acr_filter_mode,
            disable_filters=args.acr_disable_filters,
            reward_threshold=args.acr_reward_threshold,
            degenerate_filter=args.acr_degenerate_filter,
            degenerate_min_tokens=args.acr_degenerate_min_tokens,
            degenerate_rep_3_max=args.acr_degenerate_rep_3_max,
            degenerate_rep_4_max=args.acr_degenerate_rep_4_max,
            filter_model=args.acr_filter_model,
            filter_model_type=args.acr_filter_model_type,
            filter_threshold=args.acr_filter_threshold,
            filter_batch_size=args.acr_filter_batch_size,
            filter_max_length=args.acr_filter_max_length,
            filter_text_mode=args.acr_filter_text_mode,
            filter_device=args.acr_filter_device,
            filter_dtype=args.acr_filter_dtype,
            r_correct=args.acr_r_correct,
            leak_penalty=args.acr_leak_penalty,
            multilabel_match=args.acr_multilabel_match,
            enforce_no_id_in_reasoning=False,
            max_id_mentions=args.acr_max_id_mentions,
            min_reasoning_chars=args.acr_min_reasoning_chars,
            min_overlap_jaccard=args.acr_min_overlap_jaccard,
        )
    accepted_rows, state, summary = fill_missing_traces(
        rows,
        accepted_rows_v1,
        generator,
        generation_mode=args.generation_mode,
        target_accepted=args.target_accepted,
        max_attempts_per_question=args.max_attempts_per_question,
        success_threshold=args.success_threshold,
        attempts_output_path=args.attempts_output,
        acr_filter=acr_filter,
    )
    write_jsonl(args.output, accepted_rows)
    write_jsonl(args.state_output, _state_rows(state))
    Path(args.summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dart.build_sft_dataset import build_sft_dataset
from dart.common import (
    ensure_dir,
    finish_wall_clock,
    first_present_field,
    get_user_prompt,
    iter_jsonl,
    load_config,
    load_parquet_rows,
    make_uid,
    normalize_messages,
    start_wall_clock,
    to_jsonable,
    write_jsonl,
)
from dart.fill import initialize_fill_state
from dart.score import score_rows


def _boxed_answer(ground_truth: Any) -> str:
    return f"\\boxed{{{str(ground_truth)}}}"


def fill_direct_answers(
    rows: List[Dict[str, Any]],
    accepted_seed_rows: List[Dict[str, Any]],
    *,
    target_accepted: int = 2,
    success_threshold: float = 1.0,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    state, _, accepted_rows = initialize_fill_state(
        rows,
        accepted_seed_rows,
        target_accepted=target_accepted,
    )

    direct_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        uid = make_uid(row, fallback_index=idx)
        state_row = state[uid]
        base_messages = normalize_messages(
            first_present_field(row, "prompt", "base_messages", "prompt_nohint_messages", default=[])
        )
        if state_row["final_accepted_count"] >= target_accepted or not base_messages:
            continue

        reward_model = row.get("reward_model") or {}
        ground_truth = row.get("ground_truth", reward_model.get("ground_truth"))
        if ground_truth is None:
            continue

        while state_row["final_accepted_count"] + len(
            [r for r in direct_rows if str(r.get("uid")) == uid]
        ) < target_accepted:
            direct_rows.append(
                {
                    "uid": uid,
                    "mode": "direct_answer_gold",
                    "data_source": str(row.get("data_source") or row.get("reward_fn") or ""),
                    "reward_fn": str((row.get("extra_info") or {}).get("reward_fn") or row.get("data_source") or ""),
                    "task": str((row.get("extra_info") or {}).get("task") or ""),
                    "source_file": row.get("source_file"),
                    "source_index": ((row.get("extra_info") or {}).get("index", idx)),
                    "prompt_nohint": get_user_prompt(base_messages),
                    "base_messages": base_messages,
                    "prompt_used_messages": base_messages,
                    "ground_truth": ground_truth,
                    "extra_info": dict(row.get("extra_info") or {}),
                    "attempt_index": 0,
                    "prompt_meta": {"deterministic_direct_answer": True},
                    "response_text_raw": _boxed_answer(ground_truth),
                    "response_analysis": "",
                    "response_final": _boxed_answer(ground_truth),
                    "response_text": _boxed_answer(ground_truth),
                }
            )

    scored_direct_rows = score_rows(direct_rows, success_threshold=success_threshold)
    by_uid: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored_direct_rows:
        by_uid.setdefault(str(row.get("uid")), []).append(row)

    verification_failures = 0
    direct_added = 0
    for idx, row in enumerate(rows):
        uid = make_uid(row, fallback_index=idx)
        state_row = state[uid]
        for scored in by_uid.get(uid, []):
            if state_row["final_accepted_count"] >= target_accepted:
                break
            if not bool(scored.get("verifier_success")):
                verification_failures += 1
                continue
            state_row["final_accepted_count"] += 1
            accepted = dict(scored)
            accepted["accepted_rank"] = state_row["final_accepted_count"]
            accepted["trace_source"] = "direct_answer_gold"
            accepted_rows.append(accepted)
            direct_added += 1
        if state_row["final_accepted_count"] >= target_accepted:
            state_row["fill_done_reason"] = "direct_answer_completed"
        else:
            state_row["fill_done_reason"] = "direct_answer_incomplete"

    zero = sum(1 for s in state.values() if int(s.get("final_accepted_count", 0)) == 0)
    one = sum(1 for s in state.values() if int(s.get("final_accepted_count", 0)) == 1)
    target = sum(1 for s in state.values() if int(s.get("final_accepted_count", 0)) >= target_accepted)
    summary = {
        "total_questions": len(state),
        "target_accepted": int(target_accepted),
        "accepted_traces_total": len(accepted_rows),
        "accepted_trace_source_breakdown": {
            "plain": sum(1 for row in accepted_rows if row.get("trace_source") == "plain"),
            "guided_fill": sum(1 for row in accepted_rows if row.get("trace_source") == "guided_fill"),
            "direct_answer_gold": sum(1 for row in accepted_rows if row.get("trace_source") == "direct_answer_gold"),
        },
        "coverage_after_direct_fill": {
            "zero": zero,
            "one": one,
            "target": target,
        },
        "direct_answer_added": direct_added,
        "verification_failures": verification_failures,
    }
    return accepted_rows, state, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill remaining DART traces with deterministic boxed answers")
    parser.add_argument("--config", default="dart/configs/dart_cti_llama8b_v3.yaml")
    parser.add_argument("--accepted-seed", required=True)
    parser.add_argument("--group-name", default="train")
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--limit-train-rows", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg.get("output_root", "dart-artifacts/dart_cti"))
    dataset_cfg = cfg.get("dataset", {})
    teacher_cfg = cfg.get("teacher", {})

    train_paths = dataset_cfg.get("train_paths", cfg.get("train_path"))
    rows = load_parquet_rows(train_paths, limit=args.limit_train_rows)
    accepted_seed_rows = list(iter_jsonl(args.accepted_seed))

    group_root = ensure_dir(Path(output_root) / "datasets" / args.group_name)
    output_tag = str(args.output_tag)
    accepted_path = group_root / f"accepted_v2_from_{output_tag}.jsonl"
    state_path = group_root / f"state_v2_from_{output_tag}.jsonl"
    summary_path = group_root / f"summary_v2_from_{output_tag}.json"
    parquet_path = group_root / f"{args.group_name}_v2_from_{output_tag}.parquet"

    v2_started_at, v2_started_perf = start_wall_clock()
    accepted_rows, state, summary = fill_direct_answers(
        rows,
        accepted_seed_rows,
        target_accepted=int(dataset_cfg.get("target_accepted", 2)),
        success_threshold=1.0,
    )
    summary["timing"] = finish_wall_clock(v2_started_at, v2_started_perf)

    write_jsonl(accepted_path, accepted_rows)
    write_jsonl(state_path, [to_jsonable(v) for v in state.values()])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_sft_dataset(
        accepted_jsonl=accepted_path,
        output_parquet=parquet_path,
        model_path=str(teacher_cfg.get("model_path", "meta-llama/Llama-3.1-8B-Instruct")),
        response_max_tokens=int(dataset_cfg.get("response_max_tokens", 0)) or None,
        trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
    )

    print(
        json.dumps(
            {
                "accepted_seed": args.accepted_seed,
                "accepted_output": str(accepted_path),
                "state_output": str(state_path),
                "summary_output": str(summary_path),
                "parquet_output": str(parquet_path),
                "timing": summary.get("timing"),
                "metrics": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

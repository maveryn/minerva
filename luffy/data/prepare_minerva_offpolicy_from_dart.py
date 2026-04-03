import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_BASE_TRAIN = (
    "/home/jovyan/work/minerva/rlvr/mydata/minerva_base/minerva_base_train.parquet"
)
DEFAULT_STAGE_1 = (
    "/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v1.jsonl"
)
DEFAULT_STAGE_2 = (
    "/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_llama8b_v2_cap32_seed.jsonl"
)
DEFAULT_STAGE_3 = (
    "/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_llama8b_v3_cap200_seed.jsonl"
)
DEFAULT_STAGE_4 = (
    "/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v2_from_llama8b_direct_from_v3_cap200_seed.jsonl"
)
DEFAULT_OUTPUT = "/home/jovyan/work/LUFFY/data/minerva_train_dart32k_offpolicy.parquet"
DEFAULT_SUMMARY = (
    "/home/jovyan/work/LUFFY/data/minerva_train_dart32k_offpolicy.summary.json"
)


def _compose_response(response_analysis: str, response_final: str) -> str:
    analysis = (response_analysis or "").rstrip()
    final = (response_final or "").strip()
    if analysis and final:
        return f"{analysis}\n\n{final}"
    if analysis:
        return analysis
    return final


def _uid_from_extra_info(extra_info: dict) -> str:
    return (
        f"{extra_info['source_file']}:{extra_info['index']}:{extra_info['data_source']}"
    )


def _load_stage(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            grouped[row["uid"]].append(
                {
                    "uid": row["uid"],
                    "trace_source": row["trace_source"],
                    "accepted_rank": row["accepted_rank"],
                    "attempt_index": row["attempt_index"],
                    "task": row["task"],
                    "reward_fn": row["reward_fn"],
                    "data_source": row["data_source"],
                    "source_file": row["source_file"],
                    "source_index": row["source_index"],
                    "base_messages": row.get("base_messages"),
                    "response_analysis": row.get("response_analysis", ""),
                    "response_final": row.get("response_final", ""),
                    "response": _compose_response(
                        row.get("response_analysis", ""), row.get("response_final", "")
                    ),
                }
            )
    return grouped


def _select_stage_map(stage_rows: dict[str, list[dict]], excluded: set[str]) -> dict[str, list[dict]]:
    return {uid: rows for uid, rows in stage_rows.items() if uid not in excluded}


def build_dataset(
    base_train_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    stage3_path: Path,
    stage4_path: Path,
    output_path: Path,
    summary_path: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)

    base_df = pd.read_parquet(base_train_path).copy()
    base_df["uid"] = base_df["extra_info"].map(_uid_from_extra_info)

    stage1_rows = _load_stage(stage1_path)
    stage2_rows_all = _load_stage(stage2_path)
    stage3_rows_all = _load_stage(stage3_path)
    stage4_rows_all = _load_stage(stage4_path)

    stage1_uids = set(stage1_rows)
    stage2_rows = _select_stage_map(stage2_rows_all, stage1_uids)
    stage2_uids = set(stage2_rows)
    stage3_rows = _select_stage_map(stage3_rows_all, stage1_uids | stage2_uids)
    stage3_uids = set(stage3_rows)
    stage4_rows = _select_stage_map(stage4_rows_all, stage1_uids | stage2_uids | stage3_uids)
    stage4_uids = set(stage4_rows)

    selected_targets = []
    stage_counts = Counter()
    trace_source_counts = Counter()
    candidate_count_counts = Counter()

    prompt_alignment_checks = 0

    for idx, uid in enumerate(base_df["uid"]):
        stage_name = None
        candidate_rows = None
        if uid in stage1_rows:
            stage_name = "stage1_plain"
            candidate_rows = stage1_rows[uid]
        elif uid in stage2_rows:
            stage_name = "stage2_guided_filtered"
            candidate_rows = stage2_rows[uid]
        elif uid in stage3_rows:
            stage_name = "stage3_guided_verifier_only"
            candidate_rows = stage3_rows[uid]
        elif uid in stage4_rows:
            stage_name = "stage4_direct_fill"
            candidate_rows = stage4_rows[uid]
        else:
            raise KeyError(f"Missing DART trace for uid={uid}")

        chosen = rng.choice(candidate_rows)
        candidate_count = len(candidate_rows)
        base_prompt = base_df.iloc[idx]["prompt"]
        if hasattr(base_prompt, "tolist"):
            base_prompt = base_prompt.tolist()
        stage_prompt = chosen.get("base_messages")
        if stage_prompt is not None:
            if base_prompt != stage_prompt:
                raise ValueError(f"Prompt mismatch for uid={uid}")
            prompt_alignment_checks += 1

        selected_targets.append(
            {
                "target": [{"role": "assistant", "content": chosen["response"]}],
                "dart_stage": stage_name,
                "dart_trace_source": chosen["trace_source"],
                "dart_attempt_index": chosen["attempt_index"],
                "dart_accepted_rank": chosen["accepted_rank"],
                "dart_task": chosen["task"],
                "dart_reward_fn": chosen["reward_fn"],
                "dart_data_source": chosen["data_source"],
                "dart_source_file": chosen["source_file"],
                "dart_source_index": chosen["source_index"],
                "dart_response_analysis": chosen["response_analysis"],
                "dart_response_final": chosen["response_final"],
                "dart_candidate_count_in_stage": candidate_count,
            }
        )
        stage_counts[stage_name] += 1
        trace_source_counts[chosen["trace_source"]] += 1
        candidate_count_counts[candidate_count] += 1

    selected_df = pd.DataFrame(selected_targets, index=base_df.index)
    output_df = pd.concat([base_df, selected_df], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    summary = {
        "seed": seed,
        "base_train_path": str(base_train_path),
        "stage_paths": {
            "stage1_plain": str(stage1_path),
            "stage2_guided_filtered": str(stage2_path),
            "stage3_guided_verifier_only": str(stage3_path),
            "stage4_direct_fill": str(stage4_path),
        },
        "output_path": str(output_path),
        "total_rows": int(len(output_df)),
        "unique_uids": int(output_df["uid"].nunique()),
        "stage_counts": dict(stage_counts),
        "trace_source_counts": dict(trace_source_counts),
        "candidate_count_distribution": dict(sorted(candidate_count_counts.items())),
        "prompt_alignment_checks": prompt_alignment_checks,
        "stage_uid_coverage": {
            "stage1_plain": len(stage1_uids),
            "stage2_guided_filtered_only": len(stage2_uids),
            "stage3_guided_verifier_only": len(stage3_uids),
            "stage4_direct_fill_only": len(stage4_uids),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 32k LUFFY off-policy Minerva train parquet from staged DART traces."
    )
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--stage1", default=DEFAULT_STAGE_1)
    parser.add_argument("--stage2", default=DEFAULT_STAGE_2)
    parser.add_argument("--stage3", default=DEFAULT_STAGE_3)
    parser.add_argument("--stage4", default=DEFAULT_STAGE_4)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260402)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        base_train_path=Path(args.base_train),
        stage1_path=Path(args.stage1),
        stage2_path=Path(args.stage2),
        stage3_path=Path(args.stage3),
        stage4_path=Path(args.stage4),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

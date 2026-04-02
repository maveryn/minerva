from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dart.acr_filter import acr_filter_from_config
from dart.build_sft_dataset import build_sft_dataset
from dart.common import (
    ensure_dir,
    finish_wall_clock,
    iter_jsonl,
    load_config,
    load_parquet_rows,
    start_wall_clock,
    write_jsonl,
)
from dart.fill import fill_missing_traces
from dart.generate import DartGenerator


def _flatten_paths(value: Any) -> List[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _resolve_group_paths(cfg: Dict[str, Any], group_name: str) -> List[str]:
    validation_cfg = cfg.get("validation", {})
    val_sources = validation_cfg.get("sources", {}) if isinstance(validation_cfg, dict) else {}
    if isinstance(val_sources, dict):
        group_cfg = val_sources.get(group_name)
        if isinstance(group_cfg, dict):
            paths = _flatten_paths(group_cfg.get("paths"))
            if paths:
                return paths

    dataset_cfg = cfg.get("dataset", {})
    return _flatten_paths(dataset_cfg.get("train_paths", cfg.get("train_path")))


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DART v2 from an existing accepted-v1 seed")
    parser.add_argument("--config", default="dart/configs/dart_cti.yaml")
    parser.add_argument("--accepted-v1", required=True)
    parser.add_argument("--group-name", default="train")
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--generation-mode", choices=["guided_fill", "direct_answer"], default="guided_fill")
    parser.add_argument("--target-accepted", type=int)
    parser.add_argument("--max-attempts-per-question", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg.get("output_root", "dart-artifacts/dart_cti"))

    teacher_cfg = cfg.get("teacher", {})
    dataset_cfg = cfg.get("dataset", {})
    guided_cfg = cfg.get("guided_fill", {})

    group_root = ensure_dir(Path(output_root) / "datasets" / args.group_name)
    seed_path = Path(args.accepted_v1)
    seed_tag = str(args.output_tag or seed_path.stem)

    accepted_v2_path = group_root / f"accepted_v2_from_{seed_tag}.jsonl"
    state_v2_path = group_root / f"state_v2_from_{seed_tag}.jsonl"
    attempts_v2_path = group_root / f"attempts_v2_from_{seed_tag}.jsonl"
    summary_v2_path = group_root / f"summary_v2_from_{seed_tag}.json"
    parquet_v2_path = group_root / f"{args.group_name}_v2_from_{seed_tag}.parquet"

    row_paths = _resolve_group_paths(cfg, str(args.group_name))
    if not row_paths:
        raise ValueError(f"No source paths configured for group: {args.group_name}")
    rows = load_parquet_rows(row_paths, limit=args.limit_train_rows)
    accepted_rows_v1 = list(iter_jsonl(seed_path))

    generator = DartGenerator(
        model_path=str(teacher_cfg.get("model_path", "openai/gpt-oss-120b")),
        backend=str(teacher_cfg.get("backend", "vllm")),
        batch_size=int(teacher_cfg.get("batch_size", 1024)),
        max_new_tokens=int(teacher_cfg.get("max_new_tokens", 1024)),
        temperature=float(teacher_cfg.get("temperature", 0.7)),
        top_p=float(teacher_cfg.get("top_p", 0.95)),
        max_prompt_length=int(teacher_cfg.get("max_prompt_length", 0)),
        trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
        gpu_memory_utilization=float(teacher_cfg.get("gpu_memory_utilization", 0.95)),
        vllm_kwargs=teacher_cfg.get("vllm_kwargs"),
        guided_label_details_dir=guided_cfg.get("label_details_dir"),
        guided_max_details_chars=int(guided_cfg.get("max_details_chars", 8096)),
        guided_max_prompt_length=int(guided_cfg.get("max_prompt_length", 4096)),
        guided_enforce_no_id=bool(guided_cfg.get("enforce_no_id", False)),
        task_reasoning_hints=guided_cfg.get("task_reasoning_hints"),
        entity_reasoning_hints=guided_cfg.get("entity_reasoning_hints"),
    )

    acr_filter = acr_filter_from_config(generator.tokenizer, guided_cfg.get("acr_filters"))
    v2_started_at, v2_started_perf = start_wall_clock()
    accepted_rows_v2, state_v2, summary_v2 = fill_missing_traces(
        rows,
        accepted_rows_v1,
        generator,
        generation_mode=args.generation_mode,
        target_accepted=(
            int(args.target_accepted)
            if args.target_accepted is not None
            else int(dataset_cfg.get("target_accepted", 2))
        ),
        max_attempts_per_question=(
            int(args.max_attempts_per_question)
            if args.max_attempts_per_question is not None
            else int(guided_cfg.get("max_attempts_per_question", 0))
        ),
        success_threshold=1.0,
        attempts_output_path=attempts_v2_path,
        acr_filter=acr_filter,
    )
    summary_v2["timing"] = finish_wall_clock(v2_started_at, v2_started_perf)

    response_max_tokens = int(dataset_cfg.get("response_max_tokens", 0)) or None
    write_jsonl(accepted_v2_path, accepted_rows_v2)
    write_jsonl(state_v2_path, state_v2.values())
    _write_summary(summary_v2_path, summary_v2)
    build_sft_dataset(
        accepted_jsonl=accepted_v2_path,
        output_parquet=parquet_v2_path,
        model_path=str(teacher_cfg.get("model_path", "openai/gpt-oss-120b")),
        response_max_tokens=response_max_tokens,
        trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
    )

    print(
        json.dumps(
            {
                "accepted_v1": str(seed_path),
                "accepted_v2": str(accepted_v2_path),
                "state_v2": str(state_v2_path),
                "attempts_v2": str(attempts_v2_path),
                "summary_v2": str(summary_v2_path),
                "parquet_v2": str(parquet_v2_path),
                "timing": summary_v2.get("timing"),
                "metrics": summary_v2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

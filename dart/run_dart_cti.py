from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dart.build_sft_dataset import build_sft_dataset
from dart.collect import collect_uniform
from dart.common import (
    ensure_dir,
    finish_wall_clock,
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


def _artifact_paths(output_root: Path) -> Dict[str, Path]:
    teacher_root = output_root / "teacher"
    datasets_root = output_root / "datasets"
    summaries_root = output_root / "summaries"
    return {
        "teacher_root": teacher_root,
        "datasets_root": datasets_root,
        "summaries_root": summaries_root,
    }


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_group(
    *,
    group_name: str,
    rows: List[Dict[str, Any]],
    generator: DartGenerator,
    dataset_cfg: Dict[str, Any],
    output_root: Path,
    model_path: str,
    trust_remote_code: bool,
    build_v2: bool,
) -> Dict[str, Any]:
    target_accepted = int(dataset_cfg.get("target_accepted", 2))
    max_attempts = int(dataset_cfg.get("max_attempts", 100))
    response_max_tokens = int(dataset_cfg.get("response_max_tokens", 0)) or None

    group_root = ensure_dir(output_root / group_name)
    accepted_v1_path = group_root / "accepted_v1.jsonl"
    state_v1_path = group_root / "state_v1.jsonl"
    attempts_v1_path = group_root / "attempts_v1.jsonl"
    summary_v1_path = group_root / "summary_v1.json"
    parquet_v1_path = group_root / f"{group_name}_v1.parquet"

    v1_started_at, v1_started_perf = start_wall_clock()
    accepted_rows_v1, state_v1, summary_v1 = collect_uniform(
        rows,
        generator,
        target_accepted=target_accepted,
        max_attempts=max_attempts,
        attempts_output_path=attempts_v1_path,
    )
    summary_v1["timing"] = finish_wall_clock(v1_started_at, v1_started_perf)
    write_jsonl(accepted_v1_path, accepted_rows_v1)
    write_jsonl(state_v1_path, state_v1.values())
    _write_summary(summary_v1_path, summary_v1)
    build_sft_dataset(
        accepted_jsonl=accepted_v1_path,
        output_parquet=parquet_v1_path,
        model_path=model_path,
        response_max_tokens=response_max_tokens,
        trust_remote_code=trust_remote_code,
    )

    group_summary: Dict[str, Any] = {
        "group_name": group_name,
        "rows_in": len(rows),
        "v1": {
            "accepted_jsonl": str(accepted_v1_path),
            "state_jsonl": str(state_v1_path),
            "attempts_jsonl": str(attempts_v1_path),
            "summary_json": str(summary_v1_path),
            "parquet": str(parquet_v1_path),
            "metrics": summary_v1,
        },
    }

    if build_v2:
        guided_cfg = dataset_cfg.get("guided_fill", {})
        accepted_v2_path = group_root / "accepted_v2.jsonl"
        state_v2_path = group_root / "state_v2.jsonl"
        attempts_v2_path = group_root / "attempts_v2.jsonl"
        summary_v2_path = group_root / "summary_v2.json"
        parquet_v2_path = group_root / f"{group_name}_v2.parquet"

        v2_started_at, v2_started_perf = start_wall_clock()
        accepted_rows_v2, state_v2, summary_v2 = fill_missing_traces(
            rows,
            accepted_rows_v1,
            generator,
            target_accepted=target_accepted,
            max_attempts_per_question=int(guided_cfg.get("max_attempts_per_question", 0)),
            attempts_output_path=attempts_v2_path,
        )
        summary_v2["timing"] = finish_wall_clock(v2_started_at, v2_started_perf)
        write_jsonl(accepted_v2_path, accepted_rows_v2)
        write_jsonl(state_v2_path, state_v2.values())
        _write_summary(summary_v2_path, summary_v2)
        build_sft_dataset(
            accepted_jsonl=accepted_v2_path,
            output_parquet=parquet_v2_path,
            model_path=model_path,
            response_max_tokens=response_max_tokens,
            trust_remote_code=trust_remote_code,
        )
        group_summary["v2"] = {
            "accepted_jsonl": str(accepted_v2_path),
            "state_jsonl": str(state_v2_path),
            "attempts_jsonl": str(attempts_v2_path),
            "summary_json": str(summary_v2_path),
            "parquet": str(parquet_v2_path),
            "metrics": summary_v2,
        }

    return group_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DART-CTI teacher-side datasets")
    parser.add_argument("--config", default="dart/configs/dart_cti.yaml")
    parser.add_argument("--skip-v2", action="store_true")
    parser.add_argument("--limit-train-rows", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg.get("output_root", "dart-artifacts/dart_cti"))
    artifacts = _artifact_paths(output_root)

    teacher_cfg = cfg.get("teacher", {})
    dataset_cfg = cfg.get("dataset", {})
    guided_cfg = cfg.get("guided_fill", {})
    validation_cfg = cfg.get("validation", {})
    build_v2 = bool(guided_cfg.get("enabled", True)) and not args.skip_v2

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

    train_paths = _flatten_paths(dataset_cfg.get("train_paths", cfg.get("train_path")))
    if not train_paths:
        raise ValueError("No train_paths configured for DART")
    train_rows = load_parquet_rows(train_paths, limit=args.limit_train_rows)

    run_started_at, run_started_perf = start_wall_clock()
    run_summary: Dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name", "dart_cti"),
        "output_root": str(output_root),
        "teacher_model_path": teacher_cfg.get("model_path"),
        "groups": {},
    }

    run_summary["groups"]["train"] = _build_group(
        group_name="train",
        rows=train_rows,
        generator=generator,
        dataset_cfg={"guided_fill": guided_cfg, **dataset_cfg},
        output_root=artifacts["datasets_root"],
        model_path=str(teacher_cfg.get("model_path", "openai/gpt-oss-120b")),
        trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
        build_v2=build_v2,
    )

    val_sources = validation_cfg.get("sources", {})
    if not isinstance(val_sources, dict):
        val_sources = {}
    for group_name, group_cfg in val_sources.items():
        paths = _flatten_paths(group_cfg.get("paths"))
        if not paths:
            continue
        rows = load_parquet_rows(paths, limit=args.limit_val_rows)
        run_summary["groups"][group_name] = _build_group(
            group_name=group_name,
            rows=rows,
            generator=generator,
            dataset_cfg={"guided_fill": guided_cfg, **dataset_cfg},
            output_root=artifacts["datasets_root"],
            model_path=str(teacher_cfg.get("model_path", "openai/gpt-oss-120b")),
            trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
            build_v2=build_v2,
        )

    run_summary["timing"] = finish_wall_clock(run_started_at, run_started_perf)
    run_summary_path = artifacts["summaries_root"] / "run_summary.json"
    _write_summary(run_summary_path, run_summary)
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dart.common import ensure_dir, finish_wall_clock, load_config, load_parquet_rows, start_wall_clock
from dart.generate import DartGenerator
from dart.run_dart_cti import _artifact_paths, _build_group, _flatten_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DART synthetic validation datasets only")
    parser.add_argument("--config", default="dart/configs/dart_cti.yaml")
    parser.add_argument("--group-name", action="append", dest="group_names")
    parser.add_argument("--target-accepted", type=int, default=1)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--limit-val-rows", type=int)
    parser.add_argument("--build-v2", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg.get("output_root", "dart-artifacts/dart_cti"))
    artifacts = _artifact_paths(output_root)

    teacher_cfg = cfg.get("teacher", {})
    dataset_cfg = dict(cfg.get("dataset", {}))
    guided_cfg = dict(cfg.get("guided_fill", {}))
    validation_cfg = cfg.get("validation", {})
    val_sources = validation_cfg.get("sources", {})
    if not isinstance(val_sources, dict):
        val_sources = {}

    requested_groups = set(args.group_names or [])
    if requested_groups:
        val_sources = {name: group_cfg for name, group_cfg in val_sources.items() if name in requested_groups}
    if not val_sources:
        raise ValueError("No validation groups selected")

    dataset_cfg["target_accepted"] = int(args.target_accepted)
    if args.max_attempts is not None:
        dataset_cfg["max_attempts"] = int(args.max_attempts)

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

    build_started_at, build_started_perf = start_wall_clock()
    summary: Dict[str, Any] = {
        "experiment_name": cfg.get("experiment_name", "dart_cti"),
        "output_root": str(output_root),
        "teacher_model_path": teacher_cfg.get("model_path"),
        "target_accepted": int(args.target_accepted),
        "build_v2": bool(args.build_v2),
        "groups": {},
    }

    for group_name, group_cfg in val_sources.items():
        paths = _flatten_paths(group_cfg.get("paths"))
        if not paths:
            continue
        rows = load_parquet_rows(paths, limit=args.limit_val_rows)
        summary["groups"][group_name] = _build_group(
            group_name=group_name,
            rows=rows,
            generator=generator,
            dataset_cfg={"guided_fill": guided_cfg, **dataset_cfg},
            output_root=artifacts["datasets_root"],
            model_path=str(teacher_cfg.get("model_path", "openai/gpt-oss-120b")),
            trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
            build_v2=bool(args.build_v2),
        )

    summary["timing"] = finish_wall_clock(build_started_at, build_started_perf)
    summary_path = Path(artifacts["summaries_root"]) / "validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

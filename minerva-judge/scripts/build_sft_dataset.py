#!/usr/bin/env python3
"""Build an SFT parquet dataset from judged Minerva responses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    build_judge_prompt,
    iter_jsonl,
    load_cti_system_prompt,
    load_rubric_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_source_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def build_key(row: dict[str, Any]) -> str:
    uid = row.get("uid")
    model = row.get("model")
    variant = row.get("prompt_variant")
    if not variant:
        variant = "hinted" if row.get("hinted") else "plain"
    attempt = row.get("attempt")
    parts = [uid, model, variant]
    if attempt is not None:
        parts.append(attempt)
    return ":".join(str(p) for p in parts if p is not None)


def render_question(row: dict[str, Any], include_system: bool, system_prompt: str) -> str:
    question = str(row.get("prompt_used") or "")
    if not question:
        if row.get("prompt_variant") == "hinted" or row.get("hinted"):
            question = str(row.get("acr_prompt") or "")
        if not question:
            question = str(row.get("prompt", ""))
    if include_system and system_prompt:
        question = f"{system_prompt}\n\n{question}"
    return question


def main() -> None:
    parser = argparse.ArgumentParser(description="Build judge SFT parquet from judged JSONL.")
    parser.add_argument("--input", action="append", required=True, help="Judged JSONL input (repeatable)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--require-valid", action="store_true", help="Require rubric_valid == true")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on rows")
    args = parser.parse_args()

    rows = []
    judged_records: list[dict[str, Any]] = []
    source_keys_by_file: dict[str, set[str]] = {}
    for path_str in args.input:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        for record in iter_jsonl(path):
            if args.require_valid and not record.get("rubric_valid"):
                continue
            judge_response = record.get("judge_response")
            if not isinstance(judge_response, str):
                continue
            judged_records.append(record)

            judge_prompt = record.get("judge_prompt")
            if not isinstance(judge_prompt, str):
                source_file = record.get("source_file")
                judge_key = record.get("judge_key")
                if isinstance(source_file, str) and isinstance(judge_key, str):
                    source_keys_by_file.setdefault(source_file, set()).add(judge_key)
                else:
                    print(f"Skipping record missing source_file/judge_key: {record.get('uid')}")
            if args.max_samples is not None and len(judged_records) >= args.max_samples:
                break
        if args.max_samples is not None and len(judged_records) >= args.max_samples:
            break

    source_rows: dict[str, dict[str, Any]] = {}
    if source_keys_by_file:
        for source_file, needed_keys in source_keys_by_file.items():
            if not needed_keys:
                continue
            source_path = resolve_source_path(source_file)
            if not source_path.exists():
                print(f"Source file not found: {source_file}")
                continue
            for row in iter_jsonl(source_path):
                key = build_key(row)
                if key in needed_keys:
                    source_rows[key] = row
            missing = needed_keys - source_rows.keys()
            if missing:
                print(f"Missing {len(missing)} source rows from {source_file}")

    rubric_template = load_rubric_prompt()
    system_prompt = load_cti_system_prompt().strip()

    for record in judged_records:
        judge_response = record.get("judge_response")
        if not isinstance(judge_response, str):
            continue
        judge_prompt = record.get("judge_prompt")
        if not isinstance(judge_prompt, str):
            judge_key = record.get("judge_key")
            source_row = source_rows.get(judge_key or "")
            if source_row is None:
                continue
            include_system = bool(record.get("judge_include_system"))
            question = render_question(source_row, include_system, system_prompt)
            response = str(source_row.get("response", ""))
            judge_prompt = build_judge_prompt(rubric_template, question, response)

        messages = [
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": judge_response},
        ]
        rows.append(
            {
                "uid": record.get("uid"),
                "data_source": record.get("reward_fn") or record.get("task"),
                "messages": messages,
                "source_model": record.get("model"),
                "judge_model": record.get("judge_model"),
                "rubric_score": record.get("rubric_score"),
                "judge_label": record.get("judge_label"),
                "judge_category_id": record.get("judge_category_id"),
                "judge_category_title": record.get("judge_category_title"),
                "prompt_variant": record.get("prompt_variant"),
                "hinted": record.get("hinted"),
            }
        )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    print(f"Wrote {len(rows)} SFT rows to {out_path}")


if __name__ == "__main__":
    main()

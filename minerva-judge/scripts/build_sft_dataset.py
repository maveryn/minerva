#!/usr/bin/env python3
"""Build an SFT parquet dataset from judged Minerva responses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from common import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build judge SFT parquet from judged JSONL.")
    parser.add_argument("--input", action="append", required=True, help="Judged JSONL input (repeatable)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--require-valid", action="store_true", help="Require rubric_valid == true")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on rows")
    args = parser.parse_args()

    rows = []
    for path_str in args.input:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        for record in iter_jsonl(path):
            if args.require_valid and not record.get("rubric_valid"):
                continue
            judge_prompt = record.get("judge_prompt")
            judge_response = record.get("judge_response")
            if not isinstance(judge_prompt, str) or not isinstance(judge_response, str):
                continue
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
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    print(f"Wrote {len(rows)} SFT rows to {out_path}")


if __name__ == "__main__":
    main()

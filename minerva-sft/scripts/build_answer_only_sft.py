#!/usr/bin/env python3
"""Build answer-only SFT datasets from RL-style parquets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def box_answer(answer: Any) -> str:
    text = "" if answer is None else str(answer)
    if "\\boxed" in text:
        return text
    text = text.strip()
    if not text:
        return text
    return f"\\boxed{{{text}}}"


def iter_rows(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"input parquet not found: {path}")
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row


def extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", ""))
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build answer-only SFT parquet from RL parquets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input RL-style parquet files (prompt + reward_model).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output SFT parquet path.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)

    out_rows: list[dict[str, Any]] = []
    for row in iter_rows(input_paths):
        messages = list(row.get("prompt") or [])
        answer = (row.get("reward_model") or {}).get("ground_truth")
        if answer is None:
            extra_info = row.get("extra_info") or {}
            answer = extra_info.get("answer_full") or ""
        response = box_answer(answer)
        messages_with_response = [dict(m) for m in messages]
        messages_with_response.append({"role": "assistant", "content": response})

        user_prompt = extract_user_prompt(messages)
        system_prompt = extract_system_prompt(messages)
        extra_info = row.get("extra_info") or {}

        out_rows.append(
            {
                "messages": messages_with_response,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "response": response,
                "answer": "" if answer is None else str(answer),
                "final_source": "answer",
                "data_source": row.get("data_source"),
                "source_file": row.get("source_file"),
                "task": extra_info.get("task"),
                "reward_fn": extra_info.get("reward_fn"),
                "extra_info": extra_info,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, output_path)
    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

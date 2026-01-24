#!/usr/bin/env python3
"""Build an SFT dataset from judge responses, aligned to minerva_base_train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq


def load_responses(path: Path) -> dict[int, dict[bool, dict[str, Any]]]:
    by_index: dict[int, dict[bool, dict[str, Any]]] = {}
    duplicates = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("source_index")
            if idx is None:
                continue
            hinted = bool(obj.get("hinted"))
            bucket = by_index.setdefault(int(idx), {})
            if hinted in bucket:
                duplicates += 1
                continue
            bucket[hinted] = obj
    if duplicates:
        print(f"warning: skipped {duplicates} duplicate responses")
    return by_index


def extract_response(obj: dict[str, Any] | None) -> str | None:
    if not obj:
        return None
    response = obj.get("response_final")
    if response:
        return str(response)
    # response_final missing; avoid analysis-heavy raw responses unless forced
    return None


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


def normalize_prompt(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def box_answer(answer: Any) -> str:
    text = "" if answer is None else str(answer)
    if "\\boxed" in text:
        return text
    text = text.strip()
    if not text:
        return text
    return f"\\boxed{{{text}}}"


def pick_response(
    plain: dict[str, Any] | None,
    hinted: dict[str, Any] | None,
    answer: str,
) -> tuple[str, str, float]:
    if plain and plain.get("reward") == 1.0:
        response = extract_response(plain)
        if response:
            return response, "plain", float(plain.get("reward", 0.0))
    if hinted and hinted.get("reward") == 1.0:
        response = extract_response(hinted)
        if response:
            return response, "hinted", float(hinted.get("reward", 0.0))
    return box_answer(answer), "answer", 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from judge responses.")
    parser.add_argument(
        "--responses",
        default="minerva-judge/data/responses_gpt_oss_120b.jsonl",
        help="Path to judge responses JSONL.",
    )
    parser.add_argument(
        "--base-parquet",
        default="rlvr/mydata/minerva_base/minerva_base_train.parquet",
        help="Base Minerva train parquet for prompts/ordering.",
    )
    parser.add_argument(
        "--output",
        default="rlvr/mydata/minerva_base_sft/minerva_base_sft_train.parquet",
        help="Output parquet path.",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses)
    base_path = Path(args.base_parquet)
    output_path = Path(args.output)

    if not responses_path.exists():
        raise FileNotFoundError(f"responses not found: {responses_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"base parquet not found: {base_path}")

    by_index = load_responses(responses_path)

    base_table = pq.read_table(base_path)
    base_rows = base_table.to_pylist()

    out_rows: list[dict[str, Any]] = []
    missing = 0
    mismatched_prompt = 0
    counts = {"plain": 0, "hinted": 0, "answer": 0}

    for row in base_rows:
        extra_info = row.get("extra_info") or {}
        idx = extra_info.get("index")
        if idx is None:
            idx = len(out_rows)
        idx = int(idx)
        pair = by_index.get(idx)
        if not pair:
            missing += 1
            continue

        plain = pair.get(False)
        hinted = pair.get(True)

        messages = list(row.get("prompt") or [])
        user_prompt = extract_user_prompt(messages)
        system_prompt = extract_system_prompt(messages)

        if plain and plain.get("prompt"):
            plain_prompt = str(plain.get("prompt"))
            if normalize_prompt(plain_prompt) != normalize_prompt(user_prompt):
                mismatched_prompt += 1

        answer = row.get("reward_model", {}).get("ground_truth") or extra_info.get("answer_full") or ""
        response, final_source, final_reward = pick_response(plain, hinted, answer)

        messages_with_response = [dict(m) for m in messages]
        messages_with_response.append({"role": "assistant", "content": response})

        counts[final_source] += 1

        out_rows.append(
            {
                "messages": messages_with_response,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "response": response,
                "answer": answer,
                "source_index": idx,
                "final_source": final_source,
                "final_reward": final_reward,
                "data_source": row.get("data_source"),
                "task": extra_info.get("task"),
                "reward_fn": extra_info.get("reward_fn"),
                "source_file": row.get("source_file"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, output_path)

    print(f"Wrote {len(out_rows)} rows to {output_path}")
    print(f"Missing indices: {missing}")
    print(f"Prompt mismatches: {mismatched_prompt}")
    print("Final source counts:", counts)


if __name__ == "__main__":
    main()

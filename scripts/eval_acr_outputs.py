#!/usr/bin/env python3
"""Evaluate ACR outputs for parse/leak rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from rlvr.verl.utils.reward_score.reward_acr import reward_acr


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_output(record: dict) -> str:
    for key in ("output", "completion", "target_completion", "response", "text"):
        val = record.get(key)
        if isinstance(val, str):
            return val
    return ""


def _extract_ground_truth(record: dict):
    if "ground_truth" in record:
        return record.get("ground_truth")
    reward_model = record.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return reward_model.get("ground_truth")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ACR outputs JSONL.")
    parser.add_argument("--input", required=True, help="JSONL file with outputs + ground_truth.")
    parser.add_argument("--multilabel-match", default="exact", help="Match policy: exact/subset/superset.")
    parser.add_argument("--enforce-no-id", action="store_true", help="Penalize ID leak in reasoning.")
    args = parser.parse_args()

    records = _load_jsonl(Path(args.input))
    if not records:
        print("No records found.")
        return

    extracted = 0
    correct = 0
    leak_hits = 0
    id_leaks = 0
    total = 0
    scores = []

    for record in records:
        output = _extract_output(record)
        ground_truth = _extract_ground_truth(record)
        if not output or ground_truth is None:
            continue
        total += 1
        reward_info = reward_acr(
            data_source=str(record.get("data_source") or ""),
            solution_str=output,
            ground_truth=ground_truth,
            extra_info=record.get("extra_info"),
            multilabel_match=args.multilabel_match,
            enforce_no_id_in_reasoning=args.enforce_no_id,
        )
        if reward_info.get("acr_extracted"):
            extracted += 1
        if reward_info.get("acr_is_correct"):
            correct += 1
        if reward_info.get("acr_leak_hit"):
            leak_hits += 1
        if reward_info.get("acr_id_leak_hit"):
            id_leaks += 1
        scores.append(float(reward_info.get("score", 0.0)))

    if total == 0:
        print("No valid records with outputs + ground_truth.")
        return

    print(f"Total: {total}")
    print(f"Parse rate: {extracted / total:.3f}")
    print(f"Correct rate: {correct / total:.3f}")
    print(f"Leak rate: {leak_hits / total:.3f}")
    print(f"ID leak rate: {id_leaks / total:.3f}")
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Mean score: {avg_score:.3f}")


if __name__ == "__main__":
    main()

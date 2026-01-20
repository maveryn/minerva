#!/usr/bin/env python3
"""Score synthetic_bad.jsonl with Minerva rewards and update rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import extract_predicted, reward_minerva

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_source_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score synthetic BAD responses.")
    parser.add_argument(
        "--input",
        default="minerva-judge/data/synthetic_bad.jsonl",
        help="Input synthetic JSONL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL (default: overwrite input)",
    )
    parser.add_argument(
        "--mismatch-output",
        default=None,
        help="Optional JSONL path for mismatched rows",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.999,
        help="Reward threshold for correctness",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on processed rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    rows = []
    needed = defaultdict(set)
    for idx, row in enumerate(iter_jsonl(input_path), start=1):
        source_file = row.get("source_file")
        source_line = row.get("source_line")
        if isinstance(source_file, str) and source_line is not None:
            needed[source_file].add(int(source_line))
        rows.append(row)
        if args.limit is not None and idx >= args.limit:
            break

    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source_file, line_nums in needed.items():
        path = resolve_source_path(source_file)
        if not path.exists():
            print(f"Missing source file: {source_file}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if line_num not in line_nums:
                    continue
                line = line.strip()
                if not line:
                    continue
                source_rows[(source_file, line_num)] = json.loads(line)

    output_path = Path(args.output) if args.output else input_path
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    mismatch_path = Path(args.mismatch_output) if args.mismatch_output else None
    mismatch_file = mismatch_path.open("w", encoding="utf-8") if mismatch_path else None

    total = correct = mismatched = 0
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            source_file = row.get("source_file")
            source_line = row.get("source_line")
            source = None
            if isinstance(source_file, str) and source_line is not None:
                source = source_rows.get((source_file, int(source_line)))

            answer = source.get("answer") if source else row.get("correct_answer")
            reward_fn = source.get("reward_fn") if source else row.get("reward_fn")
            if not reward_fn:
                reward_fn = row.get("task")

            response = row.get("response_final") or row.get("response") or ""
            prediction = extract_predicted(reward_fn, response) if reward_fn else ""
            reward = reward_minerva(reward_fn, response, answer) if reward_fn else 0.0
            is_correct = reward >= args.correct_threshold

            row["prediction"] = prediction
            row["reward"] = reward
            row["correct"] = is_correct

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1
            if is_correct:
                correct += 1
            else:
                mismatched += 1
                if mismatch_file is not None:
                    mismatch_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if mismatch_file is not None:
        mismatch_file.close()

    temp_path.replace(output_path)
    print(f"Scored {total} rows. correct={correct} mismatched={mismatched}")
    if mismatch_path:
        print(f"Wrote mismatches to {mismatch_path}")


if __name__ == "__main__":
    main()

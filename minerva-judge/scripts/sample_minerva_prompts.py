#!/usr/bin/env python3
"""Sample prompts from a Minerva train split JSONL."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from common import iter_jsonl, write_jsonl


def reservoir_sample(path: Path, k: int, rng: random.Random) -> list[tuple[int, dict[str, Any]]]:
    sample: list[tuple[int, dict[str, Any]]] = []
    total = 0
    for idx, row in enumerate(iter_jsonl(path)):
        total += 1
        if len(sample) < k:
            sample.append((idx, row))
        else:
            j = rng.randint(0, total - 1)
            if j < k:
                sample[j] = (idx, row)
    if total < k:
        rng.shuffle(sample)
    return sample


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def project_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    return {
        "uid": f"{source_index}-{hash_prompt(prompt)}",
        "source_index": source_index,
        "task": row.get("task"),
        "prompt": prompt,
        "answer": row.get("answer", row.get("ground_truth")),
        "reward_fn": row.get("reward_fn"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample prompts from Minerva split JSONL.")
    parser.add_argument("--input", required=True, help="Input train JSONL (minerva split)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    rng = random.Random(args.seed)
    raw_sample = reservoir_sample(input_path, args.sample_size, rng)

    projected = [project_row(row, source_index) for source_index, row in raw_sample]
    write_jsonl(Path(args.output), projected, mode="w")
    print(f"Wrote {len(projected)} prompts to {args.output}")


if __name__ == "__main__":
    main()

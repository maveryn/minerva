#!/usr/bin/env python3
"""Build a prompt+response dataset for BAD/GOOD classification."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MARKER_RE = re.compile(
    r"(assistantfinal|<\|channel\|>final<\|message\|>|<\|assistant\|>final)",
    flags=re.IGNORECASE,
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def extract_final_response(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    raw = text.strip()
    if not raw:
        return ""
    match = FINAL_MARKER_RE.search(raw)
    if match:
        return raw[match.end() :].strip()
    return raw


def load_needed_keys(judged_paths: list[Path]) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}
    for path in judged_paths:
        for row in iter_jsonl(path):
            label = row.get("judge_label")
            if label not in {"GOOD", "BAD"}:
                continue
            source_file = row.get("source_file")
            if not isinstance(source_file, str):
                continue
            judge_key = row.get("judge_key")
            if not isinstance(judge_key, str):
                judge_key = build_key(row)
            needed.setdefault(source_file, set()).add(judge_key)
    return needed


def resolve_source_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def load_response_rows(needed_keys: dict[str, set[str]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for source_file, keys in needed_keys.items():
        if not keys:
            continue
        path = resolve_source_path(source_file)
        if not path.exists():
            print(f"Missing source file: {source_file}")
            continue
        for row in iter_jsonl(path):
            key = build_key(row)
            if key in keys:
                rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build binary classifier dataset from judged responses.")
    parser.add_argument(
        "--judged-dir",
        default="minerva-judge/judge_data",
        help="Directory with judged_*.jsonl files",
    )
    parser.add_argument(
        "--responses-dir",
        default="minerva-judge/data",
        help="Directory with responses_*.jsonl files",
    )
    parser.add_argument(
        "--synthetic",
        default="minerva-judge/judge_data/synthetic_bad.jsonl",
        help="Optional synthetic BAD JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        default="minerva-judge/classifier/data",
        help="Output directory for train/val JSONL",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Sample this many rows per class after balancing",
    )
    parser.add_argument(
        "--balance",
        default="downsample",
        choices=["none", "downsample"],
        help="Balance classes by downsampling the majority class",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on rows")
    args = parser.parse_args()

    judged_dir = PROJECT_ROOT / args.judged_dir
    responses_dir = PROJECT_ROOT / args.responses_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    judged_paths = sorted(judged_dir.glob("judged_*.jsonl"))
    judged_paths = [p for p in judged_paths if "gpt_oss_120b" not in p.name]
    if not judged_paths:
        raise FileNotFoundError(f"no judged files found in {judged_dir}")
    if not responses_dir.exists():
        raise FileNotFoundError(f"responses dir not found: {responses_dir}")

    needed_keys = load_needed_keys(judged_paths)
    response_rows = load_response_rows(needed_keys)

    records: list[dict[str, Any]] = []
    for path in judged_paths:
        for row in iter_jsonl(path):
            label_text = row.get("judge_label")
            if label_text not in {"GOOD", "BAD"}:
                continue
            judge_key = row.get("judge_key")
            if not isinstance(judge_key, str):
                judge_key = build_key(row)
            response_row = response_rows.get(judge_key)
            if response_row is None:
                continue
            if response_row.get("correct") is not True:
                continue
            prompt = response_row.get("prompt")
            response = extract_final_response(
                response_row.get("response_final") or response_row.get("response")
            )
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            label = 1 if label_text == "GOOD" else 0
            records.append(
                {
                    "uid": response_row.get("uid"),
                    "task": response_row.get("task"),
                    "model": response_row.get("model"),
                    "prompt_variant": response_row.get("prompt_variant"),
                    "hinted": response_row.get("hinted"),
                    "label": label,
                    "label_text": label_text,
                    "judge_category_id": row.get("judge_category_id"),
                    "judge_category_title": row.get("judge_category_title"),
                    "prompt": prompt,
                    "response": response,
                    "synthetic": False,
                }
            )
            if args.max_samples is not None and len(records) >= args.max_samples:
                break
        if args.max_samples is not None and len(records) >= args.max_samples:
            break

    if not records:
        raise RuntimeError("no training records found")

    synthetic_path = Path(args.synthetic) if args.synthetic else None
    synthetic_loaded = 0
    if synthetic_path and synthetic_path.exists():
        for row in iter_jsonl(synthetic_path):
            if row.get("correct") is not True:
                continue
            label_text = row.get("judge_label")
            if label_text not in {"GOOD", "BAD"}:
                continue
            prompt = row.get("source_prompt") or row.get("prompt")
            response = extract_final_response(row.get("response_final") or row.get("response"))
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            label = 1 if label_text == "GOOD" else 0
            records.append(
                {
                    "uid": row.get("uid"),
                    "task": row.get("task"),
                    "model": row.get("model"),
                    "prompt_variant": row.get("prompt_variant"),
                    "hinted": row.get("hinted"),
                    "label": label,
                    "label_text": label_text,
                    "judge_category_id": row.get("judge_category_id"),
                    "judge_category_title": row.get("judge_category_title"),
                    "prompt": prompt,
                    "response": response,
                    "synthetic": True,
                }
            )
            synthetic_loaded += 1

    rng = random.Random(args.seed)
    label_groups: dict[str, list[dict[str, Any]]] = {"GOOD": [], "BAD": []}
    for row in records:
        label = row.get("label_text")
        if label in label_groups:
            label_groups[label].append(row)

    original_counts = {label: len(rows) for label, rows in label_groups.items()}

    if args.balance == "downsample":
        min_count = min(len(rows) for rows in label_groups.values() if rows)
        for rows in label_groups.values():
            rng.shuffle(rows)
        label_groups = {label: rows[:min_count] for label, rows in label_groups.items()}

    balanced_counts = {label: len(rows) for label, rows in label_groups.items()}

    if args.sample_per_class is not None:
        sample_size = args.sample_per_class
        for label, rows in label_groups.items():
            if len(rows) < sample_size:
                raise ValueError(f"not enough {label} rows for sample_per_class={sample_size}")
        label_groups = {
            label: rng.sample(rows, sample_size) for label, rows in label_groups.items()
        }
        balanced_counts = {label: len(rows) for label, rows in label_groups.items()}

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for rows in label_groups.values():
        if not rows:
            continue
        split_idx = int(len(rows) * args.train_ratio)
        train_records.extend(rows[:split_idx])
        val_records.extend(rows[split_idx:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)

    def count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"GOOD": 0, "BAD": 0}
        for row in rows:
            label = row.get("label_text")
            if label in counts:
                counts[label] += 1
        return counts

    summary = {
        "total": len(records),
        "balance": args.balance,
        "sample_per_class": args.sample_per_class,
        "original_counts": original_counts,
        "balanced_counts": balanced_counts,
        "synthetic_loaded": synthetic_loaded,
        "train": len(train_records),
        "val": len(val_records),
        "train_counts": count_labels(train_records),
        "val_counts": count_labels(val_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(train_records)} train and {len(val_records)} val rows to {output_dir}")


if __name__ == "__main__":
    main()

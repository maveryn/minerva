#!/usr/bin/env python3
"""Compare human-human and human-GPT agreement for annotation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / "annotations"
VALID_LABELS = ("A", "B", "TIE")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_labels(path: Path, label_field: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in load_jsonl(path):
        item_id = str(row.get("item_id") or "")
        label = str(row.get(label_field) or "").strip().upper()
        if item_id and label in VALID_LABELS:
            labels[item_id] = label
    return labels


def percent_agreement(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a:
        return 0.0
    matches = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b)
    return matches / len(labels_a)


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a:
        return 0.0

    observed = percent_agreement(labels_a, labels_b)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    total = len(labels_a)
    expected = sum((count_a[label] / total) * (count_b[label] / total) for label in VALID_LABELS)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def make_overlap_report(
    name_a: str,
    labels_a: dict[str, str],
    name_b: str,
    labels_b: dict[str, str],
) -> dict[str, Any]:
    common_ids = sorted(set(labels_a) & set(labels_b))
    seq_a = [labels_a[item_id] for item_id in common_ids]
    seq_b = [labels_b[item_id] for item_id in common_ids]
    return {
        "left": name_a,
        "right": name_b,
        "common_items": len(common_ids),
        "percent_agreement": percent_agreement(seq_a, seq_b),
        "cohen_kappa": cohen_kappa(seq_a, seq_b),
        "left_label_counts": dict(sorted(Counter(seq_a).items())),
        "right_label_counts": dict(sorted(Counter(seq_b).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, help="subset_100 or subset_200")
    parser.add_argument("--annotator-a", required=True, help="Annotator ID for first human")
    parser.add_argument("--annotator-b", required=True, help="Annotator ID for second human")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    subset_dir = ROOT / args.subset
    key_path = subset_dir / "key.jsonl"
    if not key_path.exists():
        raise FileNotFoundError(f"Missing key file: {key_path}")

    annotator_a_path = ANNOTATIONS_DIR / args.subset / f"{args.annotator_a}.jsonl"
    annotator_b_path = ANNOTATIONS_DIR / args.subset / f"{args.annotator_b}.jsonl"
    if not annotator_a_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotator_a_path}")
    if not annotator_b_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotator_b_path}")

    human_a = load_labels(annotator_a_path, "label")
    human_b = load_labels(annotator_b_path, "label")
    gpt = load_labels(key_path, "gpt_winner")

    common_human_ids = sorted(set(human_a) & set(human_b))
    consensus_ids = sorted(item_id for item_id in common_human_ids if human_a[item_id] == human_b[item_id])
    consensus = {item_id: human_a[item_id] for item_id in consensus_ids}

    report = {
        "subset": args.subset,
        "annotator_a": {
            "name": args.annotator_a,
            "completed_items": len(human_a),
            "label_counts": dict(sorted(Counter(human_a.values()).items())),
        },
        "annotator_b": {
            "name": args.annotator_b,
            "completed_items": len(human_b),
            "label_counts": dict(sorted(Counter(human_b.values()).items())),
        },
        "human_human": make_overlap_report(args.annotator_a, human_a, args.annotator_b, human_b),
        "human_vs_gpt": {
            "annotator_a_vs_gpt": make_overlap_report(args.annotator_a, human_a, "gpt-5.2", gpt),
            "annotator_b_vs_gpt": make_overlap_report(args.annotator_b, human_b, "gpt-5.2", gpt),
            "consensus_vs_gpt": make_overlap_report("human_consensus", consensus, "gpt-5.2", gpt),
        },
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare pointwise human rubric scores with other humans and a judge model."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / "annotations"
CURRENT_SCORE_FIELDS = (
    "writing_quality_score",
    "evidence_use_score",
    "cti_concept_focus_score",
)
INTERMEDIATE_SCORE_FIELDS = (
    "response_quality_score",
    "prompt_support_score",
    "cti_framing_score",
)
LEGACY_SCORE_FIELDS = (
    "clarity_score",
    "groundedness_score",
    "alignment_score",
    "usefulness_score",
)
SUPPORTED_SCORE_SCHEMAS = (
    CURRENT_SCORE_FIELDS,
    INTERMEDIATE_SCORE_FIELDS,
    LEGACY_SCORE_FIELDS,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def detect_score_fields(rows: list[dict[str, Any]], path: Path) -> tuple[str, ...]:
    for schema in SUPPORTED_SCORE_SCHEMAS:
        for row in rows:
            if all(field in row for field in schema):
                return schema
    raise ValueError(f"Could not detect a supported score schema in {path}")


def load_pointwise_scores(path: Path) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    rows = load_jsonl(path)
    score_fields = detect_score_fields(rows, path)
    scores: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("parse_error") is True:
            continue
        item_id = str(row.get("item_id") or "")
        if not item_id:
            continue
        item_scores: dict[str, int] = {}
        try:
            for field in score_fields:
                value = int(row.get(field))
                if value < 1 or value > 4:
                    raise ValueError
                item_scores[field] = value
        except Exception:
            continue
        item_scores["total_score"] = sum(item_scores[field] for field in score_fields)
        scores[item_id] = item_scores
    return score_fields, scores


def percent_exact(values_a: list[int], values_b: list[int]) -> float:
    if not values_a:
        return 0.0
    matches = sum(1 for left, right in zip(values_a, values_b, strict=True) if left == right)
    return matches / len(values_a)


def within_one(values_a: list[int], values_b: list[int]) -> float:
    if not values_a:
        return 0.0
    close = sum(1 for left, right in zip(values_a, values_b, strict=True) if abs(left - right) <= 1)
    return close / len(values_a)


def mean_absolute_error(values_a: list[int], values_b: list[int]) -> float:
    if not values_a:
        return 0.0
    return sum(abs(left - right) for left, right in zip(values_a, values_b, strict=True)) / len(values_a)


def rankdata(values: list[int]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if not x:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def spearman(values_a: list[int], values_b: list[int]) -> float:
    if not values_a:
        return 0.0
    return pearson(rankdata(values_a), rankdata(values_b))


def quadratic_weighted_kappa(values_a: list[int], values_b: list[int], min_rating: int, max_rating: int) -> float:
    if not values_a:
        return 0.0
    num_ratings = max_rating - min_rating + 1
    conf = [[0.0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    for left, right in zip(values_a, values_b, strict=True):
        conf[left - min_rating][right - min_rating] += 1.0

    hist_a = [0.0 for _ in range(num_ratings)]
    hist_b = [0.0 for _ in range(num_ratings)]
    for value in values_a:
        hist_a[value - min_rating] += 1.0
    for value in values_b:
        hist_b[value - min_rating] += 1.0

    total = float(len(values_a))
    expected = [[0.0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected[i][j] = (hist_a[i] * hist_b[j]) / total

    def weight(i: int, j: int) -> float:
        if num_ratings == 1:
            return 0.0
        return ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    observed = 0.0
    expected_weighted = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            w = weight(i, j)
            observed += w * conf[i][j]
            expected_weighted += w * expected[i][j]

    if expected_weighted == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - (observed / expected_weighted)


def compare(
    name_a: str,
    scores_a: dict[str, dict[str, int]],
    name_b: str,
    scores_b: dict[str, dict[str, int]],
    score_fields: tuple[str, ...],
) -> dict[str, Any]:
    common_ids = sorted(set(scores_a) & set(scores_b))
    criteria: dict[str, Any] = {}
    for field in score_fields:
        seq_a = [scores_a[item_id][field] for item_id in common_ids]
        seq_b = [scores_b[item_id][field] for item_id in common_ids]
        criteria[field] = {
            "quadratic_weighted_kappa": quadratic_weighted_kappa(seq_a, seq_b, 1, 4),
            "spearman": spearman(seq_a, seq_b),
            "percent_exact": percent_exact(seq_a, seq_b),
            "within_one": within_one(seq_a, seq_b),
            "mae": mean_absolute_error(seq_a, seq_b),
            "left_score_counts": dict(sorted(Counter(seq_a).items())),
            "right_score_counts": dict(sorted(Counter(seq_b).items())),
        }

    total_a = [scores_a[item_id]["total_score"] for item_id in common_ids]
    total_b = [scores_b[item_id]["total_score"] for item_id in common_ids]
    min_total = len(score_fields)
    max_total = len(score_fields) * 4
    return {
        "left": name_a,
        "right": name_b,
        "common_items": len(common_ids),
        "criteria": criteria,
        "criteria_mean_quadratic_weighted_kappa": (
            sum(criteria[field]["quadratic_weighted_kappa"] for field in score_fields) / len(score_fields)
            if common_ids
            else 0.0
        ),
        "total_score": {
            "quadratic_weighted_kappa": quadratic_weighted_kappa(total_a, total_b, min_total, max_total),
            "spearman": spearman(total_a, total_b),
            "percent_exact": percent_exact(total_a, total_b),
            "within_one": within_one(total_a, total_b),
            "mae": mean_absolute_error(total_a, total_b),
            "left_score_counts": dict(sorted(Counter(total_a).items())),
            "right_score_counts": dict(sorted(Counter(total_b).items())),
        },
    }


def consensus_scores(
    scores_a: dict[str, dict[str, int]],
    scores_b: dict[str, dict[str, int]],
    score_fields: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    common_ids = sorted(set(scores_a) & set(scores_b))
    consensus: dict[str, dict[str, int]] = {}
    for item_id in common_ids:
        merged: dict[str, int] = {}
        for field in score_fields:
            left = scores_a[item_id][field]
            right = scores_b[item_id][field]
            if left != right:
                break
            merged[field] = left
        else:
            merged["total_score"] = sum(merged[field] for field in score_fields)
            consensus[item_id] = merged
    return consensus


def ensure_same_schema(
    left_name: str,
    left_fields: tuple[str, ...],
    right_name: str,
    right_fields: tuple[str, ...],
) -> tuple[str, ...]:
    if left_fields != right_fields:
        raise ValueError(
            f"Score schema mismatch between {left_name} and {right_name}: "
            f"{list(left_fields)} vs {list(right_fields)}"
        )
    return left_fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="pointwise_pilot_50_correct_only", help="Pointwise subset directory")
    parser.add_argument("--annotator-a", required=True, help="First annotator ID")
    parser.add_argument("--annotator-b", help="Optional second annotator ID")
    parser.add_argument(
        "--annotator-a-file",
        help="Optional path to annotator A JSONL. Defaults to annotations/<subset>/<annotator-a>.jsonl",
    )
    parser.add_argument(
        "--annotator-b-file",
        help="Optional path to annotator B JSONL. Defaults to annotations/<subset>/<annotator-b>.jsonl",
    )
    parser.add_argument(
        "--judge-file",
        help="Optional judge JSONL path. Defaults to <subset>/gpt52_scores.jsonl",
    )
    parser.add_argument(
        "--judge-name",
        help="Display name for the judge model. Defaults to judge_model from the file, else gpt-5.2",
    )
    parser.add_argument(
        "--gpt-file",
        help="Deprecated alias for --judge-file",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    subset_dir = ROOT / args.subset
    judge_path = Path(args.judge_file or args.gpt_file) if (args.judge_file or args.gpt_file) else subset_dir / "gpt52_scores.jsonl"
    if not judge_path.exists():
        raise FileNotFoundError(f"Missing judge score file: {judge_path}")

    annotator_a_path = Path(args.annotator_a_file) if args.annotator_a_file else (ANNOTATIONS_DIR / args.subset / f"{args.annotator_a}.jsonl")
    if not annotator_a_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotator_a_path}")

    human_a_fields, human_a = load_pointwise_scores(annotator_a_path)
    judge_fields, judge_scores = load_pointwise_scores(judge_path)
    score_fields = ensure_same_schema(args.annotator_a, human_a_fields, "judge", judge_fields)

    judge_name = args.judge_name
    if not judge_name:
        rows = load_jsonl(judge_path)
        for row in rows:
            candidate = str(row.get("judge_model") or "").strip()
            if candidate:
                judge_name = candidate
                break
    if not judge_name:
        judge_name = "gpt-5.2"

    report: dict[str, Any] = {
        "subset": args.subset,
        "judge_file": str(judge_path),
        "score_fields": list(score_fields),
        "annotator_a": {
            "name": args.annotator_a,
            "completed_items": len(human_a),
        },
        "judge": {
            "name": judge_name,
            "completed_items": len(judge_scores),
        },
        "annotator_a_vs_judge": compare(args.annotator_a, human_a, judge_name, judge_scores, score_fields),
    }

    if args.annotator_b:
        annotator_b_path = Path(args.annotator_b_file) if args.annotator_b_file else (ANNOTATIONS_DIR / args.subset / f"{args.annotator_b}.jsonl")
        if not annotator_b_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotator_b_path}")
        human_b_fields, human_b = load_pointwise_scores(annotator_b_path)
        ensure_same_schema(args.annotator_a, human_a_fields, args.annotator_b, human_b_fields)
        consensus = consensus_scores(human_a, human_b, score_fields)
        report["annotator_b"] = {
            "name": args.annotator_b,
            "completed_items": len(human_b),
        }
        report["human_human"] = compare(args.annotator_a, human_a, args.annotator_b, human_b, score_fields)
        report["annotator_b_vs_judge"] = compare(args.annotator_b, human_b, judge_name, judge_scores, score_fields)
        report["consensus_vs_judge"] = compare("human_consensus", consensus, judge_name, judge_scores, score_fields)
        report["human_consensus_completed_items"] = len(consensus)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

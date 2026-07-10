#!/usr/bin/env python3
"""Build compact Eval12 rollout-aware best-of-k tables for rebuttal drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from analyze_llmbench_eval12 import (
    MetricData,
    TASKS,
    bootstrap_delta_ci,
    f1_from_counts,
    metric,
    normalize_ioc_for_type,
    normalize_lance_type,
    validate_pair_compatibility,
)


DEFAULT_RUN_ROOT = Path("/home/jovyan/work/llmbench/runs/passk_eval12_t0.7_p0.95_k8")

MODELS = {
    "Llama-8B": {"GRPO": "llama8b_grpo", "MinervaRL": "llama8b_minerva"},
    "Llama-3B": {"GRPO": "llama3b_grpo", "MinervaRL": "llama3b_minerva"},
    "Qwen-8B": {"GRPO": "qwen8b_grpo", "MinervaRL": "qwen8b_minerva"},
    "Qwen-4B": {"GRPO": "qwen4b_grpo", "MinervaRL": "qwen4b_minerva"},
}

AZERG_STIX_TYPES = {
    "ATTACK_PATTERN",
    "CAMPAIGN",
    "COURSE_OF_ACTION",
    "IDENTITY",
    "INDICATOR",
    "INFRASTRUCTURE",
    "LOCATION",
    "MALWARE",
    "THREAT_ACTOR",
    "TOOL",
    "VULNERABILITY",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_value(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def record_f1(row: dict[str, Any]) -> float:
    return f1_from_counts(
        float(row.get("tp", 0.0) or 0.0),
        float(row.get("fp", 0.0) or 0.0),
        float(row.get("fn", 0.0) or 0.0),
    )


def truth_iocs(row: dict[str, Any]) -> set[str]:
    indicator_type = normalize_lance_type(row.get("indicator_type"))
    answer = row.get("answer")
    if isinstance(answer, list):
        values = answer
    elif answer:
        values = [answer]
    else:
        values = []
    out: set[str] = set()
    for value in values:
        norm = normalize_ioc_for_type(indicator_type, str(value))
        if norm:
            out.add(norm)
    return out


def prism_row_score(row: dict[str, Any]) -> float:
    truth = truth_iocs(row)
    pred = {str(value) for value in row.get("normalized_predictions") or [] if str(value)}
    return f1_from_counts(len(truth & pred), len(pred - truth), len(truth - pred))


def exact_row_score(row: dict[str, Any]) -> float:
    truth = str(row.get("truth", "") or "")
    pred = str(row.get("predicted", "") or "")
    return 1.0 if truth and pred == truth else 0.0


def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_azerg_entities(text: str) -> list[str]:
    content = extract_tag(text, "entities")
    raw_content = content if content else text or ""
    if not raw_content:
        return []
    items = [item.strip() for item in re.split(r"[|,;\n]", raw_content) if item.strip()]
    cleaned_items = []
    for item in items:
        upper = item.upper()
        if upper in AZERG_STIX_TYPES:
            continue
        if upper in {"ENTITY", "ENTITIES", "STIX", "STIX ENTITY TYPES", "ENTITY TYPES"}:
            continue
        cleaned_items.append(item)
    seen = set()
    out = []
    for item in cleaned_items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def parse_azerg_label(text: str, tag: str, label_set: list[str] | None = None) -> str:
    tagged = extract_tag(text, tag).strip()
    if tagged:
        return tagged
    upper = (text or "").upper()
    if tag == "related":
        if re.search(r"\bYES\b", upper):
            return "YES"
        if re.search(r"\bNO\b", upper):
            return "NO"
        return ""
    if tag == "entity_type":
        for label in AZERG_STIX_TYPES:
            if re.search(rf"\b{re.escape(label)}\b", upper):
                return label
        return ""
    if tag == "label" and label_set:
        for label in label_set:
            if label in text:
                return label
        for label in label_set:
            if label.lower() in (text or "").lower():
                return label
    return ""


def best_rows(
    model_dir: Path,
    task_file: str,
    k: int,
    scorer,
) -> list[dict[str, Any]]:
    by_id: dict[Any, tuple[float, int, dict[str, Any]]] = {}
    for sample_idx in range(1, k + 1):
        path = model_dir / f"sample_{sample_idx:02d}" / model_dir.name / f"{task_file}-scored.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in load_jsonl(path):
            row_id = row.get("id")
            value = scorer(row)
            previous = by_id.get(row_id)
            if previous is None or value > previous[0]:
                by_id[row_id] = (value, sample_idx, row)
    return [item[2] for item in sorted(by_id.values(), key=lambda item: item[2].get("id", 0))]


def mean_score(rows: list[dict[str, Any]]) -> float:
    return sum(score_value(row) for row in rows) / len(rows) if rows else 0.0


def count_f1(rows: list[dict[str, Any]]) -> float:
    tp = sum(float(row.get("tp", 0.0) or 0.0) for row in rows)
    fp = sum(float(row.get("fp", 0.0) or 0.0) for row in rows)
    fn = sum(float(row.get("fn", 0.0) or 0.0) for row in rows)
    return f1_from_counts(tp, fp, fn)


def prism_f1(rows: list[dict[str, Any]]) -> float:
    part = prism_part_data(rows)
    tp = float(np.sum(part["tp"]))
    fp = float(np.sum(part["fp"]))
    fn = float(np.sum(part["fn"]))
    return f1_from_counts(tp, fp, fn)


def prism_part_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_gt: dict[tuple[str, str], set[str]] = {}
    grouped_pred: dict[tuple[str, str], set[str]] = {}
    for idx, row in enumerate(rows):
        report_id = str(row.get("report_id") or row.get("id") or f"row-{idx}")
        indicator_type = normalize_lance_type(row.get("indicator_type"))
        if not indicator_type:
            continue
        key = (report_id, indicator_type)
        grouped_gt.setdefault(key, set()).update(truth_iocs(row))
        grouped_pred.setdefault(key, set()).update(
            str(value) for value in row.get("normalized_predictions") or [] if str(value)
        )

    keys = sorted(set(grouped_gt) | set(grouped_pred))
    tp = []
    fp = []
    fn = []
    for key in keys:
        pred = grouped_pred.get(key, set())
        truth = grouped_gt.get(key, set())
        tp.append(len(pred & truth))
        fp.append(len(pred - truth))
        fn.append(len(truth - pred))
    return {
        "keys": keys,
        "tp": np.array(tp, dtype=float),
        "fp": np.array(fp, dtype=float),
        "fn": np.array(fn, dtype=float),
    }


def azerg_part_metric(task_file: str, rows: list[dict[str, Any]]) -> float:
    # Use the saved scorer outputs instead of reparsing free-form responses.
    # The evaluator's fallback parser checks STIX labels from a Python set, so
    # reparsing can vary across hash seeds when a response mentions multiple
    # labels. The scored rows are the stable artifact from the original run.
    if task_file.upper().endswith("T2"):
        labels = sorted({str(row.get("truth", "") or "") for row in rows if row.get("truth")})
        f1s = []
        for label in labels:
            tp = sum(1 for row in rows if row.get("truth") == label and row.get("predicted") == label)
            fp = sum(1 for row in rows if row.get("truth") != label and row.get("predicted") == label)
            fn = sum(1 for row in rows if row.get("truth") == label and row.get("predicted") != label)
            f1s.append(f1_from_counts(tp, fp, fn))
        return sum(f1s) / len(f1s) if f1s else 0.0

    valid = [row for row in rows if row.get("truth")]
    if not valid:
        return 0.0
    correct = sum(1 for row in valid if row.get("predicted") == row.get("truth"))
    return correct / len(valid)


def task_metric(model_dir: Path, kind: str, files: list[str], k: int) -> float:
    return metric(task_metric_data(model_dir, kind, files, k))


def azerg_part_data(task_file: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.array([str(row.get("truth", "") or "") for row in rows], dtype=object)
    pred = np.array([str(row.get("predicted", "") or "") for row in rows], dtype=object)
    return {
        "task": task_file,
        "truth": truth,
        "pred": pred,
        "labels": sorted({value for value in truth.tolist() if value}),
    }


def task_metric_data(model_dir: Path, kind: str, files: list[str], k: int) -> MetricData:
    if kind in {"mean", "threat_mean"}:
        rows = best_rows(model_dir, files[0], k, score_value)
        return MetricData("mean", np.array([score_value(row) for row in rows], dtype=float))
    if kind == "counts_f1":
        rows = best_rows(model_dir, files[0], k, record_f1)
        return MetricData(
            "counts_f1",
            {
                "tp": np.array([float(row.get("tp", 0.0) or 0.0) for row in rows], dtype=float),
                "fp": np.array([float(row.get("fp", 0.0) or 0.0) for row in rows], dtype=float),
                "fn": np.array([float(row.get("fn", 0.0) or 0.0) for row in rows], dtype=float),
            },
        )
    if kind == "prism":
        return MetricData(
            "prism",
            [prism_part_data(best_rows(model_dir, task_file, k, prism_row_score)) for task_file in files],
        )
    if kind == "azerg_avg":
        return MetricData(
            "azerg_avg",
            [
                azerg_part_data(task_file, best_rows(model_dir, task_file, k, exact_row_score))
                for task_file in files
            ],
        )
    raise ValueError(kind)


def fmt(value: float) -> str:
    return f"{value * 100:.1f}"


def cell(values: dict[int, float], ks: list[int]) -> str:
    return "/".join(fmt(values[k]) for k in ks)


def seeded_rng(seed: int, *parts: str) -> np.random.Generator:
    key = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def validate_k8(results: dict[str, dict[str, dict[str, dict[int, float]]]], run_root: Path) -> None:
    summary_keys = {
        "CKT": "ckt_accuracy",
        "CyberMetric": "cybermetric_accuracy",
        "SOCEval": "soceval_avg_score",
        "RCM": "rcm_accuracy",
        "VSP": "vsp_accuracy",
        "ATE": "ate_accuracy",
        "RMS": "rms_f1",
        "ElasticRule": "elasticrule_accuracy",
        "APTNER": "aptner_f1",
        "LANCE": "lance_avg_f1",
        "AnnoCTR": "annoctr_avg",
        "AZERG": "azerg_avg",
    }
    for backbone, variants in MODELS.items():
        for model_name, dirname in variants.items():
            summary = json.loads((run_root / dirname / "passk-oracle-summary.json").read_text())
            for task_name, key in summary_keys.items():
                observed = results[backbone][model_name][task_name][8]
                expected = float(summary[key])
                if task_name in {"AnnoCTR", "AZERG"}:
                    continue
                if abs(observed - expected) > 0.005:
                    raise ValueError(
                        f"k=8 mismatch for {backbone} {model_name} {task_name}: "
                        f"{observed:.4f} vs {expected:.4f}"
                    )


def sign_summary_rows(
    data: dict[str, dict[str, dict[str, dict[int, MetricData]]]],
    ks: list[int],
    n_boot: int,
    seed: int,
) -> list[tuple[int, int, int, int, int, int]]:
    rows = []
    task_names = [task_name for task_name, _, _ in TASKS]
    for k in ks:
        point_wins = point_losses = point_ties = 0
        ci_positive = ci_negative = ci_overlap = 0
        for backbone in ["Llama-8B", "Llama-3B", "Qwen-8B", "Qwen-4B"]:
            for task_name in task_names:
                left = data[backbone]["GRPO"][task_name][k]
                right = data[backbone]["MinervaRL"][task_name][k]
                validate_pair_compatibility(left, right, task_name)
                delta = metric(right) - metric(left)
                if delta > 0:
                    point_wins += 1
                elif delta < 0:
                    point_losses += 1
                else:
                    point_ties += 1
                ci = bootstrap_delta_ci(
                    left,
                    right,
                    seeded_rng(seed, "passk", backbone, task_name, str(k)),
                    n_boot,
                )
                if ci[0] > 0:
                    ci_positive += 1
                elif ci[1] < 0:
                    ci_negative += 1
                else:
                    ci_overlap += 1
        rows.append((k, point_wins, point_losses, point_ties, ci_positive, ci_negative, ci_overlap))
    return rows


def build_table(run_root: Path, ks: list[int], n_boot: int, seed: int) -> str:
    results: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    data: dict[str, dict[str, dict[str, dict[int, MetricData]]]] = {}
    for backbone, variants in MODELS.items():
        results[backbone] = {}
        data[backbone] = {}
        for model_name, dirname in variants.items():
            model_dir = run_root / dirname
            results[backbone][model_name] = {}
            data[backbone][model_name] = {}
            for task_name, kind, files in TASKS:
                data[backbone][model_name][task_name] = {
                    k: task_metric_data(model_dir, kind, files, k) for k in ks
                }
                results[backbone][model_name][task_name] = {
                    k: metric(data[backbone][model_name][task_name][k]) for k in ks
                }

    if 8 in ks:
        validate_k8(results, run_root)

    task_names = [task_name for task_name, _, _ in TASKS]
    lines = [
        "# Eval12 Rollout-Aware Best-of-k Table",
        "",
        "Generated from saved `passk_eval12_t0.7_p0.95_k8` scored outputs.",
        "",
        "Each task cell is `GRPO/MinervaRL` and reports actual task score in percent for the specified backbone and best-of-k budget. For exact-match, MCQ, and taxonomy tasks this is verifier success/best-of-k accuracy; for graded, extraction, and multi-label tasks, the best scored sample is selected per example and the original Eval12 metric is recomputed.",
        "",
        "| Backbone | k | " + " | ".join(task_names) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(task_names)) + " |",
    ]
    backbones = ["Llama-8B", "Llama-3B", "Qwen-8B", "Qwen-4B"]
    for backbone in backbones:
        for k in ks:
            row = [backbone, str(k)]
            for task_name in task_names:
                grpo = fmt(results[backbone]["GRPO"][task_name][k])
                minerva = fmt(results[backbone]["MinervaRL"][task_name][k])
                row.append(f"{grpo}/{minerva}")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Paired Bootstrap CI Sign Summary",
            "",
            f"Generated with `{n_boot}` paired bootstrap resamples and seed `{seed}`. Counts are over 48 backbone-task comparisons at each `k`; intervals are for `MinervaRL - GRPO` under the rollout-aware score.",
            "",
        ]
    )
    summary_rows = sign_summary_rows(data, ks, n_boot, seed)
    include_ties = any(point_ties for _, _, _, point_ties, _, _, _ in summary_rows)
    if include_ties:
        lines.extend(
            [
                "| k | Point wins | Point losses | Point ties | CI positive | CI negative | CI overlaps 0 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for k, wins, losses, ties, ci_pos, ci_neg, ci_overlap in summary_rows:
            lines.append(
                f"| {k} | {wins}/48 | {losses}/48 | {ties}/48 | "
                f"{ci_pos}/48 | {ci_neg}/48 | {ci_overlap}/48 |"
            )
    else:
        lines.extend(
            [
                "| k | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for k, wins, losses, _, ci_pos, ci_neg, ci_overlap in summary_rows:
            lines.append(
                f"| {k} | {wins}/48 | {losses}/48 | "
                f"{ci_pos}/48 | {ci_neg}/48 | {ci_overlap}/48 |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8809)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal/results/eval12_passk_rollout_table.md"),
    )
    args = parser.parse_args()
    args.out.write_text(build_table(args.run_root, args.ks, args.boot, args.seed), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

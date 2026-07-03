#!/usr/bin/env python3
"""Compute Eval12 uncertainty and GRPO-vs-MinervaRL deltas from LLMBench runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LLMBENCH_ROOT = Path("/home/jovyan/work/llmbench")
RUNS_ROOT = LLMBENCH_ROOT / "runs"

LANCE_TYPE_ALIASES = {
    "ip4": "ip",
    "ipv4": "ip",
    "fqdn": "domain",
}

DEFANG_DOT_PATTERNS = [
    re.compile(r"\[(?:\.|dot)\]", re.IGNORECASE),
    re.compile(r"\((?:\.|dot)\)", re.IGNORECASE),
    re.compile(r"\{(?:\.|dot)\}", re.IGNORECASE),
]


MODELS = {
    "Llama-8B": {
        "GRPO": "xashru/minerva_grpo_llama8b_500_490",
        "MinervaRL": "xashru/minerva_noctua_llama8b_ema_0.05",
    },
    "Llama-3B": {
        "GRPO": "xashru/minerva_grpo_llama3b_500",
        "MinervaRL": "xashru/minerva_noctua_llama3b_500",
    },
    "Qwen-8B": {
        "GRPO": "xashru/minerva_grpo_qwen8b_base",
        "MinervaRL": "xashru/minerva_noctua_qwen8b_base",
    },
    "Qwen-4B": {
        "GRPO": "xashru/minerva_grpo_qwen4b_base",
        "MinervaRL": "xashru/minerva_noctua_qwen4b_base",
    },
}


TASKS = [
    ("CKT", "mean", ["MCQ3k"]),
    ("CyberMetric", "mean", ["CyberMetric"]),
    ("SOCEval", "threat_mean", ["ThreatIntelReasoning"]),
    ("RCM", "mean", ["RCM"]),
    ("VSP", "mean", ["VSP"]),
    ("ATE", "mean", ["ATE"]),
    ("RMS", "mean", ["RMS"]),
    ("ElasticRule", "mean", ["ElasticToAttack"]),
    ("APTNER", "counts_f1", ["APTNER"]),
    ("LANCE", "prism", ["PrismIP", "PrismURL", "PrismDomain", "PrismHash"]),
    ("AnnoCTR", "azerg_avg", ["ANNOCTR-T1", "ANNOCTR-T2", "ANNOCTR-T3", "ANNOCTR-T4"]),
    ("AZERG", "azerg_avg", ["AZERG-T1", "AZERG-T2", "AZERG-T3", "AZERG-T4"]),
]

TASK_NAMES = [name for name, _, _ in TASKS]

FAMILIES = {
    "Choice/reasoning": ["CKT", "CyberMetric", "SOCEval"],
    "Taxonomy mapping": ["RCM", "ATE", "ElasticRule", "RMS"],
    "Structured extraction/scoring": ["VSP", "APTNER", "LANCE", "AnnoCTR", "AZERG"],
}


@dataclass(frozen=True)
class MetricData:
    kind: str
    payload: Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def defang_text(text: str) -> str:
    text = text or ""
    text = re.sub(
        r"\bhxxps?\b",
        lambda m: "https" if m.group(0).lower() == "hxxps" else "http",
        text,
        flags=re.IGNORECASE,
    )
    for pattern in DEFANG_DOT_PATTERNS:
        text = pattern.sub(".", text)
    return text.replace("[:]", ":").replace("[/]", "/").replace("[://]", "://")


def clean_ioc_token(value: str) -> str:
    value = value.strip().strip(" \t\r\n\"'<>[](){}")
    return value.rstrip(".,;:!?)]}")


def normalize_lance_type(indicator_type: Any) -> str:
    if not indicator_type:
        return ""
    value = str(indicator_type).strip().lower()
    return LANCE_TYPE_ALIASES.get(value, value)


def normalize_ip(value: str) -> str | None:
    value = clean_ioc_token(value)
    if not value:
        return None
    if ":" in value:
        host, _, port = value.rpartition(":")
        if host and port.isdigit():
            value = host
    octets = value.split(".")
    if len(octets) != 4:
        return None
    for octet in octets:
        if not octet.isdigit() or not 0 <= int(octet) <= 255:
            return None
    return value


def normalize_domain(value: str) -> str | None:
    value = clean_ioc_token(value).lower()
    if not value:
        return None
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    value = value.split("/")[0].split(":")[0]
    value = re.sub(r"^www\.", "", value)
    return value or None


def normalize_url(value: str) -> str | None:
    value = clean_ioc_token(value)
    if not value:
        return None
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^www\.", "", value, flags=re.IGNORECASE)
    value = value.rstrip(".").lower()
    if value.endswith("/") and "/" not in value[:-1]:
        value = value[:-1]
    return value or None


def normalize_ioc_for_type(ioc_type: str, value: str) -> str | None:
    value = defang_text(str(value))
    ioc_type = normalize_lance_type(ioc_type)
    if ioc_type == "url":
        return normalize_url(value)
    if ioc_type == "domain":
        return normalize_domain(value)
    if ioc_type == "ip":
        return normalize_ip(value)
    if ioc_type == "hash":
        return clean_ioc_token(value).lower() or None
    return None


def load_mean_scores(run_dir: Path, task_file: str) -> MetricData:
    rows = load_jsonl(run_dir / f"{task_file}-scored.jsonl")
    return MetricData("mean", np.array([float(row.get("score", 0.0)) for row in rows]))


def load_threat_scores(run_dir: Path, task_file: str) -> MetricData:
    rows = load_jsonl(run_dir / f"{task_file}-scored.jsonl")
    return MetricData(
        "threat_mean",
        {
            "score": np.array([float(row.get("score", 0.0)) for row in rows]),
            "valid": np.array(
                [row.get("answered_correctly") != "parsing error" for row in rows],
                dtype=bool,
            ),
        },
    )


def load_count_f1(run_dir: Path, task_file: str) -> MetricData:
    rows = load_jsonl(run_dir / f"{task_file}-scored.jsonl")
    return MetricData(
        "counts_f1",
        {
            "tp": np.array([float(row.get("tp", 0.0)) for row in rows]),
            "fp": np.array([float(row.get("fp", 0.0)) for row in rows]),
            "fn": np.array([float(row.get("fn", 0.0)) for row in rows]),
        },
    )


def _answer_values(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(value) for value in answer if str(value)]
    if answer:
        return [str(answer)]
    return []


def parse_lance_label(line: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in line.split(",") if part.strip()]
    if len(parts) < 2:
        return None
    ioc = parts[0]
    label = parts[1].lower().replace(" ", "").replace("-", "")
    if label not in {"ioc", "nonioc"}:
        return None
    return ioc, label


def load_prism_part(run_dir: Path, task_file: str) -> dict[str, Any]:
    rows = load_jsonl(run_dir / f"{task_file}-scored.jsonl")
    grouped_gt: dict[tuple[str, str], set[str]] = {}
    vote_counts: dict[tuple[str, str], dict[str, list[int]]] = {}
    for idx, row in enumerate(rows):
        report_id = str(row.get("report_id") or row.get("id") or f"row-{idx}")
        indicator_type = normalize_lance_type(row.get("indicator_type"))
        key = (report_id, indicator_type)
        grouped_gt.setdefault(key, set())
        for value in _answer_values(row.get("answer")):
            norm = normalize_ioc_for_type(indicator_type, value)
            if norm:
                grouped_gt[key].add(norm)

        candidates = row.get("normalized_candidates")
        if isinstance(candidates, list):
            candidate_norm = {str(value) for value in candidates if str(value)}
        else:
            candidate_norm = set()
        votes = vote_counts.setdefault(key, {})
        for line in str(row.get("response", "")).splitlines():
            parsed = parse_lance_label(line)
            if not parsed:
                continue
            ioc_raw, label = parsed
            norm = normalize_ioc_for_type(indicator_type, ioc_raw)
            if not norm or (candidate_norm and norm not in candidate_norm):
                continue
            counter = votes.setdefault(norm, [0, 0])
            if label == "ioc":
                counter[0] += 1
            else:
                counter[1] += 1

    grouped_pred: dict[tuple[str, str], set[str]] = {}
    for key, ioc_votes in vote_counts.items():
        for ioc, counts in ioc_votes.items():
            if counts[0] >= counts[1]:
                grouped_pred.setdefault(key, set()).add(ioc)

    keys = sorted(set(grouped_gt) | set(grouped_pred))
    tp: list[int] = []
    fp: list[int] = []
    fn: list[int] = []
    for key in keys:
        pred = grouped_pred.get(key, set())
        gt = grouped_gt.get(key, set())
        tp.append(len(pred & gt))
        fp.append(len(pred - gt))
        fn.append(len(gt - pred))
    return {
        "keys": keys,
        "tp": np.array(tp, dtype=float),
        "fp": np.array(fp, dtype=float),
        "fn": np.array(fn, dtype=float),
    }


def load_azerg_part(run_dir: Path, task_file: str) -> dict[str, Any]:
    rows = load_jsonl(run_dir / f"{task_file}-scored.jsonl")
    truth = np.array([str(row.get("truth", "")) for row in rows], dtype=object)
    pred = np.array([str(row.get("predicted", "")) for row in rows], dtype=object)
    return {
        "task": task_file,
        "truth": truth,
        "pred": pred,
        "labels": sorted({value for value in truth.tolist() if value}),
    }


def load_task(run_dir: Path, kind: str, files: list[str]) -> MetricData:
    if kind == "mean":
        return load_mean_scores(run_dir, files[0])
    if kind == "threat_mean":
        return load_threat_scores(run_dir, files[0])
    if kind == "counts_f1":
        return load_count_f1(run_dir, files[0])
    if kind == "prism":
        return MetricData("prism", [load_prism_part(run_dir, task_file) for task_file in files])
    if kind == "azerg_avg":
        return MetricData("azerg_avg", [load_azerg_part(run_dir, task_file) for task_file in files])
    raise ValueError(f"Unsupported metric kind: {kind}")


def load_all_data() -> dict[str, dict[str, dict[str, MetricData]]]:
    data: dict[str, dict[str, dict[str, MetricData]]] = {}
    for backbone, variants in MODELS.items():
        data[backbone] = {}
        for variant, rel_path in variants.items():
            run_dir = RUNS_ROOT / rel_path
            if not run_dir.exists():
                raise FileNotFoundError(run_dir)
            data[backbone][variant] = {
                task_name: load_task(run_dir, kind, files)
                for task_name, kind, files in TASKS
            }
    return data


def sample_for(data: MetricData, rng: np.random.Generator) -> Any:
    if data.kind == "mean":
        n = len(data.payload)
        return rng.integers(0, n, n)
    if data.kind == "threat_mean":
        n = len(data.payload["score"])
        return rng.integers(0, n, n)
    if data.kind == "counts_f1":
        n = len(data.payload["tp"])
        return rng.integers(0, n, n)
    if data.kind in {"prism", "azerg_avg"}:
        samples = []
        for part in data.payload:
            n = len(part["tp"]) if data.kind == "prism" else len(part["truth"])
            samples.append(rng.integers(0, n, n))
        return samples
    raise ValueError(data.kind)


def metric(data: MetricData, sample: Any | None = None) -> float:
    if data.kind == "mean":
        scores = data.payload
        values = scores if sample is None else scores[sample]
        return float(np.mean(values)) if len(values) else 0.0

    if data.kind == "threat_mean":
        idx = np.arange(len(data.payload["score"])) if sample is None else sample
        scores = data.payload["score"][idx]
        valid = data.payload["valid"][idx]
        values = scores[valid]
        return float(np.mean(values)) if len(values) else 0.0

    if data.kind == "counts_f1":
        idx = slice(None) if sample is None else sample
        counts = data.payload
        return f1_from_counts(
            float(np.sum(counts["tp"][idx])),
            float(np.sum(counts["fp"][idx])),
            float(np.sum(counts["fn"][idx])),
        )

    if data.kind == "prism":
        f1s = []
        samples = [None] * len(data.payload) if sample is None else sample
        for part, idx in zip(data.payload, samples):
            idx = slice(None) if idx is None else idx
            f1s.append(
                f1_from_counts(
                    float(np.sum(part["tp"][idx])),
                    float(np.sum(part["fp"][idx])),
                    float(np.sum(part["fn"][idx])),
                )
            )
        return float(np.mean(f1s)) if f1s else 0.0

    if data.kind == "azerg_avg":
        scores = []
        samples = [None] * len(data.payload) if sample is None else sample
        for part, idx in zip(data.payload, samples):
            idx = np.arange(len(part["truth"])) if idx is None else idx
            truth = part["truth"][idx]
            pred = part["pred"][idx]
            if part["task"].upper().endswith("T2"):
                f1s = []
                for label in part["labels"]:
                    tp = int(np.sum((truth == label) & (pred == label)))
                    fp = int(np.sum((truth != label) & (pred == label)))
                    fn = int(np.sum((truth == label) & (pred != label)))
                    f1s.append(f1_from_counts(tp, fp, fn))
                scores.append(float(np.mean(f1s)) if f1s else 0.0)
            else:
                valid = truth != ""
                correct = (pred == truth) & valid
                scores.append(float(np.sum(correct) / len(truth)) if len(truth) else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    raise ValueError(data.kind)


def validate_pair_compatibility(left: MetricData, right: MetricData, task_name: str) -> None:
    if left.kind != right.kind:
        raise ValueError(f"{task_name}: kind mismatch {left.kind} vs {right.kind}")
    if left.kind == "mean" and len(left.payload) != len(right.payload):
        raise ValueError(f"{task_name}: mean length mismatch")
    if left.kind == "threat_mean" and len(left.payload["score"]) != len(right.payload["score"]):
        raise ValueError(f"{task_name}: threat length mismatch")
    if left.kind == "counts_f1" and len(left.payload["tp"]) != len(right.payload["tp"]):
        raise ValueError(f"{task_name}: count length mismatch")
    if left.kind == "prism":
        if len(left.payload) != len(right.payload):
            raise ValueError(f"{task_name}: PRISM part mismatch")
        for idx, (left_part, right_part) in enumerate(zip(left.payload, right.payload)):
            if left_part["keys"] != right_part["keys"]:
                raise ValueError(f"{task_name}: PRISM keys mismatch in part {idx}")
    if left.kind == "azerg_avg":
        if len(left.payload) != len(right.payload):
            raise ValueError(f"{task_name}: AZERG part mismatch")
        for idx, (left_part, right_part) in enumerate(zip(left.payload, right.payload)):
            if left_part["task"] != right_part["task"] or len(left_part["truth"]) != len(right_part["truth"]):
                raise ValueError(f"{task_name}: AZERG part mismatch in part {idx}")
            if not np.array_equal(left_part["truth"], right_part["truth"]):
                raise ValueError(f"{task_name}: AZERG truth mismatch in part {idx}")


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci(
    data: MetricData,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        values[idx] = metric(data, sample_for(data, rng))
    return percentile_ci(values)


def bootstrap_delta_ci(
    left: MetricData,
    right: MetricData,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = sample_for(left, rng)
        values[idx] = metric(right, sample) - metric(left, sample)
    return percentile_ci(values)


def average_metric(tasks: dict[str, MetricData], selected: list[str], samples: dict[str, Any] | None = None) -> float:
    return float(
        np.mean(
            [
                metric(tasks[task_name], None if samples is None else samples[task_name])
                for task_name in selected
            ]
        )
    )


def bootstrap_average_ci(
    tasks: dict[str, MetricData],
    selected: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        samples = {task_name: sample_for(tasks[task_name], rng) for task_name in selected}
        values[idx] = average_metric(tasks, selected, samples)
    return percentile_ci(values)


def bootstrap_average_delta_ci(
    left_tasks: dict[str, MetricData],
    right_tasks: dict[str, MetricData],
    selected: list[str],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    values = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        samples = {task_name: sample_for(left_tasks[task_name], rng) for task_name in selected}
        values[idx] = average_metric(right_tasks, selected, samples) - average_metric(left_tasks, selected, samples)
    return percentile_ci(values)


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def ci_pct(ci: tuple[float, float]) -> str:
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def validate_reproduction(data: dict[str, dict[str, dict[str, MetricData]]]) -> None:
    # These checks catch accidental deviations from LLMBench's summary aggregation.
    summary_keys = {
        "CKT": "mcq3k_accuracy",
        "CyberMetric": "cybermetric_accuracy",
        "SOCEval": "threat_intel_reasoning_avg_score",
        "RCM": "rcm_accuracy",
        "VSP": "vsp_accuracy",
        "ATE": "ate_accuracy",
        "RMS": "rms_f1",
        "ElasticRule": "elastictoattack_accuracy",
        "APTNER": "aptner_f1",
        "LANCE": "prism_avg_f1",
        "AnnoCTR": "annoctr_avg",
        "AZERG": "azerg_avg",
    }
    for backbone, variants in MODELS.items():
        for variant, rel_path in variants.items():
            summary_path = RUNS_ROOT / rel_path / "evalall-summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for task_name, key in summary_keys.items():
                observed = metric(data[backbone][variant][task_name])
                expected = float(summary[key])
                if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"{backbone} {variant} {task_name}: computed {observed} "
                        f"but summary {key} is {expected}"
                    )


def build_report(data: dict[str, dict[str, dict[str, MetricData]]], n_boot: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    validate_reproduction(data)

    for backbone in MODELS:
        left = data[backbone]["GRPO"]
        right = data[backbone]["MinervaRL"]
        for task_name in TASK_NAMES:
            validate_pair_compatibility(left[task_name], right[task_name], task_name)

    lines = [
        "# LLMBench Eval12 Rebuttal Statistics",
        "",
        f"Generated with `{Path(__file__).name}` using `{n_boot}` bootstrap resamples and seed `{seed}`.",
        "",
        "All values are percentages. Delta is `MinervaRL - GRPO` in percentage points.",
        "Intervals are nonparametric paired bootstrap percentile intervals over evaluation examples or grouped examples for aggregate metrics.",
        "",
        "Source root: `/home/jovyan/work/llmbench/runs`.",
        "",
        "## 12-Column Average",
        "",
        "| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    avg_delta_cis: dict[str, tuple[float, float]] = {}
    for backbone in MODELS:
        left = data[backbone]["GRPO"]
        right = data[backbone]["MinervaRL"]
        grpo_avg = average_metric(left, TASK_NAMES)
        rl_avg = average_metric(right, TASK_NAMES)
        ci = bootstrap_average_delta_ci(left, right, TASK_NAMES, rng, n_boot)
        avg_delta_cis[backbone] = ci
        lines.append(
            f"| {backbone} | {pct(grpo_avg)} | {pct(rl_avg)} | "
            f"{pct(rl_avg - grpo_avg)} | {ci_pct(ci)} |"
        )

    lines.extend(["", "## Task-Family Averages", ""])
    for family_name, selected in FAMILIES.items():
        lines.extend(
            [
                f"### {family_name}",
                "",
                "| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for backbone in MODELS:
            left = data[backbone]["GRPO"]
            right = data[backbone]["MinervaRL"]
            grpo_avg = average_metric(left, selected)
            rl_avg = average_metric(right, selected)
            ci = bootstrap_average_delta_ci(left, right, selected, rng, n_boot)
            lines.append(
                f"| {backbone} | {pct(grpo_avg)} | {pct(rl_avg)} | "
                f"{pct(rl_avg - grpo_avg)} | {ci_pct(ci)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-Task Paired Deltas",
            "",
            "| Backbone | Task | GRPO | MinervaRL | Delta | Paired 95% CI |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backbone in MODELS:
        left = data[backbone]["GRPO"]
        right = data[backbone]["MinervaRL"]
        for task_name in TASK_NAMES:
            grpo = metric(left[task_name])
            rl = metric(right[task_name])
            ci = bootstrap_delta_ci(left[task_name], right[task_name], rng, n_boot)
            lines.append(
                f"| {backbone} | {task_name} | {pct(grpo)} | {pct(rl)} | "
                f"{pct(rl - grpo)} | {ci_pct(ci)} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The script verifies every computed point estimate against `evalall-summary.json` before writing the report.",
            "- `Avg.` is recomputed over the 12 paper columns and intentionally does not use `overall_avg`, which includes additional tasks.",
            "- PRISM/LANCE CIs resample report/type groups within each IoC subtype, then average subtype F1 values.",
            "- AZERG and AnnoCTR CIs resample examples within each subtask, using macro-F1 for T2 and exact accuracy for T1/T3/T4.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=2000, help="number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=8809)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal/llmbench_eval12_stats.md"),
        help="markdown output path",
    )
    args = parser.parse_args()

    report = build_report(load_all_data(), args.boot, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

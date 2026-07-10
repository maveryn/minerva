#!/usr/bin/env python3
"""Build a markdown packet of Eval12 pairwise CI results for rebuttal planning."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from analyze_llmbench_eval12 import (
    TASK_NAMES,
    average_metric,
    bootstrap_average_delta_ci,
    bootstrap_delta_ci,
    ci_pct,
    metric,
    pct,
    validate_pair_compatibility,
)
from analyze_llmbench_eval12_all_pairs import MODEL_GROUPS, load_all, validate_summaries


BASELINES = ["Base", "STaR", "DART", "GRPO", "LUFFY"]

TASK_FAMILIES = OrderedDict(
    [
        ("MCQ", ["CKT", "CyberMetric", "SOCEval"]),
        ("Structured taxonomy mapping", ["RCM", "ATE", "ElasticRule", "RMS"]),
        ("Vulnerability scoring", ["VSP"]),
        ("Entity / IoC extraction", ["APTNER", "LANCE", "AnnoCTR", "AZERG"]),
    ]
)


def seeded_rng(seed: int, *parts: str) -> np.random.Generator:
    key = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def signed(value: float) -> str:
    return f"{value * 100:+.1f}"


def signed_ci(delta: float, ci: tuple[float, float]) -> str:
    return f"{signed(delta)} {ci_pct(ci)}"


def strongest_non_minerva(models: OrderedDict[str, dict]) -> str:
    return max(
        [name for name in models if name != "MinervaRL"],
        key=lambda name: average_metric(models[name], TASK_NAMES),
    )


def point_counts(left: dict, right: dict) -> tuple[int, int, int]:
    pos = neg = tie = 0
    for task_name in TASK_NAMES:
        delta = metric(right[task_name]) - metric(left[task_name])
        if delta > 0:
            pos += 1
        elif delta < 0:
            neg += 1
        else:
            tie += 1
    return pos, neg, tie


def ci_sign_counts(rows: list[tuple[str, str, str, float, tuple[float, float]]]) -> tuple[int, int, int]:
    pos = neg = overlap = 0
    for _, _, _, _, ci in rows:
        if ci[0] > 0:
            pos += 1
        elif ci[1] < 0:
            neg += 1
        else:
            overlap += 1
    return pos, neg, overlap


def build_packet(n_boot: int, seed: int) -> str:
    data = load_all()
    validate_summaries(data)

    per_task: dict[str, list[tuple[str, str, str, float, tuple[float, float]]]] = {
        baseline: [] for baseline in BASELINES
    }
    avg_rows: dict[str, list[tuple[str, float, tuple[float, float], tuple[int, int, int]]]] = {
        baseline: [] for baseline in BASELINES
    }

    for baseline in BASELINES:
        for backbone, models in data.items():
            left = models[baseline]
            right = models["MinervaRL"]
            for task_name in TASK_NAMES:
                validate_pair_compatibility(left[task_name], right[task_name], task_name)
                delta = metric(right[task_name]) - metric(left[task_name])
                ci = bootstrap_delta_ci(
                    left[task_name],
                    right[task_name],
                    seeded_rng(seed, "per-task", baseline, backbone, task_name),
                    n_boot,
                )
                per_task[baseline].append((backbone, task_name, baseline, delta, ci))

            avg_delta = average_metric(right, TASK_NAMES) - average_metric(left, TASK_NAMES)
            avg_ci = bootstrap_average_delta_ci(
                left,
                right,
                TASK_NAMES,
                seeded_rng(seed, "avg", baseline, backbone),
                n_boot,
            )
            avg_rows[baseline].append((backbone, avg_delta, avg_ci, point_counts(left, right)))

    lines = [
        "# Eval12 Pairwise CI Packet",
        "",
        f"Generated with `{Path(__file__).name}` using `{n_boot}` bootstrap resamples and seed `{seed}`.",
        "",
        "All deltas are `MinervaRL - baseline` in percentage points. Intervals are paired nonparametric bootstrap 95% CIs over evaluation examples or grouped examples for aggregate metrics.",
        "",
        "These intervals quantify evaluation-sample uncertainty, not training-seed variance.",
        "",
        "## 1. Average Results For All Models",
        "",
        "`Avg.` is the unweighted macro-average over the 12 Eval12 task metrics.",
        "",
        "| Backbone | Base | STaR | DART | GRPO | LUFFY | MinervaRL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for backbone, models in data.items():
        values = [pct(average_metric(models[name], TASK_NAMES)) for name in ["Base", "STaR", "DART", "GRPO", "LUFFY", "MinervaRL"]]
        lines.append(f"| {backbone} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 2. MinervaRL Versus Each Baseline: Average Delta",
            "",
            "| Baseline | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline in BASELINES:
        by_backbone = {backbone: (delta, ci) for backbone, delta, ci, _ in avg_rows[baseline]}
        lines.append(
            f"| {baseline} | "
            + " | ".join(signed_ci(*by_backbone[backbone]) for backbone in MODEL_GROUPS)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 3. MinervaRL Versus Each Baseline: Per-Task Count Summary",
            "",
            "| Baseline | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for baseline in BASELINES:
        point_pos = point_neg = point_tie = 0
        for _, _, _, counts in avg_rows[baseline]:
            point_pos += counts[0]
            point_neg += counts[1]
            point_tie += counts[2]
        ci_pos, ci_neg, ci_overlap = ci_sign_counts(per_task[baseline])
        lines.append(
            f"| {baseline} | {point_pos}/48 | {point_neg}/48 | "
            f"{ci_pos}/48 | {ci_neg}/48 | {ci_overlap}/48 |"
        )

    lines.extend(
        [
            "",
            "## 4. Strongest Non-MinervaRL Baseline By Backbone",
            "",
            "| Backbone | Strongest baseline | Baseline Avg. | MinervaRL Avg. | Delta | 95% CI |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backbone, models in data.items():
        baseline = strongest_non_minerva(models)
        left = models[baseline]
        right = models["MinervaRL"]
        delta = average_metric(right, TASK_NAMES) - average_metric(left, TASK_NAMES)
        ci = bootstrap_average_delta_ci(
            left,
            right,
            TASK_NAMES,
            seeded_rng(seed, "strongest", backbone, baseline),
            n_boot,
        )
        lines.append(
            f"| {backbone} | {baseline} | {pct(average_metric(left, TASK_NAMES))} | "
            f"{pct(average_metric(right, TASK_NAMES))} | {signed(delta)} | {ci_pct(ci)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Task-Family CIs: MinervaRL Versus GRPO",
            "",
            "| Family | Tasks | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family_name, family_tasks in TASK_FAMILIES.items():
        cells = []
        for backbone, models in data.items():
            left = models["GRPO"]
            right = models["MinervaRL"]
            delta = average_metric(right, family_tasks) - average_metric(left, family_tasks)
            ci = bootstrap_average_delta_ci(
                left,
                right,
                family_tasks,
                seeded_rng(seed, "family", "GRPO", backbone, family_name),
                n_boot,
            )
            cells.append(signed_ci(delta, ci))
        lines.append(
            f"| {family_name} | {', '.join(family_tasks)} | "
            + " | ".join(cells)
            + " |"
        )

    for baseline in BASELINES:
        title = "GRPO" if baseline == "GRPO" else baseline
        lines.extend(
            [
                "",
                f"## 6. Per-Task CIs: MinervaRL Versus {title}",
                "",
                "| Task | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        rows = {(backbone, task): (delta, ci) for backbone, task, _, delta, ci in per_task[baseline]}
        for task_name in TASK_NAMES:
            cells = [
                signed_ci(*rows[(backbone, task_name)])
                for backbone in MODEL_GROUPS
            ]
            lines.append(f"| {task_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 7. Reporting Notes",
            "",
            "- The main rebuttal should lead with paired bootstrap CIs, not descriptive win/loss counts.",
            "- The macro-average is retained for comparability with Table 1, but per-task and task-family uncertainty should be used to qualify claims.",
            "- The full per-task tables above are intended as source material; the rebuttal can quote the count summary and a few representative tasks.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8809)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal/results/eval12_pairwise_ci_packet.md"),
    )
    args = parser.parse_args()
    args.out.write_text(build_packet(args.boot, args.seed), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

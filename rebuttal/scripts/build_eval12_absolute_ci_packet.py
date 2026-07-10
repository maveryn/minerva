#!/usr/bin/env python3
"""Build Eval12 absolute score CI tables for rebuttal planning."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from analyze_llmbench_eval12 import (
    TASK_NAMES,
    average_metric,
    bootstrap_average_ci,
    bootstrap_ci,
    ci_pct,
    metric,
    pct,
)
from analyze_llmbench_eval12_all_pairs import MODEL_GROUPS, load_all, validate_summaries


def seeded_rng(seed: int, *parts: str) -> np.random.Generator:
    key = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def score_ci_cell(value: float, ci: tuple[float, float]) -> str:
    return f"{pct(value)} {ci_pct(ci)}"


def build_packet(n_boot: int, seed: int) -> str:
    data = load_all()
    validate_summaries(data)

    lines = [
        "# Eval12 Absolute Score CI Packet",
        "",
        f"Generated with `{Path(__file__).name}` using `{n_boot}` bootstrap resamples and seed `{seed}`.",
        "",
        "All values are percentages. Intervals are nonparametric bootstrap 95% CIs over evaluation examples or grouped examples for aggregate metrics.",
        "",
        "These tables answer the absolute-score uncertainty request. Use the paired-delta packet for model-comparison significance.",
        "",
        "## 1. 12-Task Average Score CIs",
        "",
        "| Backbone | Model | Avg. 95% CI |",
        "| --- | --- | ---: |",
    ]

    for backbone, models in data.items():
        for model_name, tasks in models.items():
            value = average_metric(tasks, TASK_NAMES)
            ci = bootstrap_average_ci(
                tasks,
                TASK_NAMES,
                seeded_rng(seed, "avg", backbone, model_name),
                n_boot,
            )
            lines.append(f"| {backbone} | {model_name} | {score_ci_cell(value, ci)} |")

    lines.extend(
        [
            "",
            "## 2. Per-Task Score CIs By Backbone",
            "",
        ]
    )

    for backbone, models in data.items():
        model_names = list(models)
        lines.extend(
            [
                f"### {backbone}",
                "",
                "| Task | " + " | ".join(model_names) + " |",
                "| --- |" + " ---: |" * len(model_names),
            ]
        )
        for task_name in TASK_NAMES:
            cells = []
            for model_name in model_names:
                task_data = models[model_name][task_name]
                value = metric(task_data)
                ci = bootstrap_ci(
                    task_data,
                    seeded_rng(seed, "task", backbone, model_name, task_name),
                    n_boot,
                )
                cells.append(score_ci_cell(value, ci))
            lines.append(f"| {task_name} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## 3. Reporting Notes",
            "",
            "- These are absolute score CIs for each model/task.",
            "- Model-comparison claims should use paired bootstrap delta CIs, because all models are evaluated on the same examples.",
            "- The companion pairwise packet reports `MinervaRL - baseline` deltas and paired CIs.",
            "- `Avg.` is the unweighted macro-average over the 12 Eval12 task metrics.",
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
        default=Path("rebuttal/results/eval12_absolute_score_ci_packet.md"),
    )
    args = parser.parse_args()
    args.out.write_text(build_packet(args.boot, args.seed), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

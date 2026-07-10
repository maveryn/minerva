#!/usr/bin/env python3
"""Build Llama-8B Eval12 ablation tables and paired CI summaries."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from analyze_llmbench_eval12 import (
    TASKS,
    TASK_NAMES,
    bootstrap_delta_ci,
    metric,
    load_task,
    validate_pair_compatibility,
)


RUNS = OrderedDict(
    [
        ("GRPO / no ACR", Path("/home/jovyan/work/llmbench/runs/xashru/minerva_grpo_llama8b_500_490")),
        ("GRPO-12", Path("/home/jovyan/work/llmbench/runs/eval12_ablation_grpo12_llama8b_65536")),
        ("Answer-only SFT", Path("/home/jovyan/work/llmbench/runs/eval12_ablation_sft_answer_llama8b_65536")),
        (
            "In-loop answer-only SFT",
            Path("/home/jovyan/work/llmbench/runs/minerva_llama8b_answer_sft_best320_local"),
        ),
        ("No EMA teacher", Path("/home/jovyan/work/llmbench/runs/eval12_ablation_emaoff_llama8b_65536")),
        ("No TextCNN filter", Path("/home/jovyan/work/llmbench/runs/eval12_ablation_mloff_llama8b_65536")),
        ("No filtering", Path("/home/jovyan/work/llmbench/runs/eval12_ablation_filteroff_llama8b_65536")),
        ("Full MinervaRL", Path("/home/jovyan/work/llmbench/runs/xashru/minerva_noctua_llama8b_ema_0.05")),
    ]
)


def seeded_rng(seed: int, *parts: str) -> np.random.Generator:
    key = "::".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def load_runs() -> OrderedDict[str, dict]:
    data: OrderedDict[str, dict] = OrderedDict()
    for name, run_dir in RUNS.items():
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        data[name] = {
            task_name: load_task(run_dir, kind, files)
            for task_name, kind, files in TASKS
        }
    return data


def point_counts(control: dict, full: dict) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for task_name in TASK_NAMES:
        delta = metric(full[task_name]) - metric(control[task_name])
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def ci_counts(control_name: str, control: dict, full: dict, n_boot: int, seed: int) -> tuple[int, int, int]:
    positive = negative = overlap = 0
    for task_name in TASK_NAMES:
        validate_pair_compatibility(control[task_name], full[task_name], task_name)
        ci = bootstrap_delta_ci(
            control[task_name],
            full[task_name],
            seeded_rng(seed, "ablation", control_name, task_name),
            n_boot,
        )
        if ci[0] > 0:
            positive += 1
        elif ci[1] < 0:
            negative += 1
        else:
            overlap += 1
    return positive, negative, overlap


def build_table(n_boot: int, seed: int) -> str:
    data = load_runs()
    full = data["Full MinervaRL"]

    lines = [
        "# Eval12 Llama-8B Ablation Tables",
        "",
        f"Generated with `{Path(__file__).name}` using `{n_boot}` paired bootstrap resamples and seed `{seed}`.",
        "",
        "All scores are actual Eval12 task scores in percent. The CI sign summary compares `Full MinervaRL - control/ablation` over the 12 displayed tasks.",
        "",
        "## Task Scores",
        "",
        "| Model | " + " | ".join(TASK_NAMES) + " | Avg. |",
        "| --- | " + " | ".join(["---:"] * (len(TASK_NAMES) + 1)) + " |",
    ]

    for name, tasks in data.items():
        values = [metric(tasks[task_name]) for task_name in TASK_NAMES]
        scores = [pct(value) for value in values]
        scores.append(pct(float(np.mean(values))))
        lines.append("| " + name + " | " + " | ".join(scores) + " |")

    lines.extend(
        [
            "",
            "## Paired Bootstrap CI Sign Summary",
            "",
            "| Control / ablation | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for name, tasks in data.items():
        if name == "Full MinervaRL":
            continue
        wins, losses, ties = point_counts(tasks, full)
        ci_positive, ci_negative, ci_overlap = ci_counts(name, tasks, full, n_boot, seed)
        if ties:
            point_cell = f"{wins}/12 (+ {ties} ties)"
        else:
            point_cell = f"{wins}/12"
        lines.append(
            f"| {name} | {point_cell} | {losses}/12 | "
            f"{ci_positive}/12 | {ci_negative}/12 | {ci_overlap}/12 |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8809)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal/results/eval12_ablation_ci_table.md"),
    )
    args = parser.parse_args()
    args.out.write_text(build_table(args.boot, args.seed), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

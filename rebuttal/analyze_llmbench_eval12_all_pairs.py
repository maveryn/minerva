#!/usr/bin/env python3
"""Compute all-pair Eval12 comparisons from LLMBench row-level outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from analyze_llmbench_eval12 import (
    RUNS_ROOT,
    TASKS,
    TASK_NAMES,
    average_metric,
    bootstrap_average_delta_ci,
    ci_pct,
    load_task,
    metric,
    pct,
    validate_pair_compatibility,
)


MODEL_GROUPS = {
    "Llama-8B": OrderedDict(
        [
            ("Base", "meta-llama/Llama-3.1-8B-Instruct"),
            ("STaR", "star_llama31_8b_round0"),
            ("DART", "dart_llama31_8b_dart_best470"),
            ("GRPO", "xashru/minerva_grpo_llama8b_500_490"),
            ("LUFFY", "minerva_luffy_llama31_8b_best"),
            ("MinervaRL", "xashru/minerva_noctua_llama8b_ema_0.05"),
        ]
    ),
    "Llama-3B": OrderedDict(
        [
            ("Base", "meta-llama/Llama-3.2-3B-Instruct"),
            ("STaR", "star_llama32_3b_round0"),
            ("DART", "dart_llama32_3b_dart_best430"),
            ("GRPO", "xashru/minerva_grpo_llama3b_500"),
            ("LUFFY", "minerva_luffy_llama32_3b_best490_local"),
            ("MinervaRL", "xashru/minerva_noctua_llama3b_500"),
        ]
    ),
    "Qwen-8B": OrderedDict(
        [
            ("Base", "Qwen/Qwen3-8B-Base"),
            ("STaR", "star_qwen3_8b_base_round0"),
            ("DART", "dart_qwen3_8b_base_best490"),
            ("GRPO", "xashru/minerva_grpo_qwen8b_base"),
            ("LUFFY", "minerva_luffy_qwen3_8b_best500_local"),
            ("MinervaRL", "xashru/minerva_noctua_qwen8b_base"),
        ]
    ),
    "Qwen-4B": OrderedDict(
        [
            ("Base", "Qwen/Qwen3-4B-Base"),
            ("STaR", "star_qwen3_4b_round0"),
            ("DART", "dart_qwen3_4b_base_best460"),
            ("GRPO", "xashru/minerva_grpo_qwen4b_base"),
            ("LUFFY", "minerva_luffy_qwen3_4b_best470_local"),
            ("MinervaRL", "xashru/minerva_noctua_qwen4b_base"),
        ]
    ),
}


SUMMARY_KEYS = {
    "CKT": ("ckt_accuracy", "mcq3k_accuracy"),
    "CyberMetric": ("cybermetric_accuracy",),
    "SOCEval": ("soceval_avg_score", "threat_intel_reasoning_avg_score"),
    "RCM": ("rcm_accuracy",),
    "VSP": ("vsp_accuracy",),
    "ATE": ("ate_accuracy",),
    "RMS": ("rms_f1",),
    "ElasticRule": ("elasticrule_accuracy", "elastictoattack_accuracy"),
    "APTNER": ("aptner_f1",),
    "LANCE": ("lance_avg_f1", "prism_avg_f1"),
    "AnnoCTR": ("annoctr_avg",),
    "AZERG": ("azerg_avg",),
}


def load_model(rel_path: str) -> dict[str, object]:
    run_dir = RUNS_ROOT / rel_path
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return {
        task_name: load_task(run_dir, kind, files)
        for task_name, kind, files in TASKS
    }


def load_all() -> dict[str, OrderedDict[str, dict[str, object]]]:
    return {
        backbone: OrderedDict((name, load_model(rel_path)) for name, rel_path in models.items())
        for backbone, models in MODEL_GROUPS.items()
    }


def load_summary(rel_path: str) -> dict[str, float] | None:
    run_dir = RUNS_ROOT / rel_path
    for name in ("eval12-summary.json", "evalall-summary.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def validate_summaries(data: dict[str, OrderedDict[str, dict[str, object]]]) -> None:
    for backbone, models in MODEL_GROUPS.items():
        for model_name, rel_path in models.items():
            summary = load_summary(rel_path)
            if summary is None:
                continue
            for task_name, keys in SUMMARY_KEYS.items():
                expected = None
                for key in keys:
                    if key in summary:
                        expected = float(summary[key])
                        break
                if expected is None:
                    continue
                observed = metric(data[backbone][model_name][task_name])
                if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"{backbone} {model_name} {task_name}: computed {observed}, "
                        f"summary has {expected}"
                    )


def task_sign_counts(
    left: dict[str, object],
    right: dict[str, object],
) -> tuple[int, int, int]:
    pos = neg = zero = 0
    for task_name in TASK_NAMES:
        delta = metric(right[task_name]) - metric(left[task_name])
        if delta > 0:
            pos += 1
        elif delta < 0:
            neg += 1
        else:
            zero += 1
    return pos, neg, zero


def strongest_competitor(models: OrderedDict[str, dict[str, object]]) -> str:
    candidates = [name for name in models if name != "MinervaRL"]
    return max(candidates, key=lambda name: average_metric(models[name], TASK_NAMES))


def report(
    data: dict[str, OrderedDict[str, dict[str, object]]],
    n_boot: int,
    seed: int,
    *,
    all_pair_cis: bool = False,
) -> str:
    rng = np.random.default_rng(seed)
    validate_summaries(data)
    ci_cache: dict[tuple[str, str, str], tuple[float, float]] = {}

    def cached_ci(
        backbone: str,
        left_name: str,
        right_name: str,
        left: dict[str, object],
        right: dict[str, object],
    ) -> tuple[float, float]:
        key = (backbone, left_name, right_name)
        if key not in ci_cache:
            ci_cache[key] = bootstrap_average_delta_ci(left, right, TASK_NAMES, rng, n_boot)
        return ci_cache[key]

    lines = [
        "# LLMBench Eval12 All-Pair Comparisons",
        "",
        f"Generated with `{Path(__file__).name}` using `{n_boot}` bootstrap resamples and seed `{seed}`.",
        "",
        "All values are percentages. Delta is `right model - left model` over the 12 Eval12 paper columns.",
        "Task wins/losses count point-estimate per-task deltas across the 12 columns.",
        "",
        "## 12-Column Averages",
        "",
        "| Backbone | Model | Avg. |",
        "| --- | --- | ---: |",
    ]

    for backbone, models in data.items():
        for model_name, tasks in models.items():
            lines.append(f"| {backbone} | {model_name} | {pct(average_metric(tasks, TASK_NAMES))} |")

    lines.extend(
        [
            "",
            "## MinervaRL Against Each Baseline",
            "",
            "| Backbone | Comparison | Delta | Paired 95% CI | Task >0 | Task <0 | Task =0 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for backbone, models in data.items():
        right = models["MinervaRL"]
        for left_name in [name for name in models if name != "MinervaRL"]:
            left = models[left_name]
            for task_name in TASK_NAMES:
                validate_pair_compatibility(left[task_name], right[task_name], task_name)
            delta = average_metric(right, TASK_NAMES) - average_metric(left, TASK_NAMES)
            ci = cached_ci(backbone, left_name, "MinervaRL", left, right)
            pos, neg, zero = task_sign_counts(left, right)
            lines.append(
                f"| {backbone} | MinervaRL - {left_name} | {pct(delta)} | {ci_pct(ci)} | "
                f"{pos} | {neg} | {zero} |"
            )

    lines.extend(
        [
            "",
            "## MinervaRL Against Strongest Non-MinervaRL Baseline",
            "",
            "| Backbone | Strongest baseline | Baseline Avg. | MinervaRL Avg. | Delta | Paired 95% CI |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backbone, models in data.items():
        baseline = strongest_competitor(models)
        left = models[baseline]
        right = models["MinervaRL"]
        ci = cached_ci(backbone, baseline, "MinervaRL", left, right)
        lines.append(
            f"| {backbone} | {baseline} | {pct(average_metric(left, TASK_NAMES))} | "
            f"{pct(average_metric(right, TASK_NAMES))} | "
            f"{pct(average_metric(right, TASK_NAMES) - average_metric(left, TASK_NAMES))} | {ci_pct(ci)} |"
        )

    lines.extend(["", "## All Unordered Model Pairs", ""])
    if all_pair_cis:
        lines.extend(
            [
                "| Backbone | Comparison | Delta | Paired 95% CI | Task >0 | Task <0 | Task =0 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Backbone | Comparison | Delta | Task >0 | Task <0 | Task =0 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
    for backbone, models in data.items():
        names = list(models)
        for i, left_name in enumerate(names):
            for right_name in names[i + 1 :]:
                left = models[left_name]
                right = models[right_name]
                for task_name in TASK_NAMES:
                    validate_pair_compatibility(left[task_name], right[task_name], task_name)
                delta = average_metric(right, TASK_NAMES) - average_metric(left, TASK_NAMES)
                pos, neg, zero = task_sign_counts(left, right)
                if all_pair_cis:
                    ci = cached_ci(backbone, left_name, right_name, left, right)
                    lines.append(
                        f"| {backbone} | {right_name} - {left_name} | {pct(delta)} | {ci_pct(ci)} | "
                        f"{pos} | {neg} | {zero} |"
                    )
                else:
                    lines.append(
                        f"| {backbone} | {right_name} - {left_name} | {pct(delta)} | "
                        f"{pos} | {neg} | {zero} |"
                    )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Point estimates are recomputed from row-level `*-scored.jsonl` files and validated against saved Eval12/EvalAll summaries where present.",
            "- The main rebuttal should prioritize MinervaRL-vs-GRPO because MinervaRL is introduced as a GRPO extension.",
            "- MinervaRL-vs-Base is useful for showing post-training improvement; MinervaRL-vs-strongest-baseline is useful for aggregate leaderboard claims.",
            "- Full all-pair tables are mainly a supplementary audit, not ideal for the main rebuttal text.",
            "- By default, all unordered model pairs use point estimates only; pass `--all-pair-cis` to bootstrap every pair.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8809)
    parser.add_argument("--all-pair-cis", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal/llmbench_eval12_all_pairwise_stats.md"),
    )
    args = parser.parse_args()
    out = report(load_all(), args.boot, args.seed, all_pair_cis=args.all_pair_cis)
    args.out.write_text(out, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "dart-artifacts/dart_cti_llama8b/datasets/train/attempts_v1.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent

TASK_META = {
    "cve_to_cwe": {
        "short_name": "RCM",
        "title": "RCM: CVE -> CWE",
        "output_stub": "rcm",
    },
    "scenario_to_attack_technique": {
        "short_name": "ATE",
        "title": "ATE: Scenario -> ATT&CK Technique",
        "output_stub": "ate",
    },
    "scenario_to_attack_mitigations": {
        "short_name": "RMS",
        "title": "RMS: Scenario -> ATT&CK Mitigations",
        "output_stub": "rms",
    },
    "cve_to_cvss_v31": {
        "short_name": "CVSS",
        "title": "CVSS: CVE -> CVSS v3.1",
        "output_stub": "cvss",
    },
}

TASKS_ATE_RCM = ["scenario_to_attack_technique", "cve_to_cwe"]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dedupe_preserve_order(values):
    out = []
    seen = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def parse_target_ids(raw_ground_truth) -> list[str]:
    if isinstance(raw_ground_truth, list):
        return dedupe_preserve_order(raw_ground_truth)

    if not isinstance(raw_ground_truth, str):
        return [str(raw_ground_truth)]

    text = raw_ground_truth.strip()
    if not text:
        return ["<empty>"]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return dedupe_preserve_order(parsed)

    return [text]


def collect_question_stats(path: Path):
    per_uid: dict[str, dict[str, object]] = {}

    for row in iter_jsonl(path):
        task = row.get("task")
        if task not in TASK_META:
            continue

        uid = row["uid"]
        entry = per_uid.setdefault(
            uid,
            {
                "uid": uid,
                "task": task,
                "target_ids": parse_target_ids(row.get("ground_truth")),
                "target_ids_text": " | ".join(parse_target_ids(row.get("ground_truth"))),
                "attempt_count": 0,
                "success_count": 0,
            },
        )
        entry["attempt_count"] = int(entry["attempt_count"]) + 1
        if bool(row.get("verifier_success")):
            entry["success_count"] = int(entry["success_count"]) + 1

    return list(per_uid.values())


def aggregate_target_stats(question_rows):
    by_task_target: dict[tuple[str, str], dict[str, object]] = {}

    for row in question_rows:
        attempts = int(row["attempt_count"])
        successes = int(row["success_count"])
        for target_id in row["target_ids"]:
            key = (str(row["task"]), str(target_id))
            agg = by_task_target.setdefault(
                key,
                {
                    "task": row["task"],
                    "target_id": target_id,
                    "n_questions": 0,
                    "attempt_sum": 0,
                    "success_sum": 0,
                    "cap32_count": 0,
                },
            )
            agg["n_questions"] = int(agg["n_questions"]) + 1
            agg["attempt_sum"] = int(agg["attempt_sum"]) + attempts
            agg["success_sum"] = int(agg["success_sum"]) + successes
            agg["cap32_count"] = int(agg["cap32_count"]) + (1 if attempts == 32 else 0)

    out = []
    for agg in by_task_target.values():
        n = int(agg["n_questions"])
        out.append(
            {
                "task": agg["task"],
                "target_id": agg["target_id"],
                "n_questions": n,
                "avg_attempts": int(agg["attempt_sum"]) / n,
                "avg_successes": int(agg["success_sum"]) / n,
                "frac_cap32": int(agg["cap32_count"]) / n,
            }
        )

    out.sort(key=lambda r: (r["task"], -r["n_questions"], r["target_id"]))
    return out


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_four_tasks(task_rows, out_pdf: Path, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()
    cmap = plt.cm.magma_r
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    scatter_for_cbar = None

    for ax, task in zip(axes, TASK_META):
        rows = [r for r in task_rows if r["task"] == task]
        meta = TASK_META[task]

        xs = [r["avg_attempts"] for r in rows]
        ys = [r["avg_successes"] for r in rows]
        cs = [r["frac_cap32"] for r in rows]

        scatter_for_cbar = ax.scatter(
            xs,
            ys,
            c=cs,
            s=28,
            cmap=cmap,
            norm=norm,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.35,
        )

        ax.axvline(32, color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(1.0, color="0.75", linestyle="--", linewidth=0.9)
        ax.axhline(2.0, color="0.75", linestyle="--", linewidth=0.9)
        ax.set_title(meta["title"], fontsize=11)
        ax.set_xlim(0, 33)
        ax.set_ylim(-0.05, 2.05)
        ax.grid(True, alpha=0.18, linewidth=0.6)

    for ax in axes[2:]:
        ax.set_xlabel("Avg. attempts per question")

    axes[0].set_ylabel("Avg. verifier-successful responses")
    axes[2].set_ylabel("Avg. verifier-successful responses")

    cbar = fig.colorbar(scatter_for_cbar, ax=axes, shrink=0.92, pad=0.03)
    cbar.set_label("Fraction of questions capped at 32 attempts")

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_ate_rcm_caption(question_rows) -> str:
    return (
        "Each point is a target ID from DART-CTI v1 Llama-8B plain-generation traces. "
        "x: mean attempts per question; y: mean verifier-successful responses per question; "
        "color: fraction of questions that reached the 32-attempt cap. "
        "Dashed lines mark x=32 and y in {1, 2}."
    )


def plot_ate_rcm(task_rows, question_rows, out_pdf: Path, out_png: Path):
    selected = [task for task in TASKS_ATE_RCM if task in TASK_META]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.9), sharex=True, sharey=True)
    cmap = plt.cm.magma_r
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    scatter_for_cbar = None

    for ax, task in zip(axes, selected):
        rows = [r for r in task_rows if r["task"] == task]
        meta = TASK_META[task]

        scatter_for_cbar = ax.scatter(
            [r["avg_attempts"] for r in rows],
            [r["avg_successes"] for r in rows],
            c=[r["frac_cap32"] for r in rows],
            s=28,
            cmap=cmap,
            norm=norm,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.axvline(32, color="0.45", linestyle="--", linewidth=1.0)
        ax.axhline(1.0, color="0.75", linestyle="--", linewidth=0.9)
        ax.axhline(2.0, color="0.75", linestyle="--", linewidth=0.9)
        ax.set_title(meta["title"], fontsize=11)
        ax.set_xlim(0, 33)
        ax.set_ylim(-0.05, 2.05)
        ax.grid(True, alpha=0.18, linewidth=0.6)
        ax.set_xlabel("Avg. attempts per question")

    axes[0].set_ylabel("Avg. verifier-successful responses")

    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.24, wspace=0.16)
    cax = fig.add_axes([0.88, 0.31, 0.02, 0.50])
    cbar = fig.colorbar(scatter_for_cbar, cax=cax)
    cbar.set_label("Fraction of questions capped at 32 attempts")

    caption = build_ate_rcm_caption(question_rows)
    fig.text(0.08, 0.06, caption, ha="left", va="bottom", fontsize=9, wrap=True)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    question_rows = collect_question_stats(INPUT_PATH)
    target_rows = aggregate_target_stats(question_rows)

    write_csv(
        OUTPUT_DIR / "question_level_stats_v1_4tasks.csv",
        question_rows,
        ["uid", "task", "target_ids_text", "attempt_count", "success_count"],
    )
    write_csv(
        OUTPUT_DIR / "target_level_stats_v1_4tasks.csv",
        target_rows,
        ["task", "target_id", "n_questions", "avg_attempts", "avg_successes", "frac_cap32"],
    )

    plot_four_tasks(
        target_rows,
        OUTPUT_DIR / "reward_sparsity_4tasks_v1.pdf",
        OUTPUT_DIR / "reward_sparsity_4tasks_v1.png",
    )
    plot_ate_rcm(
        target_rows,
        question_rows,
        OUTPUT_DIR / "reward_sparsity_ate_rcm_v1.pdf",
        OUTPUT_DIR / "reward_sparsity_ate_rcm_v1.png",
    )


if __name__ == "__main__":
    main()

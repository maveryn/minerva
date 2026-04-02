#!/usr/bin/env python3
"""Generate a 2x2 PDF comparing GRPO vs Noctua zero_solve_frac__MIN curves."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "zero_solve_frac_min_2x2.pdf"
OUT_PNG_PATH = ROOT / "zero_solve_frac_min_2x2.png"

PANELS = [
    ("llama8b.csv", "Llama-3.1-8B"),
    ("llama3b.csv", "Llama-3.2-3B"),
    ("qwen4b.csv", "Qwen3-4B"),
    ("qwen8b.csv", "Qwen3-8B"),
]

COLORS = {
    "GRPO": "#d94841",
    "Noctua": "#1f78b4",
}
ROLLING_WINDOW = 5
CAPTION = (
    "Zero-solve fraction over training steps for MinervaRL and GRPO across four backbone families. "
    "Curves are smoothed with a 5-step rolling mean. Lower values indicate fewer prompts with zero "
    "correct rollouts in a group, i.e. better group-level solve coverage."
)


def load_panel(csv_path: Path) -> tuple[list[float], list[float], list[float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    grpo_col = next(
        col for col in fieldnames if "grpo" in col.lower() and "zero_solve_frac__MIN" in col
    )
    noctua_col = next(
        col for col in fieldnames if "noctua" in col.lower() and "zero_solve_frac__MIN" in col
    )

    steps: list[float] = []
    grpo_vals: list[float] = []
    noctua_vals: list[float] = []
    for row in rows:
        step_text = (row.get("Step") or "").strip()
        grpo_text = (row.get(grpo_col) or "").strip()
        noctua_text = (row.get(noctua_col) or "").strip()
        if not step_text or not grpo_text or not noctua_text:
            continue
        steps.append(float(step_text))
        grpo_vals.append(float(grpo_text))
        noctua_vals.append(float(noctua_text))

    return steps, grpo_vals, noctua_vals


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values[:]

    out: list[float] = []
    running_sum = 0.0
    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(running_sum / denom)
    return out


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, (csv_name, title) in zip(axes_flat, PANELS):
        steps, grpo_vals, noctua_vals = load_panel(ROOT / csv_name)
        grpo_smooth = rolling_mean(grpo_vals, ROLLING_WINDOW)
        noctua_smooth = rolling_mean(noctua_vals, ROLLING_WINDOW)
        ax.plot(steps, noctua_smooth, color=COLORS["Noctua"], linewidth=2.0, label="MinervaRL")
        ax.plot(steps, grpo_smooth, color=COLORS["GRPO"], linewidth=2.0, label="GRPO")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Zero-solve fraction")
        ax.grid(True, alpha=0.25, linewidth=0.8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, 0.015, CAPTION, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PNG_PATH, bbox_inches="tight", dpi=300)
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {OUT_PNG_PATH}")


if __name__ == "__main__":
    main()

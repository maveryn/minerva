#!/usr/bin/env python3
"""Plot actor/entropy for selected Llama-8B runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot entropy for Llama-8B runs.")
    parser.add_argument(
        "--input",
        default="llama8b-entropy.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        default="llama8b-entropy",
        help="Output basename (without extension)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Smoothing window for time-averaged plot (1 = no smoothing)",
    )
    parser.add_argument(
        "--smooth-method",
        choices=["rolling", "ema"],
        default="rolling",
        help="Smoothing method: rolling mean or EMA",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_base = Path(args.output)
    if not output_base.is_absolute():
        output_base = Path(__file__).resolve().parent / output_base

    df = pd.read_csv(input_path)
    if "Step" not in df.columns:
        raise ValueError("Expected a 'Step' column in the CSV")

    metric_suffix = " - actor/entropy"
    model_cols = {
        "minerva_noctua_llama_3_1_8b_instruct_lr0.05_flush_bs256_mlh_t0.7_p0.9_defer_ema": "MinervaRL",
        "minerva_grpo_llama_3_1_8b_instruct": "GRPO",
        "minerva_grpo_llama_3_1_8b_instruct_rollout12": "GRPO - 12 rollout",
    }

    series = {}
    for model_name, label in model_cols.items():
        col = f"{model_name}{metric_suffix}"
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        series[label] = df[col]

    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid", font_scale=1.1)
    except Exception as exc:
        raise RuntimeError("matplotlib/seaborn is required to plot") from exc

    fig, ax = plt.subplots(figsize=(6, 2.3))

    smooth_window = max(1, int(args.smooth_window))
    style_map = {
        "MinervaRL": "-",
        "GRPO": "--",
        "GRPO - 12 rollout": ":",
    }
    for label, values in series.items():
        if smooth_window > 1:
            if args.smooth_method == "ema":
                values = values.ewm(span=smooth_window, adjust=False).mean()
            else:
                values = values.rolling(window=smooth_window, min_periods=1).mean()
        ax.plot(df["Step"], values, label=label, linewidth=2.0, linestyle=style_map.get(label, "-"))

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Entropy", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(
        frameon=True,
        fontsize=10,
        ncol=1,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=1.0,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(output_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


if __name__ == "__main__":
    main()

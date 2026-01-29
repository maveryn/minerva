#!/usr/bin/env python3
"""Plot ACR distill fractions on a single chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _load_series(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if "Step" not in df.columns:
        raise ValueError(f"Expected a 'Step' column in {path}")
    metric_cols = [
        col for col in df.columns if col != "Step" and not col.endswith("__MIN") and not col.endswith("__MAX")
    ]
    if len(metric_cols) != 1:
        raise ValueError(f"Expected exactly one metric column in {path}, found {len(metric_cols)}")
    return df["Step"], df[metric_cols[0]]


def _extend_to_step(
    steps: pd.Series, values: pd.Series, target_step: int
) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame({"Step": pd.to_numeric(steps, errors="coerce"), "Value": values})
    if frame["Step"].isna().any():
        raise ValueError("Step column must be numeric")
    frame = frame.sort_values("Step")
    max_step = frame["Step"].iloc[-1]
    if max_step < target_step:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {"Step": [target_step], "Value": [frame["Value"].iloc[-1]]}
                ),
            ],
            ignore_index=True,
        )
    return frame["Step"], frame["Value"]


def _smooth(values: pd.Series, window: int, method: str) -> pd.Series:
    if window <= 1:
        return values
    if method == "ema":
        return values.ewm(span=window, adjust=False).mean()
    return values.rolling(window=window, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ACR distill fractions.")
    parser.add_argument(
        "--heuristic",
        default="heuristic_passed_frac.csv",
        help="CSV file for heuristic_passed_frac",
    )
    parser.add_argument(
        "--ml",
        default="ml_filter_passed_frac.csv",
        help="CSV file for ml_filter_passed_frac",
    )
    parser.add_argument(
        "--uid",
        default="uid_group_frac.csv",
        help="CSV file for uid_group_frac",
    )
    parser.add_argument(
        "--output",
        default="llama8b-acr-distill-fractions",
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

    heuristic_path = _resolve(args.heuristic)
    ml_path = _resolve(args.ml)
    uid_path = _resolve(args.uid)

    output_base = Path(args.output)
    if not output_base.is_absolute():
        output_base = Path(__file__).resolve().parent / output_base

    series = {
        "Heuristic pass": _load_series(heuristic_path),
        "ML pass": _load_series(ml_path),
        "UID coverage": _load_series(uid_path),
    }

    try:
        import seaborn as sns  # type: ignore
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid", font_scale=1.1)
    except Exception as exc:
        raise RuntimeError("matplotlib/seaborn is required to plot") from exc

    fig, ax = plt.subplots(figsize=(6, 2.3))

    smooth_window = max(1, int(args.smooth_window))
    style_map = {
        "Heuristic pass": "--",
        "ML pass": ":",
        "UID coverage": "-",
    }
    for label, (steps, values) in series.items():
        steps, values = _extend_to_step(steps, values, target_step=500)
        values = _smooth(values, smooth_window, args.smooth_method)
        ax.plot(steps, values, label=label, linewidth=2.0, linestyle=style_map.get(label, "-"))

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
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

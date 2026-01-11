#!/usr/bin/env python3
"""Plot TARBA retrieval decay as a function of rank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple


DECAY_MODES = ("exp", "reciprocal", "log", "power", "linear")


def _resolve_params(
    mode: str,
    *,
    topk: int,
    floor: float,
    alpha: Optional[float],
    beta: Optional[float],
    power: Optional[float],
    linear_c: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if topk <= 1:
        return alpha or 0.0, beta or 0.0, power or 0.0, linear_c or 0.0

    if mode == "exp":
        if alpha is None:
            alpha = math.log(1.0 / floor) / (topk - 1)
        return alpha, beta, power, linear_c
    if mode == "reciprocal":
        if beta is None:
            beta = (1.0 / floor - 1.0) / (topk - 1)
        return alpha, beta, power, linear_c
    if mode == "log":
        if beta is None:
            beta = (1.0 / floor - 1.0) / math.log(topk)
        return alpha, beta, power, linear_c
    if mode == "power":
        if power is None:
            power = math.log(1.0 / floor) / math.log(topk)
        return alpha, beta, power, linear_c
    if mode == "linear":
        if linear_c is None:
            linear_c = (1.0 - floor) / (topk - 1)
        return alpha, beta, power, linear_c
    raise ValueError(f"Unknown decay mode: {mode}")


def compute_decay_scores(
    mode: str,
    *,
    topk: int,
    floor: float,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    power: Optional[float] = None,
    linear_c: Optional[float] = None,
) -> List[float]:
    alpha, beta, power, linear_c = _resolve_params(
        mode, topk=topk, floor=floor, alpha=alpha, beta=beta, power=power, linear_c=linear_c
    )
    scores = []
    for rank in range(1, topk + 1):
        if mode == "exp":
            raw = math.exp(-float(alpha) * (rank - 1))
        elif mode == "reciprocal":
            raw = 1.0 / (1.0 + float(beta) * (rank - 1))
        elif mode == "log":
            raw = 1.0 / (1.0 + float(beta) * math.log(rank))
        elif mode == "power":
            raw = float(rank) ** (-float(power))
        elif mode == "linear":
            raw = 1.0 - float(linear_c) * (rank - 1)
        else:
            raise ValueError(f"Unknown decay mode: {mode}")
        scores.append(max(floor, raw))
    return scores


def _write_csv(path: Path, scores: List[float]) -> None:
    lines = ["rank,score"]
    for rank, score in enumerate(scores, start=1):
        lines.append(f"{rank},{score:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TARBA retrieval decay vs rank.")
    parser.add_argument(
        "--budget-b",
        type=int,
        default=8,
        help="Budget B / top-k cap to plot (default: 8).",
    )
    parser.add_argument(
        "--mode",
        choices=DECAY_MODES,
        default="linear",
        help="Decay function to plot (default: linear).",
    )
    parser.add_argument("--all", action="store_true", help="Plot all decay functions.")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Deprecated; use --budget-b.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.2,
        help="Minimum reward floor (default: 0.2).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha for exp decay (auto-computed if omitted).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta for reciprocal/log decays (auto-computed if omitted).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=None,
        help="Power for power-law decay (auto-computed if omitted).",
    )
    parser.add_argument(
        "--linear-c",
        type=float,
        default=None,
        help="Slope for linear decay (auto-computed if omitted).",
    )
    parser.add_argument(
        "--out",
        default="minerva/analysis/retrieval_decay.png",
        help="Output PNG path (default: minerva/analysis/retrieval_decay.png).",
    )
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path.")
    parser.add_argument("--show", action="store_true", help="Show the plot window if supported.")
    args = parser.parse_args()

    budget_b = args.topk if args.topk is not None else args.budget_b
    if budget_b <= 0:
        raise ValueError("--budget-b must be > 0")
    if not (0.0 < args.floor <= 1.0):
        raise ValueError("--floor must be in (0, 1].")

    modes = list(DECAY_MODES) if args.all else [args.mode]

    try:
        import matplotlib.pyplot as plt
    except Exception:
        # Fall back to printing a table if matplotlib is unavailable.
        print("matplotlib not available; printing rank -> score table instead.")
        for mode in modes:
            scores = compute_decay_scores(
                mode,
                topk=budget_b,
                floor=args.floor,
                alpha=args.alpha,
                beta=args.beta,
                power=args.power,
                linear_c=args.linear_c,
            )
            print(f"mode={mode}")
            for rank, score in enumerate(scores, start=1):
                print(f"{rank}\t{score:.8f}")
        return

    out_path = Path(args.out)
    if args.all:
        out_dir = out_path if out_path.suffix == "" else out_path.parent
    else:
        out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        scores = compute_decay_scores(
            mode,
            topk=budget_b,
            floor=args.floor,
            alpha=args.alpha,
            beta=args.beta,
            power=args.power,
            linear_c=args.linear_c,
        )
        if args.csv_out:
            csv_path = Path(args.csv_out)
            if args.all:
                csv_dir = csv_path if csv_path.suffix == "" else csv_path.parent
                csv_dir.mkdir(parents=True, exist_ok=True)
                csv_path = csv_dir / f"retrieval_decay_{mode}.csv"
            _write_csv(csv_path, scores)

        ranks = list(range(1, budget_b + 1))
        plt.figure(figsize=(6, 3.5))
        plt.plot(ranks, scores, marker="o", linewidth=2)
        plt.title(f"TARBA Retrieval Decay vs Rank ({mode})")
        plt.xlabel("Retrieval Rank")
        plt.ylabel("Per-label Score (floored)")
        plt.xticks(ranks)
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.all:
            out_file = out_dir / f"retrieval_decay_{mode}.png"
        else:
            out_file = out_path
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        if args.show and not args.all:
            plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize pairwise judge results and render a triangular comparison plot."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MODEL_LABELS = {
    "llama-3-8B": "Base",
    "llama3-primus": "Primus",
    "llama3-sec": "Sec",
    "minerva_llama8b_grpo": "GRPO",
    "minerva_llama8b_noctua": "MinervaRL",
}


@dataclass
class PairStats:
    model_1: str
    model_2: str
    total_valid: int = 0
    wins_1: int = 0
    wins_2: int = 0
    ties: int = 0
    parse_errors: int = 0

    @property
    def total_seen(self) -> int:
        return self.total_valid + self.parse_errors

    def win_rate_1(self) -> float:
        return self.wins_1 / self.total_valid if self.total_valid else 0.0

    def win_rate_2(self) -> float:
        return self.wins_2 / self.total_valid if self.total_valid else 0.0

    def tie_rate(self) -> float:
        return self.ties / self.total_valid if self.total_valid else 0.0


def normalize_winner(row: Dict[str, Any], model_a: str, model_b: str) -> str:
    winner_model = row.get("winner_model")
    if isinstance(winner_model, str):
        wl = winner_model.strip().lower()
        if wl == "tie":
            return "tie"
        if wl == "parse_error":
            return "parse_error"
        if winner_model == model_a:
            return model_a
        if winner_model == model_b:
            return model_b

    winner = row.get("winner")
    if isinstance(winner, str):
        w = winner.strip().upper()
        if w == "A":
            return model_a
        if w == "B":
            return model_b
        if w in {"TIE", "T"}:
            return "tie"
        if w == "PARSE_ERROR":
            return "parse_error"

    return "parse_error"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def summarize(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], PairStats], List[str]]:
    stats: Dict[Tuple[str, str], PairStats] = {}
    models: set[str] = set()

    for row in rows:
        model_a = row.get("model_a")
        model_b = row.get("model_b")
        if not isinstance(model_a, str) or not isinstance(model_b, str):
            continue
        models.add(model_a)
        models.add(model_b)
        key = build_pair_key(model_a, model_b)
        if key not in stats:
            stats[key] = PairStats(model_1=key[0], model_2=key[1])

        outcome = normalize_winner(row, model_a, model_b)
        if outcome == "parse_error":
            stats[key].parse_errors += 1
            continue
        stats[key].total_valid += 1
        if outcome == "tie":
            stats[key].ties += 1
        elif outcome == key[0]:
            stats[key].wins_1 += 1
        elif outcome == key[1]:
            stats[key].wins_2 += 1

    return stats, sorted(models)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def compute_overall_scores(stats: Dict[Tuple[str, str], PairStats]) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    wins: Dict[str, int] = {}
    for pair, s in stats.items():
        totals[pair[0]] = totals.get(pair[0], 0) + s.total_valid
        totals[pair[1]] = totals.get(pair[1], 0) + s.total_valid
        wins[pair[0]] = wins.get(pair[0], 0) + s.wins_1
        wins[pair[1]] = wins.get(pair[1], 0) + s.wins_2
    scores = {}
    for model, total in totals.items():
        scores[model] = wins.get(model, 0) / total if total else 0.0
    return scores


def plot_heatmap(
    output_paths: List[Path],
    stats: Dict[Tuple[str, str], PairStats],
    models: List[str],
    labels: List[str],
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    try:
        import seaborn as sns  # type: ignore

        sns.set_theme(style="white", font_scale=0.9)
        use_seaborn = True
    except Exception:
        sns = None  # type: ignore
        use_seaborn = False

    n = len(models)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.2)))

    matrix = np.full((n, n), np.nan, dtype=float)
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i == j:
                continue
            if j < i:
                continue
            key = build_pair_key(model_i, model_j)
            s = stats.get(key)
            if not s or s.total_valid == 0:
                continue
            if model_i == s.model_1:
                win_i = s.wins_1
                win_j = s.wins_2
            else:
                win_i = s.wins_2
                win_j = s.wins_1
            denom = win_i + win_j
            if denom == 0:
                continue
            p = win_i / denom
            matrix[i, j] = p
            matrix[j, i] = 1.0 - p

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Model B")
    ax.set_ylabel("Model A")

    if use_seaborn and sns is not None:
        mask = np.isnan(matrix)
        sns.heatmap(
            matrix,
            mask=mask,
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.6,
            linecolor="#e0e0e0",
            cbar_kws={"label": "P(Model A > Model B)", "shrink": 0.75},
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")
    else:
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("RdBu_r")
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xticks([i - 0.5 for i in range(1, n)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(1, n)], minor=True)
        ax.grid(which="minor", color="#e0e0e0", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.75)
        cbar.set_label("P(Model A > Model B)")

    fig.tight_layout(pad=0.2)
    for path in output_paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pairwise judge results.")
    parser.add_argument(
        "--input",
        default="pairwise_judge_results.jsonl",
        help="Input JSONL file with judge results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write summary outputs",
    )
    parser.add_argument(
        "--model-order",
        default=None,
        help="Comma-separated model order override",
    )
    parser.add_argument(
        "--title",
        default="Pairwise CTI Preference (A vs B)",
        help="Plot title",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    ensure_dir(output_dir)

    stats, models = summarize(iter_jsonl(input_path))

    if args.model_order:
        order = [m.strip() for m in args.model_order.split(",") if m.strip()]
        models = order + [m for m in models if m not in order]
    else:
        # stable order by overall win rate, then name
        scores = compute_overall_scores(stats)
        models = sorted(models, key=lambda m: (-scores.get(m, 0.0), m))

    labels = [MODEL_LABELS.get(m, m) for m in models]

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(stats.keys()):
        s = stats[key]
        row = {
            "model_1": s.model_1,
            "model_2": s.model_2,
            "total_valid": s.total_valid,
            "wins_model_1": s.wins_1,
            "wins_model_2": s.wins_2,
            "ties": s.ties,
            "parse_errors": s.parse_errors,
            "win_rate_model_1": round(s.win_rate_1(), 6),
            "win_rate_model_2": round(s.win_rate_2(), 6),
            "tie_rate": round(s.tie_rate(), 6),
        }
        summary_rows.append(row)

    write_json(output_dir / "pairwise_summary.json", summary_rows)
    write_csv(output_dir / "pairwise_summary.csv", summary_rows)

    # Print a compact table
    for row in summary_rows:
        print(
            f"{row['model_1']} vs {row['model_2']}: "
            f"{row['wins_model_1']}/{row['wins_model_2']}/" 
            f"{row['ties']} (A/B/tie) over {row['total_valid']}"
        )

    if not args.no_plot:
        plot_paths = [
            output_dir / "pairwise_heatmap.png",
            output_dir / "pairwise_heatmap.pdf",
        ]
        ok = plot_heatmap(plot_paths, stats, models, labels, args.title)
        if not ok:
            print("matplotlib/numpy not available; skipping plot generation")
        else:
            print("Wrote plot to:")
            for path in plot_paths:
                print(f"  - {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot hint vs no-hint accuracy bars for multiple models from *_predictions_passk.jsonl.

For each JSONL:
  • Deduplicate by (sample_idx, k) keeping the LAST record (safety, mirrors your patch script).
  • Split rows into 'hint' vs 'no_hint' by presence of "Hint:" in prompt_text or messages[].content.
  • Verify equal number of unique prompts (sample_idx) between hint/no_hint.
  • Compute average accuracies across all k:
        answer_acc = mean(answer_match)
        ids_acc    = mean(ids_match over rows where ids_match is not None)
  • Produce two grouped bar charts (answer %, ids %): one pair of bars per model.

Usage:
  python plot_hint_effect.py  \
      --out_dir figs_hint --dpi 300 --save_pdf  \
      path/to/modelA_predictions_passk.jsonl \
      path/to/modelB_predictions_passk.jsonl ...

Outputs (in out_dir):
  hint_answer_accuracy_bar.(png|pdf)
  hint_ids_accuracy_bar.(png|pdf)
  hint_accuracy_summary.csv
"""

import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


# ------------------------------ I/O helpers ------------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedup_last_by_k_sample(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the LAST occurrence for each (sample_idx, k), sort by (k, sample_idx)."""
    last_idx = {}
    for i, r in enumerate(rows):
        try:
            key = (int(r.get("sample_idx", 0)), int(r.get("k", 1)))
        except Exception:
            continue
        last_idx[key] = i
    keep = set(last_idx.values())
    dedup = [r for i, r in enumerate(rows) if i in keep]

    def _key(r):
        try:
            return (int(r.get("k", 1)), int(r.get("sample_idx", 0)))
        except Exception:
            return (10**9, 10**9)

    dedup.sort(key=_key)
    return dedup


# ------------------------------ hint detection ------------------------------

_HINT_RE = re.compile(r"\bhint\s*:", re.IGNORECASE)

def has_hint(r: Dict[str, Any]) -> bool:
    """Detect presence of 'Hint:' either in prompt_text or user message content."""
    pt = r.get("prompt_text") or ""
    if isinstance(pt, str) and _HINT_RE.search(pt):
        return True
    msgs = r.get("messages", [])
    for m in msgs or []:
        if isinstance(m, dict):
            c = m.get("content") or ""
            if isinstance(c, str) and _HINT_RE.search(c):
                return True
    return False


# ------------------------------ metric computation ------------------------------

def avg_accuracies_across_k(rows: List[Dict[str, Any]]) -> Tuple[float, Optional[float]]:
    """
    Average instantaneous correctness across all k (i.e., mean over rows):
      answer_acc = mean(answer_match)
      ids_acc    = mean(ids_match over rows where ids_match is not None)
    """
    if not rows:
        return 0.0, None

    ans_flags = [bool(r.get("answer_match", False)) for r in rows]
    ans_acc = float(np.mean(ans_flags)) if ans_flags else 0.0

    ids_vals = [r.get("ids_match", None) for r in rows]
    ids_masked = [bool(v) for v in ids_vals if v is not None]
    ids_acc = float(np.mean(ids_masked)) if ids_masked else None

    return ans_acc, ids_acc


# ------------------------------ pretty plotting ------------------------------

BOX_LW = 1.2

def _boxify(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(BOX_LW)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

def _annotate_bars(ax, pct: bool = True):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        text = f"{h*100:.2f}%" if pct else f"{h:.3f}"
        ax.annotate(text, (p.get_x() + p.get_width()/2.0, h),
                    ha="center", va="bottom", fontsize=11,
                    xytext=(0, 4), textcoords="offset points")


# ------------------------------ main logic ------------------------------

def pretty_model_label(path: str) -> str:
    base = Path(path).name
    if base.endswith("_predictions_passk.jsonl"):
        base = base[:-len("_predictions_passk.jsonl")]
    return base  # keep simple & informative


def process_file(path: str) -> Dict[str, Any]:
    rows = dedup_last_by_k_sample(read_jsonl(path))

    # Partition by hint
    rows_hint = [r for r in rows if has_hint(r)]
    rows_no   = [r for r in rows if not has_hint(r)]

    # Verify equal prompts (unique sample_idx) per split
    uniq_hint = {int(r.get("sample_idx", 0)) for r in rows_hint}
    uniq_no   = {int(r.get("sample_idx", 0)) for r in rows_no}
    if len(uniq_hint) != len(uniq_no):
        raise ValueError(
            f"[{path}] Mismatch in unique prompt count: "
            f"hint={len(uniq_hint)} vs no_hint={len(uniq_no)}"
        )

    # Compute average accuracies across all k
    ans_no,  ids_no  = avg_accuracies_across_k(rows_no)
    ans_hint, ids_hint = avg_accuracies_across_k(rows_hint)

    return {
        "model": pretty_model_label(path),
        "n_samples": len(uniq_no),
        "k_levels": len(rows_no) // max(len(uniq_no), 1),
        "answer_no_hint": ans_no,
        "answer_hint": ans_hint,
        "ids_no_hint": ids_no,
        "ids_hint": ids_hint,
    }


def make_barplots(results: List[Dict[str, Any]],
                  out_dir: str = "figs_hint",
                  dpi: int = 300,
                  save_pdf: bool = True) -> Dict[str, str]:

    os.makedirs(out_dir, exist_ok=True)

    # Build a tidy DF for plotting
    recs = []
    for r in results:
        recs.append({"model": r["model"], "metric": "Answer", "hint": "No hint", "value": r["answer_no_hint"]})
        recs.append({"model": r["model"], "metric": "Answer", "hint": "Hint",    "value": r["answer_hint"]})
        if r["ids_no_hint"] is not None:
            recs.append({"model": r["model"], "metric": "IDs",    "hint": "No hint", "value": r["ids_no_hint"]})
        if r["ids_hint"] is not None:
            recs.append({"model": r["model"], "metric": "IDs",    "hint": "Hint",    "value": r["ids_hint"]})

    df = pd.DataFrame(recs)
    models = list(dict.fromkeys(df["model"]))  # preserve order
    n_models = len(models)

    # Styling
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("tab10", n_colors=2)

    def _figsize(nm: int) -> Tuple[float, float]:
        return (max(7.2, 1.8*nm + 1.0), 4.8)

    outputs = {}

    # 1) Answer accuracy
    dfa = df[df["metric"] == "Answer"].copy()
    fig1, ax1 = plt.subplots(figsize=_figsize(n_models))
    sns.barplot(
        data=dfa, x="model", y="value", hue="hint",
        ax=ax1, palette=palette, edgecolor="black", linewidth=0.8
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Answer accuracy (%)")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(0, min(1.0, float(dfa["value"].max()) * 1.25 if len(dfa) else 1.0))
    _boxify(ax1)
    _annotate_bars(ax1, pct=True)
    ax1.legend(title="", frameon=True, ncol=2, loc="upper left", bbox_to_anchor=(0,1.02))
    plt.tight_layout()
    out_png1 = os.path.join(out_dir, "hint_answer_accuracy_bar.png")
    out_pdf1 = os.path.join(out_dir, "hint_answer_accuracy_bar.pdf")
    fig1.savefig(out_png1, dpi=dpi)
    if save_pdf:
        fig1.savefig(out_pdf1)
    plt.close(fig1)
    outputs["answer_png"] = out_png1
    if save_pdf:
        outputs["answer_pdf"] = out_pdf1

    # 2) IDs accuracy (skip if empty)
    dfi = df[df["metric"] == "IDs"].copy()
    if not dfi.empty:
        fig2, ax2 = plt.subplots(figsize=_figsize(n_models))
        sns.barplot(
            data=dfi, x="model", y="value", hue="hint",
            ax=ax2, palette=palette, edgecolor="black", linewidth=0.8
        )
        ax2.set_xlabel("")
        ax2.set_ylabel("IDs accuracy (%)")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.set_ylim(0, min(1.0, float(dfi["value"].max()) * 1.25))
        _boxify(ax2)
        _annotate_bars(ax2, pct=True)
        ax2.legend(title="", frameon=True, ncol=2, loc="upper left", bbox_to_anchor=(0,1.02))
        plt.tight_layout()
        out_png2 = os.path.join(out_dir, "hint_ids_accuracy_bar.png")
        out_pdf2 = os.path.join(out_dir, "hint_ids_accuracy_bar.pdf")
        fig2.savefig(out_png2, dpi=dpi)
        if save_pdf:
            fig2.savefig(out_pdf2)
        plt.close(fig2)
        outputs["ids_png"] = out_png2
        if save_pdf:
            outputs["ids_pdf"] = out_pdf2

    # Save CSV summary too
    summary_csv = os.path.join(out_dir, "hint_accuracy_summary.csv")
    (pd.DataFrame(results)).to_csv(summary_csv, index=False)
    outputs["summary_csv"] = summary_csv

    return outputs


def main():
    ap = argparse.ArgumentParser(description="Hint vs no-hint accuracy bar plots for pass@k predictions JSONL.")
    ap.add_argument("jsonl_paths", nargs="+", help="Paths to *_predictions_passk.jsonl files.")
    ap.add_argument("--out_dir", default="figs_hint")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--save_pdf", action="store_true", help="Also save PDF versions.")
    args = ap.parse_args()

    results = []
    for p in args.jsonl_paths:
        r = process_file(p)
        print(f"[OK] {r['model']}: N={r['n_samples']}, K={r['k_levels']}, "
              f"Ans no-hint={r['answer_no_hint']:.4f}, Ans hint={r['answer_hint']:.4f}, "
              f"IDs no-hint={('NA' if r['ids_no_hint'] is None else f'{r['ids_no_hint']:.4f}')}, "
              f"IDs hint={('NA' if r['ids_hint'] is None else f'{r['ids_hint']:.4f}')}")

        results.append(r)

    outs = make_barplots(results, out_dir=args.out_dir, dpi=args.dpi, save_pdf=args.save_pdf)
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

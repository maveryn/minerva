#!/usr/bin/env python3
"""Summarize and plot 7-model GPT pairwise preferences from pointwise scores."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT.parent / "response_quality_eval"
DATASET_DIR = EVAL_ROOT / "full_gpt_eval_dataset"
OUTPUT_DIR = EVAL_ROOT / "judge_outputs" / "full_gpt_eval_dataset"

SCORE_PATH = OUTPUT_DIR / "gpt52_scores.jsonl"
KEY_PATH = DATASET_DIR / "key.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "gpt52_pairwise_summary.json"
MATRIX_CSV_PATH = OUTPUT_DIR / "gpt52_pairwise_matrix.csv"

PDF_EXCL = ROOT / "pairwise_preference_7models_excluding_ties.pdf"
PDF_INCL = ROOT / "pairwise_preference_7models_including_ties.pdf"
PDF_BAR = ROOT / "pairwise_preference_7models_aggregate_wins.pdf"
PDF_GRAPH = ROOT / "pairwise_preference_7models_tournament_graph.pdf"


DISPLAY_NAMES = {
    "llama3-sec-reasoning": "Sec8B-Reasoning",
    "llama3-8b-dart": "Llama8B-DART",
    "minerva_llama8b_noctua": "Llama8B-MinervaRL",
    "llama-3-8B": "Llama8B-Base",
    "llama3-8b-star": "Llama8B-STaR",
    "llama3-sec": "Foundation-Sec-8B",
    "minerva_llama8b_grpo": "Llama-3.1-8B-GRPO",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_records() -> tuple[list[dict], list[str], int]:
    key_rows = read_jsonl(KEY_PATH)
    score_rows = read_jsonl(SCORE_PATH)
    key_by_item = {row["item_id"]: row for row in key_rows}

    records: list[dict] = []
    for score in score_rows:
        key = key_by_item.get(score["item_id"])
        if key is None:
            continue
        merged = {
            **score,
            "prompt_id": key["prompt_id"],
            "model_name": key["model_name"],
            "task_label": key["task_label"],
        }
        records.append(merged)

    all_models = sorted({row["model_name"] for row in key_rows})
    all_prompts = {row["prompt_id"] for row in key_rows}
    return records, all_models, len(all_prompts)


def summarize(records: list[dict], all_models: list[str], expected_prompt_count: int) -> tuple[dict, list[dict]]:
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    model_rows: dict[str, list[dict]] = defaultdict(list)
    matrix_rows: list[dict] = []

    for row in records:
        by_prompt[row["prompt_id"]][row["model_name"]] = row
        model_rows[row["model_name"]].append(row)

    complete_prompts = [
        prompt_id for prompt_id, prompt_rows in by_prompt.items() if len(prompt_rows) == len(all_models)
    ]
    complete_prompts.sort()

    model_average_scores: dict[str, dict[str, float]] = {}
    for model in all_models:
        rows = [r for r in model_rows[model] if r["prompt_id"] in set(complete_prompts)]
        model_average_scores[model] = {
            "n": len(rows),
            "avg_total_score": mean([float(r["total_score"]) for r in rows]),
            "avg_writing_quality_score": mean([float(r["writing_quality_score"]) for r in rows]),
            "avg_evidence_use_score": mean([float(r["evidence_use_score"]) for r in rows]),
            "avg_cti_concept_focus_score": mean([float(r["cti_concept_focus_score"]) for r in rows]),
        }

    pairwise: dict[str, dict[str, float | int | str]] = {}
    aggregate_prompt_level_pairwise: dict[str, dict[str, int]] = {
        model: {"wins": 0, "losses": 0, "ties": 0} for model in all_models
    }

    complete_prompt_set = set(complete_prompts)
    for model_a, model_b in combinations(all_models, 2):
        wins_a = 0
        wins_b = 0
        ties = 0
        diffs: list[float] = []

        for prompt_id in complete_prompts:
            row_a = by_prompt[prompt_id][model_a]
            row_b = by_prompt[prompt_id][model_b]
            total_a = float(row_a["total_score"])
            total_b = float(row_b["total_score"])
            diffs.append(total_a - total_b)
            if total_a > total_b:
                wins_a += 1
            elif total_b > total_a:
                wins_b += 1
            else:
                ties += 1

        denom_excl_ties = wins_a + wins_b
        p_a_gt_b_excl_ties = wins_a / denom_excl_ties if denom_excl_ties else 0.5
        p_a_gt_b_incl_ties = wins_a / len(complete_prompts) if complete_prompts else 0.0
        mean_diff = mean(diffs)
        pair_key = f"{model_a}|{model_b}"

        pairwise[pair_key] = {
            "model_a": model_a,
            "model_b": model_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "p_a_gt_b_excl_ties": p_a_gt_b_excl_ties,
            "p_a_gt_b_incl_ties": p_a_gt_b_incl_ties,
            "mean_score_diff_a_minus_b": mean_diff,
        }

        aggregate_prompt_level_pairwise[model_a]["wins"] += wins_a
        aggregate_prompt_level_pairwise[model_a]["losses"] += wins_b
        aggregate_prompt_level_pairwise[model_a]["ties"] += ties
        aggregate_prompt_level_pairwise[model_b]["wins"] += wins_b
        aggregate_prompt_level_pairwise[model_b]["losses"] += wins_a
        aggregate_prompt_level_pairwise[model_b]["ties"] += ties

        matrix_rows.extend(
            [
                {
                    "row_model": model_a,
                    "column_model": model_b,
                    "wins_row": wins_a,
                    "wins_col": wins_b,
                    "ties": ties,
                    "p_row_gt_col_excl_ties": p_a_gt_b_excl_ties,
                    "p_row_gt_col_incl_ties": p_a_gt_b_incl_ties,
                    "mean_score_diff_row_minus_col": mean_diff,
                },
                {
                    "row_model": model_b,
                    "column_model": model_a,
                    "wins_row": wins_b,
                    "wins_col": wins_a,
                    "ties": ties,
                    "p_row_gt_col_excl_ties": (wins_b / denom_excl_ties) if denom_excl_ties else 0.5,
                    "p_row_gt_col_incl_ties": wins_b / len(complete_prompts) if complete_prompts else 0.0,
                    "mean_score_diff_row_minus_col": -mean_diff,
                },
            ]
        )

    head_to_head_match_results: dict[str, int] = {model: 0 for model in all_models}
    for pair_key, stats in pairwise.items():
        model_a = stats["model_a"]
        model_b = stats["model_b"]
        wins_a = int(stats["wins_a"])
        wins_b = int(stats["wins_b"])
        if wins_a > wins_b:
            head_to_head_match_results[model_a] += 1
        elif wins_b > wins_a:
            head_to_head_match_results[model_b] += 1

    ranking_by_prompt_level_wins = sorted(
        all_models,
        key=lambda model: (
            aggregate_prompt_level_pairwise[model]["wins"] - aggregate_prompt_level_pairwise[model]["losses"],
            aggregate_prompt_level_pairwise[model]["wins"],
            model_average_scores[model]["avg_total_score"],
        ),
        reverse=True,
    )
    ranking_by_average_total_score = sorted(
        all_models,
        key=lambda model: model_average_scores[model]["avg_total_score"],
        reverse=True,
    )
    ranking_by_head_to_head_match_results = sorted(
        all_models,
        key=lambda model: (
            head_to_head_match_results[model],
            aggregate_prompt_level_pairwise[model]["wins"] - aggregate_prompt_level_pairwise[model]["losses"],
            model_average_scores[model]["avg_total_score"],
        ),
        reverse=True,
    )

    summary = {
        "score_file": str(SCORE_PATH),
        "model_key_file": str(KEY_PATH),
        "prompt_count": len(complete_prompts),
        "expected_prompt_count": expected_prompt_count,
        "response_count": len(records),
        "models": all_models,
        "model_average_scores": model_average_scores,
        "aggregate_prompt_level_pairwise": aggregate_prompt_level_pairwise,
        "head_to_head_match_results": head_to_head_match_results,
        "ranking_by_prompt_level_wins": ranking_by_prompt_level_wins,
        "ranking_by_average_total_score": ranking_by_average_total_score,
        "ranking_by_head_to_head_match_results": ranking_by_head_to_head_match_results,
        "pairwise": pairwise,
    }
    return summary, matrix_rows


def build_matrix(summary: dict, include_ties: bool) -> pd.DataFrame:
    ordered_models = list(summary["ranking_by_prompt_level_wins"])
    prompt_count = int(summary["prompt_count"])
    pairwise = summary["pairwise"]
    value_df = pd.DataFrame(index=ordered_models, columns=ordered_models, dtype=float)

    for row_model in ordered_models:
        for col_model in ordered_models:
            if row_model == col_model:
                value_df.loc[row_model, col_model] = 0.5
                continue

            key_ab = f"{row_model}|{col_model}"
            key_ba = f"{col_model}|{row_model}"
            if key_ab in pairwise:
                stat = pairwise[key_ab]
                wins = int(stat["wins_a"])
                losses = int(stat["wins_b"])
            else:
                stat = pairwise[key_ba]
                wins = int(stat["wins_b"])
                losses = int(stat["wins_a"])

            denom = prompt_count if include_ties else wins + losses
            value_df.loc[row_model, col_model] = (wins / denom) if denom else 0.5

    return value_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)


def plot_heatmap(value_df: pd.DataFrame, title: str, subtitle: str, pdf_path: Path) -> None:
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11.4, 9.0))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)

    sns.heatmap(
        value_df,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.8,
        linecolor="#f2efe8",
        square=True,
        annot=False,
        cbar_kws={"label": "P(row model preferred over column model)"},
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=14, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_bars(summary: dict, pdf_path: Path) -> None:
    ordered_models = list(summary["ranking_by_prompt_level_wins"])
    aggregate = summary["aggregate_prompt_level_pairwise"]
    model_avg = summary["model_average_scores"]

    df = pd.DataFrame(
        [
            {
                "model": DISPLAY_NAMES[m],
                "wins": aggregate[m]["wins"],
                "losses": aggregate[m]["losses"],
                "ties": aggregate[m]["ties"],
                "avg_total": model_avg[m]["avg_total_score"],
            }
            for m in ordered_models
        ]
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    y = range(len(df))
    colors = {"wins": "#3a7d44", "ties": "#b8b3a7", "losses": "#b44b4b"}

    ax.barh(y, df["wins"], color=colors["wins"], label="Wins")
    ax.barh(y, df["ties"], left=df["wins"], color=colors["ties"], label="Ties")
    ax.barh(y, df["losses"], left=df["wins"] + df["ties"], color=colors["losses"], label="Losses")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate prompt-level pairwise outcomes across all opponents")
    ax.set_title(
        "Aggregate GPT Pairwise Outcomes Across 7 Models\n"
        "Each model is compared against all six others on the same 350 prompts",
        fontsize=14,
        pad=16,
    )
    ax.legend(loc="lower right", frameon=True)

    total = int(df.loc[0, ["wins", "ties", "losses"]].sum())
    for idx, row in df.iterrows():
        ax.text(
            row["wins"] + row["ties"] + row["losses"] + 12,
            idx,
            f"avg={row['avg_total']:.2f}",
            va="center",
            fontsize=9,
            color="#3d3428",
        )
    ax.set_xlim(0, total + 150)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_tournament_graph(summary: dict, pdf_path: Path) -> None:
    ordered_models = list(summary["ranking_by_prompt_level_wins"])
    pairwise = summary["pairwise"]
    model_avg = summary["model_average_scores"]

    graph = nx.DiGraph()
    for model in ordered_models:
        graph.add_node(model, avg_total=model_avg[model]["avg_total_score"])

    for i, model_a in enumerate(ordered_models):
        for model_b in ordered_models[i + 1 :]:
            key_ab = f"{model_a}|{model_b}"
            key_ba = f"{model_b}|{model_a}"
            if key_ab in pairwise:
                stat = pairwise[key_ab]
                wins_a = int(stat["wins_a"])
                wins_b = int(stat["wins_b"])
                p = float(stat["p_a_gt_b_excl_ties"])
            else:
                stat = pairwise[key_ba]
                wins_a = int(stat["wins_b"])
                wins_b = int(stat["wins_a"])
                p = 1.0 - float(stat["p_a_gt_b_excl_ties"])
            if wins_a > wins_b:
                winner, loser, strength = model_a, model_b, p
            else:
                winner, loser, strength = model_b, model_a, 1.0 - p
            graph.add_edge(winner, loser, strength=strength, margin=abs(wins_a - wins_b))

    positions = {model: (index, 0.0) for index, model in enumerate(ordered_models)}
    node_labels = {model: DISPLAY_NAMES[model] for model in ordered_models}
    node_colors = [model_avg[m]["avg_total_score"] for m in ordered_models]

    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(15.0, 5.0))
    ax.set_title(
        "Directed GPT Tournament Graph Across 7 Models\n"
        "Arrow points from the head-to-head winner to the loser; darker edges indicate stronger preference",
        fontsize=14,
        pad=16,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=ordered_models,
        node_color=node_colors,
        cmap=plt.cm.YlGnBu,
        node_size=2450,
        edgecolors="#2f2a24",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, labels=node_labels, font_size=10, ax=ax)

    edges = list(graph.edges(data=True))
    edge_colors = [data["strength"] for _, _, data in edges]
    edge_widths = [1.0 + 5.0 * max(0.0, data["strength"] - 0.5) for _, _, data in edges]

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[(u, v) for u, v, _ in edges],
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        edge_vmin=0.5,
        edge_vmax=1.0,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.11",
        alpha=0.7,
        ax=ax,
    )

    for index, model in enumerate(ordered_models):
        ax.text(index, -0.28, f"rank {index + 1}", ha="center", va="top", fontsize=9, color="#5d5243")

    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records, all_models, expected_prompt_count = load_records()
    summary, matrix_rows = summarize(records, all_models, expected_prompt_count)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SUMMARY_PATH, summary)
    write_csv(MATRIX_CSV_PATH, matrix_rows)

    excl_values = build_matrix(summary, include_ties=False)
    plot_heatmap(
        excl_values,
        title="Pairwise GPT Preference Across 7 Models",
        subtitle="Excluding ties from the denominator",
        pdf_path=PDF_EXCL,
    )

    incl_values = build_matrix(summary, include_ties=True)
    plot_heatmap(
        incl_values,
        title="Pairwise GPT Preference Across 7 Models",
        subtitle="Including ties in the denominator",
        pdf_path=PDF_INCL,
    )

    plot_aggregate_bars(summary, PDF_BAR)
    plot_tournament_graph(summary, PDF_GRAPH)

    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MATRIX_CSV_PATH}")
    print(f"Wrote {PDF_EXCL}")
    print(f"Wrote {PDF_INCL}")
    print(f"Wrote {PDF_BAR}")
    print(f"Wrote {PDF_GRAPH}")


if __name__ == "__main__":
    main()

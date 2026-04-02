#!/usr/bin/env python3
"""Plot a restricted 5-model GPT pairwise preference view from the full 7-model summary."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = (
    ROOT.parent
    / "response_quality_eval"
    / "judge_outputs"
    / "full_gpt_eval_dataset"
    / "gpt52_pairwise_summary.json"
)
SUBSET_SUMMARY_PATH = (
    ROOT.parent
    / "response_quality_eval"
    / "judge_outputs"
    / "full_gpt_eval_dataset"
    / "gpt52_pairwise_summary_5models.json"
)
SUBSET_MATRIX_PATH = (
    ROOT.parent
    / "response_quality_eval"
    / "judge_outputs"
    / "full_gpt_eval_dataset"
    / "gpt52_pairwise_matrix_5models.csv"
)

PDF_EXCL = ROOT / "pairwise_preference_5models_excluding_ties.pdf"
PDF_INCL = ROOT / "pairwise_preference_5models_including_ties.pdf"
PDF_BAR = ROOT / "pairwise_preference_5models_aggregate_wins.pdf"
PDF_GRAPH = ROOT / "pairwise_preference_5models_tournament_graph.pdf"

SELECTED_MODELS = [
    "llama-3-8B",
    "minerva_llama8b_grpo",
    "minerva_llama8b_noctua",
    "llama3-8b-star",
    "llama3-8b-dart",
]

DISPLAY_NAMES = {
    "llama-3-8B": "Llama8B-Base",
    "minerva_llama8b_grpo": "Llama-3.1-8B-GRPO",
    "minerva_llama8b_noctua": "Llama-3.1-8B-MinervaRL",
    "llama3-8b-star": "Llama8B-STaR",
    "llama3-8b-dart": "Llama8B-DART",
}
PAIRWISE_TITLE = "Pairwise GPT Preference on Llama-3.1-8B"
AGGREGATE_TITLE = "Aggregate GPT Pairwise Outcomes on Llama-3.1-8B"
TOURNAMENT_TITLE = "Directed GPT Tournament Graph on Llama-3.1-8B"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def extract_pair(full_summary: dict, model_a: str, model_b: str) -> dict:
    pairwise = full_summary["pairwise"]
    key_ab = f"{model_a}|{model_b}"
    key_ba = f"{model_b}|{model_a}"
    if key_ab in pairwise:
        return pairwise[key_ab]
    if key_ba in pairwise:
        stat = pairwise[key_ba]
        return {
            "model_a": model_a,
            "model_b": model_b,
            "wins_a": int(stat["wins_b"]),
            "wins_b": int(stat["wins_a"]),
            "ties": int(stat["ties"]),
            "p_a_gt_b_excl_ties": 1.0 - float(stat["p_a_gt_b_excl_ties"]),
            "mean_score_diff_a_minus_b": -float(stat["mean_score_diff_a_minus_b"]),
        }
    raise KeyError(f"Missing pairwise entry for {model_a} vs {model_b}")


def build_subset_summary(full_summary: dict) -> tuple[dict, list[dict]]:
    models = [model for model in SELECTED_MODELS if model in full_summary["models"]]
    prompt_count = int(full_summary["prompt_count"])
    model_average_scores = {
        model: full_summary["model_average_scores"][model]
        for model in models
    }

    aggregate_prompt_level_pairwise = {
        model: {"wins": 0, "losses": 0, "ties": 0} for model in models
    }
    pairwise: dict[str, dict] = {}
    matrix_rows: list[dict] = []

    for model_a, model_b in combinations(models, 2):
        stat = extract_pair(full_summary, model_a, model_b)
        pair_key = f"{model_a}|{model_b}"
        pairwise[pair_key] = stat

        wins_a = int(stat["wins_a"])
        wins_b = int(stat["wins_b"])
        ties = int(stat["ties"])

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
                    "p_row_gt_col_excl_ties": float(stat["p_a_gt_b_excl_ties"]),
                    "p_row_gt_col_incl_ties": wins_a / prompt_count,
                    "mean_score_diff_row_minus_col": float(stat["mean_score_diff_a_minus_b"]),
                },
                {
                    "row_model": model_b,
                    "column_model": model_a,
                    "wins_row": wins_b,
                    "wins_col": wins_a,
                    "ties": ties,
                    "p_row_gt_col_excl_ties": wins_b / (wins_a + wins_b) if (wins_a + wins_b) else 0.5,
                    "p_row_gt_col_incl_ties": wins_b / prompt_count,
                    "mean_score_diff_row_minus_col": -float(stat["mean_score_diff_a_minus_b"]),
                },
            ]
        )

    head_to_head_match_results = {model: 0 for model in models}
    for stat in pairwise.values():
        model_a = stat["model_a"]
        model_b = stat["model_b"]
        wins_a = int(stat["wins_a"])
        wins_b = int(stat["wins_b"])
        if wins_a > wins_b:
            head_to_head_match_results[model_a] += 1
        elif wins_b > wins_a:
            head_to_head_match_results[model_b] += 1

    ranking_by_prompt_level_wins = sorted(
        models,
        key=lambda model: (
            aggregate_prompt_level_pairwise[model]["wins"] - aggregate_prompt_level_pairwise[model]["losses"],
            aggregate_prompt_level_pairwise[model]["wins"],
            model_average_scores[model]["avg_total_score"],
        ),
        reverse=True,
    )
    ranking_by_average_total_score = sorted(
        models,
        key=lambda model: model_average_scores[model]["avg_total_score"],
        reverse=True,
    )
    ranking_by_head_to_head_match_results = sorted(
        models,
        key=lambda model: (
            head_to_head_match_results[model],
            aggregate_prompt_level_pairwise[model]["wins"] - aggregate_prompt_level_pairwise[model]["losses"],
            model_average_scores[model]["avg_total_score"],
        ),
        reverse=True,
    )

    summary = {
        "source_summary_file": str(SUMMARY_PATH),
        "prompt_count": prompt_count,
        "models": models,
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
    fig, ax = plt.subplots(figsize=(9.4, 7.8))
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
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    y = range(len(df))
    colors = {"wins": "#3a7d44", "ties": "#b8b3a7", "losses": "#b44b4b"}

    ax.barh(y, df["wins"], color=colors["wins"], label="Wins")
    ax.barh(y, df["ties"], left=df["wins"], color=colors["ties"], label="Ties")
    ax.barh(y, df["losses"], left=df["wins"] + df["ties"], color=colors["losses"], label="Losses")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate prompt-level pairwise outcomes across selected opponents")
    ax.set_title(
        AGGREGATE_TITLE,
        fontsize=14,
        pad=16,
    )
    ax.legend(loc="lower right", frameon=True)

    total = int(df.loc[0, ["wins", "ties", "losses"]].sum())
    for idx, row in df.iterrows():
        ax.text(
            row["wins"] + row["ties"] + row["losses"] + 10,
            idx,
            f"avg={row['avg_total']:.2f}",
            va="center",
            fontsize=9,
            color="#3d3428",
        )
    ax.set_xlim(0, total + 120)
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
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_title(
        TOURNAMENT_TITLE,
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
        connectionstyle="arc3,rad=0.12",
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
    full_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    summary, matrix_rows = build_subset_summary(full_summary)
    write_json(SUBSET_SUMMARY_PATH, summary)
    write_csv(SUBSET_MATRIX_PATH, matrix_rows)

    excl_values = build_matrix(summary, include_ties=False)
    plot_heatmap(
        excl_values,
        title=PAIRWISE_TITLE,
        subtitle="",
        pdf_path=PDF_EXCL,
    )

    incl_values = build_matrix(summary, include_ties=True)
    plot_heatmap(
        incl_values,
        title=PAIRWISE_TITLE,
        subtitle="",
        pdf_path=PDF_INCL,
    )

    plot_aggregate_bars(summary, PDF_BAR)
    plot_tournament_graph(summary, PDF_GRAPH)

    print(f"Wrote {SUBSET_SUMMARY_PATH}")
    print(f"Wrote {SUBSET_MATRIX_PATH}")
    print(f"Wrote {PDF_EXCL}")
    print(f"Wrote {PDF_INCL}")
    print(f"Wrote {PDF_BAR}")
    print(f"Wrote {PDF_GRAPH}")


if __name__ == "__main__":
    main()

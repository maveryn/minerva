#!/usr/bin/env python3
"""Plot 6-model pairwise GPT preference heatmaps with and without ties in the denominator."""

from __future__ import annotations

import json
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
    / "full_gpt_eval_dataset_6models_legacy"
    / "gpt52_pairwise_summary.json"
)

PNG_EXCL = ROOT / "pairwise_preference_6models_excluding_ties.png"
PDF_EXCL = ROOT / "pairwise_preference_6models_excluding_ties.pdf"
PNG_INCL = ROOT / "pairwise_preference_6models_including_ties.png"
PDF_INCL = ROOT / "pairwise_preference_6models_including_ties.pdf"
PNG_BAR = ROOT / "pairwise_preference_6models_aggregate_wins.png"
PDF_BAR = ROOT / "pairwise_preference_6models_aggregate_wins.pdf"
PNG_GRAPH = ROOT / "pairwise_preference_6models_tournament_graph.png"
PDF_GRAPH = ROOT / "pairwise_preference_6models_tournament_graph.pdf"


DISPLAY_NAMES = {
    "llama3-sec-reasoning": "Sec8B-Reasoning",
    "llama3-8b-dart": "Llama8B-DART",
    "minerva_llama8b_noctua": "Llama8B-MinervaRL",
    "llama-3-8B": "Llama8B-Base",
    "llama3-sec": "Foundation-Sec-8B",
    "minerva_llama8b_grpo": "Llama-3.1-8B-GRPO",
}


def build_matrix(summary: dict, include_ties: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_models = list(summary["ranking_by_prompt_level_wins"])
    prompt_count = int(summary["prompt_count"])
    pairwise = summary["pairwise"]

    value_df = pd.DataFrame(index=ordered_models, columns=ordered_models, dtype=float)
    annot_df = pd.DataFrame(index=ordered_models, columns=ordered_models, dtype=object)

    for row_model in ordered_models:
        for col_model in ordered_models:
            if row_model == col_model:
                value_df.loc[row_model, col_model] = 0.5
                annot_df.loc[row_model, col_model] = "-"
                continue

            key_ab = f"{row_model}|{col_model}"
            key_ba = f"{col_model}|{row_model}"
            if key_ab in pairwise:
                stat = pairwise[key_ab]
                wins = int(stat["wins_a"])
                losses = int(stat["wins_b"])
                ties = int(stat["ties"])
            else:
                stat = pairwise[key_ba]
                wins = int(stat["wins_b"])
                losses = int(stat["wins_a"])
                ties = int(stat["ties"])

            denom = prompt_count if include_ties else (wins + losses)
            value = (wins / denom) if denom else 0.5
            value_df.loc[row_model, col_model] = value
            if include_ties:
                annot_df.loc[row_model, col_model] = f"{wins}/{prompt_count}"
            else:
                annot_df.loc[row_model, col_model] = f"{wins}/{wins + losses}" if (wins + losses) else "-"

    value_df = value_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    annot_df = annot_df.rename(index=DISPLAY_NAMES, columns=DISPLAY_NAMES)
    return value_df, annot_df


def plot_heatmap(value_df: pd.DataFrame, title: str, subtitle: str, png_path: Path, pdf_path: Path) -> None:
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
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
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_bars(summary: dict, png_path: Path, pdf_path: Path) -> None:
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
    fig, ax = plt.subplots(figsize=(11, 7))
    y = range(len(df))
    colors = {
        "wins": "#3a7d44",
        "ties": "#b8b3a7",
        "losses": "#b44b4b",
    }

    ax.barh(y, df["wins"], color=colors["wins"], label="Wins")
    ax.barh(y, df["ties"], left=df["wins"], color=colors["ties"], label="Ties")
    ax.barh(y, df["losses"], left=df["wins"] + df["ties"], color=colors["losses"], label="Losses")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate prompt-level pairwise outcomes across all opponents")
    ax.set_title(
        "Aggregate GPT Pairwise Outcomes Across 6 Models\n"
        "Each model is compared against all five others on the same 350 prompts",
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
    ax.set_xlim(0, total + 130)
    plt.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_tournament_graph(summary: dict, png_path: Path, pdf_path: Path) -> None:
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
                wins_a, wins_b = int(stat["wins_a"]), int(stat["wins_b"])
                p = float(stat["p_a_gt_b_excl_ties"])
            else:
                stat = pairwise[key_ba]
                wins_a, wins_b = int(stat["wins_b"]), int(stat["wins_a"])
                p = 1.0 - float(stat["p_a_gt_b_excl_ties"])

            if wins_a > wins_b:
                winner, loser, strength = model_a, model_b, p
            else:
                winner, loser, strength = model_b, model_a, 1.0 - p
            graph.add_edge(winner, loser, strength=strength, margin=abs(wins_a - wins_b))

    positions = {
        model: (index, 0.0)
        for index, model in enumerate(ordered_models)
    }
    node_labels = {model: DISPLAY_NAMES[model] for model in ordered_models}
    node_colors = [model_avg[m]["avg_total_score"] for m in ordered_models]

    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_title(
        "Directed GPT Tournament Graph Across 6 Models\n"
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
        node_size=2400,
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
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    excl_values, _ = build_matrix(summary, include_ties=False)
    plot_heatmap(
        excl_values,
        title="Pairwise GPT Preference Across 6 Models",
        subtitle="Excluding ties from the denominator",
        png_path=PNG_EXCL,
        pdf_path=PDF_EXCL,
    )

    incl_values, _ = build_matrix(summary, include_ties=True)
    plot_heatmap(
        incl_values,
        title="Pairwise GPT Preference Across 6 Models",
        subtitle="Including ties in the denominator",
        png_path=PNG_INCL,
        pdf_path=PDF_INCL,
    )
    plot_aggregate_bars(summary, PNG_BAR, PDF_BAR)
    plot_tournament_graph(summary, PNG_GRAPH, PDF_GRAPH)

    print(f"Wrote {PNG_EXCL}")
    print(f"Wrote {PDF_EXCL}")
    print(f"Wrote {PNG_INCL}")
    print(f"Wrote {PDF_INCL}")
    print(f"Wrote {PNG_BAR}")
    print(f"Wrote {PDF_BAR}")
    print(f"Wrote {PNG_GRAPH}")
    print(f"Wrote {PDF_GRAPH}")


if __name__ == "__main__":
    main()

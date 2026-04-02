#!/usr/bin/env python3
"""Summarize and plot Qwen 8B-family 5-model GPT pairwise preferences from pointwise scores."""

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
DATASET_DIR = EVAL_ROOT / "full_gpt_eval_dataset_qwen8b5models"
OUTPUT_DIR = EVAL_ROOT / "judge_outputs" / "full_gpt_eval_dataset_qwen8b5models"

SCORE_PATH = OUTPUT_DIR / "gpt52_scores.jsonl"
KEY_PATH = DATASET_DIR / "key.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "gpt52_pairwise_summary.json"
MATRIX_CSV_PATH = OUTPUT_DIR / "gpt52_pairwise_matrix.csv"

PDF_EXCL = ROOT / "pairwise_preference_qwen8b5models_excluding_ties.pdf"
PDF_INCL = ROOT / "pairwise_preference_qwen8b5models_including_ties.pdf"
PDF_BAR = ROOT / "pairwise_preference_qwen8b5models_aggregate_wins.pdf"
PDF_GRAPH = ROOT / "pairwise_preference_qwen8b5models_tournament_graph.pdf"


DISPLAY_NAMES = {
    "qwen3-8b-base": "Qwen3-8B-Base",
    "minerva_qwen8b_grpo": "Qwen3-8B-GRPO",
    "qwen3-8b-star": "Qwen3-8B-STaR",
    "qwen3-8b-dart": "Qwen3-8B-DART",
    "minerva_qwen8b_noctua": "Qwen3-8B-MinervaRL",
}
PAIRWISE_TITLE = "Pairwise GPT Preference on Qwen3-8B"
AGGREGATE_TITLE = "Aggregate GPT Pairwise Outcomes on Qwen3-8B"
TOURNAMENT_TITLE = "Directed GPT Tournament Graph on Qwen3-8B"


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
        records.append(
            {
                **score,
                "prompt_id": key["prompt_id"],
                "model_name": key["model_name"],
                "task_label": key["task_label"],
            }
        )

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
    prompt_set = set(complete_prompts)
    for model in all_models:
        rows = [row for row in model_rows[model] if row["prompt_id"] in prompt_set]
        model_average_scores[model] = {
            "n": len(rows),
            "avg_total_score": mean([float(row["total_score"]) for row in rows]),
            "avg_writing_quality_score": mean([float(row["writing_quality_score"]) for row in rows]),
            "avg_evidence_use_score": mean([float(row["evidence_use_score"]) for row in rows]),
            "avg_cti_concept_focus_score": mean([float(row["cti_concept_focus_score"]) for row in rows]),
        }

    pairwise: dict[str, dict[str, float | int | str]] = {}
    aggregate_prompt_level_pairwise = {
        model: {"wins": 0, "losses": 0, "ties": 0} for model in all_models
    }

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
        pairwise[f"{model_a}|{model_b}"] = {
            "model_a": model_a,
            "model_b": model_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "p_a_gt_b_excl_ties": (wins_a / denom_excl_ties) if denom_excl_ties else 0.5,
            "p_a_gt_b_incl_ties": wins_a / len(complete_prompts) if complete_prompts else 0.0,
            "mean_score_diff_a_minus_b": mean(diffs),
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
                    "p_row_gt_col_excl_ties": (wins_a / denom_excl_ties) if denom_excl_ties else 0.5,
                    "p_row_gt_col_incl_ties": wins_a / len(complete_prompts) if complete_prompts else 0.0,
                    "mean_score_diff_row_minus_col": mean(diffs),
                },
                {
                    "row_model": model_b,
                    "column_model": model_a,
                    "wins_row": wins_b,
                    "wins_col": wins_a,
                    "ties": ties,
                    "p_row_gt_col_excl_ties": (wins_b / denom_excl_ties) if denom_excl_ties else 0.5,
                    "p_row_gt_col_incl_ties": wins_b / len(complete_prompts) if complete_prompts else 0.0,
                    "mean_score_diff_row_minus_col": -mean(diffs),
                },
            ]
        )

    head_to_head_match_results = {model: 0 for model in all_models}
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
    df["net_wins"] = df["wins"] - df["losses"]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sns.barplot(data=df, x="net_wins", y="model", palette="crest", ax=ax)
    ax.set_title(AGGREGATE_TITLE, fontsize=14, pad=12)
    ax.set_xlabel("Net wins (wins - losses)")
    ax.set_ylabel("")
    for y, row in enumerate(df.itertuples(index=False)):
        ax.text(row.net_wins, y, f"  avg={row.avg_total:.2f}", va="center", ha="left", fontsize=10)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_tournament(summary: dict, pdf_path: Path) -> None:
    ordered_models = list(summary["ranking_by_prompt_level_wins"])
    G = nx.DiGraph()
    for model in ordered_models:
        G.add_node(DISPLAY_NAMES[model])
    for stat in summary["pairwise"].values():
        a = stat["model_a"]
        b = stat["model_b"]
        wins_a = int(stat["wins_a"])
        wins_b = int(stat["wins_b"])
        if wins_a > wins_b:
            G.add_edge(DISPLAY_NAMES[a], DISPLAY_NAMES[b], weight=wins_a - wins_b)
        elif wins_b > wins_a:
            G.add_edge(DISPLAY_NAMES[b], DISPLAY_NAMES[a], weight=wins_b - wins_a)

    pos = nx.circular_layout(G)
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    nx.draw_networkx_nodes(G, pos, node_size=2600, node_color="#d8ecf3", edgecolors="#345f73", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edge_widths = [1.0 + 0.18 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="#4c7c92",
        ax=ax,
        connectionstyle="arc3,rad=0.08",
    )
    ax.set_title(TOURNAMENT_TITLE, fontsize=14, pad=14)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_matrix_csv(matrix_rows: list[dict]) -> None:
    MATRIX_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MATRIX_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)


def main() -> None:
    records, all_models, expected_prompt_count = load_records()
    summary, matrix_rows = summarize(records, all_models, expected_prompt_count)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_matrix_csv(matrix_rows)

    excl_df = build_matrix(summary, include_ties=False)
    incl_df = build_matrix(summary, include_ties=True)
    plot_heatmap(
        excl_df,
        PAIRWISE_TITLE,
        "",
        PDF_EXCL,
    )
    plot_heatmap(
        incl_df,
        PAIRWISE_TITLE,
        "",
        PDF_INCL,
    )
    plot_aggregate_bars(summary, PDF_BAR)
    plot_tournament(summary, PDF_GRAPH)

    print(json.dumps({
        "summary_path": str(SUMMARY_PATH),
        "matrix_csv_path": str(MATRIX_CSV_PATH),
        "prompt_count": summary["prompt_count"],
        "models": summary["models"],
    }, indent=2))


if __name__ == "__main__":
    main()

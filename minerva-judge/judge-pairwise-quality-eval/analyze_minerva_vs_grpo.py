#!/usr/bin/env python3
"""Analyze MinervaRL vs GRPO pairwise judge results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


MINERVA = "minerva_llama8b_noctua"
GRPO = "minerva_llama8b_grpo"

MODEL_LABELS = {
    MINERVA: "MinervaRL",
    GRPO: "GRPO",
}

STOPWORDS = set(ENGLISH_STOP_WORDS)
STOPWORDS.update(
    {
        "cti",
        "cyber",
        "threat",
        "intelligence",
        "model",
        "response",
        "answer",
        "answers",
        "question",
        "based",
        "also",
        "use",
    }
)


@dataclass
class PairRow:
    prompt: str
    minerva_response: str
    grpo_response: str
    winner_model: str
    rationale: str


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_pairs(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    for obj in iter_jsonl(path):
        model_a = obj.get("model_a")
        model_b = obj.get("model_b")
        if {model_a, model_b} != {MINERVA, GRPO}:
            continue
        response_a = str(obj.get("response_a") or "")
        response_b = str(obj.get("response_b") or "")
        prompt = str(obj.get("prompt") or "")
        winner_model = obj.get("winner_model") or ""
        rationale = str(obj.get("rationale") or "")

        if model_a == MINERVA:
            minerva_response = response_a
            grpo_response = response_b
        else:
            minerva_response = response_b
            grpo_response = response_a

        rows.append(
            PairRow(
                prompt=prompt,
                minerva_response=minerva_response,
                grpo_response=grpo_response,
                winner_model=winner_model,
                rationale=rationale,
            )
        )
    return rows


def summarize(rows: List[PairRow]) -> Dict[str, Any]:
    total = len(rows)
    wins_minerva = sum(1 for r in rows if r.winner_model == MINERVA)
    wins_grpo = sum(1 for r in rows if r.winner_model == GRPO)
    ties = sum(1 for r in rows if r.winner_model == "tie")
    denom = wins_minerva + wins_grpo
    win_rate_minerva = wins_minerva / denom if denom else 0.0
    win_rate_grpo = wins_grpo / denom if denom else 0.0

    lengths = {
        "minerva_words_mean": float(np.mean([count_words(r.minerva_response) for r in rows])) if rows else 0.0,
        "grpo_words_mean": float(np.mean([count_words(r.grpo_response) for r in rows])) if rows else 0.0,
    }

    return {
        "total_pairs": total,
        "wins_minerva": wins_minerva,
        "wins_grpo": wins_grpo,
        "ties": ties,
        "win_rate_minerva": win_rate_minerva,
        "win_rate_grpo": win_rate_grpo,
        **lengths,
    }


def build_wordcloud(text: str, output_path: Path, title: str) -> bool:
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except Exception:
        return False

    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        stopwords=STOPWORDS,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    fig.tight_layout(pad=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return True


def top_terms(diff_vec: np.ndarray, vocab: List[str], n: int = 20) -> List[Tuple[str, float]]:
    idx = np.argsort(diff_vec)[::-1]
    out = []
    for i in idx[:n]:
        out.append((vocab[i], float(diff_vec[i])))
    return out


def embedding_analysis(rows: List[PairRow], output_dir: Path) -> None:
    minerva_texts = [normalize_text(r.minerva_response) for r in rows]
    grpo_texts = [normalize_text(r.grpo_response) for r in rows]

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=sorted(STOPWORDS),
    )
    all_texts = minerva_texts + grpo_texts
    tfidf = vectorizer.fit_transform(all_texts)
    vocab = vectorizer.get_feature_names_out().tolist()

    minerva_vec = tfidf[: len(minerva_texts)]
    grpo_vec = tfidf[len(minerva_texts) :]

    diff = minerva_vec - grpo_vec

    win_mask = np.array([r.winner_model == MINERVA for r in rows])
    lose_mask = np.array([r.winner_model == GRPO for r in rows])

    if win_mask.any():
        mean_diff_win = diff[win_mask].mean(axis=0).A1
    else:
        mean_diff_win = np.zeros(diff.shape[1])
    if lose_mask.any():
        mean_diff_lose = diff[lose_mask].mean(axis=0).A1
    else:
        mean_diff_lose = np.zeros(diff.shape[1])

    top_win = top_terms(mean_diff_win, vocab, n=20)
    top_lose = top_terms(-mean_diff_lose, vocab, n=20)

    rows_out = []
    for term, score in top_win:
        rows_out.append({"group": "minerva_win", "term": term, "score": score})
    for term, score in top_lose:
        rows_out.append({"group": "grpo_win", "term": term, "score": score})
    write_csv(output_dir / "top_terms_by_winner.csv", rows_out)

    # 2D scatter of difference vectors
    if diff.shape[0] >= 3:
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(diff)
        labels = [
            "MinervaRL" if r.winner_model == MINERVA else "GRPO" if r.winner_model == GRPO else "Tie"
            for r in rows
        ]
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns  # type: ignore

            sns.set_theme(style="white", font_scale=0.9)
            fig, ax = plt.subplots(figsize=(7, 5))
            palette = {"MinervaRL": "#1f77b4", "GRPO": "#ff7f0e", "Tie": "#999999"}
            for label in sorted(set(labels)):
                mask = np.array([l == label for l in labels])
                ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6, label=label, color=palette[label])
            ax.set_title("Response-difference embeddings (Minerva - GRPO)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout(pad=0.4)
            fig.savefig(output_dir / "diff_embedding_scatter.png", dpi=300, bbox_inches="tight", pad_inches=0.0)
            fig.savefig(output_dir / "diff_embedding_scatter.pdf", dpi=300, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MinervaRL vs GRPO pairwise judgments.")
    parser.add_argument(
        "--input",
        default="pairwise_judge_results.jsonl",
        help="Input JSONL file with pairwise judge results",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_minerva_vs_grpo",
        help="Directory to write analysis outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pairs(input_path)
    if not rows:
        raise ValueError("No MinervaRL vs GRPO rows found.")

    summary = summarize(rows)
    write_json(output_dir / "summary.json", summary)

    # Word clouds from rationales
    rationales_minerva = " ".join(
        normalize_text(r.rationale) for r in rows if r.winner_model == MINERVA and r.rationale
    )
    rationales_grpo = " ".join(
        normalize_text(r.rationale) for r in rows if r.winner_model == GRPO and r.rationale
    )

    wordcloud_ok = True
    if rationales_minerva:
        ok = build_wordcloud(
            rationales_minerva,
            output_dir / "wordcloud_minerva_wins.png",
            "Judge rationales when MinervaRL wins",
        )
        wordcloud_ok = wordcloud_ok and ok
    if rationales_grpo:
        ok = build_wordcloud(
            rationales_grpo,
            output_dir / "wordcloud_grpo_wins.png",
            "Judge rationales when GRPO wins",
        )
        wordcloud_ok = wordcloud_ok and ok

    if not wordcloud_ok:
        print("wordcloud not available; install with `pip install wordcloud`")

    embedding_analysis(rows, output_dir)

    print(f"Wrote analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()

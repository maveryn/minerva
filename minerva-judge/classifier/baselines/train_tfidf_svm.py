#!/usr/bin/env python3
"""Train a TF-IDF + Linear SVM baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_text(row: dict, mode: str, max_tokens: int | None) -> str:
    prompt = row.get("prompt") or ""
    response = row.get("response") or ""
    if mode == "response":
        tokens = tokenize(response)
        return " ".join(tokens[:max_tokens] if max_tokens else tokens)
    if mode == "prompt":
        tokens = tokenize(prompt)
        return " ".join(tokens[:max_tokens] if max_tokens else tokens)

    resp_tokens = tokenize(response)
    prompt_tokens = tokenize(prompt)
    if max_tokens is not None:
        if len(resp_tokens) >= max_tokens:
            resp_tokens = resp_tokens[:max_tokens]
            prompt_tokens = []
        else:
            budget = max_tokens - len(resp_tokens)
            prompt_tokens = prompt_tokens[:budget]
    combined = resp_tokens + ["[sep]"] + prompt_tokens
    return " ".join(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + Linear SVM baseline.")
    parser.add_argument(
        "--train-file",
        default="minerva-judge/classifier/data/train.jsonl",
        help="Training JSONL file",
    )
    parser.add_argument(
        "--eval-file",
        default="minerva-judge/classifier/data/val.jsonl",
        help="Validation JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="minerva-judge/classifier/outputs/baselines/tfidf_svm",
        help="Output directory",
    )
    parser.add_argument(
        "--text-mode",
        choices=["response_prompt", "response", "prompt"],
        default="response_prompt",
        help="How to combine prompt/response",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Optional max tokens for response+prompt (0 disables cap)",
    )
    parser.add_argument("--max-features", type=int, default=200000, help="Max vocab size")
    parser.add_argument("--min-df", type=int, default=2, help="Min document freq")
    parser.add_argument("--max-df", type=float, default=0.95, help="Max document freq")
    parser.add_argument("--ngram-min", type=int, default=1, help="Min ngram size")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max ngram size")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)
    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    train_rows = list(iter_jsonl(train_path))
    eval_rows = list(iter_jsonl(eval_path))
    train_texts = [build_text(row, args.text_mode, max_tokens) for row in train_rows]
    eval_texts = [build_text(row, args.text_mode, max_tokens) for row in eval_rows]
    y_train = np.array([row.get("label", 0) for row in train_rows], dtype=np.int64)
    y_eval = np.array([row.get("label", 0) for row in eval_rows], dtype=np.int64)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(args.ngram_min, args.ngram_max),
        lowercase=False,
        token_pattern=r"\S+",
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)

    clf = LinearSVC(C=args.C)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)

    acc = float(accuracy_score(y_eval, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_eval, preds, average="binary", zero_division=0
    )

    metrics = {
        "eval_accuracy": acc,
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "eval_samples": int(len(y_eval)),
        "text_mode": args.text_mode,
        "max_tokens": max_tokens,
        "max_features": args.max_features,
        "ngram_range": [args.ngram_min, args.ngram_max],
        "C": args.C,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

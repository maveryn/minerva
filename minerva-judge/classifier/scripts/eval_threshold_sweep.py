#!/usr/bin/env python3
"""Evaluate threshold sweep for a binary classifier on the val set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep for classifier outputs.")
    parser.add_argument(
        "--model-dir",
        default="minerva-judge/classifier/outputs/modernbert-2k-lr5e-05",
        help="HF model directory",
    )
    parser.add_argument(
        "--data-file",
        default="minerva-judge/classifier/data/val.jsonl",
        help="Validation JSONL file",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--threshold-start", type=float, default=0.5, help="Start threshold")
    parser.add_argument("--threshold-end", type=float, default=0.95, help="End threshold (inclusive)")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Step size")
    parser.add_argument(
        "--output",
        default="minerva-judge/classifier/experiments/threshold_sweep_modernbert-2k-lr5e-05.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_file = Path(args.data_file)
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    if not data_file.exists():
        raise FileNotFoundError(f"data file not found: {data_file}")

    rows = []
    for row in iter_jsonl(data_file):
        prompt = row.get("prompt")
        response = row.get("response")
        label = row.get("label")
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        if label not in (0, 1):
            continue
        rows.append((prompt, response, int(label)))
    if not rows:
        raise RuntimeError("no valid rows found")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    good_id = 1
    label2id = getattr(model.config, "label2id", {}) or {}
    for label, idx in label2id.items():
        if str(label).upper() == "GOOD":
            good_id = int(idx)
            break

    labels = np.array([row[2] for row in rows], dtype=np.int64)
    probs = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [item[0] for item in batch]
        responses = [item[1] for item in batch]
        enc = tokenizer(
            prompts,
            responses,
            truncation="only_first",
            max_length=int(args.max_length),
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, good_id].detach().cpu().numpy()
            probs.append(batch_probs)
    probs = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    if probs.size != labels.size:
        raise RuntimeError("probability count does not match labels")

    thresholds = []
    t = float(args.threshold_start)
    end = float(args.threshold_end) + 1e-9
    step = float(args.threshold_step)
    if step <= 0:
        raise ValueError("threshold-step must be positive")
    while t <= end:
        thresholds.append(round(t, 4))
        t += step

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for t in thresholds:
            preds = (probs >= t).astype(np.int64)
            metrics = compute_metrics(labels, preds)
            record = {
                "threshold": t,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print("threshold\tprecision\trecall\tf1")
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        metrics = compute_metrics(labels, preds)
        print(f"{t:.2f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

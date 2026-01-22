#!/usr/bin/env python3
"""Evaluate threshold sweep for a TextCNN baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_tokens(prompt: str, response: str, mode: str, max_tokens: int | None) -> list[str]:
    prompt_tokens = tokenize(prompt)
    response_tokens = tokenize(response)
    if mode == "response":
        tokens = response_tokens
        return tokens[:max_tokens] if max_tokens else tokens
    if mode == "prompt":
        tokens = prompt_tokens
        return tokens[:max_tokens] if max_tokens else tokens

    if max_tokens is not None:
        if len(response_tokens) >= max_tokens:
            response_tokens = response_tokens[:max_tokens]
            prompt_tokens = []
        else:
            budget = max_tokens - len(response_tokens)
            prompt_tokens = prompt_tokens[:budget]
    return response_tokens + ["sep"] + prompt_tokens


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, kernel_sizes: list[int], num_filters: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        emb = self.embed(x)
        emb = emb.transpose(1, 2)
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(emb))
            y = torch.max(y, dim=2).values
            feats.append(y)
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out)


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
    parser = argparse.ArgumentParser(description="Threshold sweep for TextCNN.")
    parser.add_argument(
        "--model-dir",
        default="minerva-judge/classifier/outputs/baselines/textcnn_lr5e-4",
        help="TextCNN output dir with model.pt and metrics.json",
    )
    parser.add_argument(
        "--data-file",
        default="minerva-judge/classifier/data/val.jsonl",
        help="Validation JSONL file",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--text-mode",
        choices=["response_prompt", "response", "prompt"],
        default="response_prompt",
        help="How to combine prompt/response",
    )
    parser.add_argument("--max-tokens", type=int, default=0, help="Max tokens (0 uses metrics.json)")
    parser.add_argument("--threshold-start", type=float, default=0.5, help="Start threshold")
    parser.add_argument("--threshold-end", type=float, default=0.95, help="End threshold (inclusive)")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Step size")
    parser.add_argument(
        "--output",
        default="minerva-judge/classifier/experiments/threshold_sweep_textcnn_lr5e-4.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_file = Path(args.data_file)
    if not model_dir.exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    if not data_file.exists():
        raise FileNotFoundError(f"data file not found: {data_file}")

    payload = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    vocab = payload.get("vocab") or {}
    state_dict = payload.get("state_dict")
    if not vocab or state_dict is None:
        raise RuntimeError("model.pt missing vocab or state_dict")

    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    embed_dim = int(metrics.get("embed_dim", 200))
    kernel_sizes = metrics.get("kernel_sizes", [3, 4, 5])
    if isinstance(kernel_sizes, str):
        kernel_sizes = [int(x) for x in kernel_sizes.split(",") if x.strip()]
    num_filters = int(metrics.get("num_filters", 256))

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    if max_tokens is None:
        mt = metrics.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(len(vocab), embed_dim, kernel_sizes, num_filters).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    for row in iter_jsonl(data_file):
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        label = row.get("label")
        if label not in (0, 1):
            continue
        rows.append((str(prompt), str(response), int(label)))
    if not rows:
        raise RuntimeError("no valid rows found")

    labels = np.array([r[2] for r in rows], dtype=np.int64)
    probs = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        seqs = []
        for prompt, response, _ in batch:
            tokens = build_tokens(prompt, response, args.text_mode, max_tokens)
            if not tokens:
                tokens = ["<unk>"]
            seqs.append([vocab.get(tok, 1) for tok in tokens])
        max_len = max(len(seq) for seq in seqs)
        if max_tokens:
            max_len = min(max_len, max_tokens)
        if max_len <= 0:
            max_len = 1
        padded = []
        for seq in seqs:
            if len(seq) >= max_len:
                padded.append(seq[:max_len])
            else:
                padded.append(seq + [0] * (max_len - len(seq)))
        x = torch.tensor(padded, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(batch_probs)
    probs = np.concatenate(probs) if probs else np.array([], dtype=np.float32)

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
            metrics_out = compute_metrics(labels, preds)
            record = {
                "threshold": t,
                "accuracy": metrics_out["accuracy"],
                "precision": metrics_out["precision"],
                "recall": metrics_out["recall"],
                "f1": metrics_out["f1"],
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print("threshold\tprecision\trecall\tf1")
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        metrics_out = compute_metrics(labels, preds)
        print(f"{t:.2f}\t{metrics_out['precision']:.4f}\t{metrics_out['recall']:.4f}\t{metrics_out['f1']:.4f}")


if __name__ == "__main__":
    main()

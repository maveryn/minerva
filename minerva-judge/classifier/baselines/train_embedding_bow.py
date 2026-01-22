#!/usr/bin/env python3
"""Train a simple word-embedding bag-of-words classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


def build_tokens(row: dict, mode: str, max_tokens: int | None) -> list[str]:
    prompt = row.get("prompt") or ""
    response = row.get("response") or ""
    if mode == "response":
        tokens = tokenize(response)
        return tokens[:max_tokens] if max_tokens else tokens
    if mode == "prompt":
        tokens = tokenize(prompt)
        return tokens[:max_tokens] if max_tokens else tokens

    resp_tokens = tokenize(response)
    prompt_tokens = tokenize(prompt)
    if max_tokens is not None:
        if len(resp_tokens) >= max_tokens:
            resp_tokens = resp_tokens[:max_tokens]
            prompt_tokens = []
        else:
            budget = max_tokens - len(resp_tokens)
            prompt_tokens = prompt_tokens[:budget]
    return resp_tokens + ["[sep]"] + prompt_tokens


def add_ngrams(tokens: list[str], ngram: int) -> list[str]:
    if ngram <= 1:
        return tokens
    out = list(tokens)
    for n in range(2, ngram + 1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i : i + n]))
    return out


class BowDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: dict[str, int], mode: str, max_tokens: int | None, ngram: int):
        self.rows = rows
        self.vocab = vocab
        self.mode = mode
        self.max_tokens = max_tokens
        self.ngram = ngram

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        tokens = build_tokens(row, self.mode, self.max_tokens)
        tokens = add_ngrams(tokens, self.ngram)
        indices = [self.vocab.get(tok, 0) for tok in tokens] or [0]
        label = int(row.get("label", 0))
        return indices, label


def collate_batch(batch):
    indices, labels = zip(*batch)
    offsets = [0]
    flat = []
    for item in indices:
        flat.extend(item)
        offsets.append(len(flat))
    offsets = offsets[:-1]
    return (
        torch.tensor(flat, dtype=torch.long),
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_vocab(rows: list[dict], mode: str, max_tokens: int | None, ngram: int, max_vocab: int, min_freq: int):
    counter: Counter[str] = Counter()
    for row in rows:
        tokens = build_tokens(row, mode, max_tokens)
        tokens = add_ngrams(tokens, ngram)
        counter.update(tokens)
    vocab = {"<unk>": 0}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for flat, offsets, y in loader:
            flat = flat.to(device)
            offsets = offsets.to(device)
            logits = model(flat, offsets)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred)
            labels.append(y.numpy())
    preds = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    labels = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    if labels.size == 0:
        return {"eval_accuracy": 0.0, "eval_precision": 0.0, "eval_recall": 0.0, "eval_f1": 0.0}

    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
    }


class EmbeddingBowClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, flat, offsets):
        x = self.embed(flat, offsets)
        return self.fc(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding bag-of-words baseline.")
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
        default="minerva-judge/classifier/outputs/baselines/embedding_bow",
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
    parser.add_argument("--word-ngrams", type=int, default=1, help="Max word n-gram length")
    parser.add_argument("--max-vocab", type=int, default=200000, help="Max vocab size")
    parser.add_argument("--min-freq", type=int, default=2, help="Min token frequency")
    parser.add_argument("--embed-dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file)
    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    train_rows = list(iter_jsonl(train_path))
    eval_rows = list(iter_jsonl(eval_path))

    vocab = build_vocab(
        train_rows,
        args.text_mode,
        max_tokens,
        args.word_ngrams,
        args.max_vocab,
        args.min_freq,
    )

    train_ds = BowDataset(train_rows, vocab, args.text_mode, max_tokens, args.word_ngrams)
    eval_ds = BowDataset(eval_rows, vocab, args.text_mode, max_tokens, args.word_ngrams)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingBowClassifier(len(vocab), args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for flat, offsets, labels in train_loader:
            flat = flat.to(device)
            offsets = offsets.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(flat, offsets)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(model, eval_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    **metrics,
                },
                ensure_ascii=False,
            )
        )

    final_metrics = evaluate(model, eval_loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                **final_metrics,
                "vocab_size": len(vocab),
                "embed_dim": args.embed_dim,
                "word_ngrams": args.word_ngrams,
                "max_tokens": max_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"state_dict": model.state_dict(), "vocab": vocab}, output_dir / "model.pt")


if __name__ == "__main__":
    main()

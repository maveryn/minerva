#!/usr/bin/env python3
"""Train a simple TextCNN classifier."""

from __future__ import annotations

import argparse
import json
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
    return resp_tokens + ["sep"] + prompt_tokens


def build_vocab(rows: list[dict], mode: str, max_tokens: int | None, max_vocab: int, min_freq: int):
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(build_tokens(row, mode, max_tokens))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


class TextDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        vocab: dict[str, int],
        mode: str,
        max_tokens: int | None,
    ):
        self.rows = rows
        self.vocab = vocab
        self.mode = mode
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        tokens = build_tokens(row, self.mode, self.max_tokens)
        ids = [self.vocab.get(tok, 1) for tok in tokens]
        label = int(row.get("label", 0))
        return ids, label


def collate_batch(batch, pad_id: int, max_len: int | None):
    ids, labels = zip(*batch)
    if max_len is None:
        max_len = max(len(seq) for seq in ids)
    padded = []
    for seq in ids:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_id] * (max_len - len(seq)))
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, kernel_sizes: list[int], num_filters: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):
        emb = self.embed(x)  # (batch, seq, dim)
        emb = emb.transpose(1, 2)  # (batch, dim, seq)
        feats = []
        for conv in self.convs:
            y = torch.relu(conv(emb))
            y = torch.max(y, dim=2).values
            feats.append(y)
        out = torch.cat(feats, dim=1)
        out = self.dropout(out)
        return self.fc(out)


def _maybe_skip_header(parts: list[str]) -> bool:
    if len(parts) < 2:
        return False
    try:
        int(parts[0])
        int(parts[1])
        return True
    except ValueError:
        return False


def load_pretrained_embeddings(path: Path, vocab: dict[str, int], embed_dim: int):
    rng = np.random.default_rng(1337)
    weights = rng.normal(0.0, 0.02, size=(len(vocab), embed_dim)).astype(np.float32)
    weights[vocab.get("<pad>", 0)] = 0.0
    found = 0
    total = len(vocab)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if line_num == 0 and _maybe_skip_header(parts):
                continue
            if len(parts) <= embed_dim:
                continue
            word = parts[0]
            if word not in vocab:
                continue
            vec = parts[1:]
            if len(vec) != embed_dim:
                continue
            try:
                weights[vocab[word]] = np.asarray(vec, dtype=np.float32)
                found += 1
            except ValueError:
                continue
    coverage = found / max(1, total)
    print(f"Loaded pretrained vectors for {found}/{total} tokens ({coverage:.2%}).")
    return weights


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="TextCNN baseline.")
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
        default="minerva-judge/classifier/outputs/baselines/textcnn",
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
        default=2048,
        help="Optional max tokens for response+prompt (0 disables cap)",
    )
    parser.add_argument("--max-vocab", type=int, default=100000, help="Max vocab size")
    parser.add_argument("--min-freq", type=int, default=2, help="Min token frequency")
    parser.add_argument("--embed-dim", type=int, default=200, help="Embedding dimension")
    parser.add_argument("--kernel-sizes", type=str, default="3,4,5", help="Comma-separated kernel sizes")
    parser.add_argument("--num-filters", type=int, default=256, help="Filters per kernel")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--pretrained-embeddings",
        type=str,
        default=None,
        help="Path to pretrained embeddings (GloVe/fastText .vec/.txt)",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embedding weights when using pretrained embeddings",
    )
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

    vocab = build_vocab(train_rows, args.text_mode, max_tokens, args.max_vocab, args.min_freq)
    train_ds = TextDataset(train_rows, vocab, args.text_mode, max_tokens)
    eval_ds = TextDataset(eval_rows, vocab, args.text_mode, max_tokens)

    pad_id = vocab.get("<pad>", 0)
    max_len = max_tokens
    collate = lambda batch: collate_batch(batch, pad_id, max_len)  # noqa: E731
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    kernel_sizes = [int(x) for x in args.kernel_sizes.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(len(vocab), args.embed_dim, kernel_sizes, args.num_filters, args.dropout).to(device)
    if args.pretrained_embeddings:
        emb_path = Path(args.pretrained_embeddings)
        if not emb_path.exists():
            raise FileNotFoundError(f"pretrained embeddings not found: {emb_path}")
        weights = load_pretrained_embeddings(emb_path, vocab, args.embed_dim)
        model.embed.weight.data.copy_(torch.tensor(weights, device=device))
        if args.freeze_embeddings:
            model.embed.weight.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
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
                "kernel_sizes": kernel_sizes,
                "num_filters": args.num_filters,
                "max_tokens": max_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"state_dict": model.state_dict(), "vocab": vocab}, output_dir / "model.pt")


if __name__ == "__main__":
    main()

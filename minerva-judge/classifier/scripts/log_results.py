#!/usr/bin/env python3
"""Append eval metrics and config metadata to a JSONL results log."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Log classifier experiment results.")
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--max-length", type=int, required=True, help="Max sequence length")
    parser.add_argument("--learning-rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch-size", type=int, required=True, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, required=True, help="Per-device eval batch size")
    parser.add_argument("--grad-accumulation", type=int, required=True, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, required=True, help="Epoch count")
    parser.add_argument(
        "--train-file",
        default="minerva-judge/classifier/data/train.jsonl",
        help="Training data path",
    )
    parser.add_argument(
        "--eval-file",
        default="minerva-judge/classifier/data/val.jsonl",
        help="Validation data path",
    )
    parser.add_argument("--run-tag", default="", help="Optional run tag")
    parser.add_argument(
        "--results-file",
        default="minerva-judge/classifier/experiments/results.jsonl",
        help="Results JSONL path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"eval metrics not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = {
        "timestamp": int(time.time()),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accumulation": args.grad_accumulation,
        "epochs": args.epochs,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "run_tag": args.run_tag,
        "metrics": metrics,
    }

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Logged results to {results_path}")


if __name__ == "__main__":
    main()

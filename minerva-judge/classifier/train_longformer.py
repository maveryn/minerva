#!/usr/bin/env python3
"""Train a Longformer classifier on prompt+response pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ClassifierCollator(DataCollatorWithPadding):
    def __init__(self, *args, use_global_attention: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_global_attention = use_global_attention

    def __call__(self, features):
        batch = super().__call__(features)
        if self.use_global_attention and "global_attention_mask" not in batch:
            global_mask = torch.zeros_like(batch["input_ids"])
            global_mask[:, 0] = 1
            batch["global_attention_mask"] = global_mask
        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    acc = float((preds == labels).mean()) if labels.size else 0.0
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Longformer classifier.")
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
        "--model-name",
        default="allenai/longformer-base-4096",
        help="Base model name",
    )
    parser.add_argument(
        "--output-dir",
        default="minerva-judge/classifier/outputs/longformer-base-4096",
        help="Output directory",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Train batch size per device")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Eval batch size per device")
    parser.add_argument("--grad-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
    args = parser.parse_args()

    set_seed(args.seed)

    train_path = PROJECT_ROOT / args.train_file
    eval_path = PROJECT_ROOT / args.eval_file
    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(eval_path)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    def tokenize_batch(batch):
        return tokenizer(
            batch["prompt"],
            batch["response"],
            truncation="only_first",
            max_length=args.max_length,
        )

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col not in {"label"}],
        desc="tokenizing",
    )

    id2label = {0: "BAD", 1: "GOOD"}
    label2id = {"BAD": 0, "GOOD": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    if hasattr(model.config, "max_position_embeddings") and args.max_length > model.config.max_position_embeddings:
        if hasattr(model, "resize_position_embeddings"):
            model.resize_position_embeddings(args.max_length)
            model.config.max_position_embeddings = args.max_length
            print(f"Resized position embeddings to {args.max_length}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=[],
    )

    use_global_attention = getattr(model.config, "model_type", "") == "longformer"
    collator = ClassifierCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        use_global_attention=use_global_attention,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()

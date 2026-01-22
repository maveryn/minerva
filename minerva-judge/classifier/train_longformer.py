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
import torch.nn.functional as F

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


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float,
    alpha_bad: float | None,
    alpha_good: float | None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    labels = labels.long()
    log_pt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    loss = -((1.0 - pt) ** gamma) * log_pt
    if alpha_bad is not None or alpha_good is not None:
        alpha_bad = 1.0 if alpha_bad is None else float(alpha_bad)
        alpha_good = 1.0 if alpha_good is None else float(alpha_good)
        alpha = torch.where(
            labels == 0,
            torch.tensor(alpha_bad, device=labels.device),
            torch.tensor(alpha_good, device=labels.device),
        )
        loss = loss * alpha
    return loss.mean()


class LossTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str,
        focal_gamma: float,
        focal_alpha_bad: float | None,
        focal_alpha_good: float | None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha_bad = focal_alpha_bad
        self.focal_alpha_good = focal_alpha_good

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
        elif self.loss_type == "focal":
            loss = focal_loss(
                logits,
                labels,
                gamma=self.focal_gamma,
                alpha_bad=self.focal_alpha_bad,
                alpha_good=self.focal_alpha_good,
            )
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def maybe_resize_position_embeddings(model, max_length: int) -> None:
    current = getattr(model.config, "max_position_embeddings", None)
    if current is None or max_length <= current:
        return
    try:
        model.resize_position_embeddings(max_length)
        model.config.max_position_embeddings = max_length
        print(f"Resized position embeddings to {max_length}")
        return
    except NotImplementedError:
        pass

    base_prefix = getattr(model, "base_model_prefix", "")
    base_model = getattr(model, base_prefix, None) if base_prefix else None
    embeddings = None
    if base_model is not None and hasattr(base_model, "embeddings"):
        embeddings = getattr(base_model, "embeddings")
    if embeddings is None or not hasattr(embeddings, "position_embeddings"):
        raise NotImplementedError(
            f"Position embedding resize not supported for model type {model.__class__.__name__}"
        )

    old_weight = embeddings.position_embeddings.weight.data
    old_len, hidden = old_weight.shape
    new_embed = torch.nn.Embedding(max_length, hidden).to(old_weight.device)
    new_embed.weight.data[:old_len] = old_weight
    if max_length > old_len:
        torch.nn.init.normal_(
            new_embed.weight.data[old_len:],
            mean=0.0,
            std=getattr(model.config, "initializer_range", 0.02),
        )
    embeddings.position_embeddings = new_embed
    embeddings.position_ids = torch.arange(max_length, device=old_weight.device).unsqueeze(0)
    model.config.max_position_embeddings = max_length
    print(f"Resized position embeddings from {old_len} to {max_length} (manual).")


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
        "--tokenizer-name",
        default=None,
        help="Tokenizer name (defaults to model name)",
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
    parser.add_argument(
        "--loss-type",
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
        help="Loss function type",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--focal-alpha-bad", type=float, default=None, help="Focal loss alpha for BAD class")
    parser.add_argument("--focal-alpha-good", type=float, default=None, help="Focal loss alpha for GOOD class")
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

    tokenizer_name = args.tokenizer_name or args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        if args.tokenizer_name is None and "mosaic-bert" in args.model_name:
            print(
                f"Tokenizer load failed for {tokenizer_name}; "
                "falling back to bert-base-uncased."
            )
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise exc
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
        maybe_resize_position_embeddings(model, args.max_length)

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
    trainer = LossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha_bad=args.focal_alpha_bad,
        focal_alpha_good=args.focal_alpha_good,
    )

    trainer.train()
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()

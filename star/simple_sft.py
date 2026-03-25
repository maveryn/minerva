from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from star.common import normalize_messages


class StarSFTDataset(Dataset):
    def __init__(self, parquet_path: str | Path, tokenizer, max_length: int):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx].to_dict()
        messages = normalize_messages(row.get("messages"))
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Expected SFT parquet rows to end with an assistant message")

        prompt_messages = messages[:-1]
        response_text = messages[-1]["content"] + (self.tokenizer.eos_token or "")

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        labels = ([-100] * len(prompt_ids)) + response_ids[:]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def train(
    *,
    model_path: str,
    train_path: str | Path,
    output_root: str | Path,
    max_length: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    total_epochs: float,
    total_steps: int | None,
    save_steps: int,
    logging_steps: int,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataset = StarSFTDataset(train_path, tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    if total_steps is None:
        steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, train_batch_size)))
        total_steps = max(1, int(math.ceil(steps_per_epoch * total_epochs)))
    else:
        total_steps = max(1, int(total_steps))

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    epoch = 0
    while global_step < total_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= total_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, gradient_accumulation_steps)
            loss.backward()
            running_loss += float(loss.detach().cpu())

            if (global_step + 1) % max(1, gradient_accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if logging_steps > 0 and global_step % logging_steps == 0:
                avg = running_loss / max(1, logging_steps)
                print(json.dumps({"step": global_step, "loss": avg}))
                running_loss = 0.0

    final_dir = output_root / "final" / "hf_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_path_in": model_path,
        "train_path": str(train_path),
        "output_root": str(output_root),
        "final_model_path": str(final_dir),
        "train_rows": len(dataset),
        "global_step": global_step,
        "epochs_seen": epoch,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple full-finetuning fallback for STaR SFT")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--total-epochs", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    args = parser.parse_args()

    summary = train(
        model_path=args.model_path,
        train_path=args.train_path,
        output_root=args.output_root,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        total_epochs=args.total_epochs,
        total_steps=args.total_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

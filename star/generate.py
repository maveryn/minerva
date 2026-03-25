from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from star.common import get_user_prompt, make_uid, normalize_messages, to_jsonable, write_jsonl
from star.prompting import build_star_original_messages, build_star_rationalization_messages


def _load_model(model_path: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


def _load_tokenizer(model_path: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def _prepare_messages(row: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    base_messages = normalize_messages(row.get("prompt"))
    if mode == "original":
        return build_star_original_messages(base_messages)
    if mode == "rationalization":
        gold_answer = row.get("reward_model", {}).get("ground_truth")
        if gold_answer is None:
            raise ValueError("Missing reward_model.ground_truth for rationalization mode")
        return build_star_rationalization_messages(base_messages, str(gold_answer))
    raise ValueError(f"Unknown mode: {mode}")


def generate_rows(
    *,
    parquet_path: str | Path,
    model_path: str,
    mode: str,
    round_id: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prompt_length: int,
    trust_remote_code: bool = False,
    limit: int | None = None,
    backend: str = "hf",
    gpu_memory_utilization: float = 0.9,
) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)
    rows = df.to_dict(orient="records")
    backend = str(backend).lower()
    if backend not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported generation backend: {backend}")
    tokenizer = _load_tokenizer(model_path, trust_remote_code=trust_remote_code)
    model = None
    llm = None
    device = None
    if backend == "hf":
        tokenizer, model = _load_model(model_path, trust_remote_code=trust_remote_code)
        device = next(model.parameters()).device
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    outputs: List[Dict[str, Any]] = []
    total_batches = (len(rows) + batch_size - 1) // batch_size if batch_size > 0 else 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        messages_batch = [_prepare_messages(row, mode) for row in batch_rows]
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "mode": mode,
                    "backend": backend,
                    "batch_index": (start // batch_size) + 1,
                    "total_batches": total_batches,
                    "rows_done": min(start + len(batch_rows), len(rows)),
                    "rows_total": len(rows),
                }
            ),
            flush=True,
        )
        if backend == "hf":
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": False,
            }
            if max_prompt_length and max_prompt_length > 0:
                tokenizer_kwargs.update(
                    {
                        "truncation": True,
                        "max_length": max_prompt_length,
                    }
                )
            else:
                tokenizer_kwargs["truncation"] = False
            encoded = tokenizer(texts, **tokenizer_kwargs)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p

            with torch.no_grad():
                generated = model.generate(**generate_kwargs)

            input_lengths = attention_mask.sum(dim=1).tolist()
            batch_outputs = []
            for generated_ids, input_length in zip(generated, input_lengths):
                response_ids = generated_ids[int(input_length) :]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                batch_outputs.append(response_text)
        else:
            prompt_texts = []
            for text in texts:
                if max_prompt_length and max_prompt_length > 0:
                    tokenized = tokenizer(
                        text,
                        truncation=True,
                        max_length=max_prompt_length,
                        add_special_tokens=False,
                    )
                    prompt_texts.append(tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False))
                else:
                    prompt_texts.append(text)
            generated = llm.generate(prompt_texts, sampling_params=sampling_params)
            batch_outputs = []
            for output in generated:
                if not output.outputs:
                    batch_outputs.append("")
                else:
                    batch_outputs.append((output.outputs[0].text or "").strip())

        for row_idx, (row, prompt_messages, response_text) in enumerate(
            zip(batch_rows, messages_batch, batch_outputs)
        ):
            extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            uid = make_uid(row, fallback_index=start + row_idx)
            base_messages = normalize_messages(row.get("prompt"))
            outputs.append(
                {
                    "uid": uid,
                    "round_id": round_id,
                    "mode": mode,
                    "data_source": row.get("data_source"),
                    "reward_fn": extra_info.get("reward_fn", row.get("data_source")),
                    "task": extra_info.get("task"),
                    "source_file": row.get("source_file"),
                    "source_index": extra_info.get("index", start + row_idx),
                    "prompt_nohint": get_user_prompt(base_messages),
                    "base_messages": base_messages,
                    "prompt_used_messages": prompt_messages,
                    "ground_truth": row.get("reward_model", {}).get("ground_truth"),
                    "extra_info": extra_info,
                    "response_text": response_text,
                    "generation_config": {
                        "model_path": model_path,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_prompt_length": max_prompt_length,
                        "backend": backend,
                    },
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate STaR traces from Minerva parquet")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["original", "rationalization"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-length", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    rows = generate_rows(
        parquet_path=args.parquet,
        model_path=args.model_path,
        mode=args.mode,
        round_id=args.round_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "mode": args.mode}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate LiveSQLBench SQL predictions with local HF/vLLM models."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional runtime path
    LLM = None
    SamplingParams = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_JSONL = ROOT / "sql-benchmark" / "livesqlbench" / "artifacts" / "livesqlbench_base_full_v1_full.jsonl"
DEFAULT_DATASET_ROOT = ROOT / "sql-benchmark" / "livesqlbench-base-full-v1"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_db_texts(dataset_root: Path, db_name: str) -> tuple[str, dict[str, str], list[dict]]:
    db_dir = dataset_root / db_name
    schema = (db_dir / f"{db_name}_schema.txt").read_text(encoding="utf-8")
    column_meanings = json.loads((db_dir / f"{db_name}_column_meaning_base.json").read_text(encoding="utf-8"))

    kb = []
    kb_path = db_dir / f"{db_name}_kb.jsonl"
    with kb_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            visible = {k: item[k] for k in ("id", "knowledge", "description", "definition") if k in item}
            kb.append(visible)

    return schema, {k.lower(): v for k, v in column_meanings.items()}, kb


def build_prompt(row: dict, dataset_root: Path) -> str:
    db_name = row["selected_database"]
    schema, column_meanings, kb = load_db_texts(dataset_root, db_name)
    return f"""# Database Schema:
{schema}

# Column Meanings:
{json.dumps(column_meanings, indent=2)}

# External Knowledge:
{json.dumps(kb, indent=2)}

# User Task:
{row['query']}

Generate the correct PostgreSQL to handle the user task above:
(FORMAT: You should enclose your final PostgreSQL in '```postgresql\\n[Your Generated SQLs]\\n```' in the end. Could use semicolon to separate multiple statements.)

# Your Generated SQL: 
```postgresql"""


def parse_sql(response: str) -> str:
    if "`" in response:
        for pattern in (
            r"```postgresql(.*?)```",
            r"```sql(.*?)```",
            r"```(.*?)```",
            r"(.*?)```",
            r"`(.*?)`",
        ):
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
    else:
        match = re.search(r"(SELECT\s+.*?;)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).strip(";").strip().lower()


def pick_final_sql(candidate_sqls: list[str]) -> tuple[str, int]:
    if not candidate_sqls:
        return "", 0
    norm_to_original = []
    for sql in candidate_sqls:
        norm_to_original.append((normalize_sql(sql), sql))
    counts = Counter(norm for norm, _ in norm_to_original)
    best_norm = max(counts, key=lambda key: (counts[key], -next(i for i, (norm, _) in enumerate(norm_to_original) if norm == key)))
    for norm, original in norm_to_original:
        if norm == best_norm:
            return original, counts[best_norm]
    return candidate_sqls[0], 1


def maybe_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_with_vllm(prompts: list[str], args: argparse.Namespace) -> list[list[str]]:
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vllm is not available in this environment.")

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    chunk_size = args.vllm_chunk_size if args.vllm_chunk_size and args.vllm_chunk_size > 0 else len(prompts)
    all_outputs: list[list[str]] = []
    for start in tqdm(range(0, len(prompts), chunk_size), desc="Generating with vLLM"):
        chunk_prompts = prompts[start : start + chunk_size]
        outputs = llm.generate(chunk_prompts, sampling_params)
        all_outputs.extend([[candidate.text for candidate in result.outputs] for result in outputs])
    return all_outputs


def generate_with_transformers(
    prompts: list[str], tokenizer: AutoTokenizer, args: argparse.Namespace
) -> list[list[str]]:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    results: list[list[str]] = []
    for prompt in tqdm(prompts, desc="Generating with transformers"):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        do_sample = args.temperature > 0
        generation_kwargs = {
            "max_new_tokens": args.max_tokens,
            "num_return_sequences": args.n,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
        )
        results.append(decoded)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LiveSQLBench SQL predictions with vLLM.")
    parser.add_argument("--model", required=True, help="HF model id or local HF model directory.")
    parser.add_argument("--output-jsonl", required=True, help="Path to write predictions JSONL.")
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_JSONL))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm-chunk-size",
        type=int,
        default=20,
        help="Number of prompts to submit to each vLLM generate() call. Smaller values are more stable on long SQL prompts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers"),
        default="auto",
        help="Generation backend. 'auto' tries vllm first, then falls back to transformers.",
    )
    args = parser.parse_args()

    dataset_jsonl = Path(args.dataset_jsonl)
    dataset_root = Path(args.dataset_root)
    rows = load_jsonl(dataset_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompts = [maybe_chat_template(tokenizer, build_prompt(row, dataset_root)) for row in rows]

    if args.backend == "transformers":
        all_responses = generate_with_transformers(prompts, tokenizer, args)
    elif args.backend == "vllm":
        all_responses = generate_with_vllm(prompts, args)
    else:
        try:
            all_responses = generate_with_vllm(prompts, args)
        except Exception as e:
            print(f"vllm generation failed, falling back to transformers: {e}")
            all_responses = generate_with_transformers(prompts, tokenizer, args)

    out_rows = []
    empty_sqls = 0

    for row, prompt, responses in tqdm(zip(rows, prompts, all_responses), total=len(rows), desc="Post-processing"):
        candidate_sqls = [parse_sql(resp) for resp in responses]
        final_sql, vote_count = pick_final_sql(candidate_sqls)
        if not normalize_sql(final_sql):
            empty_sqls += 1

        out_item = dict(row)
        out_item["prompt"] = prompt
        out_item["response"] = responses[0] if responses else ""
        out_item["responses"] = responses
        out_item["candidate_sqls"] = candidate_sqls
        out_item["pred_sqls"] = final_sql
        out_item["vote_count"] = vote_count
        out_rows.append(out_item)

    output_jsonl = Path(args.output_jsonl)
    write_jsonl(out_rows, output_jsonl)

    summary = {
        "model": args.model,
        "rows": len(out_rows),
        "limit": args.limit,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "empty_final_sql_count": empty_sqls,
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_root": str(dataset_root),
    }
    summary_path = output_jsonl.with_name("generation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_jsonl}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

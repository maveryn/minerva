#!/usr/bin/env python3
"""Generate responses with and without ACR answer hints for Minerva prompts."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import (
    ModelConfig,
    build_acr_prompt_text,
    build_chat_messages,
    extract_predicted,
    iter_jsonl,
    load_cti_system_prompt,
    load_model,
    render_messages_for_model,
    resolve_backend,
    reward_minerva,
)
from minerva.label_details_store import LabelDetailsStore


def load_prompts(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def ensure_row_fields(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    normalized = dict(row)
    prompt = str(normalized.get("prompt", ""))
    if not normalized.get("uid"):
        normalized["uid"] = f"{source_index}-{hash_prompt(prompt)}"
    normalized.setdefault("source_index", source_index)
    if "answer" not in normalized and "ground_truth" in normalized:
        normalized["answer"] = normalized.get("ground_truth")
    return normalized


def row_variant_key(uid: str, variant: str) -> str:
    return f"{uid}:{variant}"


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for row in iter_jsonl(path):
        uid = row.get("uid")
        if not uid:
            continue
        variant = row.get("prompt_variant") or ("hinted" if row.get("hinted") else "plain")
        keys.add(row_variant_key(str(uid), str(variant)))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate responses with and without ACR answer hints for Minerva prompts."
    )
    parser.add_argument("--input", required=True, help="Input prompts JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Model name (HF or OpenAI/Gemini)")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "hf", "vllm", "openai", "gemini"],
        help="Model backend",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens for HF models")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size")
    parser.add_argument(
        "--vllm-args",
        type=str,
        default=None,
        help="Optional JSON dict of vLLM engine args",
    )
    parser.add_argument(
        "--variants",
        default="both",
        choices=["both", "hinted", "plain"],
        help="Which prompt variants to generate",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start index into the input prompts")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of prompts to process")
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not prepend the CTI system prompt",
    )
    parser.add_argument(
        "--acr-max-details-chars",
        type=int,
        default=4048,
        help="Max chars for LABEL_REFERENCE before truncation",
    )
    parser.add_argument(
        "--acr-max-prompt-chars",
        type=int,
        default=0,
        help="Optional max prompt length (chars). 0 disables the check.",
    )
    parser.add_argument(
        "--acr-allow-id",
        action="store_true",
        help="Allow exact labels in the reasoning section",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.999,
        help="Reward threshold for considering a response correct",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    prompts = load_prompts(input_path)
    if not prompts:
        raise RuntimeError(f"no prompts found in {input_path}")
    prompts = [ensure_row_fields(row, idx) for idx, row in enumerate(prompts)]
    if args.start_index:
        prompts = prompts[max(0, args.start_index) :]
    if args.limit is not None:
        prompts = prompts[: max(0, args.limit)]

    vllm_kwargs = {}
    if args.vllm_args:
        import json

        vllm_kwargs = json.loads(args.vllm_args)
        if not isinstance(vllm_kwargs, dict):
            raise ValueError("--vllm-args must be a JSON object")
    backend = resolve_backend(args.model, args.backend)
    model = load_model(
        ModelConfig(
            args.model,
            backend,
            args.max_new_tokens,
            batch_size=args.batch_size or 1,
            vllm_kwargs=vllm_kwargs,
        )
    )
    batch_size = args.batch_size or getattr(model, "batch_size", 1) or 1

    system_prompt = "" if args.no_system_prompt else load_cti_system_prompt().strip()
    details_store = LabelDetailsStore()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variants: list[str]
    if args.variants == "both":
        variants = ["plain", "hinted"]
    elif args.variants == "hinted":
        variants = ["hinted"]
    else:
        variants = ["plain"]

    seen = existing_keys(output_path)
    total = len(prompts) * len(variants)
    pbar = tqdm(total=total, desc=f"{args.model} generations")
    written = 0
    skipped_acr = 0

    def flush(batch: list[dict[str, Any]], apply_chat_template: bool) -> None:
        nonlocal written
        if not batch:
            return
        prompts_batch = [item["full_prompt"] for item in batch]
        responses = model.generate_batch(
            prompts_batch,
            temperature=args.temperature,
            apply_chat_template=apply_chat_template,
        )
        ts = int(time.time())
        for item, response in zip(batch, responses, strict=False):
            answer = item["answer"]
            data_source = item["reward_fn"]
            prediction = extract_predicted(data_source, response)
            reward = reward_minerva(data_source, response, answer)
            correct = reward >= args.correct_threshold
            record = {
                "uid": item["uid"],
                "source_index": item["row"].get("source_index"),
                "task": item["row"].get("task"),
                "prompt": item["prompt_used"],
                "prompt_variant": item["prompt_variant"],
                "hinted": item["hinted"],
                "answer": answer,
                "reward_fn": data_source,
                "model": args.model,
                "response": response,
                "prediction": prediction,
                "reward": reward,
                "correct": correct,
                "timestamp": ts,
            }
            acr_meta = item.get("acr_meta") or {}
            if acr_meta:
                record.update(acr_meta)
            f.write(json_dumps(record) + "\n")
            f.flush()
            seen.add(row_variant_key(item["uid"], item["prompt_variant"]))
            written += 1
            pbar.update(1)

    with output_path.open("a", encoding="utf-8") as f:
        batch: list[dict[str, Any]] = []
        batch_apply_chat_template = True
        for row in prompts:
            uid = str(row.get("uid", ""))
            prompt = str(row.get("prompt", ""))
            answer = row.get("answer")
            data_source = row.get("reward_fn") or row.get("task") or ""

            for variant in variants:
                key = row_variant_key(uid, variant)
                if key in seen:
                    pbar.update(1)
                    continue

                acr_meta = {}
                if variant == "hinted":
                    acr_prompt, acr_meta = build_acr_prompt_text(
                        prompt,
                        answer,
                        data_source,
                        row.get("task"),
                        details_store=details_store,
                        max_details_chars=args.acr_max_details_chars,
                        enforce_no_id=not args.acr_allow_id,
                        max_prompt_chars=args.acr_max_prompt_chars,
                    )
                    if acr_prompt is None:
                        skipped_acr += 1
                        pbar.update(1)
                        continue
                    prompt_used = acr_prompt
                else:
                    acr_prompt = None
                    prompt_used = prompt

                messages = build_chat_messages(system_prompt, prompt_used)
                full_prompt, used_template = render_messages_for_model(model, messages)
                apply_chat_template = not used_template
                if batch and apply_chat_template != batch_apply_chat_template:
                    flush(batch, batch_apply_chat_template)
                    batch = []
                item = {
                    "uid": uid,
                    "row": row,
                    "prompt": prompt,
                    "prompt_used": prompt_used,
                    "prompt_variant": variant,
                    "hinted": variant == "hinted",
                    "answer": answer,
                    "reward_fn": data_source,
                    "acr_prompt": acr_prompt,
                    "acr_meta": acr_meta,
                    "full_prompt": full_prompt,
                }
                batch_apply_chat_template = apply_chat_template
                batch.append(item)
                if len(batch) >= batch_size:
                    flush(batch, batch_apply_chat_template)
                    batch = []
        if batch:
            flush(batch, batch_apply_chat_template)

    pbar.close()
    print(
        f"Wrote {written} responses to {output_path} "
        f"(skipped {skipped_acr} hinted prompts)."
    )


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()

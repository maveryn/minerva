#!/usr/bin/env python3
"""Generate ACRD-style answer-conditioned responses and keep sampling until enough correct outputs."""

from __future__ import annotations

import argparse
import random
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


def count_existing(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    correct = 0
    total = 0
    for row in iter_jsonl(path):
        total += 1
        if row.get("correct") is True:
            correct += 1
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ACRD-style answer-conditioned responses for Minerva prompts."
    )
    parser.add_argument("--input", required=True, help="Input prompts JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Model name (HF or OpenAI/Gemini)")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "openai", "gemini"], help="Model backend")
    parser.add_argument("--target-correct", type=int, default=1024, help="Number of correct samples to collect")
    parser.add_argument("--max-attempts", type=int, default=20000, help="Maximum total generation attempts")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens for HF models")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not prepend the CTI system prompt",
    )
    parser.add_argument(
        "--acr-max-details-chars",
        type=int,
        default=4048,
        help="Max chars for CANONICAL_LABEL_DETAILS before truncation",
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

    backend = resolve_backend(args.model, args.backend)
    model = load_model(ModelConfig(args.model, backend, args.max_new_tokens))

    system_prompt = "" if args.no_system_prompt else load_cti_system_prompt().strip()
    details_store = LabelDetailsStore()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct_count, total_attempts = count_existing(output_path)

    if correct_count >= args.target_correct:
        print(f"Already have {correct_count} correct samples in {output_path}")
        return

    rng = random.Random(args.seed)
    pbar = tqdm(total=args.target_correct, initial=correct_count, desc=f"{args.model} correct")

    with output_path.open("a", encoding="utf-8") as f:
        while correct_count < args.target_correct and total_attempts < args.max_attempts:
            row = rng.choice(prompts)
            prompt = str(row.get("prompt", ""))
            answer = row.get("answer")
            data_source = row.get("reward_fn") or row.get("task") or ""

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
                total_attempts += 1
                continue
            messages = build_chat_messages(system_prompt, acr_prompt)
            full_prompt, used_template = render_messages_for_model(model, messages)

            response = model.generate(
                full_prompt,
                temperature=args.temperature,
                apply_chat_template=not used_template,
            )
            prediction = extract_predicted(data_source, response)
            reward = reward_minerva(data_source, response, answer)
            correct = reward >= args.correct_threshold

            record = {
                "uid": row.get("uid"),
                "source_index": row.get("source_index"),
                "task": row.get("task"),
                "prompt": prompt,
                "answer": answer,
                "reward_fn": row.get("reward_fn"),
                "model": args.model,
                "acr_prompt": acr_prompt,
                "full_prompt": full_prompt,
                "response": response,
                "prediction": prediction,
                "reward": reward,
                "correct": correct,
                "attempt": total_attempts + 1,
                "timestamp": int(time.time()),
            }
            if acr_meta:
                record.update(acr_meta)
            f.write(json_dumps(record) + "\n")
            f.flush()

            total_attempts += 1
            if correct:
                correct_count += 1
                pbar.update(1)

    pbar.close()
    print(
        f"Finished with {correct_count} correct samples after {total_attempts} attempts. "
        f"Output: {output_path}"
    )


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()

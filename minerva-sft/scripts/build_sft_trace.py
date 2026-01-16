#!/usr/bin/env python3
"""Generate SFT traces from Minerva train split with ACRD-style retry."""

from __future__ import annotations

import argparse
import json
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


def existing_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx = obj.get("source_index")
            if isinstance(idx, int):
                done.add(idx)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT traces from Minerva split JSONL.")
    parser.add_argument("--input", required=True, help="Input train split JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL for SFT traces")
    parser.add_argument("--model", required=True, help="Model name (HF/OpenAI/Gemini)")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "openai", "gemini"], help="Model backend")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens for HF models")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples")
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not prepend the CTI system prompt",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=0.999,
        help="Reward threshold for considering a response correct",
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
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = resolve_backend(args.model, args.backend)
    model = load_model(ModelConfig(args.model, backend, args.max_new_tokens))

    system_prompt = "" if args.no_system_prompt else load_cti_system_prompt().strip()
    details_store = LabelDetailsStore()

    done = existing_indices(output_path)

    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))
    to_process = total_lines - len(done)
    if args.limit is not None:
        to_process = min(to_process, args.limit)

    processed = 0
    with output_path.open("a", encoding="utf-8") as fout:
        pbar = tqdm(total=to_process, desc="sft-trace")
        for idx, row in enumerate(iter_jsonl(input_path)):
            if idx in done:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            prompt = str(row.get("prompt", ""))
            answer = row.get("answer", row.get("ground_truth"))
            data_source = row.get("reward_fn") or row.get("task") or ""

            messages = build_chat_messages(system_prompt, prompt)
            full_prompt, used_template = render_messages_for_model(model, messages)
            response_1 = model.generate(
                full_prompt,
                temperature=args.temperature,
                apply_chat_template=not used_template,
            )
            pred_1 = extract_predicted(data_source, response_1)
            reward_1 = reward_minerva(data_source, response_1, answer)
            correct_1 = reward_1 >= args.correct_threshold

            response_2 = None
            pred_2 = None
            reward_2 = None
            correct_2 = None
            acr_prompt = None
            acr_meta: dict[str, Any] = {}

            if not correct_1:
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
                if acr_prompt is not None:
                    messages = build_chat_messages(system_prompt, acr_prompt)
                    full_prompt_2, used_template_2 = render_messages_for_model(model, messages)
                    response_2 = model.generate(
                        full_prompt_2,
                        temperature=args.temperature,
                        apply_chat_template=not used_template_2,
                    )
                    pred_2 = extract_predicted(data_source, response_2)
                    reward_2 = reward_minerva(data_source, response_2, answer)
                    correct_2 = reward_2 >= args.correct_threshold

            if correct_1:
                final_source = "first"
                final_response = response_1
                final_reward = reward_1
            elif correct_2:
                final_source = "second"
                final_response = response_2
                final_reward = reward_2
            else:
                final_source = "ground_truth"
                final_response = answer
                final_reward = 1.0

            record = {
                "source_index": idx,
                "task": row.get("task"),
                "prompt": prompt,
                "answer": answer,
                "reward_fn": row.get("reward_fn"),
                "model": args.model,
                "first_prompt": full_prompt,
                "response_first": response_1,
                "prediction_first": pred_1,
                "reward_first": reward_1,
                "correct_first": correct_1,
                "acr_prompt": acr_prompt,
                "response_second": response_2,
                "prediction_second": pred_2,
                "reward_second": reward_2,
                "correct_second": correct_2,
                "final_source": final_source,
                "final_response": final_response,
                "final_reward": final_reward,
                "timestamp": int(time.time()),
            }
            if acr_meta:
                record.update(acr_meta)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            processed += 1
            pbar.update(1)

        pbar.close()

    print(f"Wrote {processed} rows to {output_path}")


if __name__ == "__main__":
    main()

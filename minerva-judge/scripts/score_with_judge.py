#!/usr/bin/env python3
"""Score responses with the ACR rubric using a judge model (e.g., GPT-5)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import (
    ModelConfig,
    build_judge_prompt,
    compute_rubric_score,
    extract_json_object,
    iter_jsonl,
    load_cti_system_prompt,
    load_model,
    load_rubric_prompt,
    parse_rubric,
    resolve_backend,
)


def iter_inputs(paths: list[Path]):
    for path in paths:
        yield from iter_jsonl(path)


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for row in iter_jsonl(path):
        key = row.get("judge_key")
        if isinstance(key, str):
            keys.add(key)
    return keys


def build_key(row: dict[str, Any]) -> str:
    uid = row.get("uid")
    model = row.get("model")
    attempt = row.get("attempt")
    return f"{uid}:{model}:{attempt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Minerva responses with a judge model.")
    parser.add_argument("--input", action="append", required=True, help="Input responses JSONL (repeatable)")
    parser.add_argument("--output", required=True, help="Output judged JSONL")
    parser.add_argument("--judge-model", required=True, help="Judge model name (e.g., gpt-5)")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "openai", "gemini"], help="Judge backend")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens for HF judge models")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include incorrect responses (default: only correct responses)",
    )
    parser.add_argument("--include-system", action="store_true", help="Prefix the CTI system prompt in QUESTION")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = resolve_backend(args.judge_model, args.backend)
    model = load_model(ModelConfig(args.judge_model, backend, args.max_new_tokens))

    rubric_template = load_rubric_prompt()
    system_prompt = load_cti_system_prompt().strip() if args.include_system else ""

    seen = existing_keys(output_path)

    total = 0
    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for row in tqdm(iter_inputs(input_paths), desc="judging"):
            total += 1
            if args.limit is not None and total > args.limit:
                break
            if (not args.include_incorrect) and not row.get("correct"):
                continue
            judge_key = build_key(row)
            if judge_key in seen:
                continue

            question = str(row.get("prompt", ""))
            if system_prompt:
                question = f"{system_prompt}\n\n{question}"
            response = str(row.get("response", ""))

            judge_prompt = build_judge_prompt(rubric_template, question, response)
            judge_response = model.generate(judge_prompt, temperature=args.temperature)

            obj = extract_json_object(judge_response)
            rubric = parse_rubric(obj or {})
            rubric_valid = rubric is not None
            rubric_score = compute_rubric_score(rubric) if rubric_valid else None

            out = dict(row)
            out.update(
                {
                    "judge_key": judge_key,
                    "judge_model": args.judge_model,
                    "judge_prompt": judge_prompt,
                    "judge_response": judge_response,
                    "rubric_valid": rubric_valid,
                    "rubric_score": rubric_score,
                    "rubric": rubric,
                }
            )

            f.write(json_dumps(out) + "\n")
            f.flush()
            written += 1

    print(f"Wrote {written} judged rows to {output_path}")


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect ChatGPT responses for Minerva split data and store verified outputs."""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "rlvr"))

from athena_eval.models import OpenAIModel  # noqa: E402
from verl.utils.reward_score import reward_minerva as reward_minerva_mod  # noqa: E402


def load_cti_system_prompt() -> str:
    import importlib.util

    prompt_path = PROJECT_ROOT / "rlvr" / "mydata" / "data_prepare" / "prompt.py"
    spec = importlib.util.spec_from_file_location("minerva_cti_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prompt module from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "CTI_SYSTEM_PROMPT")


def load_api_key_from_file(path: Path, var_name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"api key file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("export "):
                raw = raw[len("export ") :].strip()
            if "=" not in raw:
                continue
            key, val = raw.split("=", 1)
            if key.strip() == var_name:
                return val.strip()
    raise ValueError(f"{var_name} not found in {path}")


def existing_indices(path: Path) -> set[int]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            idx = obj.get("index")
            if isinstance(idx, int):
                done.add(idx)
    return done


def format_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(a) for a in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def build_chatgpt_prompt(system_prompt: str, prompt: str) -> str:
    system_text = (system_prompt or "").strip()
    if not system_text:
        return prompt
    return f"{system_text}\n\n{prompt}"


def build_second_prompt(prompt: str, answer: Any) -> str:
    answer_text = format_answer(answer)
    suffix = (
        "\n\n"
        f"You are given the correct answer: {answer_text}.\n"
        "Reason step by step to arrive at the correct answer, but do not mention it explicitly.\n"
        "Do not mention the ground truth label in the reasoning process.\n"
        "Provide the final answer."
    )
    return prompt.rstrip() + suffix


def call_model(
    model: OpenAIModel,
    prompt: str,
    max_retries: int,
    base_sleep: float,
) -> str:
    delay = max(0.0, base_sleep)
    for attempt in range(max_retries):
        try:
            return model.generate(prompt)
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            jitter = random.uniform(0.0, 0.25)
            time.sleep(delay + jitter)
            delay = delay * 2 if delay > 0 else 1.0
    return ""


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def find_duplicate_prompt_hashes(path: Path) -> dict[str, list[int]]:
    seen: dict[str, int] = {}
    dupes: dict[str, list[int]] = {}
    for idx, row in enumerate(iter_jsonl(path)):
        prompt = str(row.get("prompt", ""))
        digest = hash_prompt(prompt)
        if digest in seen:
            dupes.setdefault(digest, [seen[digest]]).append(idx)
        else:
            seen[digest] = idx
    return dupes


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ChatGPT responses for Minerva split data")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "dataset" / "minerva_split" / "train.jsonl"),
        help="Input JSONL (default: dataset/minerva_split/train.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "dataset" / "minerva_split" / "train_chatgpt.jsonl"),
        help="Output JSONL (default: dataset/minerva_split/train_chatgpt.jsonl)",
    )
    parser.add_argument("--model", default="gpt-nano", help="OpenAI model name (default: gpt-nano)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    parser.add_argument("--sleep", type=float, default=0.0, help="Base sleep between retries")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume and overwrite output")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    dupes = find_duplicate_prompt_hashes(input_path)
    if dupes:
        sample = next(iter(dupes.values()))
        print(
            f"Warning: duplicate prompt hashes found; example indices: {sample}",
            file=sys.stderr,
        )

    if args.no_resume and output_path.exists():
        output_path.unlink()

    done = existing_indices(output_path)

    api_key_path = PROJECT_ROOT / "api_keys"
    api_key = load_api_key_from_file(api_key_path, "OPENAI_API_KEY")
    model = OpenAIModel(args.model, api_key=api_key)

    system_prompt = load_cti_system_prompt()

    reward_fn = reward_minerva_mod.reward_minerva
    extract_pred = getattr(reward_minerva_mod, "_extract_predicted")

    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))

    processed = 0
    with output_path.open("a", encoding="utf-8") as fout:
        pbar = tqdm(total=total_lines, desc=f"{args.model} minerva_split")
        for idx, row in enumerate(iter_jsonl(input_path)):
            pbar.update(1)
            if idx in done:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            prompt = row.get("prompt", "")
            answer = row.get("answer", "")
            data_source = row.get("reward_fn") or row.get("task") or ""
            prompt_hash = hash_prompt(str(prompt))

            chatgpt_prompt_first = build_chatgpt_prompt(system_prompt, str(prompt))
            first_response = call_model(model, chatgpt_prompt_first, args.max_retries, args.sleep)
            first_pred = extract_pred(data_source, first_response)
            first_reward = reward_fn(data_source, first_response, answer)
            first_correct = first_reward >= 0.999

            chatgpt_prompt_second = None
            second_response = None
            second_pred = None
            second_reward = None
            second_correct = None

            if not first_correct:
                second_prompt = build_second_prompt(str(prompt), answer)
                chatgpt_prompt_second = build_chatgpt_prompt(system_prompt, second_prompt)
                second_response = call_model(model, chatgpt_prompt_second, args.max_retries, args.sleep)
                second_pred = extract_pred(data_source, second_response)
                second_reward = reward_fn(data_source, second_response, answer)
                second_correct = second_reward >= 0.999

            if first_correct:
                correct_response = first_response
                correct_source = "first"
            elif second_correct:
                correct_response = second_response
                correct_source = "second"
            else:
                correct_response = answer
                correct_source = "ground_truth"

            record = dict(row)
            record.update(
                {
                    "index": idx,
                    "model": args.model,
                    "prompt_hash": prompt_hash,
                    "chatgpt_prompt_first": chatgpt_prompt_first,
                    "response_first": first_response,
                    "prediction_first": first_pred,
                    "reward_first": first_reward,
                    "correct_first": first_correct,
                    "chatgpt_prompt_second": chatgpt_prompt_second,
                    "response_second": second_response,
                    "prediction_second": second_pred,
                    "reward_second": second_reward,
                    "correct_second": second_correct,
                    "correct_response": correct_response,
                    "correct_response_source": correct_source,
                }
            )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            processed += 1

        pbar.close()


if __name__ == "__main__":
    main()

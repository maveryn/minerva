#!/usr/bin/env python3
"""Run pairwise CTI preference judging over all_models_correct_samples.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

PAIRWISE_TEMPLATE = (
    "You are a cyber threat intelligence (CTI) analyst expert. "
    "You are given a question and two LLM responses. "
    "Both responses are verifiably correct. "
    "Select the response you would prefer to receive for CTI analysis.\n\n"
    "Question:\n{question}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Return JSON only:\n"
    "{{\"winner\": \"A\"|\"B\"|\"tie\", \"rationale\": \"<short rationale>\"}}"
)


@dataclass
class Job:
    sample_index: int
    sample_key: str
    task: str
    subtask: str | None
    prompt: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    pair_key: str
    judge_prompt: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_script_path() -> None:
    scripts_dir = repo_root() / "minerva-judge" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))


def resolve_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    lowered = model_name.lower()
    if lowered.startswith("gpt-") or lowered.startswith("o1") or lowered.startswith("o3"):
        return "openai"
    if lowered.startswith("gemini"):
        return "gemini"
    return "hf"


def compute_sample_key(task: str, subtask: str | None, prompt: str) -> str:
    blob = f"{task}||{subtask or ''}||{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def choose_order(seed: int, sample_key: str, model_1: str, model_2: str) -> tuple[str, str]:
    token = f"{seed}|{sample_key}|{model_1}|{model_2}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    if digest[0] % 2 == 0:
        return model_1, model_2
    return model_2, model_1


def build_pair_key(sample_key: str, model_1: str, model_2: str) -> str:
    a, b = sorted([model_1, model_2])
    return f"{sample_key}:{a}:{b}"


def build_prompt(question: str, response_a: str, response_b: str) -> str:
    return PAIRWISE_TEMPLATE.format(
        question=question.strip(),
        response_a=response_a.strip(),
        response_b=response_b.strip(),
    )


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def iter_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("pair_key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def chunked(items: list[Job], size: int) -> Iterable[list[Job]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_winner(obj: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not obj:
        return None, None
    winner_raw = obj.get("winner")
    rationale = obj.get("rationale")
    winner = str(winner_raw or "").strip().upper()
    if winner in {"A", "B"}:
        return winner, str(rationale or "").strip()
    if winner in {"TIE", "T"}:
        return "TIE", str(rationale or "").strip()
    return None, str(rationale or "").strip() if rationale is not None else None


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


class OpenAIJudge:
    def __init__(self, name: str) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise ImportError("openai package is required for OpenAI judge models") from exc
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        self.name = name

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if self.name.startswith("gpt-5"):
            resp = self.client.responses.create(
                model=self.name,
                input=prompt,
            )
            return (getattr(resp, "output_text", "") or "").strip()
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def generate_batch(self, prompts: list[str], temperature: float = 0.0, **_: object) -> list[str]:
        return [self.generate(prompt, temperature=temperature) for prompt in prompts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise CTI preference judging.")
    parser.add_argument(
        "--input",
        default="all_models_correct_samples.json",
        help="Input JSON file with prompts + responses",
    )
    parser.add_argument(
        "--output",
        default="pairwise_judge_results.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.2",
        help="Judge model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "openai", "hf", "vllm", "gemini"],
        help="Backend for judge model",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for HF models")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed for A/B order shuffling")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total pairs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).resolve().parent / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    samples = load_samples(input_path)
    if not samples:
        raise ValueError("No samples found")

    model_names = sorted(samples[0].get("responses", {}).keys())
    if not model_names:
        raise ValueError("No model responses found in samples")

    pairs = list(combinations(model_names, 2))

    backend = resolve_backend(args.judge_model, args.backend)
    if backend == "openai":
        model = OpenAIJudge(args.judge_model)
    else:
        add_script_path()
        from common import ModelConfig, load_model  # noqa: E402

        model = load_model(
            ModelConfig(
                args.judge_model,
                backend,
                args.max_new_tokens,
                batch_size=max(1, args.batch_size),
            )
        )

    existing = iter_existing_keys(output_path)

    jobs: list[Job] = []
    for sample_index, sample in enumerate(samples):
        task = str(sample.get("task") or "")
        subtask = sample.get("subtask")
        prompt = str(sample.get("prompt") or "")
        responses = sample.get("responses") or {}
        if not isinstance(responses, dict):
            continue
        sample_key = compute_sample_key(task, subtask, prompt)
        for model_1, model_2 in pairs:
            resp_1 = responses.get(model_1)
            resp_2 = responses.get(model_2)
            if not isinstance(resp_1, str) or not isinstance(resp_2, str):
                continue
            pair_key = build_pair_key(sample_key, model_1, model_2)
            if pair_key in existing:
                continue
            model_a, model_b = choose_order(args.seed, sample_key, model_1, model_2)
            response_a = responses[model_a]
            response_b = responses[model_b]
            judge_prompt = build_prompt(prompt, response_a, response_b)
            jobs.append(
                Job(
                    sample_index=sample_index,
                    sample_key=sample_key,
                    task=task,
                    subtask=subtask,
                    prompt=prompt,
                    model_a=model_a,
                    model_b=model_b,
                    response_a=response_a,
                    response_b=response_b,
                    pair_key=pair_key,
                    judge_prompt=judge_prompt,
                )
            )
            if args.limit is not None and len(jobs) >= args.limit:
                break
        if args.limit is not None and len(jobs) >= args.limit:
            break

    total = len(jobs)
    if total == 0:
        print("No new pairs to judge.")
        return

    batch_size = max(1, args.batch_size)
    iterator = chunked(jobs, batch_size)
    if tqdm is not None:
        total_batches = (total + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=total_batches, desc="judging", unit="batch")

    with output_path.open("a", encoding="utf-8") as out_f:
        for batch in iterator:
            prompts = [job.judge_prompt for job in batch]
            outputs = model.generate_batch(prompts, temperature=args.temperature, apply_chat_template=True)
            for job, judge_response in zip(batch, outputs, strict=False):
                parsed = extract_json_object(judge_response)
                winner, rationale = parse_winner(parsed)
                if winner == "A":
                    winner_model = job.model_a
                elif winner == "B":
                    winner_model = job.model_b
                elif winner == "TIE":
                    winner_model = "tie"
                else:
                    winner_model = "parse_error"

                row = {
                    "pair_key": job.pair_key,
                    "sample_index": job.sample_index,
                    "sample_key": job.sample_key,
                    "task": job.task,
                    "subtask": job.subtask,
                    "prompt": job.prompt,
                    "model_a": job.model_a,
                    "model_b": job.model_b,
                    "response_a": job.response_a,
                    "response_b": job.response_b,
                    "judge_model": args.judge_model,
                    "judge_prompt": job.judge_prompt,
                    "judge_response": judge_response,
                    "parsed": parsed,
                    "winner": winner if winner is not None else "parse_error",
                    "winner_model": winner_model,
                    "rationale": rationale,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"Wrote {total} judgments to {output_path}")


if __name__ == "__main__":
    main()

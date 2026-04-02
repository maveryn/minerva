#!/usr/bin/env python3
"""Score blind pointwise annotation items with a judge model using the human rubric."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


ROOT = Path(__file__).resolve().parent
RUBRIC_PATH = ROOT / "POINTWISE_RUBRIC.md"
DEFAULT_SUBSET = "pointwise_pilot_50_correct_only"
DEFAULT_MODEL = "gpt-5.2"

SCORE_FIELDS = (
    "writing_quality_score",
    "evidence_use_score",
    "cti_concept_focus_score",
)


PROMPT_TEMPLATE = """You are an expert cyber threat intelligence (CTI) analyst evaluating the quality of a single model response.

Your job is to score the response using the rubric below.

Rubric:
{rubric}

Task: {task}
Subtask: {subtask}

Prompt:
{prompt}

Model Response:
{response}

Return JSON only with this exact schema:
{{
  "writing_quality_score": 1,
  "evidence_use_score": 1,
  "cti_concept_focus_score": 1,
  "notes": "short explanation"
}}

Rules:
- Each score must be an integer from 1 to 4.
- Use the rubric exactly.
- Keep notes short and concrete.
- Return JSON only. No markdown, no code fences, no extra text.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def normalize_scores(obj: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in SCORE_FIELDS:
        value = obj.get(field)
        try:
            score = int(value)
        except Exception as exc:
            raise ValueError(f"Missing or invalid {field}: {value!r}") from exc
        if score < 1 or score > 4:
            raise ValueError(f"{field} out of range: {score}")
        normalized[field] = score
    normalized["notes"] = str(obj.get("notes") or "").strip()
    normalized["total_score"] = sum(normalized[field] for field in SCORE_FIELDS)
    return normalized


class OpenAIJudge:
    def __init__(self, model_name: str) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("openai package is required") from exc
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if self.model_name.startswith("gpt-5"):
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
            )
            return (getattr(response, "output_text", "") or "").strip()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class AnthropicJudge:
    def __init__(self, model_name: str) -> None:
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("anthropic package is required") from exc
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()


def resolve_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    lowered = model_name.lower()
    if lowered.startswith("gpt-") or lowered.startswith("o1") or lowered.startswith("o3"):
        return "openai"
    if lowered.startswith("claude"):
        return "anthropic"
    raise ValueError(
        f"Could not infer backend for model {model_name!r}. Pass --backend explicitly."
    )


def default_output_name(model_name: str) -> str:
    if model_name == "gpt-5.2":
        return "gpt52_scores.jsonl"
    slug = "".join(ch if ch.isalnum() else "_" for ch in model_name.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return f"{slug}_scores.jsonl"


def load_judge(model_name: str, backend: str) -> Any:
    resolved = resolve_backend(model_name, backend)
    if resolved == "openai":
        return OpenAIJudge(model_name)
    if resolved == "anthropic":
        return AnthropicJudge(model_name)
    raise ValueError(f"Unsupported backend: {resolved}")


def build_prompt(item: dict[str, Any], rubric_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        rubric=rubric_text.strip(),
        task=str(item.get("task") or ""),
        subtask=str(item.get("subtask") or "") or "(none)",
        prompt=str(item.get("prompt") or "").strip(),
        response=str(item.get("response") or "").strip(),
    )


def iter_existing_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    item_ids: set[str] = set()
    for row in load_jsonl(path):
        item_id = str(row.get("item_id") or "")
        parse_error = bool(row.get("parse_error"))
        if item_id and not parse_error:
            item_ids.add(item_id)
    return item_ids


def score_item(
    judge: Any,
    item: dict[str, Any],
    rubric_text: str,
    max_attempts: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    prompt = build_prompt(item, rubric_text)
    last_error: str | None = None
    raw_output = ""

    for attempt in range(1, max_attempts + 1):
        try:
            raw_output = judge.generate(prompt)
            parsed = extract_json_object(raw_output)
            if parsed is None:
                raise ValueError("Could not extract JSON object from model output")
            normalized = normalize_scores(parsed)
            return {
                "item_id": str(item.get("item_id") or ""),
                "source_item_id": str(item.get("source_item_id") or ""),
                "source_prompt_id": str(item.get("source_prompt_id") or ""),
                "task": str(item.get("task") or ""),
                "subtask": str(item.get("subtask") or ""),
                "word_count": int(item.get("word_count") or 0),
                "judge_model": judge.model_name,
                **normalized,
                "parse_error": False,
                "raw_output": raw_output,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_attempts:
                time.sleep(retry_sleep_s)

    return {
        "item_id": str(item.get("item_id") or ""),
        "source_item_id": str(item.get("source_item_id") or ""),
        "source_prompt_id": str(item.get("source_prompt_id") or ""),
        "task": str(item.get("task") or ""),
        "subtask": str(item.get("subtask") or ""),
        "word_count": int(item.get("word_count") or 0),
        "judge_model": judge.model_name,
        "writing_quality_score": None,
        "evidence_use_score": None,
        "cti_concept_focus_score": None,
        "total_score": None,
        "notes": "",
        "parse_error": True,
        "error": last_error or "unknown error",
        "raw_output": raw_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=DEFAULT_SUBSET, help="Subset directory name")
    parser.add_argument(
        "--input",
        help="Optional override for the blind input JSONL. Defaults to <subset>/blind_responses.jsonl",
    )
    parser.add_argument(
        "--output",
        help="Optional override for the judge scores JSONL. Defaults to <subset>/<model>_scores.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory for judge outputs. Used only when --output is not set.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_MODEL, help="Judge model name")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "openai", "anthropic"],
        help="Judge backend",
    )
    parser.add_argument("--limit", type=int, help="Optional cap on number of items")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per item")
    parser.add_argument("--retry-sleep-s", type=float, default=1.5, help="Sleep between retries")
    args = parser.parse_args()

    subset_dir = ROOT / args.subset
    input_path = Path(args.input) if args.input else subset_dir / "blind_responses.jsonl"
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_path = Path(args.output_dir) / default_output_name(args.judge_model)
    else:
        output_path = subset_dir / default_output_name(args.judge_model)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(f"Missing rubric file: {RUBRIC_PATH}")

    if args.overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(input_path)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("No items to score")

    rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")
    existing_item_ids = iter_existing_item_ids(output_path)
    pending_items = [item for item in items if str(item.get("item_id") or "") not in existing_item_ids]

    judge = load_judge(args.judge_model, args.backend)
    iterator: Any = pending_items
    if tqdm is not None:
        iterator = tqdm(pending_items, desc=f"Scoring with {args.judge_model}")

    with output_path.open("a", encoding="utf-8") as f:
        for item in iterator:
            row = score_item(
                judge=judge,
                item=item,
                rubric_text=rubric_text,
                max_attempts=args.max_attempts,
                retry_sleep_s=args.retry_sleep_s,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(json.dumps(
        {
            "subset": args.subset,
            "judge_model": args.judge_model,
            "backend": resolve_backend(args.judge_model, args.backend),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "requested_items": len(items),
            "scored_this_run": len(pending_items),
            "existing_completed_before_run": len(existing_item_ids),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

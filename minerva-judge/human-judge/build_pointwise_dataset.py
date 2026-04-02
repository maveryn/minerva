#!/usr/bin/env python3
"""Build a pointwise human-judge dataset from the shared-correct llmbench pool."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import runpy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
SOURCE_SCRIPT = LLMBENCH_ROOT / "response-quality-judge" / "generate_all_correct_samples.py"
OUTPUT_DIR = ROOT / "pointwise_subset_238"

TASKS = ("VSP", "RCM", "ATE", "MCQ3k", "CyberMetric")
MIN_WORDS = 20
MIN_MODELS_AT_THRESHOLD = 4
PER_TASK_CAP = 50

SELECTION_SEED = 20260330
SLOT_ASSIGNMENT_SEED = 20260331
ANNOTATION_ORDER_SEED = 20260332


@dataclass(frozen=True)
class PromptEntry:
    task: str
    subtask: str | None
    prompt: str
    responses: dict[str, str]
    sample_key: str
    eligible_model_count: int


def compute_sample_key(task: str, subtask: str | None, prompt: str) -> str:
    blob = f"{task}||{subtask or ''}||{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def word_count(text: str) -> int:
    return len((text or "").split())


def task_seed(task: str) -> int:
    return SELECTION_SEED + sum((index + 1) * ord(ch) for index, ch in enumerate(task))


def load_source_namespace() -> dict[str, Any]:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing source script: {SOURCE_SCRIPT}")
    return runpy.run_path(str(SOURCE_SCRIPT))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prompt_pool(
    collect_base_task: Any,
) -> tuple[dict[str, int], dict[str, list[PromptEntry]], list[str]]:
    eligible_counts: dict[str, int] = {}
    selected_by_task: dict[str, list[PromptEntry]] = {}
    model_names: list[str] | None = None

    for task in TASKS:
        pool: list[PromptEntry] = []
        for raw in collect_base_task(task):
            responses = {
                str(model_name): str(response or "")
                for model_name, response in (raw.get("responses") or {}).items()
            }
            if model_names is None:
                model_names = sorted(responses.keys())

            eligible_model_count = sum(
                word_count(response) >= MIN_WORDS for response in responses.values()
            )
            if eligible_model_count < MIN_MODELS_AT_THRESHOLD:
                continue

            prompt = str(raw.get("prompt") or "")
            subtask = raw.get("subtask")
            pool.append(
                PromptEntry(
                    task=task,
                    subtask=subtask if subtask is not None else None,
                    prompt=prompt,
                    responses=responses,
                    sample_key=compute_sample_key(task, subtask, prompt),
                    eligible_model_count=eligible_model_count,
                )
            )

        pool.sort(key=lambda entry: entry.sample_key)
        rng = random.Random(task_seed(task))
        rng.shuffle(pool)

        selected = pool[: min(PER_TASK_CAP, len(pool))]
        selected.sort(key=lambda entry: entry.sample_key)

        eligible_counts[task] = len(pool)
        selected_by_task[task] = selected

    if not model_names:
        raise ValueError("No models found in source entries")

    return eligible_counts, selected_by_task, model_names


def build_outputs(
    selected_by_task: dict[str, list[PromptEntry]],
    model_names: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prompts = [
        entry
        for task in TASKS
        for entry in selected_by_task.get(task, [])
    ]
    prompts.sort(key=lambda entry: (TASKS.index(entry.task), entry.sample_key))

    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_prompts: list[dict[str, Any]] = []

    for prompt_index, entry in enumerate(prompts, start=1):
        prompt_id = f"pw_prompt_{prompt_index:04d}"
        model_rows = [
            {
                "model_name": model_name,
                "response": entry.responses[model_name],
                "word_count": word_count(entry.responses[model_name]),
            }
            for model_name in model_names
        ]

        slot_rows = list(model_rows)
        random.Random(SLOT_ASSIGNMENT_SEED + prompt_index).shuffle(slot_rows)

        blind_prompt_responses: list[dict[str, Any]] = []
        named_prompt_responses: list[dict[str, Any]] = []

        for slot_index, model_row in enumerate(slot_rows, start=1):
            item_id = f"{prompt_id}_resp_{slot_index}"
            blind_row = {
                "item_id": item_id,
                "prompt_id": prompt_id,
                "task": entry.task,
                "subtask": entry.subtask or "",
                "prompt": entry.prompt,
                "response": model_row["response"],
                "word_count": model_row["word_count"],
            }
            blind_responses.append(blind_row)
            blind_prompt_responses.append(
                {
                    "item_id": item_id,
                    "slot": slot_index,
                    "response": model_row["response"],
                    "word_count": model_row["word_count"],
                }
            )
            named_prompt_responses.append(
                {
                    "item_id": item_id,
                    "slot": slot_index,
                    "model_name": model_row["model_name"],
                    "response": model_row["response"],
                    "word_count": model_row["word_count"],
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "task": entry.task,
                    "subtask": entry.subtask or "",
                    "sample_key": entry.sample_key,
                    "model_name": model_row["model_name"],
                    "word_count": model_row["word_count"],
                }
            )

        blind_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": entry.task,
                "subtask": entry.subtask or "",
                "prompt": entry.prompt,
                "eligible_model_count_ge20": entry.eligible_model_count,
                "responses": blind_prompt_responses,
            }
        )
        internal_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": entry.task,
                "subtask": entry.subtask or "",
                "sample_key": entry.sample_key,
                "prompt": entry.prompt,
                "eligible_model_count_ge20": entry.eligible_model_count,
                "responses": named_prompt_responses,
            }
        )

    ordered_blind_responses = list(blind_responses)
    ordered_blind_responses.sort(key=lambda row: row["item_id"])
    random.Random(ANNOTATION_ORDER_SEED).shuffle(ordered_blind_responses)

    csv_rows = [
        {
            "item_id": row["item_id"],
            "prompt_id": row["prompt_id"],
            "task": row["task"],
            "subtask": row["subtask"],
            "word_count": row["word_count"],
            "prompt": row["prompt"],
            "response": row["response"],
        }
        for row in ordered_blind_responses
    ]

    return blind_prompts, ordered_blind_responses, key_rows, internal_prompts, csv_rows


def main() -> None:
    namespace = load_source_namespace()
    collect_base_task = namespace["collect_base_task"]
    models = list(namespace["MODELS"].keys())

    eligible_counts, selected_by_task, discovered_models = build_prompt_pool(collect_base_task)
    if sorted(models) != sorted(discovered_models):
        raise ValueError("Model mismatch between source config and collected entries")

    blind_prompts, blind_responses, key_rows, internal_prompts, csv_rows = build_outputs(
        selected_by_task,
        sorted(models),
    )
    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in csv_rows
    ]

    selected_prompt_counts = {
        task: len(selected_by_task.get(task, []))
        for task in TASKS
    }
    model_response_counts = Counter(row["model_name"] for row in key_rows)
    response_counts_by_task = Counter(row["task"] for row in key_rows)

    summary = {
        "source_script": str(SOURCE_SCRIPT),
        "tasks": list(TASKS),
        "model_names": sorted(models),
        "filter": {
            "all_models_correct": True,
            "min_words": MIN_WORDS,
            "min_models_at_threshold": MIN_MODELS_AT_THRESHOLD,
            "per_task_cap": PER_TASK_CAP,
        },
        "seeds": {
            "selection_seed": SELECTION_SEED,
            "slot_assignment_seed": SLOT_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
        "eligible_prompt_counts_by_task": eligible_counts,
        "selected_prompt_counts_by_task": selected_prompt_counts,
        "selected_prompt_count": sum(selected_prompt_counts.values()),
        "selected_response_count": len(key_rows),
        "selected_response_counts_by_task": dict(sorted(response_counts_by_task.items())),
        "selected_response_counts_by_model": dict(sorted(model_response_counts.items())),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_responses)
    write_csv(OUTPUT_DIR / "blind_responses.csv", csv_rows)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", sorted(key_rows, key=lambda row: row["item_id"]))
    write_jsonl(
        OUTPUT_DIR / "selected_prompts_with_models.jsonl",
        internal_prompts,
    )


if __name__ == "__main__":
    main()

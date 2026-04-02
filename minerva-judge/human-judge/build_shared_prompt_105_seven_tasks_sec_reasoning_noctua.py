#!/usr/bin/env python3
"""Build a 105-prompt nested subset from the 350-prompt seven-task shared set."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "shared_prompt_350_seven_tasks_security5"
OUTPUT_DIR = ROOT / "shared_prompt_105_seven_tasks_sec_reasoning_noctua"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)
PROMPTS_PER_TASK = 15

SELECTED_MODELS = (
    "llama3-sec",
    "llama3-sec-reasoning",
    "minerva_llama8b_noctua",
)

PROMPT_SELECTION_SEED = 20260430
SLOT_ASSIGNMENT_SEED = 20260431
ANNOTATION_ORDER_SEED = 20260432


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def task_seed(task: str) -> int:
    return PROMPT_SELECTION_SEED + sum((index + 1) * ord(ch) for index, ch in enumerate(task))


def main() -> None:
    source_summary = load_json(SOURCE_DIR / "summary.json")
    source_prompts = load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")

    prompts_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_prompts:
        task = str(row["task"])
        if task in TASKS:
            prompts_by_task[task].append(row)

    selected_source_prompts: list[dict[str, Any]] = []
    for task in TASKS:
        prompt_rows = sorted(prompts_by_task[task], key=lambda row: str(row["prompt_key"]))
        rng = random.Random(task_seed(task))
        rng.shuffle(prompt_rows)
        if len(prompt_rows) < PROMPTS_PER_TASK:
            raise ValueError(f"Task {task} has only {len(prompt_rows)} prompts; need {PROMPTS_PER_TASK}")
        chosen = sorted(prompt_rows[:PROMPTS_PER_TASK], key=lambda row: str(row["prompt_key"]))
        selected_source_prompts.extend(chosen)

    selected_source_prompts.sort(key=lambda row: (TASKS.index(str(row["task"])), str(row["prompt_key"])))

    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_prompts: list[dict[str, Any]] = []

    for prompt_index, source_row in enumerate(selected_source_prompts, start=1):
        prompt_id = f"shared105_prompt_{prompt_index:04d}"
        task = str(source_row["task"])
        task_label = str(source_row.get("task_label") or source_summary.get("task_labels", {}).get(task, task))

        responses = [
            response
            for response in source_row["responses"]
            if response["model_name"] in SELECTED_MODELS
        ]
        if len(responses) != len(SELECTED_MODELS):
            found = sorted(response["model_name"] for response in responses)
            raise ValueError(f"Prompt {source_row['prompt_id']} missing expected models; found {found}")

        random.Random(SLOT_ASSIGNMENT_SEED + prompt_index).shuffle(responses)

        blind_prompt_responses: list[dict[str, Any]] = []
        internal_prompt_responses: list[dict[str, Any]] = []

        for slot_index, response in enumerate(responses, start=1):
            item_id = f"{prompt_id}_resp_{slot_index}"
            blind_entry = {
                "item_id": item_id,
                "source_item_id": str(response["item_id"]),
                "source_prompt_id": str(source_row["prompt_id"]),
                "prompt_id": prompt_id,
                "task": task,
                "task_label": task_label,
                "subtask": str(source_row.get("subtask") or ""),
                "prompt": str(source_row["prompt"]),
                "response": str(response["response"]),
                "word_count": int(response["word_count"]),
            }
            blind_responses.append(blind_entry)
            blind_prompt_responses.append(
                {
                    "item_id": item_id,
                    "slot": slot_index,
                    "response": str(response["response"]),
                    "word_count": int(response["word_count"]),
                }
            )
            internal_prompt_responses.append(
                {
                    "item_id": item_id,
                    "source_item_id": str(response["item_id"]),
                    "slot": slot_index,
                    "model_name": str(response["model_name"]),
                    "response": str(response["response"]),
                    "word_count": int(response["word_count"]),
                    "source_id": response.get("source_id"),
                    "source_answer": response.get("source_answer"),
                    "source_score": response.get("source_score"),
                    "source_mad": response.get("source_mad"),
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "source_item_id": str(response["item_id"]),
                    "source_prompt_id": str(source_row["prompt_id"]),
                    "prompt_id": prompt_id,
                    "task": task,
                    "task_label": task_label,
                    "subtask": str(source_row.get("subtask") or ""),
                    "prompt_key": str(source_row["prompt_key"]),
                    "canonical_source_id": str(source_row["canonical_source_id"]),
                    "model_name": str(response["model_name"]),
                    "word_count": int(response["word_count"]),
                    "source_id": response.get("source_id"),
                    "source_answer": response.get("source_answer"),
                    "source_score": response.get("source_score"),
                    "source_mad": response.get("source_mad"),
                }
            )

        blind_prompts.append(
            {
                "prompt_id": prompt_id,
                "source_prompt_id": str(source_row["prompt_id"]),
                "task": task,
                "task_label": task_label,
                "subtask": str(source_row.get("subtask") or ""),
                "prompt": str(source_row["prompt"]),
                "responses": blind_prompt_responses,
            }
        )
        internal_prompts.append(
            {
                "prompt_id": prompt_id,
                "source_prompt_id": str(source_row["prompt_id"]),
                "task": task,
                "task_label": task_label,
                "subtask": str(source_row.get("subtask") or ""),
                "prompt_key": str(source_row["prompt_key"]),
                "canonical_source_id": str(source_row["canonical_source_id"]),
                "prompt": str(source_row["prompt"]),
                "responses": internal_prompt_responses,
            }
        )

    ordered_blind_responses = list(blind_responses)
    random.Random(ANNOTATION_ORDER_SEED).shuffle(ordered_blind_responses)

    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in ordered_blind_responses
    ]

    summary = {
        "source_dir": str(SOURCE_DIR),
        "selected_models": list(SELECTED_MODELS),
        "tasks": list(TASKS),
        "task_labels": source_summary.get("task_labels", {}),
        "prompts_per_task": PROMPTS_PER_TASK,
        "selected_prompt_count": len(blind_prompts),
        "selected_response_count": len(key_rows),
        "selected_prompt_counts_by_task": dict(sorted(Counter(row["task"] for row in blind_prompts).items())),
        "selected_response_counts_by_model": dict(sorted(Counter(row["model_name"] for row in key_rows).items())),
        "selected_response_counts_by_task": dict(sorted(Counter(row["task"] for row in key_rows).items())),
        "source_prompt_counts_by_task": source_summary.get("selected_prompt_counts_by_task", {}),
        "seeds": {
            "prompt_selection_seed": PROMPT_SELECTION_SEED,
            "slot_assignment_seed": SLOT_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
        "model_dirs": {k: v for k, v in source_summary["model_dirs"].items() if k in SELECTED_MODELS},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", ordered_blind_responses)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_prompts_with_models.jsonl", internal_prompts)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    print(f"Wrote {len(blind_prompts)} prompts and {len(key_rows)} responses to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

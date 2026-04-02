#!/usr/bin/env python3
"""Build a small balanced pointwise pilot set for human/GPT alignment checks."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "pointwise_subset_238"
OUTPUT_DIR = ROOT / "pointwise_pilot_50"

TASKS = ("VSP", "RCM", "ATE", "MCQ3k", "CyberMetric")
TASK_QUOTA = 10
MIN_WORDS = 20

SELECTION_SEED = 20260333
MODEL_TARGET_SEED = 20260334
ANNOTATION_ORDER_SEED = 20260335


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def model_targets(model_names: list[str], total_items: int) -> dict[str, int]:
    base = total_items // len(model_names)
    remainder = total_items % len(model_names)
    targets = {model: base for model in model_names}
    order = list(model_names)
    random.Random(MODEL_TARGET_SEED).shuffle(order)
    for model in order[:remainder]:
        targets[model] += 1
    return targets


def sample_prompt_pool(
    prompts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        task = str(prompt.get("task") or "")
        if task in TASKS:
            by_task[task].append(prompt)

    selected: list[dict[str, Any]] = []
    for task in TASKS:
        pool = list(by_task[task])
        if len(pool) < TASK_QUOTA:
            raise ValueError(f"Task {task} only has {len(pool)} prompts; need {TASK_QUOTA}")
        pool.sort(key=lambda row: str(row.get("sample_key") or row.get("prompt_id") or ""))
        rng = random.Random(SELECTION_SEED + sum(ord(ch) for ch in task))
        rng.shuffle(pool)
        chosen = pool[:TASK_QUOTA]
        chosen.sort(key=lambda row: str(row.get("prompt_id") or ""))
        selected.extend(chosen)

    selected_counts = Counter(str(prompt.get("task") or "") for prompt in selected)
    return selected, dict(sorted(selected_counts.items()))


def assign_models(
    prompts: list[dict[str, Any]],
    targets: dict[str, int],
) -> dict[str, str]:
    eligible_by_prompt: dict[str, list[str]] = {}
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        eligible_models = [
            str(resp.get("model_name") or "")
            for resp in prompt.get("responses") or []
            if int(resp.get("word_count") or 0) >= MIN_WORDS
        ]
        eligible_models = sorted(model for model in eligible_models if model)
        if not prompt_id or not eligible_models:
            raise ValueError(f"Prompt missing eligible models: {prompt_id}")
        eligible_by_prompt[prompt_id] = eligible_models

    ordered_prompts = sorted(
        prompts,
        key=lambda prompt: (
            len(eligible_by_prompt[str(prompt.get("prompt_id") or "")]),
            str(prompt.get("task") or ""),
            str(prompt.get("prompt_id") or ""),
        ),
    )

    remaining = dict(targets)
    assignment: dict[str, str] = {}
    candidate_priority = list(targets.keys())
    random.Random(MODEL_TARGET_SEED).shuffle(candidate_priority)

    def feasible(index: int) -> bool:
        remaining_prompt_ids = [
            str(prompt.get("prompt_id") or "")
            for prompt in ordered_prompts[index:]
        ]
        for model_name, needed in remaining.items():
            if needed < 0:
                return False
            capacity = sum(
                1 for prompt_id in remaining_prompt_ids if model_name in eligible_by_prompt[prompt_id]
            )
            if needed > capacity:
                return False
        return True

    def backtrack(index: int) -> bool:
        if index == len(ordered_prompts):
            return all(value == 0 for value in remaining.values())

        prompt = ordered_prompts[index]
        prompt_id = str(prompt.get("prompt_id") or "")
        candidates = sorted(
            eligible_by_prompt[prompt_id],
            key=lambda model_name: (
                -remaining[model_name],
                candidate_priority.index(model_name),
                model_name,
            ),
        )
        for model_name in candidates:
            if remaining[model_name] <= 0:
                continue
            assignment[prompt_id] = model_name
            remaining[model_name] -= 1
            if feasible(index + 1) and backtrack(index + 1):
                return True
            remaining[model_name] += 1
            assignment.pop(prompt_id, None)
        return False

    if not feasible(0) or not backtrack(0):
        raise RuntimeError("Failed to assign model targets for the 50-item pilot set")

    return assignment


def build_rows(
    prompts: list[dict[str, Any]],
    assignment: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    ordered_prompts = sorted(prompts, key=lambda row: (TASKS.index(str(row["task"])), str(row["prompt_id"])))
    for index, prompt in enumerate(ordered_prompts, start=1):
        prompt_id = str(prompt["prompt_id"])
        assigned_model = assignment[prompt_id]
        chosen_response = next(
            response
            for response in prompt["responses"]
            if str(response.get("model_name") or "") == assigned_model
        )

        item_id = f"pilot50_{index:04d}"
        row = {
            "item_id": item_id,
            "source_item_id": str(chosen_response["item_id"]),
            "source_prompt_id": prompt_id,
            "task": str(prompt["task"]),
            "subtask": str(prompt.get("subtask") or ""),
            "prompt": str(prompt["prompt"]),
            "response": str(chosen_response["response"]),
            "word_count": int(chosen_response["word_count"]),
        }
        blind_rows.append(row)
        prompt_rows.append(
            {
                "item_id": item_id,
                "source_item_id": row["source_item_id"],
                "source_prompt_id": prompt_id,
                "task": row["task"],
                "subtask": row["subtask"],
                "sample_key": str(prompt.get("sample_key") or ""),
                "model_name": assigned_model,
                "prompt": row["prompt"],
                "response": row["response"],
                "word_count": row["word_count"],
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "source_item_id": row["source_item_id"],
                "source_prompt_id": prompt_id,
                "task": row["task"],
                "subtask": row["subtask"],
                "sample_key": str(prompt.get("sample_key") or ""),
                "model_name": assigned_model,
                "word_count": row["word_count"],
            }
        )

    annotation_order = list(blind_rows)
    random.Random(ANNOTATION_ORDER_SEED).shuffle(annotation_order)
    return prompt_rows, annotation_order, key_rows


def main() -> None:
    source_summary = load_json(SOURCE_DIR / "summary.json")
    prompts = load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")

    selected_prompts, task_counts = sample_prompt_pool(prompts)
    models = list(source_summary["model_names"])
    targets = model_targets(models, len(selected_prompts))
    assignment = assign_models(selected_prompts, targets)

    prompt_rows, blind_rows, key_rows = build_rows(selected_prompts, assignment)
    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_rows
    ]

    key_by_item = {row["item_id"]: row for row in key_rows}
    model_counts = Counter(key_by_item[row["item_id"]]["model_name"] for row in blind_rows)

    summary = {
        "source_dataset": str(SOURCE_DIR),
        "tasks": list(TASKS),
        "task_quota": TASK_QUOTA,
        "min_words": MIN_WORDS,
        "selection_seed": SELECTION_SEED,
        "model_target_seed": MODEL_TARGET_SEED,
        "annotation_order_seed": ANNOTATION_ORDER_SEED,
        "model_names": models,
        "model_target_counts": dict(sorted(targets.items())),
        "selected_item_count": len(blind_rows),
        "selected_prompt_counts_by_task": task_counts,
        "selected_counts_by_model": dict(sorted(model_counts.items())),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_rows)
    write_csv(
        OUTPUT_DIR / "blind_responses.csv",
        [
            {
                "item_id": row["item_id"],
                "source_item_id": row["source_item_id"],
                "source_prompt_id": row["source_prompt_id"],
                "task": row["task"],
                "subtask": row["subtask"],
                "word_count": row["word_count"],
                "prompt": row["prompt"],
                "response": row["response"],
            }
            for row in blind_rows
        ],
    )
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", sorted(key_rows, key=lambda row: row["item_id"]))
    write_jsonl(OUTPUT_DIR / "selected_items_with_models.jsonl", prompt_rows)


if __name__ == "__main__":
    main()

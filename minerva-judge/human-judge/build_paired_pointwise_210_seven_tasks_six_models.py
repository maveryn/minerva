#!/usr/bin/env python3
"""Build a balanced 210-item human set with one prompt and two model responses per item.

Design:
- 7 tasks
- 6 models => 15 unordered pairs
- 2 prompts per pair per task
- 30 unique prompts per task, 210 total items
- each prompt appears once
- each item contains two blind responses (A/B) for the same prompt
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "shared_prompt_350_seven_tasks_six_models"
OUTPUT_DIR = ROOT / "paired_pointwise_210_seven_tasks_six_models"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)
PROMPTS_PER_TASK = 30
PAIR_REPEATS_PER_TASK = 2

TASK_PROMPT_SELECTION_SEED = 20260431
PAIR_ASSIGNMENT_SEED = 20260432
SIDE_ASSIGNMENT_SEED = 20260433
ANNOTATION_ORDER_SEED = 20260434


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


def all_pairs(model_names: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(model_names):
        for right in model_names[i + 1 :]:
            pairs.append((left, right))
    return pairs


def task_seed(base: int, task: str) -> int:
    return base + sum((idx + 1) * ord(ch) for idx, ch in enumerate(task))


def select_task_prompts(
    rows: list[dict[str, Any]],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}

    for task in TASKS:
        pool = [row for row in rows if row["task"] == task]
        available_counts[task] = len(pool)
        if len(pool) < PROMPTS_PER_TASK:
            raise ValueError(f"Task {task} has only {len(pool)} prompts; need {PROMPTS_PER_TASK}")

        pool = sorted(pool, key=lambda row: row["prompt_key"])
        rng = random.Random(task_seed(TASK_PROMPT_SELECTION_SEED, task))
        rng.shuffle(pool)
        chosen = pool[:PROMPTS_PER_TASK]
        chosen.sort(key=lambda row: row["prompt_key"])
        selected.extend(chosen)

    selected.sort(key=lambda row: (TASKS.index(row["task"]), row["prompt_key"]))
    return selected, available_counts


def build_items(
    selected_prompts: list[dict[str, Any]],
    model_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = all_pairs(model_names)
    if len(pairs) * PAIR_REPEATS_PER_TASK != PROMPTS_PER_TASK:
        raise ValueError("Prompt count per task does not match pair repetition design")

    blind_items: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_items: list[dict[str, Any]] = []

    item_index = 0
    for task in TASKS:
        task_prompts = [row for row in selected_prompts if row["task"] == task]
        repeated_pairs = list(pairs) * PAIR_REPEATS_PER_TASK
        rng_pairs = random.Random(task_seed(PAIR_ASSIGNMENT_SEED, task))
        rng_pairs.shuffle(repeated_pairs)

        for prompt_row, pair in zip(task_prompts, repeated_pairs, strict=True):
            item_index += 1
            item_id = f"pp210_{item_index:04d}"
            response_map = {resp["model_name"]: resp for resp in prompt_row["responses"]}
            if pair[0] not in response_map or pair[1] not in response_map:
                raise KeyError(f"Missing pair {pair} for prompt {prompt_row['prompt_id']}")

            model_left, model_right = pair
            response_left = response_map[model_left]
            response_right = response_map[model_right]

            rng_side = random.Random(SIDE_ASSIGNMENT_SEED + item_index)
            if rng_side.random() < 0.5:
                model_a, model_b = model_left, model_right
                response_a, response_b = response_left, response_right
            else:
                model_a, model_b = model_right, model_left
                response_a, response_b = response_right, response_left

            blind_items.append(
                {
                    "item_id": item_id,
                    "task": prompt_row["task"],
                    "task_label": prompt_row["task_label"],
                    "subtask": prompt_row.get("subtask", ""),
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt": prompt_row["prompt"],
                    "response_a": response_a["response"],
                    "response_b": response_b["response"],
                    "word_count_a": response_a["word_count"],
                    "word_count_b": response_b["word_count"],
                }
            )

            internal_item = {
                "item_id": item_id,
                "task": prompt_row["task"],
                "task_label": prompt_row["task_label"],
                "subtask": prompt_row.get("subtask", ""),
                "prompt_id": prompt_row["prompt_id"],
                "prompt_key": prompt_row["prompt_key"],
                "canonical_source_id": prompt_row["canonical_source_id"],
                "prompt": prompt_row["prompt"],
                "model_a": model_a,
                "model_b": model_b,
                "source_item_id_a": response_a["item_id"],
                "source_item_id_b": response_b["item_id"],
                "response_a": response_a["response"],
                "response_b": response_b["response"],
                "word_count_a": response_a["word_count"],
                "word_count_b": response_b["word_count"],
                "source_id_a": response_a["source_id"],
                "source_id_b": response_b["source_id"],
                "source_answer_a": response_a["source_answer"],
                "source_answer_b": response_b["source_answer"],
                "source_score_a": response_a["source_score"],
                "source_score_b": response_b["source_score"],
                "source_mad_a": response_a["source_mad"],
                "source_mad_b": response_b["source_mad"],
            }
            internal_items.append(internal_item)
            key_rows.append(
                {
                    "item_id": item_id,
                    "task": prompt_row["task"],
                    "task_label": prompt_row["task_label"],
                    "subtask": prompt_row.get("subtask", ""),
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt_key": prompt_row["prompt_key"],
                    "canonical_source_id": prompt_row["canonical_source_id"],
                    "model_a": model_a,
                    "model_b": model_b,
                    "source_item_id_a": response_a["item_id"],
                    "source_item_id_b": response_b["item_id"],
                    "word_count_a": response_a["word_count"],
                    "word_count_b": response_b["word_count"],
                    "source_id_a": response_a["source_id"],
                    "source_id_b": response_b["source_id"],
                    "source_answer_a": response_a["source_answer"],
                    "source_answer_b": response_b["source_answer"],
                    "source_score_a": response_a["source_score"],
                    "source_score_b": response_b["source_score"],
                    "source_mad_a": response_a["source_mad"],
                    "source_mad_b": response_b["source_mad"],
                }
            )

    return blind_items, key_rows, internal_items


def main() -> None:
    source_summary = json.loads((SOURCE_DIR / "summary.json").read_text(encoding="utf-8"))
    model_names = list(source_summary["selected_models"])
    prompt_rows = load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")

    selected_prompts, available_counts = select_task_prompts(prompt_rows, model_names)
    blind_items, key_rows, internal_items = build_items(selected_prompts, model_names)

    ordered_blind_items = list(blind_items)
    random.Random(ANNOTATION_ORDER_SEED).shuffle(ordered_blind_items)

    annotation_template_rows = [
        {
            **row,
            "response_a_writing_quality_score": "",
            "response_a_evidence_use_score": "",
            "response_a_cti_concept_focus_score": "",
            "response_b_writing_quality_score": "",
            "response_b_evidence_use_score": "",
            "response_b_cti_concept_focus_score": "",
        }
        for row in ordered_blind_items
    ]

    pair_counts = Counter(
        tuple(sorted((row["model_a"], row["model_b"])))
        for row in internal_items
    )
    pair_counts_by_task = {
        task: dict(
            sorted(
                Counter(
                    tuple(sorted((row["model_a"], row["model_b"])))
                    for row in internal_items
                    if row["task"] == task
                ).items()
            )
        )
        for task in TASKS
    }
    model_exposures = Counter()
    for row in internal_items:
        model_exposures[row["model_a"]] += 1
        model_exposures[row["model_b"]] += 1

    summary = {
        "source_subset": str(SOURCE_DIR),
        "selected_models": model_names,
        "tasks": list(TASKS),
        "prompts_per_task": PROMPTS_PER_TASK,
        "pair_repeats_per_task": PAIR_REPEATS_PER_TASK,
        "selected_item_count": len(blind_items),
        "selected_prompt_count": len(blind_items),
        "selected_counts_by_task": dict(sorted(Counter(row["task"] for row in internal_items).items())),
        "selected_pair_counts_overall": {f"{a}|{b}": count for (a, b), count in sorted(pair_counts.items())},
        "selected_pair_counts_by_task": {
            task: {f"{a}|{b}": count for (a, b), count in counts.items()}
            for task, counts in pair_counts_by_task.items()
        },
        "selected_model_exposures": dict(sorted(model_exposures.items())),
        "available_prompt_counts_by_task": available_counts,
        "seeds": {
            "task_prompt_selection_seed": TASK_PROMPT_SELECTION_SEED,
            "pair_assignment_seed": PAIR_ASSIGNMENT_SEED,
            "side_assignment_seed": SIDE_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_items.jsonl", ordered_blind_items)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_items_with_models.jsonl", internal_items)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)

    print(f"Wrote {len(blind_items)} items to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

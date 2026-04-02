#!/usr/bin/env python3
"""Build a 168-item human subset from the 7-model shared 350-prompt set.

Design:
- 7 tasks, 7 models, 21 unordered model pairs
- 12 prompts per task -> 84 prompts total
- each prompt assigned to one model pair
- each pair appears exactly 4 times overall
- each model appears exactly 24 times overall
- each prompt expands to 2 independent pointwise items
- final annotation load: 168 single-response items
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "shared_prompt_350_seven_tasks_seven_models"
OUTPUT_DIR = ROOT / "pointwise_168_seven_tasks_seven_models"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)
PROMPTS_PER_TASK = 12
DIFFICULTY_BUCKETS = 3
SAMPLES_PER_BUCKET = 4
PAIR_REPEAT_OVERALL = 4

PROMPT_SELECTION_SEED = 20260461
PAIR_TASK_ASSIGNMENT_SEED = 20260462
PAIR_TO_PROMPT_ASSIGNMENT_SEED = 20260463
SIDE_ASSIGNMENT_SEED = 20260464
ANNOTATION_ORDER_SEED = 20260465


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


def task_seed(base: int, task: str) -> int:
    return base + sum((idx + 1) * ord(ch) for idx, ch in enumerate(task))


def pair_key(pair: tuple[str, str]) -> str:
    return "|".join(sorted(pair))


def all_pairs(model_names: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(model_names):
        for right in model_names[i + 1 :]:
            pairs.append((left, right))
    return pairs


def prompt_difficulty_score(prompt_row: dict[str, Any]) -> float:
    values: list[float] = []
    for response in prompt_row["responses"]:
        value = response.get("source_score")
        if value is None:
            value = 0.0
        values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def split_into_buckets(rows: list[dict[str, Any]], bucket_count: int) -> list[list[dict[str, Any]]]:
    total = len(rows)
    base = total // bucket_count
    remainder = total % bucket_count
    buckets: list[list[dict[str, Any]]] = []
    start = 0
    for bucket_index in range(bucket_count):
        size = base + (1 if bucket_index < remainder else 0)
        buckets.append(rows[start : start + size])
        start += size
    return buckets


def select_task_prompts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    selected: list[dict[str, Any]] = []
    bucket_sizes: dict[str, list[int]] = {}

    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        if len(task_rows) < 50:
            raise ValueError(f"Task {task} has only {len(task_rows)} prompts; expected 50")

        ordered = sorted(
            task_rows,
            key=lambda row: (prompt_difficulty_score(row), row["prompt_key"]),
        )
        buckets = split_into_buckets(ordered, DIFFICULTY_BUCKETS)
        bucket_sizes[task] = [len(bucket) for bucket in buckets]
        task_selected: list[dict[str, Any]] = []
        for bucket_index, bucket in enumerate(buckets):
            if len(bucket) < SAMPLES_PER_BUCKET:
                raise ValueError(
                    f"Task {task} bucket {bucket_index} has {len(bucket)} prompts; need {SAMPLES_PER_BUCKET}"
                )
            bucket = list(bucket)
            random.Random(task_seed(PROMPT_SELECTION_SEED, f"{task}|bucket{bucket_index}")).shuffle(bucket)
            chosen = bucket[:SAMPLES_PER_BUCKET]
            chosen.sort(key=lambda row: row["prompt_key"])
            task_selected.extend(chosen)

        task_selected.sort(key=lambda row: row["prompt_key"])
        if len(task_selected) != PROMPTS_PER_TASK:
            raise ValueError(f"Task {task} selected {len(task_selected)} prompts; expected {PROMPTS_PER_TASK}")
        selected.extend(task_selected)

    selected.sort(key=lambda row: (TASKS.index(row["task"]), row["prompt_key"]))
    return selected, bucket_sizes


def assign_pairs_to_tasks(model_names: list[str]) -> dict[str, list[tuple[str, str]]]:
    pairs = all_pairs(model_names)
    for attempt in range(1, 20001):
        rng = random.Random(PAIR_TASK_ASSIGNMENT_SEED + attempt)
        remaining = {pair: PAIR_REPEAT_OVERALL for pair in pairs}
        task_pairs = {task: [] for task in TASKS}

        pair_order = list(pairs)
        rng.shuffle(pair_order)
        ok = True

        for pair in pair_order:
            need = remaining[pair]
            candidates = [task for task in TASKS if len(task_pairs[task]) < PROMPTS_PER_TASK and pair not in task_pairs[task]]
            if len(candidates) < need:
                ok = False
                break

            candidates.sort(key=lambda task: (PROMPTS_PER_TASK - len(task_pairs[task]), rng.random()), reverse=True)
            chosen = candidates[:need]
            for task in chosen:
                task_pairs[task].append(pair)
            remaining[pair] = 0

        if not ok:
            continue

        if any(len(task_pairs[task]) != PROMPTS_PER_TASK for task in TASKS):
            continue
        if any(len({pair_key(pair) for pair in task_pairs[task]}) != PROMPTS_PER_TASK for task in TASKS):
            continue

        counts = Counter(pair_key(pair) for task in TASKS for pair in task_pairs[task])
        expected = len(pairs)
        if len(counts) != expected or any(count != PAIR_REPEAT_OVERALL for count in counts.values()):
            continue

        for task in TASKS:
            task_pairs[task].sort(key=pair_key)
        return task_pairs

    raise RuntimeError("Could not find a valid pair-to-task assignment")


def build_source_pair_rows(
    selected_prompts: list[dict[str, Any]],
    model_names: list[str],
) -> list[dict[str, Any]]:
    prompts_by_task: dict[str, list[dict[str, Any]]] = {task: [row for row in selected_prompts if row["task"] == task] for task in TASKS}
    task_pairs = assign_pairs_to_tasks(model_names)
    source_pair_rows: list[dict[str, Any]] = []

    item_index = 0
    for task in TASKS:
        task_prompts = list(prompts_by_task[task])
        task_pair_list = list(task_pairs[task])
        random.Random(task_seed(PAIR_TO_PROMPT_ASSIGNMENT_SEED, task)).shuffle(task_pair_list)

        for prompt_row, pair in zip(task_prompts, task_pair_list, strict=True):
            item_index += 1
            item_id = f"pw168pair_{item_index:04d}"
            response_map = {response["model_name"]: response for response in prompt_row["responses"]}
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

            source_pair_rows.append(
                {
                    "item_id": item_id,
                    "task": prompt_row["task"],
                    "task_label": prompt_row["task_label"],
                    "subtask": prompt_row.get("subtask", ""),
                    "prompt_id": prompt_row["prompt_id"],
                    "prompt_key": prompt_row["prompt_key"],
                    "canonical_source_id": prompt_row["canonical_source_id"],
                    "prompt": prompt_row["prompt"],
                    "difficulty_score_mean": prompt_difficulty_score(prompt_row),
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
            )

    if len(source_pair_rows) != len(TASKS) * PROMPTS_PER_TASK:
        raise ValueError(f"Expected 84 source pair rows, got {len(source_pair_rows)}")
    return source_pair_rows


def shuffle_without_adjacent_duplicate_prompts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = list(rows)
    random.Random(ANNOTATION_ORDER_SEED).shuffle(ordered)

    for index in range(1, len(ordered)):
        if ordered[index]["prompt_id"] != ordered[index - 1]["prompt_id"]:
            continue

        swap_index: int | None = None
        next_prompt = ordered[index + 1]["prompt_id"] if index + 1 < len(ordered) else None
        for candidate_index in range(index + 1, len(ordered)):
            candidate_prompt = ordered[candidate_index]["prompt_id"]
            if candidate_prompt == ordered[index - 1]["prompt_id"]:
                continue
            if next_prompt is not None and candidate_prompt == next_prompt:
                continue
            swap_index = candidate_index
            break

        if swap_index is None:
            for candidate_index in range(index + 1, len(ordered)):
                if ordered[candidate_index]["prompt_id"] != ordered[index - 1]["prompt_id"]:
                    swap_index = candidate_index
                    break

        if swap_index is None:
            raise ValueError("Could not separate duplicate prompts in randomized order")

        ordered[index], ordered[swap_index] = ordered[swap_index], ordered[index]

    for index in range(1, len(ordered)):
        if ordered[index]["prompt_id"] == ordered[index - 1]["prompt_id"]:
            raise ValueError("Adjacent duplicate prompts remain after shuffle")

    return ordered


def build_pointwise_rows(
    source_pair_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []
    for pair_row in source_pair_rows:
        for side in ("a", "b"):
            response_rows.append(
                {
                    "source_pair_item_id": pair_row["item_id"],
                    "source_side": side,
                    "task": pair_row["task"],
                    "task_label": pair_row["task_label"],
                    "subtask": pair_row["subtask"],
                    "prompt_id": pair_row["prompt_id"],
                    "prompt_key": pair_row["prompt_key"],
                    "canonical_source_id": pair_row["canonical_source_id"],
                    "difficulty_score_mean": pair_row["difficulty_score_mean"],
                    "prompt": pair_row["prompt"],
                    "response": pair_row[f"response_{side}"],
                    "word_count": pair_row[f"word_count_{side}"],
                    "model_name": pair_row[f"model_{side}"],
                    "paired_model_name": pair_row[f"model_{'b' if side == 'a' else 'a'}"],
                    "source_item_id": pair_row[f"source_item_id_{side}"],
                    "paired_source_item_id": pair_row[f"source_item_id_{'b' if side == 'a' else 'a'}"],
                    "source_id": pair_row[f"source_id_{side}"],
                    "paired_source_id": pair_row[f"source_id_{'b' if side == 'a' else 'a'}"],
                    "source_answer": pair_row[f"source_answer_{side}"],
                    "paired_source_answer": pair_row[f"source_answer_{'b' if side == 'a' else 'a'}"],
                    "source_score": pair_row[f"source_score_{side}"],
                    "paired_source_score": pair_row[f"source_score_{'b' if side == 'a' else 'a'}"],
                    "source_mad": pair_row[f"source_mad_{side}"],
                    "paired_source_mad": pair_row[f"source_mad_{'b' if side == 'a' else 'a'}"],
                }
            )

    ordered = shuffle_without_adjacent_duplicate_prompts(response_rows)
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []

    for index, row in enumerate(ordered, start=1):
        item_id = f"pw168_{index:04d}"
        blind_rows.append(
            {
                "item_id": item_id,
                "task": row["task"],
                "task_label": row["task_label"],
                "subtask": row["subtask"],
                "prompt": row["prompt"],
                "response": row["response"],
                "word_count": row["word_count"],
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "task": row["task"],
                "task_label": row["task_label"],
                "subtask": row["subtask"],
                "prompt_id": row["prompt_id"],
                "prompt_key": row["prompt_key"],
                "canonical_source_id": row["canonical_source_id"],
                "difficulty_score_mean": row["difficulty_score_mean"],
                "source_pair_item_id": row["source_pair_item_id"],
                "source_side": row["source_side"],
                "model_name": row["model_name"],
                "paired_model_name": row["paired_model_name"],
                "source_item_id": row["source_item_id"],
                "paired_source_item_id": row["paired_source_item_id"],
                "word_count": row["word_count"],
                "source_id": row["source_id"],
                "paired_source_id": row["paired_source_id"],
                "source_answer": row["source_answer"],
                "paired_source_answer": row["paired_source_answer"],
                "source_score": row["source_score"],
                "paired_source_score": row["paired_source_score"],
                "source_mad": row["source_mad"],
                "paired_source_mad": row["paired_source_mad"],
            }
        )
        internal_rows.append({"item_id": item_id, **row})

    return blind_rows, key_rows, internal_rows


def main() -> None:
    source_summary = json.loads((SOURCE_DIR / "summary.json").read_text(encoding="utf-8"))
    source_prompts = load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")
    model_names = list(source_summary["selected_models"])

    selected_prompts, bucket_sizes = select_task_prompts(source_prompts)
    source_pair_rows = build_source_pair_rows(selected_prompts, model_names)
    blind_rows, key_rows, internal_rows = build_pointwise_rows(source_pair_rows)

    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_rows
    ]

    pair_counts = dict(sorted(Counter(pair_key((row["model_name"], row["paired_model_name"])) for row in internal_rows if row["source_side"] == "a").items()))
    counts_by_task = dict(sorted(Counter(row["task"] for row in blind_rows).items()))
    counts_by_model = dict(sorted(Counter(row["model_name"] for row in internal_rows).items()))
    counts_by_task_and_model = {
        task: dict(sorted(Counter(row["model_name"] for row in internal_rows if row["task"] == task).items()))
        for task in TASKS
    }
    adjacent_duplicate_prompts = sum(
        1
        for left, right in zip(internal_rows, internal_rows[1:], strict=False)
        if left["prompt_id"] == right["prompt_id"]
    )

    summary = {
        "source_subset": str(SOURCE_DIR),
        "selected_models": model_names,
        "tasks": list(TASKS),
        "prompts_per_task": PROMPTS_PER_TASK,
        "selected_prompt_count": len(source_pair_rows),
        "selected_item_count": len(blind_rows),
        "selected_counts_by_task": counts_by_task,
        "selected_counts_by_model": counts_by_model,
        "selected_counts_by_task_and_model": counts_by_task_and_model,
        "selected_pair_counts_overall": pair_counts,
        "difficulty_bucket_sizes_by_task": bucket_sizes,
        "seeds": {
            "prompt_selection_seed": PROMPT_SELECTION_SEED,
            "pair_task_assignment_seed": PAIR_TASK_ASSIGNMENT_SEED,
            "pair_to_prompt_assignment_seed": PAIR_TO_PROMPT_ASSIGNMENT_SEED,
            "side_assignment_seed": SIDE_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
        "adjacent_duplicate_prompt_count": adjacent_duplicate_prompts,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_pairs_with_models.jsonl", source_pair_rows)
    write_jsonl(OUTPUT_DIR / "selected_responses_with_models.jsonl", internal_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    print(f"Wrote {len(blind_rows)} pointwise items to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a balanced 210-item single-response human set from the paired 210 source set.

Design:
- Source paired set has 2 prompts per model-pair per task.
- Keep 1 prompt per model-pair per task.
- 6 models => 15 unordered pairs.
- 7 tasks => 105 prompts total.
- Expand each prompt to its 2 responses as independent pointwise items.
- Final annotation load: 210 single-response items.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "paired_pointwise_210_seven_tasks_six_models"
OUTPUT_DIR = ROOT / "pointwise_210_from_paired210_seven_tasks_six_models"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)

PAIR_SELECTION_SEED = 20260441
ANNOTATION_ORDER_SEED = 20260442


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


def pair_key(row: dict[str, Any]) -> str:
    return "|".join(sorted((str(row["model_a"]), str(row["model_b"]))))


def select_source_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        rows_by_pair: dict[str, list[dict[str, Any]]] = {}
        for row in task_rows:
            rows_by_pair.setdefault(pair_key(row), []).append(row)

        if len(rows_by_pair) != 15:
            raise ValueError(f"Task {task} has {len(rows_by_pair)} pairs; expected 15")

        for key, pair_rows in rows_by_pair.items():
            if len(pair_rows) != 2:
                raise ValueError(f"Task {task} pair {key} has {len(pair_rows)} rows; expected 2")
            pair_rows = sorted(pair_rows, key=lambda row: str(row["item_id"]))
            rng = random.Random(task_seed(PAIR_SELECTION_SEED, f"{task}|{key}"))
            selected.append(rng.choice(pair_rows))

    selected.sort(key=lambda row: (TASKS.index(str(row["task"])), str(row["item_id"])))
    if len(selected) != 105:
        raise ValueError(f"Expected 105 selected source rows, got {len(selected)}")
    return selected


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


def build_response_rows(
    selected_pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []

    for pair_row in selected_pairs:
        for side in ("a", "b"):
            response_rows.append(
                {
                    "source_pair_item_id": pair_row["item_id"],
                    "source_side": side,
                    "task": pair_row["task"],
                    "task_label": pair_row["task_label"],
                    "subtask": pair_row.get("subtask", ""),
                    "prompt_id": pair_row["prompt_id"],
                    "prompt_key": pair_row["prompt_key"],
                    "canonical_source_id": pair_row["canonical_source_id"],
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
        item_id = f"pw210_{index:04d}"
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
    selected_pair_rows = load_jsonl(SOURCE_DIR / "selected_items_with_models.jsonl")

    chosen_pairs = select_source_pairs(selected_pair_rows)
    blind_rows, key_rows, internal_rows = build_response_rows(chosen_pairs)

    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_rows
    ]

    counts_by_task = dict(sorted(Counter(row["task"] for row in blind_rows).items()))
    counts_by_model = dict(sorted(Counter(row["model_name"] for row in internal_rows).items()))
    counts_by_task_and_model = {
        task: dict(
            sorted(
                Counter(row["model_name"] for row in internal_rows if row["task"] == task).items()
            )
        )
        for task in TASKS
    }
    adjacent_duplicate_prompts = sum(
        1
        for left, right in zip(internal_rows, internal_rows[1:], strict=False)
        if left["prompt_id"] == right["prompt_id"]
    )

    summary = {
        "source_subset": str(SOURCE_DIR),
        "selected_models": source_summary["selected_models"],
        "tasks": TASKS,
        "source_pair_item_count": len(selected_pair_rows),
        "selected_source_pair_count": len(chosen_pairs),
        "selected_item_count": len(blind_rows),
        "selected_prompt_count": len({row["prompt_id"] for row in internal_rows}),
        "selected_counts_by_task": counts_by_task,
        "selected_counts_by_model": counts_by_model,
        "selected_counts_by_task_and_model": counts_by_task_and_model,
        "pair_selection_seed": PAIR_SELECTION_SEED,
        "annotation_order_seed": ANNOTATION_ORDER_SEED,
        "adjacent_duplicate_prompt_count": adjacent_duplicate_prompts,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_rows)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_responses_with_models.jsonl", internal_rows)
    print(f"Wrote {len(blind_rows)} pointwise items to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

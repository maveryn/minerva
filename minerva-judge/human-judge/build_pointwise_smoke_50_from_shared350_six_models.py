#!/usr/bin/env python3
"""Build a 50-item pointwise smoke-test subset from the shared 350-prompt six-model set."""

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
OUTPUT_DIR = ROOT / "pointwise_smoke_50_from_shared350_six_models"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)
TOTAL_ITEMS = 50
BASE_PER_CELL = 1

TASK_EXTRA_SEED = 20260361
MODEL_EXTRA_SEED = 20260362
POOL_SHUFFLE_SEED = 20260363
ANNOTATION_ORDER_SEED = 20260364


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


def compute_task_quotas(tasks: list[str]) -> dict[str, int]:
    base = TOTAL_ITEMS // len(tasks)
    remainder = TOTAL_ITEMS % len(tasks)
    quotas = {task: base for task in tasks}
    order = list(tasks)
    random.Random(TASK_EXTRA_SEED).shuffle(order)
    for task in order[:remainder]:
        quotas[task] += 1
    return quotas


def compute_model_quotas(model_names: list[str]) -> dict[str, int]:
    base = TOTAL_ITEMS // len(model_names)
    remainder = TOTAL_ITEMS % len(model_names)
    quotas = {name: base for name in model_names}
    order = list(model_names)
    random.Random(MODEL_EXTRA_SEED).shuffle(order)
    for model_name in order[:remainder]:
        quotas[model_name] += 1
    return quotas


def assign_extra_cells(
    tasks: list[str],
    model_names: list[str],
    task_quotas: dict[str, int],
    model_quotas: dict[str, int],
) -> dict[tuple[str, str], int]:
    row_remaining = {
        task: task_quotas[task] - BASE_PER_CELL * len(model_names)
        for task in tasks
    }
    col_remaining = {
        model_name: model_quotas[model_name] - BASE_PER_CELL * len(tasks)
        for model_name in model_names
    }
    if any(value < 0 for value in row_remaining.values()) or any(value < 0 for value in col_remaining.values()):
        raise ValueError("Base per-cell count is too large for requested quotas")

    extras: dict[tuple[str, str], int] = {(task, model_name): 0 for task in tasks for model_name in model_names}

    def backtrack(task_index: int, current_cols: dict[str, int]) -> bool:
        if task_index == len(tasks):
            return all(value == 0 for value in current_cols.values())

        task = tasks[task_index]
        needed = row_remaining[task]
        for chosen_models in itertools.combinations(model_names, needed):
            if any(current_cols[model_name] <= 0 for model_name in chosen_models):
                continue
            next_cols = dict(current_cols)
            for model_name in chosen_models:
                next_cols[model_name] -= 1
            if any(value < 0 for value in next_cols.values()):
                continue
            for model_name in chosen_models:
                extras[(task, model_name)] = 1
            if backtrack(task_index + 1, next_cols):
                return True
            for model_name in chosen_models:
                extras[(task, model_name)] = 0
        return False

    if not backtrack(0, col_remaining):
        raise RuntimeError("Failed to assign extra cell quotas")

    return {
        (task, model_name): BASE_PER_CELL + extras[(task, model_name)]
        for task in tasks
        for model_name in model_names
    }


def build_pools() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str], dict[str, str]]:
    rows = load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")
    model_names = [response["model_name"] for response in rows[0]["responses"]]
    task_labels = {row["task"]: row["task_label"] for row in rows}

    pools: dict[tuple[str, str], list[dict[str, Any]]] = {(task, model_name): [] for task in TASKS for model_name in model_names}
    for prompt_row in rows:
        task = prompt_row["task"]
        for response in prompt_row["responses"]:
            pools[(task, response["model_name"])].append(
                {
                    "prompt_id": prompt_row["prompt_id"],
                    "task": task,
                    "task_label": prompt_row["task_label"],
                    "subtask": prompt_row["subtask"],
                    "prompt_key": prompt_row["prompt_key"],
                    "canonical_source_id": prompt_row["canonical_source_id"],
                    "prompt": prompt_row["prompt"],
                    "model_name": response["model_name"],
                    "response": response["response"],
                    "word_count": response["word_count"],
                    "source_item_id": response["item_id"],
                    "source_slot": response["slot"],
                    "source_id": response["source_id"],
                    "source_answer": response["source_answer"],
                    "source_score": response["source_score"],
                    "source_mad": response["source_mad"],
                }
            )

    for task in TASKS:
        for model_name in model_names:
            pool = pools[(task, model_name)]
            pool.sort(key=lambda row: row["prompt_id"])
            random.Random(f"{POOL_SHUFFLE_SEED}:{task}:{model_name}").shuffle(pool)

    return pools, model_names, task_labels


def sample_items(
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    cell_quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    used_prompt_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    cells = sorted(
        cell_quotas.keys(),
        key=lambda cell: (
            len(pools[cell]),
            -cell_quotas[cell],
            TASKS.index(cell[0]),
            cell[1],
        ),
    )

    for task, model_name in cells:
        quota = cell_quotas[(task, model_name)]
        picked = 0
        for row in pools[(task, model_name)]:
            if row["prompt_id"] in used_prompt_ids:
                continue
            used_prompt_ids.add(row["prompt_id"])
            selected.append(dict(row))
            picked += 1
            if picked == quota:
                break
        if picked != quota:
            raise RuntimeError(
                f"Could not satisfy quota for {(task, model_name)}; needed {quota}, got {picked}"
            )

    return selected


def main() -> None:
    task_names = list(TASKS)
    pools, model_names, task_labels = build_pools()
    task_quotas = compute_task_quotas(task_names)
    model_quotas = compute_model_quotas(model_names)
    cell_quotas = assign_extra_cells(task_names, model_names, task_quotas, model_quotas)
    selected = sample_items(pools, cell_quotas)

    selected.sort(key=lambda row: (TASKS.index(row["task"]), row["model_name"], row["prompt_id"]))

    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        item_id = f"smoke50_{index:04d}"
        blind_row = {
            "item_id": item_id,
            "task": row["task"],
            "task_label": row["task_label"],
            "subtask": row["subtask"],
            "prompt": row["prompt"],
            "response": row["response"],
            "word_count": row["word_count"],
        }
        blind_rows.append(blind_row)
        key_rows.append(
            {
                "item_id": item_id,
                "task": row["task"],
                "task_label": row["task_label"],
                "subtask": row["subtask"],
                "model_name": row["model_name"],
                "prompt_id": row["prompt_id"],
                "prompt_key": row["prompt_key"],
                "canonical_source_id": row["canonical_source_id"],
                "word_count": row["word_count"],
                "source_item_id": row["source_item_id"],
                "source_slot": row["source_slot"],
                "source_id": row["source_id"],
                "source_answer": row["source_answer"],
                "source_score": row["source_score"],
                "source_mad": row["source_mad"],
            }
        )
        internal_rows.append({"item_id": item_id, **row})

    blind_rows_for_annotation = list(blind_rows)
    random.Random(ANNOTATION_ORDER_SEED).shuffle(blind_rows_for_annotation)

    annotation_template = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_rows_for_annotation
    ]

    summary = {
        "source_subset": str(SOURCE_DIR),
        "tasks": list(TASKS),
        "task_labels": task_labels,
        "task_quotas": task_quotas,
        "model_quotas": model_quotas,
        "cell_quotas": {
            f"{task}|{model_name}": count
            for (task, model_name), count in sorted(cell_quotas.items())
        },
        "selected_item_count": len(blind_rows_for_annotation),
        "selected_counts_by_task": dict(sorted(Counter(row["task"] for row in blind_rows_for_annotation).items())),
        "selected_counts_by_model": dict(sorted(Counter(row["model_name"] for row in internal_rows).items())),
        "pool_sizes": {
            f"{task}|{model_name}": len(pools[(task, model_name)])
            for task in TASKS
            for model_name in model_names
        },
        "seeds": {
            "task_extra_seed": TASK_EXTRA_SEED,
            "model_extra_seed": MODEL_EXTRA_SEED,
            "pool_shuffle_seed": POOL_SHUFFLE_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_rows_for_annotation)
    write_csv(OUTPUT_DIR / "blind_responses.csv", blind_rows_for_annotation)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_items_with_models.jsonl", internal_rows)


if __name__ == "__main__":
    main()

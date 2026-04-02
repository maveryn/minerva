#!/usr/bin/env python3
"""Build a 75-item pointwise smoke-test set from 3 selected models."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import runpy
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
SOURCE_SCRIPT = LLMBENCH_ROOT / "response-quality-judge" / "generate_all_correct_samples.py"
OUTPUT_DIR = ROOT / "pointwise_pilot_75_three_models"

TASKS = ("VSP", "RCM", "ATE")
TASK_QUOTAS = {
    "VSP": 25,
    "RCM": 25,
    "ATE": 25,
}
SELECTED_MODELS = (
    "llama3-sec",
    "llama3-sec-reasoning",
    "minerva_llama8b_grpo",
)
TOTAL_ITEMS = sum(TASK_QUOTAS.values())
BASE_PER_CELL = 8

POOL_SHUFFLE_SEED = 20260339
ANNOTATION_ORDER_SEED = 20260340


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


def prompt_key(task: str, prompt_hash: str | None, prompt: str) -> str:
    if prompt_hash:
        return f"{task}:{prompt_hash}"
    blob = f"{task}||{prompt}".encode("utf-8")
    return f"{task}:{hashlib.sha256(blob).hexdigest()}"


def assign_extra_cells(tasks: list[str], model_names: list[str]) -> dict[tuple[str, str], int]:
    row_remaining = {task: TASK_QUOTAS[task] - BASE_PER_CELL * len(model_names) for task in tasks}
    col_remaining = {model_name: 25 - BASE_PER_CELL * len(tasks) for model_name in model_names}

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


def build_pools() -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[str]]:
    namespace = runpy.run_path(str(SOURCE_SCRIPT))
    models = namespace["MODELS"]
    is_correct_base = namespace["is_correct_base"]

    model_names = [model_name for model_name in SELECTED_MODELS if model_name in models]
    if model_names != list(SELECTED_MODELS):
        missing = sorted(set(SELECTED_MODELS) - set(model_names))
        raise ValueError(f"Missing models in source script: {missing}")

    pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model_name in model_names:
        model_dir = models[model_name]
        for task in TASKS:
            rows = []
            for row in load_jsonl(model_dir / f"{task}-scored.jsonl"):
                if not is_correct_base(task, row):
                    continue
                rows.append(
                    {
                        "model_name": model_name,
                        "task": task,
                        "subtask": "",
                        "prompt": str(row.get("prompt") or ""),
                        "response": str(row.get("response") or ""),
                        "word_count": len(str(row.get("response") or "").split()),
                        "source_score": row.get("score"),
                        "source_mad": row.get("mad"),
                        "source_prompt_hash": str(row.get("prompt_hash") or ""),
                        "source_id": row.get("id"),
                        "prompt_key": prompt_key(task, row.get("prompt_hash"), str(row.get("prompt") or "")),
                    }
                )
            rows.sort(key=lambda row: (row["prompt_key"], str(row.get("source_id") or "")))
            random.Random(f"{POOL_SHUFFLE_SEED}:{task}:{model_name}").shuffle(rows)
            pools[(task, model_name)] = rows

    return pools, model_names


def sample_items(
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    cell_quotas: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    used_prompt_keys: set[str] = set()
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
            if row["prompt_key"] in used_prompt_keys:
                continue
            used_prompt_keys.add(row["prompt_key"])
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
    pools, model_names = build_pools()
    task_names = list(TASKS)
    cell_quotas = assign_extra_cells(task_names, model_names)
    selected = sample_items(pools, cell_quotas)

    selected.sort(key=lambda row: (TASKS.index(row["task"]), row["model_name"], row["prompt_key"]))

    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        item_id = f"three75_{index:04d}"
        blind_row = {
            "item_id": item_id,
            "task": row["task"],
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
                "subtask": row["subtask"],
                "model_name": row["model_name"],
                "word_count": row["word_count"],
                "prompt_key": row["prompt_key"],
                "source_score": row["source_score"],
                "source_mad": row["source_mad"],
                "source_prompt_hash": row["source_prompt_hash"],
                "source_id": row["source_id"],
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
        "source_script": str(SOURCE_SCRIPT),
        "tasks": list(TASKS),
        "task_quotas": TASK_QUOTAS,
        "selected_models": list(model_names),
        "model_quotas": {model_name: 25 for model_name in model_names},
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

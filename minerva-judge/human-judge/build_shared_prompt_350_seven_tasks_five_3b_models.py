#!/usr/bin/env python3
"""Build the 3B five-model response-quality dataset on the same 350 questions as the 8B study."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
RUNS_ROOT = LLMBENCH_ROOT / "runs"

SOURCE_DATASET_DIR = ROOT / "shared_prompt_350_seven_tasks_seven_models"
OUTPUT_DIR = ROOT / "shared_prompt_350_seven_tasks_five_3b_models"

TASKS = (
    "MCQ3k",
    "CyberMetric",
    "RCM",
    "VSP",
    "ATE",
    "RMS",
    "ElasticToAttack",
)
TASK_LABELS = {
    "MCQ3k": "CKT",
    "CyberMetric": "CyberMetric",
    "RCM": "RCM",
    "VSP": "VSP",
    "ATE": "ATE",
    "RMS": "RMS",
    "ElasticToAttack": "ElasticRule",
}

SELECTED_MODELS = (
    "llama-3-3B",
    "minerva_llama3b_grpo",
    "llama3-3b-star",
    "llama3-3b-dart",
    "minerva_llama3b_noctua",
)
MODEL_DIRS = {
    "llama-3-3B": RUNS_ROOT / "meta-llama" / "Llama-3.2-3B-Instruct",
    "minerva_llama3b_grpo": RUNS_ROOT / "xashru" / "minerva_grpo_llama3b_500",
    "llama3-3b-star": RUNS_ROOT / "star_llama32_3b_round0",
    "llama3-3b-dart": RUNS_ROOT / "dart_llama32_3b_dart_best430",
    "minerva_llama3b_noctua": RUNS_ROOT / "xashru" / "minerva_noctua_llama3b_500",
}

SLOT_ASSIGNMENT_SEED = 20260461
ANNOTATION_ORDER_SEED = 20260462


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def word_count(text: str) -> int:
    return len((text or "").split())


def load_source_prompts() -> tuple[list[dict[str, Any]], dict[str, str]]:
    prompt_rows = [
        json.loads(line)
        for line in (SOURCE_DATASET_DIR / "blind_prompts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    prompt_meta: dict[str, dict[str, Any]] = {}
    for line in (SOURCE_DATASET_DIR / "key.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = row["prompt_id"]
        prompt_meta.setdefault(
            prompt_id,
            {
                "prompt_id": prompt_id,
                "task": row["task"],
                "task_label": row["task_label"],
                "subtask": row.get("subtask") or "",
                "prompt_key": row["prompt_key"],
                "canonical_source_id": str(row["canonical_source_id"]),
            },
        )

    ordered: list[dict[str, Any]] = []
    canonical_source_ids_by_task: dict[str, list[str]] = {task: [] for task in TASKS}
    for prompt_row in prompt_rows:
        meta = prompt_meta[prompt_row["prompt_id"]]
        entry = {
            **meta,
            "prompt": str(prompt_row["prompt"]).strip(),
        }
        ordered.append(entry)
        canonical_source_ids_by_task[entry["task"]].append(entry["canonical_source_id"])

    return ordered, {task: str(len(ids)) for task, ids in canonical_source_ids_by_task.items()}


def load_model_task_map(task: str, model_name: str) -> dict[str, dict[str, Any]]:
    path = MODEL_DIRS[model_name] / f"{task}-scored.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scored file for {model_name} {task}: {path}")

    rows_by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row.get("id"))
            rows_by_id[source_id] = {
                "model_name": model_name,
                "response": str(row.get("response") or ""),
                "word_count": word_count(str(row.get("response") or "")),
                "source_id": source_id,
                "source_answer": row.get("answer"),
                "source_score": row.get("score"),
                "source_mad": row.get("mad"),
            }
    return rows_by_id


def load_model_maps() -> dict[str, dict[str, dict[str, Any]]]:
    task_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASKS:
        task_maps[task] = {
            model_name: load_model_task_map(task, model_name)
            for model_name in SELECTED_MODELS
        }
    return task_maps


def build_outputs(selected_prompts: list[dict[str, Any]], task_maps: dict[str, dict[str, dict[str, Any]]]) -> None:
    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_prompts: list[dict[str, Any]] = []

    for prompt_index, prompt_entry in enumerate(selected_prompts, start=1):
        prompt_id = f"shared3503b5_prompt_{prompt_index:04d}"
        task = prompt_entry["task"]
        canonical_source_id = prompt_entry["canonical_source_id"]

        slot_rows: list[dict[str, Any]] = []
        for model_name in SELECTED_MODELS:
            model_map = task_maps[task][model_name]
            if canonical_source_id not in model_map:
                raise KeyError(
                    f"Source id {canonical_source_id} missing for task {task} in model {model_name}"
                )
            slot_rows.append(model_map[canonical_source_id])

        random.Random(SLOT_ASSIGNMENT_SEED + prompt_index).shuffle(slot_rows)

        blind_prompt_responses: list[dict[str, Any]] = []
        internal_prompt_responses: list[dict[str, Any]] = []
        for slot_index, response_row in enumerate(slot_rows, start=1):
            item_id = f"{prompt_id}_resp_{slot_index}"
            blind_entry = {
                "item_id": item_id,
                "prompt_id": prompt_id,
                "task": task,
                "task_label": TASK_LABELS[task],
                "subtask": prompt_entry["subtask"],
                "prompt": prompt_entry["prompt"],
                "response": response_row["response"],
                "word_count": response_row["word_count"],
            }
            blind_responses.append(blind_entry)
            blind_prompt_responses.append(
                {
                    "item_id": item_id,
                    "slot": slot_index,
                    "response": response_row["response"],
                    "word_count": response_row["word_count"],
                }
            )
            internal_prompt_responses.append(
                {
                    "item_id": item_id,
                    "slot": slot_index,
                    "model_name": response_row["model_name"],
                    "response": response_row["response"],
                    "word_count": response_row["word_count"],
                    "source_id": response_row["source_id"],
                    "source_answer": response_row["source_answer"],
                    "source_score": response_row["source_score"],
                    "source_mad": response_row["source_mad"],
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "subtask": prompt_entry["subtask"],
                    "prompt_key": prompt_entry["prompt_key"],
                    "canonical_source_id": canonical_source_id,
                    "model_name": response_row["model_name"],
                    "word_count": response_row["word_count"],
                    "source_id": response_row["source_id"],
                    "source_answer": response_row["source_answer"],
                    "source_score": response_row["source_score"],
                    "source_mad": response_row["source_mad"],
                }
            )

        blind_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "task_label": TASK_LABELS[task],
                "subtask": prompt_entry["subtask"],
                "prompt": prompt_entry["prompt"],
                "responses": blind_prompt_responses,
            }
        )
        internal_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "task_label": TASK_LABELS[task],
                "subtask": prompt_entry["subtask"],
                "prompt_key": prompt_entry["prompt_key"],
                "canonical_source_id": canonical_source_id,
                "prompt": prompt_entry["prompt"],
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
        "source_dataset": str(SOURCE_DATASET_DIR),
        "selected_models": list(SELECTED_MODELS),
        "tasks": list(TASKS),
        "task_labels": TASK_LABELS,
        "prompts_per_task": 50,
        "selected_prompt_count": len(selected_prompts),
        "selected_response_count": len(key_rows),
        "selected_prompt_counts_by_task": dict(sorted(Counter(row["task"] for row in selected_prompts).items())),
        "selected_response_counts_by_model": dict(sorted(Counter(row["model_name"] for row in key_rows).items())),
        "selected_response_counts_by_task": dict(sorted(Counter(row["task"] for row in key_rows).items())),
        "seeds": {
            "slot_assignment_seed": SLOT_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
        "model_dirs": {model_name: str(MODEL_DIRS[model_name]) for model_name in SELECTED_MODELS},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", ordered_blind_responses)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_prompts_with_models.jsonl", internal_prompts)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)


def main() -> None:
    selected_prompts, _ = load_source_prompts()
    if len(selected_prompts) != 350:
        raise ValueError(f"Expected 350 prompts from source dataset, found {len(selected_prompts)}")
    task_maps = load_model_maps()
    build_outputs(selected_prompts, task_maps)
    print(f"Wrote {len(selected_prompts)} prompts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

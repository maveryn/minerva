#!/usr/bin/env python3
"""Build a shared-prompt 250-item sample across five CTI tasks and five models."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
RUNS_ROOT = LLMBENCH_ROOT / "runs"
OUTPUT_DIR = ROOT / "shared_prompt_250_five_tasks_security5"

TASKS = ("VSP", "RCM", "ATE", "ElasticToAttack", "RMS")
PROMPTS_PER_TASK = 50

SELECTED_MODELS = (
    "llama3-primus",
    "llama3-sec",
    "llama3-sec-reasoning",
    "minerva_llama8b_grpo",
    "minerva_llama8b_noctua",
)
MODEL_DIRS = {
    "llama3-primus": RUNS_ROOT / "trendmicro-ailab" / "Llama-Primus-Merged",
    "llama3-sec": RUNS_ROOT / "fdtn-ai" / "Foundation-Sec-8B-Instruct",
    "llama3-sec-reasoning": RUNS_ROOT / "foundation-sec-8b-reasoning",
    "minerva_llama8b_grpo": RUNS_ROOT / "xashru" / "minerva_grpo_llama8b_500_490",
    "minerva_llama8b_noctua": RUNS_ROOT / "xashru" / "minerva_noctua_llama8b_ema_0.05",
}

PROMPT_SELECTION_SEED = 20260399
SLOT_ASSIGNMENT_SEED = 20260400
ANNOTATION_ORDER_SEED = 20260401


@dataclass(frozen=True)
class PromptEntry:
    task: str
    subtask: str
    prompt: str
    prompt_key: str
    canonical_source_id: str


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


def compute_prompt_key(task: str, prompt: str, prompt_hash: str | None = None) -> str:
    if prompt_hash:
        return str(prompt_hash)
    blob = f"{task}||{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def task_seed(task: str) -> int:
    return PROMPT_SELECTION_SEED + sum((index + 1) * ord(ch) for index, ch in enumerate(task))


def word_count(text: str) -> int:
    return len((text or "").split())


def load_model_task_map(task: str, model_name: str) -> dict[str, dict[str, Any]]:
    path = MODEL_DIRS[model_name] / f"{task}-scored.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scored file for {model_name} {task}: {path}")

    rows_by_key: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            prompt_key = compute_prompt_key(task, prompt, row.get("prompt_hash"))
            source_id = row.get("id")
            if source_id is None:
                source_id = line_index
            candidate = {
                "model_name": model_name,
                "prompt": prompt,
                "response": str(row.get("response") or ""),
                "word_count": word_count(str(row.get("response") or "")),
                "source_id": str(source_id),
                "source_answer": row.get("answer"),
                "source_score": row.get("score"),
                "source_mad": row.get("mad"),
            }
            current = rows_by_key.get(prompt_key)
            if current is None or candidate["source_id"] < current["source_id"]:
                rows_by_key[prompt_key] = candidate
    return rows_by_key


def load_model_maps() -> dict[str, dict[str, dict[str, Any]]]:
    task_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASKS:
        model_maps = {model_name: load_model_task_map(task, model_name) for model_name in SELECTED_MODELS}
        task_maps[task] = model_maps
    return task_maps


def select_shared_prompts(task_maps: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[PromptEntry], dict[str, int]]:
    selected_prompts: list[PromptEntry] = []
    available_counts: dict[str, int] = {}

    for task in TASKS:
        shared_keys = set.intersection(*(set(task_maps[task][model_name]) for model_name in SELECTED_MODELS))
        prompt_pool: list[PromptEntry] = []
        canonical_model = SELECTED_MODELS[0]
        for prompt_key in shared_keys:
            canonical = task_maps[task][canonical_model][prompt_key]
            prompt_pool.append(
                PromptEntry(
                    task=task,
                    subtask="",
                    prompt=str(canonical["prompt"]),
                    prompt_key=prompt_key,
                    canonical_source_id=str(canonical["source_id"]),
                )
            )

        prompt_pool.sort(key=lambda entry: entry.prompt_key)
        rng = random.Random(task_seed(task))
        rng.shuffle(prompt_pool)
        available_counts[task] = len(prompt_pool)
        if len(prompt_pool) < PROMPTS_PER_TASK:
            raise ValueError(f"Task {task} has only {len(prompt_pool)} shared prompts; need {PROMPTS_PER_TASK}")

        chosen = prompt_pool[:PROMPTS_PER_TASK]
        chosen.sort(key=lambda entry: entry.prompt_key)
        selected_prompts.extend(chosen)

    selected_prompts.sort(key=lambda entry: (TASKS.index(entry.task), entry.prompt_key))
    return selected_prompts, available_counts


def build_outputs(
    selected_prompts: list[PromptEntry],
    task_maps: dict[str, dict[str, dict[str, Any]]],
    available_counts: dict[str, int],
) -> None:
    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_prompts: list[dict[str, Any]] = []

    for prompt_index, prompt_entry in enumerate(selected_prompts, start=1):
        prompt_id = f"shared250_prompt_{prompt_index:04d}"
        task = prompt_entry.task

        slot_rows: list[dict[str, Any]] = []
        for model_name in SELECTED_MODELS:
            model_map = task_maps[task][model_name]
            if prompt_entry.prompt_key not in model_map:
                raise KeyError(
                    f"Prompt {prompt_entry.prompt_key} missing for task {task} in model {model_name}"
                )
            slot_rows.append(model_map[prompt_entry.prompt_key])

        random.Random(SLOT_ASSIGNMENT_SEED + prompt_index).shuffle(slot_rows)

        blind_prompt_responses: list[dict[str, Any]] = []
        internal_prompt_responses: list[dict[str, Any]] = []
        for slot_index, response_row in enumerate(slot_rows, start=1):
            item_id = f"{prompt_id}_resp_{slot_index}"
            blind_entry = {
                "item_id": item_id,
                "prompt_id": prompt_id,
                "task": task,
                "subtask": prompt_entry.subtask,
                "prompt": prompt_entry.prompt,
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
                    "subtask": prompt_entry.subtask,
                    "prompt_key": prompt_entry.prompt_key,
                    "canonical_source_id": prompt_entry.canonical_source_id,
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
                "subtask": prompt_entry.subtask,
                "prompt": prompt_entry.prompt,
                "responses": blind_prompt_responses,
            }
        )
        internal_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "subtask": prompt_entry.subtask,
                "prompt_key": prompt_entry.prompt_key,
                "canonical_source_id": prompt_entry.canonical_source_id,
                "prompt": prompt_entry.prompt,
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
        "selected_models": list(SELECTED_MODELS),
        "tasks": list(TASKS),
        "prompts_per_task": PROMPTS_PER_TASK,
        "selected_prompt_count": len(selected_prompts),
        "selected_response_count": len(key_rows),
        "selected_prompt_counts_by_task": dict(sorted(Counter(row.task for row in selected_prompts).items())),
        "selected_response_counts_by_model": dict(sorted(Counter(row["model_name"] for row in key_rows).items())),
        "selected_response_counts_by_task": dict(sorted(Counter(row["task"] for row in key_rows).items())),
        "available_shared_prompt_counts_by_task": available_counts,
        "seeds": {
            "prompt_selection_seed": PROMPT_SELECTION_SEED,
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
    task_maps = load_model_maps()
    selected_prompts, available_counts = select_shared_prompts(task_maps)
    build_outputs(selected_prompts, task_maps, available_counts)
    print(f"Wrote {len(selected_prompts)} prompts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

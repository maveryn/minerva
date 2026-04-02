#!/usr/bin/env python3
"""Project the shared 4-task prompt sample onto the base Llama 3.1 8B Instruct run."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "shared_prompt_250_sec_vs_noctua"
OUTPUT_DIR = ROOT / "shared_prompt_200_llama3_8b_four_tasks"
MODEL_NAME = "llama-3-8B"
MODEL_DIR = ROOT.parents[2] / "llmbench" / "runs" / "meta-llama" / "Llama-3.1-8B-Instruct"
TASKS = ("VSP", "RCM", "ATE", "RMS")


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


def compute_prompt_key(task: str, prompt: str, prompt_hash: str | None = None) -> str:
    if prompt_hash:
        return str(prompt_hash)
    blob = f"{task}||{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def word_count(text: str) -> int:
    return len((text or "").split())


def load_model_maps() -> dict[str, dict[str, dict[str, Any]]]:
    task_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASKS:
        path = MODEL_DIR / f"{task}-scored.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing scored file for {MODEL_NAME} {task}: {path}")
        mapping: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_index, line in enumerate(f, start=1):
                row = json.loads(line)
                prompt = str(row.get("prompt") or "").strip()
                prompt_key = compute_prompt_key(task, prompt, row.get("prompt_hash"))
                source_id = row.get("id")
                if source_id is None:
                    source_id = line_index
                candidate = {
                    "response": str(row.get("response") or ""),
                    "word_count": word_count(str(row.get("response") or "")),
                    "source_id": str(source_id),
                    "source_answer": row.get("answer"),
                    "source_score": row.get("score"),
                    "source_mad": row.get("mad"),
                }
                current = mapping.get(prompt_key)
                if current is None or candidate["source_id"] < current["source_id"]:
                    mapping[prompt_key] = candidate
        task_maps[task] = mapping
    return task_maps


def main() -> None:
    source_summary = json.loads((SOURCE_DIR / "summary.json").read_text(encoding="utf-8"))
    source_prompts = [
        row
        for row in load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl")
        if row["task"] in TASKS
    ]
    source_prompts.sort(key=lambda row: (TASKS.index(row["task"]), row["prompt_id"]))

    model_maps = load_model_maps()

    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    internal_prompts: list[dict[str, Any]] = []

    for prompt_row in source_prompts:
        task = prompt_row["task"]
        prompt_key = prompt_row["prompt_key"]
        prompt_id = prompt_row["prompt_id"]
        if prompt_key not in model_maps[task]:
            raise KeyError(f"Prompt {prompt_key} missing for {MODEL_NAME} in task {task}")
        response_row = model_maps[task][prompt_key]
        item_id = f"{prompt_id}_resp_1"

        blind_entry = {
            "item_id": item_id,
            "prompt_id": prompt_id,
            "task": task,
            "subtask": prompt_row.get("subtask", ""),
            "prompt": prompt_row["prompt"],
            "response": response_row["response"],
            "word_count": response_row["word_count"],
        }
        blind_responses.append(blind_entry)
        blind_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "subtask": prompt_row.get("subtask", ""),
                "prompt": prompt_row["prompt"],
                "responses": [
                    {
                        "item_id": item_id,
                        "slot": 1,
                        "response": response_row["response"],
                        "word_count": response_row["word_count"],
                    }
                ],
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "prompt_id": prompt_id,
                "task": task,
                "subtask": prompt_row.get("subtask", ""),
                "prompt_key": prompt_key,
                "model_name": MODEL_NAME,
                "word_count": response_row["word_count"],
                "source_id": response_row["source_id"],
                "source_answer": response_row["source_answer"],
                "source_score": response_row["source_score"],
                "source_mad": response_row["source_mad"],
            }
        )
        internal_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "subtask": prompt_row.get("subtask", ""),
                "prompt_key": prompt_key,
                "prompt": prompt_row["prompt"],
                "responses": [
                    {
                        "item_id": item_id,
                        "slot": 1,
                        "model_name": MODEL_NAME,
                        "response": response_row["response"],
                        "word_count": response_row["word_count"],
                        "source_id": response_row["source_id"],
                        "source_answer": response_row["source_answer"],
                        "source_score": response_row["source_score"],
                        "source_mad": response_row["source_mad"],
                    }
                ],
            }
        )

    annotation_template = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_responses
    ]

    summary = {
        "source_dir": str(SOURCE_DIR),
        "selected_model": MODEL_NAME,
        "tasks": list(TASKS),
        "selected_prompt_count": len(source_prompts),
        "selected_response_count": len(blind_responses),
        "selected_prompt_counts_by_task": dict(sorted(Counter(row["task"] for row in source_prompts).items())),
        "selected_response_counts_by_task": dict(sorted(Counter(row["task"] for row in blind_responses).items())),
        "model_dir": str(MODEL_DIR),
        "source_prompt_pool_summary": source_summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_responses)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_prompts_with_models.jsonl", internal_prompts)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template)
    print(f"Wrote {len(blind_responses)} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

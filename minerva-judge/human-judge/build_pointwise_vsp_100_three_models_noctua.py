#!/usr/bin/env python3
"""Build a prompt-aligned VSP subset with 100 shared prompts for three selected models."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
OUTPUT_DIR = ROOT / "pointwise_vsp_100_three_models_noctua"

TASK = "VSP"
MAD_THRESHOLD = 1.0
PROMPT_QUOTA = 100
SELECTED_MODELS = (
    "llama3-sec",
    "llama3-sec-reasoning",
    "minerva_llama8b_noctua",
)
MODEL_RUN_DIRS = {
    "llama3-sec": LLMBENCH_ROOT / "runs" / "fdtn-ai" / "Foundation-Sec-8B-Instruct",
    "llama3-sec-reasoning": LLMBENCH_ROOT / "runs" / "foundation-sec-8b-reasoning",
    "minerva_llama8b_noctua": LLMBENCH_ROOT / "runs" / "xashru" / "minerva_noctua_llama8b_ema_0.05",
}

PROMPT_SELECTION_SEED = 20260394
SLOT_ASSIGNMENT_SEED = 20260395
ANNOTATION_ORDER_SEED = 20260396


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


def prompt_key(prompt_hash: str | None, prompt: str) -> str:
    if prompt_hash:
        return f"{TASK}:{prompt_hash}"
    blob = f"{TASK}||{prompt}".encode("utf-8")
    return f"{TASK}:{hashlib.sha256(blob).hexdigest()}"


def build_correct_maps() -> dict[str, dict[str, Any]]:
    task_maps: dict[str, dict[str, Any]] = {}
    for model_name in SELECTED_MODELS:
        rows_by_key: dict[str, Any] = {}
        for row in load_jsonl(MODEL_RUN_DIRS[model_name] / f"{TASK}-scored.jsonl"):
            mad = row.get("mad")
            try:
                if mad is None or float(mad) > MAD_THRESHOLD:
                    continue
            except Exception:
                continue
            key = prompt_key(row.get("prompt_hash"), str(row.get("prompt") or ""))
            candidate = {
                "prompt": str(row.get("prompt") or ""),
                "prompt_hash": str(row.get("prompt_hash") or ""),
                "response": str(row.get("response") or ""),
                "word_count": len(str(row.get("response") or "").split()),
                "source_score": row.get("score"),
                "source_mad": row.get("mad"),
                "source_id": row.get("id"),
            }
            existing = rows_by_key.get(key)
            if existing is None or str(candidate["source_id"] or "") < str(existing["source_id"] or ""):
                rows_by_key[key] = candidate
        task_maps[model_name] = rows_by_key
    return task_maps


def build_aligned_prompt_pool(task_maps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    shared_keys = set.intersection(*(set(task_maps[model_name]) for model_name in SELECTED_MODELS))
    pool: list[dict[str, Any]] = []
    for key in sorted(shared_keys):
        prompt = task_maps[SELECTED_MODELS[0]][key]["prompt"]
        responses = []
        for model_name in SELECTED_MODELS:
            row = task_maps[model_name][key]
            responses.append(
                {
                    "model_name": model_name,
                    "response": row["response"],
                    "word_count": row["word_count"],
                    "source_score": row["source_score"],
                    "source_mad": row["source_mad"],
                    "source_id": row["source_id"],
                }
            )
        pool.append(
            {
                "task": TASK,
                "subtask": "",
                "prompt": prompt,
                "prompt_key": key,
                "responses": responses,
            }
        )
    rng = random.Random(PROMPT_SELECTION_SEED)
    rng.shuffle(pool)
    return pool


def build_outputs(selected_prompts: list[dict[str, Any]]) -> None:
    blind_prompts: list[dict[str, Any]] = []
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for prompt_index, prompt_row in enumerate(selected_prompts, start=1):
        prompt_id = f"vsp100_prompt_{prompt_index:04d}"
        slot_rows = list(prompt_row["responses"])
        random.Random(SLOT_ASSIGNMENT_SEED + prompt_index).shuffle(slot_rows)

        blind_prompt_responses: list[dict[str, Any]] = []
        for slot_index, response_row in enumerate(slot_rows, start=1):
            item_id = f"{prompt_id}_resp_{slot_index}"
            blind_entry = {
                "item_id": item_id,
                "prompt_id": prompt_id,
                "task": prompt_row["task"],
                "subtask": prompt_row["subtask"],
                "prompt": prompt_row["prompt"],
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
            key_rows.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "task": prompt_row["task"],
                    "subtask": prompt_row["subtask"],
                    "prompt_key": prompt_row["prompt_key"],
                    "model_name": response_row["model_name"],
                    "word_count": response_row["word_count"],
                    "source_score": response_row["source_score"],
                    "source_mad": response_row["source_mad"],
                    "source_id": response_row["source_id"],
                }
            )

        blind_prompts.append(
            {
                "prompt_id": prompt_id,
                "task": prompt_row["task"],
                "subtask": prompt_row["subtask"],
                "prompt": prompt_row["prompt"],
                "responses": blind_prompt_responses,
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
        "task": TASK,
        "mad_threshold": MAD_THRESHOLD,
        "selected_models": list(SELECTED_MODELS),
        "selected_prompt_count": len(selected_prompts),
        "selected_response_count": len(key_rows),
        "selected_response_counts_by_model": {
            model_name: sum(1 for row in key_rows if row["model_name"] == model_name)
            for model_name in SELECTED_MODELS
        },
        "seeds": {
            "prompt_selection_seed": PROMPT_SELECTION_SEED,
            "slot_assignment_seed": SLOT_ASSIGNMENT_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", ordered_blind_responses)
    write_csv(OUTPUT_DIR / "blind_responses.csv", ordered_blind_responses)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", sorted(key_rows, key=lambda row: row["item_id"]))


def main() -> None:
    task_maps = build_correct_maps()
    selected_prompts = build_aligned_prompt_pool(task_maps)
    if len(selected_prompts) < PROMPT_QUOTA:
        raise ValueError(f"Only found {len(selected_prompts)} shared prompts; need {PROMPT_QUOTA}")
    build_outputs(selected_prompts[:PROMPT_QUOTA])


if __name__ == "__main__":
    main()

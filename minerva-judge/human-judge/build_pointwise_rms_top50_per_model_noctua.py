#!/usr/bin/env python3
"""Build a pointwise RMS subset using the top 50 scored rows per model including Noctua."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LLMBENCH_ROOT = ROOT.parents[2] / "llmbench"
OUTPUT_DIR = ROOT / "pointwise_rms_top50_per_model_noctua"

TASK = "RMS"
TOP_K = 50
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

SELECTION_BASE_SEED = 20260381
ANNOTATION_ORDER_SEED = 20260382


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


def select_top_rows(model_name: str) -> list[dict[str, Any]]:
    path = MODEL_RUN_DIRS[model_name] / f"{TASK}-scored.jsonl"
    rows = load_jsonl(path)
    rng = random.Random(SELECTION_BASE_SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(model_name)))

    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for row in rows:
        score = float(row.get("score") or 0.0)
        ranked.append((score, rng.random(), row))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in ranked[:TOP_K]]


def build_outputs() -> None:
    blind_responses: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []

    for model_name in SELECTED_MODELS:
        chosen_rows = select_top_rows(model_name)
        if len(chosen_rows) < TOP_K:
            raise ValueError(f"Model {model_name} only has {len(chosen_rows)} rows; need {TOP_K}")

        for index, row in enumerate(chosen_rows, start=1):
            item_id = f"rms_top50n_{model_name}_{index:03d}"
            prompt_id = f"rms_top50n_{model_name}_{index:03d}"
            response = str(row.get("response") or "")
            blind_responses.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "task": TASK,
                    "subtask": "",
                    "prompt": str(row.get("prompt") or ""),
                    "response": response,
                    "word_count": len(response.split()),
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "prompt_id": prompt_id,
                    "task": TASK,
                    "subtask": "",
                    "model_name": model_name,
                    "word_count": len(response.split()),
                    "source_score": row.get("score"),
                    "source_id": row.get("id"),
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
        "selected_models": list(SELECTED_MODELS),
        "top_k_per_model": TOP_K,
        "selected_response_count": len(key_rows),
        "selected_response_counts_by_model": dict(sorted(Counter(row["model_name"] for row in key_rows).items())),
        "source_score_min_by_model": {
            model_name: min(float(row["source_score"]) for row in key_rows if row["model_name"] == model_name)
            for model_name in SELECTED_MODELS
        },
        "seeds": {
            "selection_base_seed": SELECTION_BASE_SEED,
            "annotation_order_seed": ANNOTATION_ORDER_SEED,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", ordered_blind_responses)
    write_csv(OUTPUT_DIR / "blind_responses.csv", ordered_blind_responses)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "annotation_template.jsonl", annotation_template_rows)
    write_jsonl(OUTPUT_DIR / "key.jsonl", sorted(key_rows, key=lambda row: row["item_id"]))


def main() -> None:
    build_outputs()


if __name__ == "__main__":
    main()

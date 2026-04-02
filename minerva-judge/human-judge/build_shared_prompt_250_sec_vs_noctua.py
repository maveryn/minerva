#!/usr/bin/env python3
"""Filter the shared 250-prompt five-model set down to sec vs noctua."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "shared_prompt_250_five_tasks_security5"
OUTPUT_DIR = ROOT / "shared_prompt_250_sec_vs_noctua"
SELECTED_MODELS = ("llama3-sec", "minerva_llama8b_noctua")


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


def main() -> None:
    summary = json.loads((SOURCE_DIR / "summary.json").read_text(encoding="utf-8"))
    key_rows = [row for row in load_jsonl(SOURCE_DIR / "key.jsonl") if row["model_name"] in SELECTED_MODELS]
    keep_item_ids = {row["item_id"] for row in key_rows}

    blind_responses = [row for row in load_jsonl(SOURCE_DIR / "blind_responses.jsonl") if row["item_id"] in keep_item_ids]
    blind_prompts = []
    for row in load_jsonl(SOURCE_DIR / "blind_prompts.jsonl"):
        responses = [resp for resp in row["responses"] if resp["item_id"] in keep_item_ids]
        blind_prompts.append({**row, "responses": responses})

    selected_prompts_with_models = []
    for row in load_jsonl(SOURCE_DIR / "selected_prompts_with_models.jsonl"):
        responses = [resp for resp in row["responses"] if resp["model_name"] in SELECTED_MODELS]
        selected_prompts_with_models.append({**row, "responses": responses})

    annotation_template_rows = [
        {
            **row,
            "writing_quality_score": "",
            "evidence_use_score": "",
            "cti_concept_focus_score": "",
        }
        for row in blind_responses
    ]

    out_summary = {
        "source_dir": str(SOURCE_DIR),
        "selected_models": list(SELECTED_MODELS),
        "tasks": summary["tasks"],
        "prompts_per_task": summary["prompts_per_task"],
        "selected_prompt_count": summary["selected_prompt_count"],
        "selected_response_count": len(key_rows),
        "selected_prompt_counts_by_task": summary["selected_prompt_counts_by_task"],
        "selected_response_counts_by_model": dict(sorted(Counter(row["model_name"] for row in key_rows).items())),
        "selected_response_counts_by_task": dict(sorted(Counter(row["task"] for row in key_rows).items())),
        "available_shared_prompt_counts_by_task": summary.get("available_shared_prompt_counts_by_task", {}),
        "model_dirs": {k: v for k, v in summary["model_dirs"].items() if k in SELECTED_MODELS},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "summary.json", out_summary)
    write_jsonl(OUTPUT_DIR / "blind_prompts.jsonl", blind_prompts)
    write_jsonl(OUTPUT_DIR / "blind_responses.jsonl", blind_responses)
    write_jsonl(OUTPUT_DIR / "key.jsonl", key_rows)
    write_jsonl(OUTPUT_DIR / "selected_prompts_with_models.jsonl", selected_prompts_with_models)
    write_csv(OUTPUT_DIR / "annotation_template.csv", annotation_template_rows)
    print(f"Wrote {len(key_rows)} rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build nested human-annotation subsets from pairwise GPT judge results."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PAIRWISE_DIR = ROOT.parent / "judge-pairwise-quality-eval"
SAMPLES_PATH = PAIRWISE_DIR / "all_models_correct_samples.json"
PAIRWISE_RESULTS_PATH = PAIRWISE_DIR / "pairwise_judge_results.jsonl"

SELECTION_SEED = 20260327
PAIR_ASSIGNMENT_SEED_100 = 20260328
PAIR_ASSIGNMENT_SEED_EXTRA_100 = 20260329

@dataclass(frozen=True)
class Sample:
    sample_index: int
    sample_key: str
    task: str
    subtask: str | None
    prompt: str


def compute_sample_key(task: str, subtask: str | None, prompt: str) -> str:
    blob = f"{task}||{subtask or ''}||{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pairwise_rows(
    path: Path,
) -> tuple[dict[tuple[str, tuple[str, str]], dict[str, Any]], dict[str, set[tuple[str, str]]]]:
    pairwise_rows: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
    available_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            winner_model = str(row.get("winner_model") or "")
            if winner_model == "parse_error":
                continue
            sample_key = str(row.get("sample_key") or "")
            model_a = str(row.get("model_a") or "")
            model_b = str(row.get("model_b") or "")
            if not sample_key or not model_a or not model_b:
                continue
            pair = tuple(sorted((model_a, model_b)))
            pairwise_rows[(sample_key, pair)] = row
            available_pairs[sample_key].add(pair)

    return pairwise_rows, available_pairs


def load_samples(
    samples_path: Path,
    available_pairs: dict[str, set[tuple[str, str]]],
) -> tuple[list[Sample], dict[str, list[Sample]], list[tuple[str, str]]]:
    raw_samples = load_json(samples_path)
    if not isinstance(raw_samples, list):
        raise ValueError(f"Expected a JSON list in {samples_path}")

    samples: list[Sample] = []
    by_task: dict[str, list[Sample]] = defaultdict(list)
    model_names: list[str] | None = None

    for idx, obj in enumerate(raw_samples):
        if not isinstance(obj, dict):
            continue
        task = str(obj.get("task") or "")
        subtask = obj.get("subtask")
        prompt = str(obj.get("prompt") or "")
        responses = obj.get("responses") or {}
        if not task or not prompt or not isinstance(responses, dict):
            continue

        if model_names is None:
            model_names = sorted(str(name) for name in responses.keys())

        sample_key = compute_sample_key(task, subtask, prompt)
        if sample_key not in available_pairs:
            continue

        sample = Sample(
            sample_index=idx,
            sample_key=sample_key,
            task=task,
            subtask=subtask if subtask is not None else None,
            prompt=prompt,
        )
        samples.append(sample)
        by_task[task].append(sample)

    if not model_names:
        raise ValueError("No model names found in all_models_correct_samples.json")

    pairs = list(_all_pairs(model_names))
    return samples, by_task, pairs


def _all_pairs(model_names: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            pairs.append(tuple(sorted((model_a, model_b))))
    return pairs


def allocate_task_quotas(task_counts: dict[str, int], target_size: int) -> dict[str, int]:
    total = sum(task_counts.values())
    if target_size > total:
        raise ValueError(f"target_size={target_size} exceeds pool size={total}")

    raw = {task: count * target_size / total for task, count in task_counts.items()}
    quotas = {task: int(value) for task, value in raw.items()}
    remainder = target_size - sum(quotas.values())

    ranked = sorted(
        task_counts.keys(),
        key=lambda task: (raw[task] - quotas[task], raw[task], task),
        reverse=True,
    )
    for task in ranked[:remainder]:
        quotas[task] += 1
    return quotas


def sample_by_task(
    by_task: dict[str, list[Sample]],
    quotas: dict[str, int],
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    selected: dict[str, list[Sample]] = {}

    for task in sorted(quotas):
        quota = quotas[task]
        pool = list(by_task[task])
        pool.sort(key=lambda sample: sample.sample_key)
        rng.shuffle(pool)
        if quota > len(pool):
            raise ValueError(f"Task {task} quota {quota} exceeds pool size {len(pool)}")
        selected[task] = pool[:quota]

    return selected


def flatten_task_samples(task_map: dict[str, list[Sample]]) -> list[Sample]:
    flat: list[Sample] = []
    for task in sorted(task_map):
        flat.extend(task_map[task])
    return flat


def assign_pairs(
    samples: list[Sample],
    pairs: list[tuple[str, str]],
    pair_target: int,
    available_pairs: dict[str, set[tuple[str, str]]],
    seed: int,
) -> dict[str, tuple[str, str]]:
    expected = len(pairs) * pair_target
    if len(samples) != expected:
        raise ValueError(
            f"Need exactly {expected} samples for pair_target={pair_target}, got {len(samples)}"
        )

    sample_keys = [sample.sample_key for sample in samples]
    for attempt in range(500):
        rng = random.Random(seed + attempt)
        order = list(sample_keys)
        rng.shuffle(order)
        remaining = Counter({pair: pair_target for pair in pairs})
        assignment: dict[str, tuple[str, str]] = {}

        for sample_key in order:
            candidates = [
                pair
                for pair in pairs
                if remaining[pair] > 0 and pair in available_pairs[sample_key]
            ]
            if not candidates:
                break
            max_remaining = max(remaining[pair] for pair in candidates)
            top = [pair for pair in candidates if remaining[pair] == max_remaining]
            chosen = rng.choice(top)
            assignment[sample_key] = chosen
            remaining[chosen] -= 1

        if len(assignment) == len(samples) and all(count == 0 for count in remaining.values()):
            return assignment

    raise RuntimeError("Failed to assign pairs while satisfying exact pair quotas")


def canonical_sort(samples: list[Sample]) -> list[Sample]:
    return sorted(
        samples,
        key=lambda sample: (sample.task, sample.subtask or "", sample.sample_key),
    )


def blind_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": item["item_id"],
        "task": item["task"],
        "subtask": item["subtask"] or "",
        "prompt": item["prompt"],
        "response_a": item["response_a"],
        "response_b": item["response_b"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_model_appearances(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts[item["model_a"]] += 1
        counts[item["model_b"]] += 1
    return dict(sorted(counts.items()))


def count_pairs(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        pair = tuple(sorted((item["model_a"], item["model_b"])))
        counts[" vs ".join(pair)] += 1
    return dict(sorted(counts.items()))


def count_tasks(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(item["task"] for item in items)
    return dict(sorted(counts.items()))


def count_subtasks(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        if item["subtask"]:
            counts[f"{item['task']}::{item['subtask']}"] += 1
    return dict(sorted(counts.items()))


def build_items(
    ordered_samples: list[Sample],
    pair_assignment: dict[str, tuple[str, str]],
    pairwise_rows: dict[tuple[str, tuple[str, str]], dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for offset, sample in enumerate(ordered_samples, start=1):
        pair = pair_assignment[sample.sample_key]
        pairwise = pairwise_rows[(sample.sample_key, pair)]
        item_number = start_index + offset - 1
        items.append(
            {
                "item_id": f"HJ-{item_number:03d}",
                "sample_index": sample.sample_index,
                "sample_key": sample.sample_key,
                "pair_key": pairwise["pair_key"],
                "task": sample.task,
                "subtask": sample.subtask,
                "prompt": sample.prompt,
                "model_a": pairwise["model_a"],
                "model_b": pairwise["model_b"],
                "response_a": pairwise["response_a"],
                "response_b": pairwise["response_b"],
                "judge_model": pairwise["judge_model"],
                "gpt_winner": pairwise["winner"],
                "gpt_winner_model": pairwise["winner_model"],
                "gpt_rationale": pairwise.get("rationale") or "",
                "judge_response": pairwise.get("judge_response") or "",
            }
        )

    return items


def build_summary(
    subset_name: str,
    items: list[dict[str, Any]],
    task_quotas: dict[str, int],
    parent_subset_name: str | None,
) -> dict[str, Any]:
    return {
        "subset_name": subset_name,
        "size": len(items),
        "parent_subset": parent_subset_name,
        "source_samples": str(SAMPLES_PATH),
        "source_pairwise_results": str(PAIRWISE_RESULTS_PATH),
        "selection_seed": SELECTION_SEED,
        "pair_assignment_seed_100": PAIR_ASSIGNMENT_SEED_100,
        "pair_assignment_seed_extra_100": PAIR_ASSIGNMENT_SEED_EXTRA_100,
        "task_quotas": dict(sorted(task_quotas.items())),
        "task_counts": count_tasks(items),
        "subtask_counts": count_subtasks(items),
        "pair_counts": count_pairs(items),
        "model_appearance_counts": count_model_appearances(items),
        "judge_models": dict(sorted(Counter(item["judge_model"] for item in items).items())),
    }


def main() -> None:
    pairwise_rows, available_pairs = load_pairwise_rows(PAIRWISE_RESULTS_PATH)
    _, by_task, pairs = load_samples(SAMPLES_PATH, available_pairs)

    task_counts = {task: len(samples) for task, samples in by_task.items()}
    quotas_100 = allocate_task_quotas(task_counts, 100)
    quotas_200 = allocate_task_quotas(task_counts, 200)

    selected_200_by_task = sample_by_task(by_task, quotas_200, SELECTION_SEED)
    selected_100_by_task = sample_by_task(selected_200_by_task, quotas_100, SELECTION_SEED + 1)

    selected_200 = flatten_task_samples(selected_200_by_task)
    selected_100 = flatten_task_samples(selected_100_by_task)
    selected_100_keys = {sample.sample_key for sample in selected_100}
    extra_100 = [sample for sample in selected_200 if sample.sample_key not in selected_100_keys]

    assignment_100 = assign_pairs(
        selected_100,
        pairs,
        pair_target=10,
        available_pairs=available_pairs,
        seed=PAIR_ASSIGNMENT_SEED_100,
    )
    assignment_extra_100 = assign_pairs(
        extra_100,
        pairs,
        pair_target=10,
        available_pairs=available_pairs,
        seed=PAIR_ASSIGNMENT_SEED_EXTRA_100,
    )

    ordered_100 = canonical_sort(selected_100)
    ordered_extra_100 = canonical_sort(extra_100)

    items_100 = build_items(ordered_100, assignment_100, pairwise_rows, start_index=1)
    items_extra_100 = build_items(
        ordered_extra_100,
        assignment_extra_100,
        pairwise_rows,
        start_index=101,
    )
    items_200 = items_100 + items_extra_100

    subset_payloads = {
        "subset_100": (items_100, quotas_100, None),
        "subset_200": (items_200, quotas_200, "subset_100"),
    }

    for subset_name, (items, quotas, parent_subset) in subset_payloads.items():
        subset_dir = ROOT / subset_name
        blind_rows = [blind_row(item) for item in items]
        key_rows = [
            {
                "item_id": item["item_id"],
                "sample_index": item["sample_index"],
                "sample_key": item["sample_key"],
                "pair_key": item["pair_key"],
                "task": item["task"],
                "subtask": item["subtask"],
                "model_a": item["model_a"],
                "model_b": item["model_b"],
                "gpt_winner": item["gpt_winner"],
                "gpt_winner_model": item["gpt_winner_model"],
                "judge_model": item["judge_model"],
                "gpt_rationale": item["gpt_rationale"],
            }
            for item in items
        ]
        summary = build_summary(subset_name, items, quotas, parent_subset)

        write_csv(subset_dir / "blind_master.csv", blind_rows)
        write_jsonl(subset_dir / "blind_master.jsonl", blind_rows)
        write_jsonl(subset_dir / "key.jsonl", key_rows)
        write_json(subset_dir / "summary.json", summary)
        for old_path in (subset_dir / "annotator_a.csv", subset_dir / "annotator_b.csv"):
            if old_path.exists():
                old_path.unlink()

    top_level_summary = {
        "subset_100_items": len(items_100),
        "subset_200_items": len(items_200),
        "subset_100_is_nested_in_subset_200": True,
        "subset_100_item_ids": [item["item_id"] for item in items_100],
        "subset_200_item_ids": [item["item_id"] for item in items_200],
    }
    write_json(ROOT / "summary.json", top_level_summary)

    print("Built human-judge subsets:")
    print(f"  subset_100: {len(items_100)} items")
    print(f"  subset_200: {len(items_200)} items")


if __name__ == "__main__":
    main()

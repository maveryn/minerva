from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dart.common import ensure_dir, is_athena_row, is_seceval_row, load_parquet_rows, make_uid, write_jsonl
from dart.generate import DartGenerator
from dart.score import score_rows


def evaluate_model(
    *,
    model_path: str,
    val_paths: List[str],
    output_dir: str | Path,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: int,
    trust_remote_code: bool = False,
    backend: str = "hf",
    gpu_memory_utilization: float = 0.9,
    limit_per_val_path: int | None = None,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    combined_rows: List[Dict[str, Any]] = []
    uid_to_dataset: Dict[str, str] = {}
    for val_path in val_paths:
        dataset_name = Path(val_path).stem
        rows = load_parquet_rows(val_path, limit=limit_per_val_path)
        for idx, row in enumerate(rows):
            uid = make_uid(row, fallback_index=idx)
            row = dict(row)
            row["uid"] = uid
            uid_to_dataset[uid] = dataset_name
            combined_rows.append(row)

    generator = DartGenerator(
        model_path=model_path,
        backend=backend,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        max_prompt_length=max_prompt_length,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    generated, skipped = generator.generate_rows(combined_rows, mode="plain")
    if skipped:
        write_jsonl(output_dir / "skipped.jsonl", skipped)
    scored_all = score_rows(generated, success_threshold=1.0)
    per_dataset_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_all:
        per_dataset_rows[uid_to_dataset[str(row.get("uid") or "")]].append(row)

    per_dataset = {}
    for idx, val_path in enumerate(val_paths):
        dataset_name = Path(val_path).stem
        scored = per_dataset_rows.get(dataset_name, [])
        write_jsonl(output_dir / f"{idx:02d}_{dataset_name}_scored.jsonl", scored)
        dataset_mean = (
            sum(float(row.get("verifier_score", 0.0)) for row in scored) / len(scored)
            if scored
            else 0.0
        )
        per_dataset[dataset_name] = {
            "path": str(val_path),
            "count": len(scored),
            "mean_score": dataset_mean,
        }

    by_data_source: Dict[str, List[float]] = defaultdict(list)
    for row in scored_all:
        by_data_source[str(row.get("data_source") or "")].append(float(row.get("verifier_score", 0.0)))

    per_source = {
        key: {
            "count": len(values),
            "mean_score": (sum(values) / len(values)) if values else 0.0,
        }
        for key, values in sorted(by_data_source.items())
    }

    all_scores = [float(row.get("verifier_score", 0.0)) for row in scored_all]
    global_mean = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

    minerva_dataset = None
    athena_dataset_means: List[float] = []
    for dataset_name, info in per_dataset.items():
        mean_score = info.get("mean_score")
        if mean_score is None:
            continue
        if dataset_name.startswith("athena_cti_"):
            athena_dataset_means.append(float(mean_score))
        elif "minerva" in dataset_name and "dev" in dataset_name and minerva_dataset is None:
            minerva_dataset = dataset_name
    minerva_mean = (
        float(per_dataset[minerva_dataset]["mean_score"])
        if minerva_dataset is not None
        else None
    )
    athena_mean = (
        sum(athena_dataset_means) / len(athena_dataset_means)
        if athena_dataset_means
        else None
    )
    rl_global_val_mean = None
    if minerva_mean is not None and athena_mean is not None:
        rl_global_val_mean = (minerva_mean + athena_mean) / 2.0

    summary = {
        "model_path": model_path,
        "global_mean_score": global_mean,
        "rl_minerva_dev_mean_score": minerva_mean,
        "rl_athena_bench_mean_score": athena_mean,
        "rl_global_val_mean_score": rl_global_val_mean,
        "count": len(scored_all),
        "per_dataset": per_dataset,
        "per_data_source": per_source,
    }
    with (Path(output_dir) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DART checkpoint on RL-matched CTI validation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--val-path", action="append", required=True, dest="val_paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--limit-per-val-path", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    summary = evaluate_model(
        model_path=args.model_path,
        val_paths=args.val_paths,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_per_val_path=args.limit_per_val_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

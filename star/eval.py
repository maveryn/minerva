from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List

import pandas as pd

from star.common import ensure_dir, iter_jsonl, make_uid, write_jsonl
from star.score import score_rows


def evaluate_model(
    *,
    model_path: str,
    val_paths: List[str],
    output_dir: str | Path,
    round_id: int,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_length: int,
    limit_per_val_path: int | None = None,
    trust_remote_code: bool = False,
    backend: str = "hf",
    gpu_memory_utilization: float = 0.9,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    combined_rows: List[Dict[str, Any]] = []
    uid_to_dataset: Dict[str, str] = {}
    per_dataset: Dict[str, Dict[str, Any]] = {}
    dataset_order: List[str] = []
    for val_path in val_paths:
        dataset_name = Path(val_path).stem
        dataset_order.append(dataset_name)
        df = pd.read_parquet(val_path)
        if limit_per_val_path is not None:
            df = df.head(limit_per_val_path)
        rows = df.to_dict(orient="records")
        for row_idx, row in enumerate(rows):
            uid = make_uid(row, fallback_index=row_idx)
            uid_to_dataset[uid] = dataset_name
            combined_rows.append(row)

    combined_parquet = output_dir / "combined_eval_input.parquet"
    pd.DataFrame(combined_rows).to_parquet(combined_parquet, index=False)

    generated_path = output_dir / "combined_generated.jsonl"
    command = [
        "python",
        "-m",
        "star.generate",
        "--parquet",
        str(combined_parquet),
        "--model-path",
        model_path,
        "--mode",
        "original",
        "--output",
        str(generated_path),
        "--round-id",
        str(round_id),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--max-prompt-length",
        str(max_prompt_length),
        "--backend",
        str(backend),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])

    generated = list(iter_jsonl(generated_path))
    scored_all = score_rows(generated, success_threshold=1.0)
    scored_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_all:
        dataset_name = uid_to_dataset.get(str(row.get("uid") or ""))
        if dataset_name is None:
            continue
        scored_by_dataset[dataset_name].append(row)

    all_scored: List[Dict[str, Any]] = []
    for idx, val_path in enumerate(val_paths):
        dataset_name = dataset_order[idx]
        scored = scored_by_dataset.get(dataset_name, [])
        all_scored.extend(scored)
        write_jsonl(output_dir / f"{idx:02d}_{dataset_name}_scored.jsonl", scored)
        dataset_mean = (
            sum(float(row.get("verifier_score", 0.0)) for row in scored) / len(scored)
            if scored
            else 0.0
        )
        by_source: Dict[str, List[float]] = defaultdict(list)
        for row in scored:
            by_source[str(row.get("data_source") or "")].append(float(row.get("verifier_score", 0.0)))
        per_dataset[dataset_name] = {
            "path": str(val_path),
            "count": len(scored),
            "mean_score": dataset_mean,
            "per_data_source": {
                key: {
                    "count": len(values),
                    "mean_score": (sum(values) / len(values)) if values else 0.0,
                }
                for key, values in sorted(by_source.items())
            },
        }

    by_source: Dict[str, List[float]] = defaultdict(list)
    for row in all_scored:
        by_source[str(row.get("data_source") or "")].append(float(row.get("verifier_score", 0.0)))

    per_source = {
        key: {
            "count": len(values),
            "mean_score": (sum(values) / len(values)) if values else 0.0,
        }
        for key, values in sorted(by_source.items())
    }
    global_mean = (
        sum(float(row.get("verifier_score", 0.0)) for row in all_scored) / len(all_scored)
        if all_scored
        else 0.0
    )
    minerva_dataset = None
    athena_scores: List[float] = []
    for dataset_name, info in per_dataset.items():
        if dataset_name.startswith("athena_cti_"):
            athena_scores.append(float(info.get("mean_score", 0.0)))
        elif "minerva" in dataset_name and "dev" in dataset_name and minerva_dataset is None:
            minerva_dataset = dataset_name
    minerva_dev_mean = (
        float(per_dataset[minerva_dataset]["mean_score"])
        if minerva_dataset is not None
        else None
    )
    athena_bench_mean = (sum(athena_scores) / len(athena_scores)) if athena_scores else None
    rl_global_val_mean = None
    if minerva_dev_mean is not None and athena_bench_mean is not None:
        rl_global_val_mean = (minerva_dev_mean + athena_bench_mean) / 2.0
    summary = {
        "model_path": model_path,
        "round_id": round_id,
        "global_mean_score": global_mean,
        "rl_minerva_dev_mean_score": minerva_dev_mean,
        "rl_athena_bench_mean_score": athena_bench_mean,
        "rl_global_val_mean_score": rl_global_val_mean,
        "count": len(all_scored),
        "per_dataset": per_dataset,
        "per_data_source": per_source,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a STaR checkpoint on Minerva validation parquet")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--val-path", action="append", required=True, dest="val_paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-id", type=int, default=0)
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
        round_id=args.round_id,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        limit_per_val_path=args.limit_per_val_path,
        trust_remote_code=args.trust_remote_code,
        backend=args.backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

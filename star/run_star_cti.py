from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from star.build_sft_dataset import build_sft_dataset
from star.common import ensure_dir, load_config
from star.eval import evaluate_model
from star.common import iter_jsonl
from star.score import score_rows
from star.select import select_rows


def _stage_paths(round_dir: Path) -> Dict[str, Path]:
    return {
        "original_samples": round_dir / "original_samples.jsonl",
        "original_scored": round_dir / "original_scored.jsonl",
        "rationalization_samples": round_dir / "rationalization_samples.jsonl",
        "rationalization_scored": round_dir / "rationalization_scored.jsonl",
        "selected": round_dir / "selected.jsonl",
        "round_sft_parquet": round_dir / "star_sft_train.parquet",
        "train_root": round_dir / "train",
        "checkpoint_root": round_dir / "checkpoint",
        "eval_root": round_dir / "eval",
        "summary": round_dir / "summary.json",
    }


def _resolve_best_model_path(train_root: Path) -> Optional[Path]:
    candidates = [
        train_root / "best" / "hf_model",
        train_root / "best",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    tracker = train_root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step_name = tracker.read_text(encoding="utf-8").strip()
        if step_name:
            step_dir = train_root / step_name
            for candidate in [step_dir / "huggingface", step_dir / "hf_model", step_dir]:
                if (candidate / "config.json").exists():
                    return candidate
    for path in sorted(train_root.rglob("config.json")):
        if "best" in path.parts or "hf_model" in path.parts or "huggingface" in path.parts:
            return path.parent
    return None


def _materialize_round_checkpoint(model_path: Path, checkpoint_root: Path) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    target = checkpoint_root / "hf_model"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    shutil.copytree(model_path, target)
    return target


def _prune_training_outputs(train_root: Path, keep_model_path: Optional[Path]) -> None:
    if not train_root.exists():
        return
    keep_path = keep_model_path.resolve() if keep_model_path is not None and keep_model_path.exists() else None
    for child in train_root.iterdir():
        try:
            if keep_path is not None and child.resolve() == keep_path:
                continue
        except OSError:
            pass
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _run_training(cfg: Dict[str, Any], *, train_parquet: Path, model_path: str, train_root: Path, round_id: int) -> None:
    ensure_dir(train_root)
    training_cfg = cfg.get("training", {})
    train_rows = len(pd.read_parquet(train_parquet))
    if train_rows <= 0:
        raise ValueError(f"STaR training parquet has no rows: {train_parquet}")
    requested_train_batch = int(training_cfg.get("train_batch_size", 128))
    requested_micro_batch = int(training_cfg.get("micro_batch_size_per_gpu", 8))
    effective_train_batch = min(requested_train_batch, train_rows)
    effective_micro_batch = min(requested_micro_batch, effective_train_batch)
    if effective_micro_batch <= 0:
        effective_micro_batch = 1
    if effective_train_batch <= 0:
        effective_train_batch = 1
    effective_train_batch = max(
        effective_micro_batch,
        (effective_train_batch // effective_micro_batch) * effective_micro_batch,
    )
    backend = str(training_cfg.get("backend", "verl")).lower()
    if backend == "simple":
        command = [
            "python",
            "-m",
            "star.simple_sft",
            "--model-path",
            model_path,
            "--train-path",
            str(train_parquet),
            "--output-root",
            str(train_root),
            "--max-length",
            str(training_cfg.get("max_length", 4096)),
            "--train-batch-size",
            str(effective_micro_batch),
            "--gradient-accumulation-steps",
            str(training_cfg.get("gradient_accumulation_steps", 1)),
            "--learning-rate",
            str(training_cfg.get("learning_rate", 1e-5)),
            "--total-epochs",
            str(training_cfg.get("total_epochs", 1)),
            "--save-steps",
            str(training_cfg.get("save_freq", 50)),
            "--logging-steps",
            str(training_cfg.get("logging_steps", 1)),
        ]
        if "total_steps" in training_cfg:
            command.extend(["--total-steps", str(training_cfg.get("total_steps"))])
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, env=os.environ.copy())
        return

    env = {
        "SFT_MODEL_PATH": model_path,
        "SFT_TRAIN_PATH": str(train_parquet),
        "SFT_OUTPUT_ROOT": str(train_root),
        "SFT_EXPERIMENT_NAME": f"{cfg['experiment_name']}_round_{round_id}",
        "SFT_TOTAL_EPOCHS": str(training_cfg.get("total_epochs", 1)),
        "SFT_TRAIN_BATCH_SIZE": str(effective_train_batch),
        "SFT_MICRO_BATCH_SIZE_PER_GPU": str(effective_micro_batch),
        "SFT_MAX_LENGTH": str(training_cfg.get("max_length", 4096)),
        "SFT_MODEL_DTYPE": str(training_cfg.get("model_dtype", "bfloat16")),
        "SFT_MODEL_STRATEGY": str(training_cfg.get("model_strategy", "fsdp")),
        "SFT_ENABLE_GRADIENT_CHECKPOINTING": str(training_cfg.get("enable_gradient_checkpointing", True)).lower(),
        "SFT_REWARD_EVAL_BATCH_SIZE": str(training_cfg.get("reward_eval_batch_size", 64)),
        "SFT_REWARD_EVAL_MAX_PROMPT_LEN": str(training_cfg.get("reward_eval_max_prompt_length", 2048)),
        "SFT_REWARD_EVAL_MAX_RESPONSE_LEN": str(training_cfg.get("reward_eval_max_response_length", 1024)),
        "SFT_N_GPUS_PER_NODE": str(training_cfg.get("n_gpus", 1)),
        "SFT_SAVE_FREQ": str(training_cfg.get("save_freq", 50)),
        "SFT_TEST_FREQ": str(training_cfg.get("test_freq", 50)),
        "SFT_SAVE_BEST_ONLY": str(training_cfg.get("save_best_only", True)).lower(),
    }
    if "total_steps" in training_cfg:
        env["SFT_TOTAL_STEPS"] = str(training_cfg.get("total_steps"))
    val_paths = list(cfg.get("val_paths", []))
    for idx, path in enumerate(val_paths, start=1):
        env[f"SFT_VAL_PATH_{idx}"] = str(path)
    command = ["bash", "star/train_round.sh"]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, env={**os.environ, **env})


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    from star.common import write_jsonl

    write_jsonl(path, rows)


def _run_generation_phase(
    *,
    cfg: Dict[str, Any],
    mode: str,
    round_id: int,
    model_path: str,
    output_path: Path,
) -> List[Dict[str, Any]]:
    gen_cfg = cfg.get("generation", {})
    command = [
        "python",
        "-m",
        "star.generate",
        "--parquet",
        str(cfg["train_path"]),
        "--model-path",
        model_path,
        "--mode",
        mode,
        "--output",
        str(output_path),
        "--round-id",
        str(round_id),
        "--batch-size",
        str(int(gen_cfg.get("batch_size", 8))),
        "--max-new-tokens",
        str(int(gen_cfg.get("max_new_tokens", 512))),
        "--temperature",
        str(float(gen_cfg.get("temperature", 0.7))),
        "--top-p",
        str(float(gen_cfg.get("top_p", 0.95))),
        "--max-prompt-length",
        str(int(gen_cfg.get("max_prompt_length", 2048))),
        "--backend",
        str(gen_cfg.get("backend", "hf")),
        "--gpu-memory-utilization",
        str(float(gen_cfg.get("gpu_memory_utilization", 0.9))),
    ]
    if bool(gen_cfg.get("trust_remote_code", False)):
        command.append("--trust-remote-code")
    limit = cfg.get("limit_train_rows")
    if limit is not None:
        command.extend(["--limit", str(limit)])
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, env=os.environ.copy())
    return list(iter_jsonl(output_path))


def _round_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "success_count": sum(1 for row in rows if bool(row.get("verifier_success"))),
    }


def _init_wandb(cfg: Dict[str, Any]):
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    import wandb

    run = wandb.init(
        project=str(wandb_cfg.get("project", "minerva")),
        entity=wandb_cfg.get("entity"),
        name=str(wandb_cfg.get("name", cfg.get("experiment_name", "star_cti"))),
        group=wandb_cfg.get("group"),
        job_type=str(wandb_cfg.get("job_type", "star")),
        config=cfg,
    )
    return run


def _log_wandb_round(wandb_run, summary: Dict[str, Any]) -> None:
    if wandb_run is None:
        return
    eval_info = summary.get("eval") or {}
    payload: Dict[str, Any] = {
        "star/round": int(summary.get("round_id", 0)),
        "star/selected_count": int(summary.get("selected_count", 0)),
        "star/round_sft_rows": int(summary.get("round_sft_rows", 0)),
        "star/original_success_count": int((summary.get("original") or {}).get("success_count", 0)),
        "star/rationalization_success_count": int((summary.get("rationalization") or {}).get("success_count", 0)),
    }
    if eval_info.get("rl_minerva_dev_mean_score") is not None:
        payload["val-core/minerva-dev/reward/mean"] = float(eval_info["rl_minerva_dev_mean_score"])
    if eval_info.get("rl_athena_bench_mean_score") is not None:
        payload["val-core/athena-bench/reward/mean"] = float(eval_info["rl_athena_bench_mean_score"])
    if eval_info.get("rl_global_val_mean_score") is not None:
        payload["val-core/global-val/reward/mean"] = float(eval_info["rl_global_val_mean_score"])
    per_dataset = eval_info.get("per_dataset") or {}
    for dataset_name, dataset_info in per_dataset.items():
        mean_score = dataset_info.get("mean_score")
        if mean_score is not None:
            payload[f"val/{dataset_name}/reward/mean"] = float(mean_score)
    wandb_run.log(payload, step=int(summary.get("round_id", 0)))


def run_round(cfg: Dict[str, Any], *, round_id: int, model_path: str, skip_train: bool, skip_eval: bool) -> Dict[str, Any]:
    output_root = ensure_dir(cfg["output_root"])
    round_dir = ensure_dir(output_root / f"round_{round_id:02d}")
    paths = _stage_paths(round_dir)
    gen_cfg = cfg.get("generation", {})

    if paths["original_scored"].exists():
        original_scored = list(iter_jsonl(paths["original_scored"]))
    else:
        original_rows = _run_generation_phase(
            cfg=cfg,
            mode="original",
            round_id=round_id,
            model_path=model_path,
            output_path=paths["original_samples"],
        )
        original_scored = score_rows(original_rows, success_threshold=float(cfg.get("success_threshold", 1.0)))
        _write_jsonl(paths["original_scored"], original_scored)

    if paths["rationalization_scored"].exists():
        rationalization_scored = list(iter_jsonl(paths["rationalization_scored"]))
    else:
        rationalization_rows = _run_generation_phase(
            cfg=cfg,
            mode="rationalization",
            round_id=round_id,
            model_path=model_path,
            output_path=paths["rationalization_samples"],
        )
        rationalization_scored = score_rows(rationalization_rows, success_threshold=float(cfg.get("success_threshold", 1.0)))
        _write_jsonl(paths["rationalization_scored"], rationalization_scored)

    if paths["selected"].exists():
        selected = list(iter_jsonl(paths["selected"]))
    else:
        selected = select_rows(original_scored, rationalization_scored)
        _write_jsonl(paths["selected"], selected)

    if paths["round_sft_parquet"].exists():
        total_sft_rows = len(pd.read_parquet(paths["round_sft_parquet"]))
    else:
        training_cfg = cfg.get("training", {})
        total_sft_rows = build_sft_dataset(
            selected_jsonl=paths["selected"],
            output_parquet=paths["round_sft_parquet"],
            existing_parquet=None,
            model_path=str(cfg["model_path"]),
            response_max_tokens=int(training_cfg.get("response_max_tokens", 0)) or None,
            trust_remote_code=bool(gen_cfg.get("trust_remote_code", False)),
        )

    trained_model_path: Optional[Path] = None
    round_checkpoint_path: Optional[Path] = None
    if not skip_train:
        _run_training(
            cfg,
            train_parquet=paths["round_sft_parquet"],
            model_path=str(cfg["model_path"]),
            train_root=paths["train_root"],
            round_id=round_id,
        )
        trained_model_path = _resolve_best_model_path(paths["train_root"])
        if trained_model_path is not None:
            round_checkpoint_path = _materialize_round_checkpoint(trained_model_path, paths["checkpoint_root"])

    eval_summary = None
    eval_model_path = round_checkpoint_path if round_checkpoint_path is not None else trained_model_path
    if not skip_eval and eval_model_path is not None:
        eval_cfg = cfg.get("evaluation", {})
        eval_summary = evaluate_model(
            model_path=str(eval_model_path),
            val_paths=list(cfg.get("val_paths", [])),
            output_dir=paths["eval_root"],
            round_id=round_id,
            batch_size=int(eval_cfg.get("batch_size", 8)),
            max_new_tokens=int(eval_cfg.get("max_new_tokens", 512)),
            max_prompt_length=int(eval_cfg.get("max_prompt_length", 2048)),
            limit_per_val_path=(
                int(eval_cfg.get("limit_per_val_path"))
                if eval_cfg.get("limit_per_val_path") is not None
                else None
            ),
            trust_remote_code=bool(gen_cfg.get("trust_remote_code", False)),
            backend=str(eval_cfg.get("backend", gen_cfg.get("backend", "hf"))),
            gpu_memory_utilization=float(eval_cfg.get("gpu_memory_utilization", gen_cfg.get("gpu_memory_utilization", 0.9))),
        )
    if round_checkpoint_path is not None:
        _prune_training_outputs(paths["train_root"], trained_model_path)

    summary = {
        "round_id": round_id,
        "generator_model_path": model_path,
        "train_init_model_path": str(cfg["model_path"]),
        "model_path_out": str(round_checkpoint_path) if round_checkpoint_path else (str(trained_model_path) if trained_model_path else None),
        "round_checkpoint_path": str(round_checkpoint_path) if round_checkpoint_path else None,
        "original": _round_counts(original_scored),
        "rationalization": _round_counts(rationalization_scored),
        "selected_count": len(selected),
        "round_sft_rows": total_sft_rows,
        "eval": eval_summary,
    }
    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline STaR-CTI loop")
    parser.add_argument("--config", default="star/configs/star_cti.yaml")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(cfg["output_root"])
    rounds = int(args.rounds if args.rounds is not None else cfg.get("num_rounds", 1))
    current_model_path = str(cfg["model_path"])
    best_score = None
    best_round = None
    best_model_path = None
    summaries = []
    wandb_run = _init_wandb(cfg)

    try:
        for round_id in range(rounds):
            summary = run_round(
                cfg,
                round_id=round_id,
                model_path=current_model_path,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval,
            )
            summaries.append(summary)
            _log_wandb_round(wandb_run, summary)
            if summary.get("model_path_out"):
                current_model_path = str(summary["model_path_out"])
            eval_info = summary.get("eval") or {}
            score = eval_info.get("rl_global_val_mean_score")
            if score is not None and (best_score is None or float(score) > float(best_score)):
                best_score = float(score)
                best_round = round_id
                best_model_path = summary.get("model_path_out")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    final_summary = {
        "experiment_name": cfg.get("experiment_name"),
        "best_round": best_round,
        "best_score": best_score,
        "best_model_path": best_model_path,
        "round_summaries": summaries,
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()

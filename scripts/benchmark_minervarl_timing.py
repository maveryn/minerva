#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RLVR_ROOT = REPO_ROOT / "rlvr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small GRPO vs Noctua timing benchmark and save structured timing summaries."
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps for each run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs and summaries. Defaults to artifacts/benchmarks/minervarl_timing/<timestamp>.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Override GPUs per node for both runs. If unset, keep the wrapper defaults.",
    )
    parser.add_argument(
        "--echo-logs",
        action="store_true",
        help="Echo subprocess logs to stdout while also writing them to files.",
    )
    parser.add_argument(
        "--grpo-wrapper",
        type=Path,
        default=Path("rlvr/cti-scripts/train_minerva_base_llama8b.sh"),
        help="GRPO wrapper to execute, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--noctua-wrapper",
        type=Path,
        default=Path("rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.05_textcnn_mlh_t0.7_p0.9_defer_ema.sh"),
        help="Noctua wrapper to execute, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def parse_local_logger_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    match = re.search(r"(?<!\S)step:(\d+)\b", line)
    if match is None:
        return None
    line = line[match.start() :]

    parts = [part.strip() for part in line.split(" - ") if part.strip()]
    if not parts:
        return None

    step_part = parts[0]
    if ":" not in step_part:
        return None
    _, step_text = step_part.split(":", 1)
    try:
        step = int(step_text)
    except ValueError:
        return None

    metrics: dict[str, float] = {}
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value_text = part.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not key:
            continue
        value = parse_numeric_value(value_text)
        if value is None:
            continue
        metrics[key] = value

    return {"step": step, "metrics": metrics, "raw_line": line}


def parse_numeric_value(value_text: str) -> float | None:
    try:
        value = ast.literal_eval(value_text)
    except Exception:
        try:
            value = float(value_text)
        except ValueError:
            return None

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def build_common_overrides(run_output_dir: Path) -> list[str]:
    return [
        "trainer.logger=[console]",
        "trainer.project_name=minerva_timing_benchmark",
        "trainer.val_before_train=false",
        "trainer.test_freq=0",
        "trainer.save_freq=0",
        f"trainer.default_local_dir={run_output_dir}",
    ]


def build_grpo_spec(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "GRPO_TOTAL_STEPS": str(args.steps),
        "GRPO_TEST_FREQ": "0",
        "GRPO_SAVE_FREQ": "0",
        "GRPO_VAL_BEFORE_TRAIN": "false",
        "GRPO_OUTPUT_ROOT": str(output_dir / "grpo_checkpoints"),
        "GRPO_EXPERIMENT_NAME": f"benchmark_grpo_llama8b_{args.steps}steps",
    }
    if args.gpus_per_node is not None:
        env["GRPO_N_GPUS_PER_NODE"] = str(args.gpus_per_node)

    run_output_dir = output_dir / "grpo_run"
    cmd = [
        "bash",
        str(resolve_path(args.grpo_wrapper)),
        *build_common_overrides(run_output_dir),
    ]
    return {"name": "grpo", "cmd": cmd, "env": env}


def build_noctua_spec(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "VERL_VAL_DEBUG": "0",
        "ACRD_DEBUG_SAMPLES": "0",
        "ACRD_DETAILS_DEBUG_SAMPLES": "0",
        "ACRD_ML_DEBUG_SAMPLES": "0",
        "ACRD_TOTAL_STEPS": str(args.steps),
        "ACRD_TEST_FREQ": "0",
        "ACRD_SAVE_FREQ": "0",
        "ACRD_OUTPUT_ROOT": str(output_dir / "noctua_checkpoints"),
        "ACRD_EXPERIMENT_NAME": f"benchmark_noctua_llama8b_{args.steps}steps",
        "ACRD_ACR_DISTILL_INTERVAL": str(args.steps),
    }
    if args.gpus_per_node is not None:
        env["ACRD_N_GPUS_PER_NODE"] = str(args.gpus_per_node)

    run_output_dir = output_dir / "noctua_run"
    cmd = [
        "bash",
        str(resolve_path(args.noctua_wrapper)),
        *build_common_overrides(run_output_dir),
    ]
    return {"name": "noctua", "cmd": cmd, "env": env}


def run_benchmark(spec: dict[str, Any], output_dir: Path, echo_logs: bool) -> dict[str, Any]:
    name = spec["name"]
    log_path = output_dir / f"{name}.log"
    steps_path = output_dir / f"{name}.steps.jsonl"
    env = os.environ.copy()
    env.update(spec["env"])
    pythonpath_entries = [str(RLVR_ROOT), str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    process_start = time.perf_counter()
    step_records: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            spec["cmd"],
            cwd=RLVR_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            now = time.perf_counter()
            log_file.write(raw_line)
            log_file.flush()
            if echo_logs:
                sys.stdout.write(f"[{name}] {raw_line}")
                sys.stdout.flush()

            parsed = parse_local_logger_line(raw_line)
            if parsed is None:
                continue
            parsed["wall_time_from_process_start_s"] = now - process_start
            step_records.append(parsed)

        return_code = proc.wait()
    process_wall_s = time.perf_counter() - process_start

    if return_code != 0:
        raise RuntimeError(f"{name} benchmark failed with exit code {return_code}. See {log_path}.")

    with steps_path.open("w", encoding="utf-8") as f:
        for record in step_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    if not step_records:
        raise RuntimeError(f"{name} benchmark produced no parsed step logs. See {log_path}.")

    summary = summarize_run(name, spec, step_records, process_wall_s)
    summary_path = output_dir / f"{name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def summarize_run(
    name: str,
    spec: dict[str, Any],
    step_records: list[dict[str, Any]],
    process_wall_s: float,
) -> dict[str, Any]:
    timing_sums: dict[str, float] = defaultdict(float)
    metric_sums: dict[str, float] = defaultdict(float)

    for record in step_records:
        for key, value in record["metrics"].items():
            metric_sums[key] += value
            if key.startswith("timing_s/"):
                timing_sums[key] += value

    last_step = max(record["step"] for record in step_records)
    step_full_sum = timing_sums.get("timing_s/step_full")
    step_sum = timing_sums.get("timing_s/step")
    effective_sum = step_full_sum if step_full_sum is not None else step_sum
    startup_overhead_s = None
    if effective_sum is not None:
        startup_overhead_s = process_wall_s - effective_sum

    component_keys = [
        "timing_s/acr_build",
        "timing_s/gen",
        "timing_s/reward",
        "timing_s/update_actor",
        "timing_s/update_critic",
        "timing_s/acr_gen",
        "timing_s/acr_reward",
        "timing_s/acr_distill_sft",
        "timing_s/acr_distill_dpo",
    ]
    component_sums = {key: timing_sums[key] for key in component_keys if key in timing_sums}

    return {
        "name": name,
        "command": spec["cmd"],
        "env_overrides": spec["env"],
        "steps_observed": len(step_records),
        "last_step_logged": last_step,
        "process_wall_s": process_wall_s,
        "timing_step_s_sum": step_sum,
        "timing_step_full_s_sum": step_full_sum,
        "startup_and_nonstep_overhead_s": startup_overhead_s,
        "component_sums_s": component_sums,
        "step_records": step_records,
        "metric_sums": dict(metric_sums),
        "metric_means": compute_metric_means(step_records),
    }


def compute_metric_means(step_records: list[dict[str, Any]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for record in step_records:
        for key, value in record["metrics"].items():
            sums[key] += value
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}


def build_comparison(grpo: dict[str, Any], noctua: dict[str, Any]) -> dict[str, Any]:
    grpo_total = grpo.get("timing_step_full_s_sum") or grpo.get("timing_step_s_sum")
    noctua_total = noctua.get("timing_step_full_s_sum") or noctua.get("timing_step_s_sum")
    overhead_s = None
    overhead_pct = None
    if grpo_total is not None and noctua_total is not None:
        overhead_s = noctua_total - grpo_total
        if grpo_total > 0:
            overhead_pct = 100.0 * overhead_s / grpo_total

    return {
        "grpo_training_wall_s": grpo_total,
        "noctua_training_wall_s": noctua_total,
        "noctua_minus_grpo_s": overhead_s,
        "noctua_over_grpo_pct": overhead_pct,
        "grpo_process_wall_s": grpo.get("process_wall_s"),
        "noctua_process_wall_s": noctua.get("process_wall_s"),
        "grpo_startup_and_nonstep_overhead_s": grpo.get("startup_and_nonstep_overhead_s"),
        "noctua_startup_and_nonstep_overhead_s": noctua.get("startup_and_nonstep_overhead_s"),
        "noctua_components_s": noctua.get("component_sums_s", {}),
    }


def write_report(output_dir: Path, args: argparse.Namespace, grpo: dict[str, Any], noctua: dict[str, Any]) -> None:
    comparison = build_comparison(grpo, noctua)
    report_path = output_dir / "report.md"

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# MinervaRL Timing Benchmark",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Steps per run: {args.steps}",
        "",
        "## Totals",
        "",
        "| Run | Training wall s (startup excluded) | Process wall s | Startup/non-step overhead s |",
        "| --- | ---: | ---: | ---: |",
        f"| GRPO | {fmt(comparison['grpo_training_wall_s'])} | {fmt(comparison['grpo_process_wall_s'])} | {fmt(comparison['grpo_startup_and_nonstep_overhead_s'])} |",
        f"| Noctua | {fmt(comparison['noctua_training_wall_s'])} | {fmt(comparison['noctua_process_wall_s'])} | {fmt(comparison['noctua_startup_and_nonstep_overhead_s'])} |",
        "",
        "## Noctua Components",
        "",
        "| Component | Sum s |",
        "| --- | ---: |",
    ]

    for key, value in sorted(noctua.get("component_sums_s", {}).items()):
        lines.append(f"| `{key}` | {fmt(value)} |")

    lines.extend(
        [
            "",
            "## Overhead",
            "",
            f"- Noctua minus GRPO: {fmt(comparison['noctua_minus_grpo_s'])} s",
            f"- Noctua over GRPO: {fmt(comparison['noctua_over_grpo_pct'])} %",
            "",
            "## Files",
            "",
            f"- Raw GRPO log: `{display_path(output_dir / 'grpo.log')}`",
            f"- Raw Noctua log: `{display_path(output_dir / 'noctua.log')}`",
            f"- GRPO summary: `{display_path(output_dir / 'grpo.summary.json')}`",
            f"- Noctua summary: `{display_path(output_dir / 'noctua.summary.json')}`",
            f"- Comparison: `{display_path(output_dir / 'comparison.json')}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "artifacts" / "benchmarks" / "minervarl_timing" / timestamp
    elif not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "steps": args.steps,
        "gpus_per_node": args.gpus_per_node,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    specs = [
        build_grpo_spec(args, output_dir),
        build_noctua_spec(args, output_dir),
    ]

    summaries = {}
    for spec in specs:
        summaries[spec["name"]] = run_benchmark(spec, output_dir, echo_logs=args.echo_logs)

    comparison = build_comparison(summaries["grpo"], summaries["noctua"])
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    write_report(output_dir, args, summaries["grpo"], summaries["noctua"])

    print(json.dumps({"output_dir": str(output_dir), "comparison": comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

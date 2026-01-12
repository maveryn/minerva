#!/usr/bin/env python3
"""Orchestrate the ACRD 3-stage pipeline with configurable commands."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StagePaths:
    rlvr_dir: Path
    acr_dir: Path
    sft_dir: Path


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _latest_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    latest_file = root / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        try:
            step = int(latest_file.read_text(encoding="utf-8").strip())
            candidate = root / f"global_step_{step}"
            if candidate.exists():
                return candidate
        except ValueError:
            pass
    candidates = sorted(root.glob("global_step_*"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.name.split("global_step_")[-1]))


def _render_cmd(template: List[str], mapping: Dict[str, Any]) -> List[str]:
    safe = _SafeDict(mapping)
    return [str(item).format_map(safe) for item in template]


def _run(cmd: List[str], *, dry_run: bool = False) -> None:
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _stage_paths(output_root: Path, cycle: int) -> StagePaths:
    return StagePaths(
        rlvr_dir=output_root / f"rlvr_cycle{cycle}",
        acr_dir=output_root / f"acr_cycle{cycle}",
        sft_dir=output_root / f"sft_cycle{cycle}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ACRD pipeline with staged commands.")
    parser.add_argument("--config", required=True, help="Pipeline YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    pipeline = cfg.get("pipeline", {}) or {}
    commands = cfg.get("commands", {}) or {}

    mode = str(pipeline.get("mode", "cycle"))
    num_cycles = int(pipeline.get("num_cycles", 1))
    rlvr_steps = int(pipeline.get("rlvr_steps_per_cycle", 0) or 0)
    acr_steps = int(pipeline.get("acr_steps_per_cycle", 0) or 0)
    sft_steps = int(pipeline.get("sft_steps_per_cycle", 0) or 0)
    output_root = Path(pipeline.get("output_root", "outputs/acrd"))
    sft_buffer = pipeline.get("sft_buffer_path", "outputs/acr_sft_buffer.jsonl")
    sft_parquet = pipeline.get("sft_parquet_path", "outputs/acr_sft.parquet")
    collect_each_cycle = bool(pipeline.get("collect_traces_every_cycle", True))
    clear_buffer = bool(pipeline.get("clear_buffer_each_cycle", True))

    output_root.mkdir(parents=True, exist_ok=True)

    if not commands:
        raise ValueError("commands section is required in the pipeline config.")

    prev_sft_ckpt: Optional[Path] = None
    cycles = range(1, num_cycles + 1) if mode == "cycle" else [1]

    last_cycle = cycles[-1]
    for cycle in cycles:
        stage_dirs = _stage_paths(output_root, cycle)
        stage_dirs.rlvr_dir.mkdir(parents=True, exist_ok=True)
        stage_dirs.acr_dir.mkdir(parents=True, exist_ok=True)
        stage_dirs.sft_dir.mkdir(parents=True, exist_ok=True)

        mapping = {
            "cycle": cycle,
            "rlvr_dir": str(stage_dirs.rlvr_dir),
            "acr_dir": str(stage_dirs.acr_dir),
            "sft_dir": str(stage_dirs.sft_dir),
            "sft_buffer": str(sft_buffer),
            "sft_parquet": str(sft_parquet),
            "output_root": str(output_root),
        }

        if clear_buffer and Path(sft_buffer).exists():
            Path(sft_buffer).unlink()

        # RLVR stage
        rlvr_cmd = commands.get("rlvr")
        if rlvr_cmd:
            mapping.update(
                {
                    "steps": rlvr_steps,
                    "default_local_dir": str(stage_dirs.rlvr_dir),
                    "resume_path": str(prev_sft_ckpt) if prev_sft_ckpt else "",
                    "resume_mode": "resume_path" if prev_sft_ckpt else "disable",
                }
            )
            _run(_render_cmd(list(rlvr_cmd), mapping), dry_run=args.dry_run)

        rlvr_ckpt = _latest_checkpoint(stage_dirs.rlvr_dir)
        if rlvr_ckpt is None:
            raise RuntimeError(f"No RLVR checkpoint found in {stage_dirs.rlvr_dir}")

        # ACR stage
        acr_cmd = commands.get("acr")
        if acr_cmd:
            mapping.update(
                {
                    "steps": acr_steps,
                    "default_local_dir": str(stage_dirs.acr_dir),
                    "resume_path": str(rlvr_ckpt),
                    "resume_mode": "resume_path",
                }
            )
            _run(_render_cmd(list(acr_cmd), mapping), dry_run=args.dry_run)

        acr_ckpt = _latest_checkpoint(stage_dirs.acr_dir)
        if acr_ckpt is None:
            raise RuntimeError(f"No ACR checkpoint found in {stage_dirs.acr_dir}")

        # Collect traces
        if collect_each_cycle or cycle == last_cycle:
            collect_cmd = commands.get("collect")
            if collect_cmd:
                mapping.update({"ckpt": str(acr_ckpt)})
                _run(_render_cmd(list(collect_cmd), mapping), dry_run=args.dry_run)

        # Build SFT dataset
        build_cmd = commands.get("build_sft")
        if build_cmd:
            _run(_render_cmd(list(build_cmd), mapping), dry_run=args.dry_run)

        # SFT stage
        sft_cmd = commands.get("sft")
        if sft_cmd:
            mapping.update(
                {
                    "steps": sft_steps,
                    "default_local_dir": str(stage_dirs.sft_dir),
                    "resume_path": str(acr_ckpt),
                    "resume_mode": "resume_path",
                }
            )
            _run(_render_cmd(list(sft_cmd), mapping), dry_run=args.dry_run)

        prev_sft_ckpt = _latest_checkpoint(stage_dirs.sft_dir)

    print("ACRD pipeline complete.")


if __name__ == "__main__":
    main()

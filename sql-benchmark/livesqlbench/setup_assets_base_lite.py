#!/usr/bin/env python3
"""Prepare local LiveSQLBench assets for Base-Lite."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SQL_BENCHMARK_ROOT = ROOT / "sql-benchmark"
SELF_DIR = ROOT / "sql-benchmark" / "livesqlbench"
ARTIFACTS_DIR = SELF_DIR / "artifacts"
OFFICIAL_DIR = SELF_DIR / "official"

PUBLIC_DATASET_DIR = SQL_BENCHMARK_ROOT / "livesqlbench-base-lite"
PUBLIC_JSONL = PUBLIC_DATASET_DIR / "livesqlbench_data.jsonl"
GT_JSONL = SQL_BENCHMARK_ROOT / "livesqlbench_gt_kg_testcases_0528.jsonl"
DUMPS_ZIP = SQL_BENCHMARK_ROOT / "livesqlbench-base-lite-dumps.zip"

INTEGRATED_JSONL = ARTIFACTS_DIR / "livesqlbench_base_lite_full.jsonl"
UNPACK_DIR = ARTIFACTS_DIR / "postgre_table_dumps_base_lite"

OFFICIAL_FILES = {
    "baseline/src/prompt.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/baseline/src/prompt.py",
    "baseline/src/prompt_generator.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/baseline/src/prompt_generator.py",
    "baseline/src/post_process.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/baseline/src/post_process.py",
    "evaluation/src/db_config.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/db_config.py",
    "evaluation/src/db_utils.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/db_utils.py",
    "evaluation/src/evaluation.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/evaluation.py",
    "evaluation/src/logger.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/logger.py",
    "evaluation/src/test_utils.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/test_utils.py",
    "evaluation/src/utils.py": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/src/utils.py",
    "evaluation/run/run_eval.sh": "https://raw.githubusercontent.com/bird-bench/livesqlbench/main/evaluation/run/run_eval.sh",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def integrate_dataset(force: bool) -> None:
    if not PUBLIC_JSONL.exists():
        raise FileNotFoundError(f"Missing public dataset JSONL: {PUBLIC_JSONL}")
    if not GT_JSONL.exists():
        raise FileNotFoundError(f"Missing GT JSONL: {GT_JSONL}")

    if INTEGRATED_JSONL.exists() and not force:
        print(f"Integrated dataset already exists: {INTEGRATED_JSONL}")
        return

    public_rows = load_jsonl(PUBLIC_JSONL)
    gt_rows = load_jsonl(GT_JSONL)
    gt_lookup = {row["instance_id"]: row for row in gt_rows}

    merged_rows = []
    for row in public_rows:
        item = dict(row)
        gt = gt_lookup.get(item["instance_id"])
        if gt is not None:
            for field in ("sol_sql", "test_cases", "external_knowledge"):
                if field in gt:
                    item[field] = gt[field]

        if "clean_up_sql" not in item and "clean_up_sqls" in item:
            item["clean_up_sql"] = item.get("clean_up_sqls") or []
        elif "clean_up_sql" not in item:
            item["clean_up_sql"] = []

        if "preprocess_sql" not in item:
            item["preprocess_sql"] = []
        if "sol_sql" not in item:
            item["sol_sql"] = []
        if "test_cases" not in item:
            item["test_cases"] = []
        if "external_knowledge" not in item:
            item["external_knowledge"] = []

        merged_rows.append(item)

    write_jsonl(merged_rows, INTEGRATED_JSONL)
    print(f"Wrote integrated dataset: {INTEGRATED_JSONL}")
    print(f"Rows: {len(merged_rows)}")


def download_official_files(force: bool) -> None:
    for rel_path, url in OFFICIAL_FILES.items():
        target = OFFICIAL_DIR / rel_path
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp, target.open("wb") as out:
            shutil.copyfileobj(resp, out)
        print(f"Downloaded {rel_path}")


def unpack_dumps(force: bool) -> None:
    if not DUMPS_ZIP.exists():
        raise FileNotFoundError(f"Missing dump zip: {DUMPS_ZIP}")

    extracted_root = UNPACK_DIR / "livesqlbench-base-lite-dumps"
    if extracted_root.exists() and not force:
        print(f"Dumps already unpacked: {extracted_root}")
        return

    if force and UNPACK_DIR.exists():
        shutil.rmtree(UNPACK_DIR)
    UNPACK_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DUMPS_ZIP) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.startswith("__MACOSX/"):
                continue
            zf.extract(member, UNPACK_DIR)

    print(f"Unpacked dumps to: {extracted_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local LiveSQLBench Base-Lite assets.")
    parser.add_argument("--force", action="store_true", help="Overwrite integrated files and re-download/re-unpack assets.")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)

    integrate_dataset(force=args.force)
    download_official_files(force=args.force)
    unpack_dumps(force=args.force)


if __name__ == "__main__":
    main()

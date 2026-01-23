"""
Prepare a SecEval mini validation dataset in VeRL Parquet format.

Usage:
  python seceval.py --input dataset/seceval/seceval.jsonl --out-dir rlvr/mydata/seceval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from prompt import ATHENABENCH_SYSTEM_PROMPT


def sanitize_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    text = ILLEGAL_CHARACTERS_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_messages(problem: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": sanitize_text(system_prompt)},
        {"role": "user", "content": sanitize_text(problem)},
    ]


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def write_parquet(rows: List[Dict[str, Any]], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def write_excel(df: pd.DataFrame, out_path: Path, *, system_prompt: str) -> None:
    df_out = df.copy()
    df_out["system_prompt"] = sanitize_text(system_prompt)
    df_out["user_prompt"] = df_out["problem"].map(sanitize_text)
    df_out["data_source"] = df_out.get("data_source", "")
    for column in df_out.columns:
        if pd.api.types.is_object_dtype(df_out[column]):
            df_out[column] = df_out[column].map(sanitize_text)
    df_out.to_excel(out_path, index=False)


def rows_to_verl(df: pd.DataFrame, *, data_source: str, split: str, source_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    extra_cols = [c for c in df.columns if c not in {"problem", "answer"}]

    for idx, record in df.reset_index(drop=True).iterrows():
        messages = build_messages(record["problem"], ATHENABENCH_SYSTEM_PROMPT)
        extra_info: Dict[str, Any] = {
            "split": split,
            "index": int(idx),
            "source_file": source_file,
            "answer_full": sanitize_text(record["answer"]),
            "system_prompt": sanitize_text(ATHENABENCH_SYSTEM_PROMPT),
        }
        for col in extra_cols:
            value = record[col]
            if isinstance(value, (list, dict)):
                extra_info[col] = value
            else:
                extra_info[col] = sanitize_text(value)

        rows.append(
            {
                "data_source": data_source,
                "source_file": source_file,
                "prompt": messages,
                "ability": "cti",
                "reward_model": {"style": "rule", "ground_truth": sanitize_text(record["answer"])},
                "extra_info": extra_info,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a SecEval mini dataset for validation.")
    parser.add_argument(
        "--input",
        default="dataset/seceval/seceval.jsonl",
        help="Path to the SecEval JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default="rlvr/mydata/seceval",
        help="Output directory for parquet/xlsx.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=400,
        help="Number of rows to sample for seceval-mini.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--name",
        default="seceval_mini",
        help="Output dataset name (used for parquet/xlsx filenames).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(input_path)
    if df.empty:
        raise ValueError(f"No records found in {input_path}")

    if "answer" not in df.columns and "correct_answer" in df.columns:
        df["answer"] = df["correct_answer"]
    if "prompt" not in df.columns or "answer" not in df.columns:
        missing = {"prompt", "answer"} - set(df.columns)
        raise ValueError(f"{input_path} missing required columns: {', '.join(sorted(missing))}")

    df = df.rename(columns={"prompt": "problem"})
    df["problem"] = df["problem"].map(sanitize_text)
    df["answer"] = df["answer"].map(sanitize_text)

    sample_size = int(args.sample_size)
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=int(args.seed)).reset_index(drop=True)

    df["data_source"] = "seceval-mini"

    rows = rows_to_verl(
        df,
        data_source="seceval-mini",
        split="mini",
        source_file=input_path.name,
    )

    parquet_path = out_dir / f"{args.name}.parquet"
    excel_path = out_dir / f"{args.name}.xlsx"
    write_parquet(rows, parquet_path)
    write_excel(df, excel_path, system_prompt=ATHENABENCH_SYSTEM_PROMPT)

    print(f"{args.name}: {len(df)} rows -> {parquet_path}, {excel_path}")


if __name__ == "__main__":
    main()

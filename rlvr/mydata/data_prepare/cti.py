"""
Convert local cyber threat intelligence JSONL datasets into VeRL-style Parquet
and Excel files.

Usage:
    python cti.py --out_dir ./out/cti

Optional:
    --data_dir path/to/jsonl/dir  # defaults to ../cti relative to this file

Dependencies: pandas, pyarrow, openpyxl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from prompt import ATHENABENCH_SYSTEM_PROMPT, CTI_SYSTEM_PROMPT


@dataclass(frozen=True)
class DatasetSpec:
    name: str          # e.g. "athena_cti_ate_test"
    source_path: Path  # e.g. Path(.../"ate-test.jsonl")
    data_source: str   # e.g. "athena-cti-ate" or "use_reward_fn" for minerva files
    split: str         # e.g. "test"
    system_prompt: str


def derive_dataset_spec(path: Path) -> DatasetSpec:
    """Build a dataset specification by inferring metadata from the file name."""
    stem = path.stem
    prefix, sep, split = stem.rpartition("-")

    if not sep:  # No hyphen -> treat the whole stem as the dataset name.
        prefix = stem
        split = "full"
    elif not prefix:  # Hyphen at the start -> fall back to the full stem.
        prefix = stem
        split = "full"

    safe_prefix = prefix.replace("-", "_")
    safe_split = split.replace("-", "_")

    # For minerva-* files we use reward_fn as the data_source per-row.
    if stem.startswith("minerva-"):
        data_source = "use_reward_fn"
        name = f"{safe_prefix}_{safe_split}"
        system_prompt = CTI_SYSTEM_PROMPT
    else:
        data_source = stem  # use dataset file name for athena-* files
        name = f"{safe_prefix}_{safe_split}"
        system_prompt = ATHENABENCH_SYSTEM_PROMPT

    return DatasetSpec(name=name, source_path=path, data_source=data_source, split=safe_split, system_prompt=system_prompt)


def discover_dataset_specs(data_dir: Path) -> List[DatasetSpec]:
    """Find all JSONL files in the data directory and derive dataset specs for each."""
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    return [derive_dataset_spec(path) for path in jsonl_files]


def sanitize_text(value) -> str:
    """Remove characters Excel cannot store and normalise newlines."""
    if value is None:
        return ""
    text = str(value)
    text = ILLEGAL_CHARACTERS_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_messages(problem: str, system_prompt: str) -> List[Dict[str, str]]:
    """Create (system, user) message list for VeRL format."""
    return [
        {"role": "system", "content": sanitize_text(system_prompt)},
        {"role": "user", "content": sanitize_text(problem)},
    ]


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a dataframe, renaming prompt->problem."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return pd.DataFrame(columns=["problem", "answer"])

    df = pd.DataFrame(records)
    if "prompt" not in df.columns or "answer" not in df.columns:
        missing = {"prompt", "answer"} - set(df.columns)
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")

    df = df.rename(columns={"prompt": "problem"})
    df["problem"] = df["problem"].map(sanitize_text)
    df["answer"] = df["answer"].map(sanitize_text)
    return df


def add_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Add ground_truth column as the sanitized answer (no shortening)."""
    return df.assign(ground_truth=df["answer"].map(sanitize_text))


def rows_to_verl(df: pd.DataFrame, *, spec: DatasetSpec) -> List[Dict]:
    """Transform dataframe rows into VeRL schema dictionaries."""
    rows: List[Dict] = []
    extra_cols = [c for c in df.columns if c not in {"problem", "answer", "ground_truth"}]

    for idx, record in df.reset_index(drop=True).iterrows():
        messages = build_messages(record["problem"], spec.system_prompt)
        ds_value = spec.data_source if spec.data_source != "use_reward_fn" else sanitize_text(record.get("reward_fn", ""))
        extra_info: Dict = {
            "split": spec.split,
            "index": int(idx),
            "source_file": spec.source_path.name,
            "answer_full": sanitize_text(record["answer"]),
        }
        for col in extra_cols:
            value = record[col]
            if isinstance(value, (list, dict)):
                extra_info[col] = value
            else:
                extra_info[col] = sanitize_text(value)

        rows.append(
            {
                "data_source": ds_value,
                "source_file": spec.source_path.name,
                "prompt": messages,
                "ability": "cti",
                "reward_model": {"style": "rule", "ground_truth": record["ground_truth"]},
                "extra_info": extra_info,
            }
        )
    return rows


def write_parquet(rows: List[Dict], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def write_excel(df: pd.DataFrame, out_path: Path) -> None:
    """Write a user-facing Excel file with problem/answer/ground_truth columns."""
    df_out = df.copy()
    if "system_prompt" in df_out.columns:
        df_out["system_prompt"] = df_out["system_prompt"].map(sanitize_text)
    else:
        df_out["system_prompt"] = sanitize_text(CTI_SYSTEM_PROMPT)
    df_out["user_prompt"] = df_out["problem"].map(sanitize_text)
    # Preserve data_source for visibility
    if "data_source" in df_out.columns:
        df_out["data_source"] = df_out["data_source"].map(sanitize_text)
    else:
        df_out["data_source"] = ""

    # Ensure all object columns are sanitised for Excel safety.
    for column in df_out.columns:
        if pd.api.types.is_object_dtype(df_out[column]):
            df_out[column] = df_out[column].map(sanitize_text)

    df_out.to_excel(out_path, index=False)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "cti-in"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CTI datasets in Parquet and Excel formats.")
    parser.add_argument("--out_dir", required=True, help="Directory for the generated files.")
    parser.add_argument(
        "--data_dir",
        default=str(default_data_dir()),
        help="Directory containing source JSONL files (defaults to ../cti).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = discover_dataset_specs(data_dir)

    for spec in specs:
        if not spec.source_path.is_file():
            raise FileNotFoundError(f"Expected source file not found: {spec.source_path}")

        df = load_jsonl(spec.source_path)
        if spec.source_path.name == "minerva-dev.jsonl" and len(df) > 2000:
            df = df.sample(n=2000, random_state=1337).reset_index(drop=True)
        df = add_ground_truth(df)
        df["system_prompt"] = sanitize_text(spec.system_prompt)
        if spec.data_source == "use_reward_fn":
            df["data_source"] = df.get("reward_fn", "").map(sanitize_text)
        else:
            df["data_source"] = sanitize_text(spec.data_source)

        parquet_path = out_dir / f"{spec.name}.parquet"
        excel_path = out_dir / f"{spec.name}.xlsx"

        verl_rows = rows_to_verl(df, spec=spec)
        write_parquet(verl_rows, parquet_path)
        write_excel(df, excel_path)

        print(f"{spec.name}: {len(df)} rows -> {parquet_path.name}, {excel_path.name}")


if __name__ == "__main__":
    main()

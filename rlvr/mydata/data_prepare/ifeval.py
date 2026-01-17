"""
Convert Instruction-Following Eval (IFEval) JSONL to VeRL-style Parquet + Excel.

Usage:
  python rlvr/mydata/data_prepare/ifeval.py \
    --input_data /path/to/input_data.jsonl \
    --out_dir rlvr/mydata/ifeval
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

from prompt import INSTRUCTION_FOLLOWING_SYSTEM_PROMPT


DATA_SOURCE = "reward_instruction_following"
ABILITY = "instruction_following"
DEFAULT_SPLIT = "dev"
DEFAULT_TASK = "instruction_following_eval"
DEFAULT_MAX_SAMPLES = 300
DEFAULT_SEED = 1337


def sanitize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = ILLEGAL_CHARACTERS_RE.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def to_excel_cell(value: Any) -> str:
    if isinstance(value, (list, dict)):
        try:
            return sanitize_text(json.dumps(value, ensure_ascii=False))
        except Exception:
            return sanitize_text(value)
    return sanitize_text(value)


def build_messages(prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": sanitize_text(system_prompt)},
        {"role": "user", "content": sanitize_text(prompt)},
    ]


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "key": obj.get("key"),
                    "prompt": sanitize_text(obj.get("prompt", "")),
                    "instruction_id_list": obj.get("instruction_id_list", []),
                    "kwargs": obj.get("kwargs", []),
                }
            )
    return pd.DataFrame(records)


def rows_to_verl(
    df: pd.DataFrame,
    *,
    system_prompt: str,
    split_name: str,
    data_source: str,
    ability: str,
    source_file: str,
) -> List[Dict]:
    rows: List[Dict] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        prompt = row.get("prompt", "")
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])
        messages = build_messages(prompt, system_prompt)
        ground_truth = {
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs_list,
            "key": row.get("key"),
        }
        extra_info = {
            "split": split_name,
            "index": int(idx),
            "source_file": source_file,
            "task": DEFAULT_TASK,
            "key": row.get("key"),
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs_list,
        }
        rows.append(
            {
                "data_source": data_source,
                "source_file": source_file,
                "prompt": messages,
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": extra_info,
            }
        )
    return rows


def write_parquet(rows: List[Dict], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)


def write_excel(df: pd.DataFrame, out_path: Path, system_prompt: str) -> None:
    df_out = df.copy()
    df_out["system_prompt"] = sanitize_text(system_prompt)
    df_out["user_prompt"] = df_out["prompt"].map(sanitize_text)
    df_out["data_source"] = DATA_SOURCE
    for col in df_out.columns:
        if pd.api.types.is_object_dtype(df_out[col]):
            df_out[col] = df_out[col].map(to_excel_cell)
    df_out.to_excel(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IFEval dataset in Parquet + Excel.")
    parser.add_argument("--input_data", required=True, help="Path to IFEval input_data.jsonl.")
    parser.add_argument("--out_dir", required=True, help="Output directory for Parquet/Excel.")
    parser.add_argument("--system_prompt", default=INSTRUCTION_FOLLOWING_SYSTEM_PROMPT, help="System prompt text.")
    parser.add_argument("--split_name", default=DEFAULT_SPLIT, help="Split label stored in extra_info.")
    parser.add_argument("--data_source", default=DATA_SOURCE, help="data_source field for VeRL rows.")
    parser.add_argument("--ability", default=ABILITY, help="ability field for VeRL rows.")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Max rows to keep.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling.")
    args = parser.parse_args()

    input_path = Path(args.input_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_jsonl(input_path)
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    rows = rows_to_verl(
        df,
        system_prompt=args.system_prompt,
        split_name=args.split_name,
        data_source=args.data_source,
        ability=args.ability,
        source_file=input_path.name,
    )

    parquet_path = out_dir / "ifeval_dev.parquet"
    excel_path = out_dir / "ifeval_dev.xlsx"
    write_parquet(rows, parquet_path)
    write_excel(df, excel_path, args.system_prompt)

    print(f"ifeval_dev: {len(df)} rows -> {parquet_path.name}, {excel_path.name}")


if __name__ == "__main__":
    main()

# math500_to_verl.py
# Usage:
#   python math500_to_verl.py --out_dir ./out/math500 --seed 1337
# Options:
#   --val_n 100                # how many items for validation (default 100)
#   --instruction_prefix ""    # optional text prepended to the USER message
# Requires:
#   pip install datasets pandas pyarrow openpyxl

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from prompt import DEFAULT_SYSTEM_PROMPT  # your OpenR1-style system prompt

HF_DATASET = "HuggingFaceH4/MATH-500"
HF_SPLIT   = "test"   # single split with 500 rows on HF

# ---------- helpers ----------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def to_excel_cell(v: Any) -> str:
    """Readable Excel cell (JSON for lists/dicts; sanitized text otherwise)."""
    if isinstance(v, (list, dict)):
        try:
            return sanitize_text(json.dumps(v, ensure_ascii=False))
        except Exception:
            return sanitize_text(str(v))
    return sanitize_text(v)

def build_messages(problem_text: str, system_prompt: str, instr_prefix: str) -> List[Dict]:
    sys = {"role": "system", "content": sanitize_text(system_prompt)}
    user_content = sanitize_text(problem_text)
    if instr_prefix:
        user_content = f"{instr_prefix.strip()}\n\n{user_content}"
    usr = {"role": "user", "content": user_content}
    return [sys, usr]

def rows_to_verl(
    df: pd.DataFrame,
    *,
    data_source: str,
    split_name: str,
    problem_col: str,
    answer_col: str,
    system_prompt: str,
    instruction_prefix: str,
    hf_dataset: str,
    hf_split: str,
) -> List[Dict]:
    """Convert a dataframe to VeRL-style rows with OpenR1 (system+user) messages."""
    rows: List[Dict] = []
    extras_cols = [c for c in df.columns if c not in (problem_col, answer_col)]
    for idx, r in df.reset_index(drop=True).iterrows():
        messages = build_messages(r[problem_col], system_prompt, instruction_prefix)
        ground_truth = sanitize_text(r[answer_col])

        extra_info: Dict = {
            "split": split_name,
            "index": int(idx),
            "hf_dataset": hf_dataset,
            "hf_split": hf_split,
        }
        for c in extras_cols:
            v = r[c]
            if isinstance(v, (list, dict)) or v is None:
                extra_info[c] = v
            else:
                extra_info[c] = sanitize_text(v)

        rows.append({
            "data_source": data_source,
            "prompt": messages,  # system + user messages
            "ability": "reason",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": extra_info,
        })
    return rows

def write_parquet(rows: List[Dict], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

def write_excel_keep_all(
    df: pd.DataFrame,
    out_path: Path,
    system_prompt: str,
    instruction_prefix: str,
    problem_col: str,
    answer_col: str,
) -> None:
    """Keep all original columns and add system_prompt, user_prompt, ground_truth."""
    df2 = df.copy()
    df2["system_prompt"] = sanitize_text(system_prompt)
    df2["user_prompt"] = df2[problem_col].map(
        lambda s: sanitize_text(f"{instruction_prefix.strip()}\n\n{s}" if instruction_prefix else s)
    )
    df2["ground_truth"] = df2[answer_col].map(to_excel_cell)
    for c in df2.columns:
        col = df2[c]
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
            df2[c] = col.map(to_excel_cell)
    df2.to_excel(out_path, index=False)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--val_n", type=int, default=100, help="Validation size (random sample from the single split)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    ap.add_argument("--instruction_prefix", default="", help="Optional prefix for the USER message")
    ap.add_argument("--data_source", default="math_500", help="data_source field for VeRL rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HF dataset: single 'test' split (500 rows) with columns problem/solution/answer/subject/level/unique_id
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    df = ds.to_pandas()
    for col in ("problem", "answer"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in {HF_DATASET}.")

    # Sample 100 for validation (deterministic), and use ALL rows for test
    idxs = list(range(len(df)))
    random.Random(args.seed).shuffle(idxs)
    val_n  = min(args.val_n, len(df))
    val_df = df.iloc[idxs[:val_n]].reset_index(drop=True)
    test_df = df.copy().reset_index(drop=True)   # includes validation items

    # Build VeRL rows
    val_rows = rows_to_verl(
        val_df, data_source=args.data_source, split_name="validation",
        problem_col="problem", answer_col="answer",
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )
    test_rows = rows_to_verl(
        test_df, data_source=args.data_source, split_name="test",
        problem_col="problem", answer_col="answer",
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )

    # Paths
    val_pq   = out_dir / "math500_validation.parquet"
    val_xlsx = out_dir / "math500_validation.xlsx"
    test_pq  = out_dir / "math500_test.parquet"
    test_xlsx= out_dir / "math500_test.xlsx"

    # Write files
    write_parquet(val_rows,  val_pq)
    write_parquet(test_rows, test_pq)
    write_excel_keep_all(val_df,  val_xlsx,  DEFAULT_SYSTEM_PROMPT, args.instruction_prefix, "problem", "answer")
    write_excel_keep_all(test_df, test_xlsx, DEFAULT_SYSTEM_PROMPT, args.instruction_prefix, "problem", "answer")

    print(f"MATH-500 rows total: {len(df)}  (validation {len(val_df)}; test {len(test_df)} includes all)")
    print(f"Wrote:\n  - {val_pq}\n  - {test_pq}\n  - {val_xlsx}\n  - {test_xlsx}")
    print(f"Saved under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

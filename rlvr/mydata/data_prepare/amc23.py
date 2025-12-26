# amc23_to_verl.py
# Usage:
#   python amc23_to_verl.py --out_dir ./out/amc23 --instruction_prefix ""
# Requires:
#   pip install datasets pandas pyarrow openpyxl

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from prompt import DEFAULT_SYSTEM_PROMPT  # <-- your system prompt source

HF_DATASET = "knoveleng/AMC-23"
HF_SPLIT   = "train"  # dataset exposes a single 'train' split (~40 rows)

# ---------- helpers ----------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def pick_problem_col(columns) -> str:
    # Prefer "problem"; fall back to "question"
    if "problem" in columns:
        return "problem"
    if "question" in columns:
        return "question"
    raise KeyError("Neither 'problem' nor 'question' column found in AMC-23 dataset.")

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
            extra_info[c] = v if isinstance(v, (list, dict)) else sanitize_text(v)

        rows.append({
            "data_source": data_source,
            "prompt": messages,                     # <-- system + user messages
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
    problem_col: str,
    answer_col: str,
    system_prompt: str,
    instruction_prefix: str,
) -> None:
    """Keep all original columns and add system_prompt, user_prompt, ground_truth."""
    df2 = df.copy()
    df2["system_prompt"] = sanitize_text(system_prompt)
    df2["user_prompt"] = df2[problem_col].map(
        lambda s: sanitize_text(f"{instruction_prefix.strip()}\n\n{s}" if instruction_prefix else s)
    )
    df2["ground_truth"] = df2[answer_col].map(sanitize_text)
    # sanitize object/string cols for Excel safety
    for c in df2.columns:
        try:
            if pd.api.types.is_string_dtype(df2[c]) or df2[c].dtype == object:
                df2[c] = df2[c].map(sanitize_text)
        except Exception:
            pass
    df2.to_excel(out_path, index=False)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--instruction_prefix", default="", help="Optional prefix prepended to the USER message")
    ap.add_argument("--data_source", default="amc_23", help="data_source field for VeRL rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HF dataset (train split, 40 rows)
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    df = ds.to_pandas()

    # Identify columns
    problem_col = pick_problem_col(df.columns)
    answer_col  = "answer"
    if answer_col not in df.columns:
        raise KeyError("Expected 'answer' column not found in AMC-23 dataset.")

    # Build VeRL rows (two splits with identical content)
    val_rows = rows_to_verl(
        df, data_source=args.data_source, split_name="validation",
        problem_col=problem_col, answer_col=answer_col,
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )
    test_rows = rows_to_verl(
        df, data_source=args.data_source, split_name="test",
        problem_col=problem_col, answer_col=answer_col,
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )

    # Paths
    val_pq   = out_dir / "amc23_validation.parquet"
    val_xlsx = out_dir / "amc23_validation.xlsx"
    test_pq  = out_dir / "amc23_test.parquet"
    test_xlsx= out_dir / "amc23_test.xlsx"

    # Write Parquet
    write_parquet(val_rows,  val_pq)
    write_parquet(test_rows, test_pq)

    # Write Excel (keeps all original cols + system/user/ground_truth)
    write_excel_keep_all(df, val_xlsx,  problem_col, answer_col, DEFAULT_SYSTEM_PROMPT, args.instruction_prefix)
    write_excel_keep_all(df, test_xlsx, problem_col, answer_col, DEFAULT_SYSTEM_PROMPT, args.instruction_prefix)

    print(f"AMC-23 rows: {len(df)}")
    print(f"Wrote:\n  - {val_pq}\n  - {test_pq}\n  - {val_xlsx}\n  - {test_xlsx}")
    print(f"Saved under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

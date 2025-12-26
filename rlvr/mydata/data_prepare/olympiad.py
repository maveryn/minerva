# olympiadbench_to_verl.py
# Usage:
#   python olympiadbench_to_verl.py --out_dir ./out/olympiadbench --val_n 50 --seed 1337
# Options:
#   --instruction_prefix ""   # optional text prepended to the USER message
#   --disjoint_test           # if set, removes the 50 validation items from test export
# Requires:
#   pip install datasets pandas pyarrow openpyxl

from __future__ import annotations
import argparse, random, json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from prompt import DEFAULT_SYSTEM_PROMPT  # OpenR1-style system message

HF_DATASET = "knoveleng/OlympiadBench"
HF_SPLIT   = "train"   # single split; ~675 rows on HF

# ---------------- helpers ----------------
def sanitize_text(x) -> str:
    """Strip Excel-illegal control chars and normalize newlines."""
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

def build_messages(question_text: str, system_prompt: str, instr_prefix: str) -> List[Dict]:
    sys = {"role": "system", "content": sanitize_text(system_prompt)}
    user_content = sanitize_text(question_text)
    if instr_prefix:
        user_content = f"{instr_prefix.strip()}\n\n{user_content}"
    usr = {"role": "user", "content": user_content}
    return [sys, usr]

def pick_question_col(columns) -> str:
    # Prefer 'question'; fall back to 'problem'
    if "question" in columns:
        return "question"
    if "problem" in columns:
        return "problem"
    raise KeyError("Neither 'question' nor 'problem' column found in OlympiadBench.")

def pick_answer_col(columns) -> str:
    # Prefer 'answer'; fall back to 'final_answer'
    if "answer" in columns:
        return "answer"
    if "final_answer" in columns:
        return "final_answer"
    raise KeyError("Neither 'answer' nor 'final_answer' column found in OlympiadBench.")

def to_ground_truth_text(v: Any) -> str:
    """Ensure reward_model.ground_truth is a string."""
    if isinstance(v, (list, dict)):
        try:
            return sanitize_text(json.dumps(v, ensure_ascii=False))
        except Exception:
            return sanitize_text(str(v))
    return sanitize_text(v)

def rows_to_verl(
    df: pd.DataFrame,
    *,
    data_source: str,
    split_name: str,
    question_col: str,
    answer_col: str,
    system_prompt: str,
    instruction_prefix: str,
    hf_dataset: str,
    hf_split: str,
) -> List[Dict]:
    """Convert a dataframe to VeRL-style rows with OpenR1 (system+user) messages."""
    rows: List[Dict] = []
    extras_cols = [c for c in df.columns if c not in (question_col, answer_col)]
    for idx, r in df.reset_index(drop=True).iterrows():
        messages = build_messages(r[question_col], system_prompt, instruction_prefix)
        ground_truth = to_ground_truth_text(r[answer_col])

        extra_info: Dict = {
            "split": split_name,
            "index": int(idx),
            "hf_dataset": hf_dataset,
            "hf_split": hf_split,
        }
        for c in extras_cols:
            v = r[c]
            # Keep lists/dicts as-is in extra_info for Parquet; sanitize strings
            if isinstance(v, (list, dict)) or v is None:
                extra_info[c] = v
            else:
                extra_info[c] = sanitize_text(v)

        rows.append({
            "data_source": data_source,
            "prompt": messages,  # <-- system + user messages
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
    question_col: str,
    answer_col: str,
) -> None:
    """Keep all original columns and add system_prompt, user_prompt, ground_truth."""
    df2 = df.copy()
    df2["system_prompt"] = sanitize_text(system_prompt)
    df2["user_prompt"] = df2[question_col].map(
        lambda s: sanitize_text(f"{instruction_prefix.strip()}\n\n{s}" if instruction_prefix else s)
    )
    df2["ground_truth"] = df2[answer_col].map(to_excel_cell)
    # Sanitize/JSON-encode other object-like columns for readability in Excel
    for c in df2.columns:
        col = df2[c]
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
            df2[c] = col.map(to_excel_cell)
    df2.to_excel(out_path, index=False)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--val_n", type=int, default=50, help="Validation size (sampled from the single split)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for sampling")
    ap.add_argument("--instruction_prefix", default="", help="Optional prefix for the USER message")
    ap.add_argument("--data_source", default="olympiadbench", help="data_source field for VeRL rows")
    ap.add_argument("--disjoint_test", action="store_true",
                    help="If set, exclude validation items from the test export")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HF dataset (train split ~675 rows). HF page confirms row count & recent schema. :contentReference[oaicite:1]{index=1}
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)
    df = ds.to_pandas()

    question_col = pick_question_col(df.columns)
    answer_col   = pick_answer_col(df.columns)

    # Deterministic sample for validation
    idxs = list(range(len(df)))
    random.Random(args.seed).shuffle(idxs)
    val_n = min(args.val_n, len(df))
    val_idx = set(idxs[:val_n])

    val_df  = df.iloc[list(val_idx)].reset_index(drop=True)
    test_df = df.copy() if not args.disjoint_test else df.drop(index=list(val_idx)).reset_index(drop=True)

    # Paths
    val_pq   = out_dir / "olympiadbench_validation.parquet"
    val_xlsx = out_dir / "olympiadbench_validation.xlsx"
    test_pq  = out_dir / "olympiadbench_test.parquet"
    test_xlsx= out_dir / "olympiadbench_test.xlsx"

    # Build & write Parquet (VeRL-style, OpenR1 messages)
    val_rows = rows_to_verl(
        val_df, data_source=args.data_source, split_name="validation",
        question_col=question_col, answer_col=answer_col,
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )
    test_rows = rows_to_verl(
        test_df, data_source=args.data_source, split_name="test",
        question_col=question_col, answer_col=answer_col,
        system_prompt=DEFAULT_SYSTEM_PROMPT, instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_split=HF_SPLIT,
    )
    write_parquet(val_rows,  val_pq)
    write_parquet(test_rows, test_pq)

    # Excel (keep all original cols + system/user/ground_truth)
    write_excel_keep_all(val_df,  val_xlsx,  DEFAULT_SYSTEM_PROMPT, args.instruction_prefix,
                         question_col=question_col, answer_col=answer_col)
    write_excel_keep_all(test_df, test_xlsx, DEFAULT_SYSTEM_PROMPT, args.instruction_prefix,
                         question_col=question_col, answer_col=answer_col)

    # Summary
    print(f"Saved to: {out_dir.resolve()}")
    print(f"HF dataset/split: {HF_DATASET} / {HF_SPLIT}")
    print(f"Total rows: {len(df)}  (validation {len(val_df)}; test {len(test_df)} "
          f"{'disjoint' if args.disjoint_test else 'includes all'})")
    print(f"Wrote:\n  - {val_pq}\n  - {test_pq}\n  - {val_xlsx}\n  - {test_xlsx}")

if __name__ == "__main__":
    main()

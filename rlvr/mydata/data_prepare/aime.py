# aime_to_verl.py
# Usage:
#   python aime_to_verl.py --out_dir ./out/aime
# Options:
#   --system_prompt "..."            # override system message (string)
#   --system_prompt_path path.txt    # or load system message from file
#   --instruction_prefix ""          # optional extra to prepend inside the *user* message
# Requires: pip install datasets pandas pyarrow openpyxl

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from prompt import DEFAULT_SYSTEM_PROMPT


# ---------- utilities ----------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def build_messages(problem: str, system_prompt: str, user_prefix: str) -> List[Dict]:
    sys = {"role": "system", "content": sanitize_text(system_prompt)}
    user_content = sanitize_text(problem)
    if user_prefix:
        user_content = f"{user_prefix.strip()}\n\n{user_content}"
    usr = {"role": "user", "content": user_content}
    return [sys, usr]

def rows_to_verl(df: pd.DataFrame, *, data_source: str, split_name: str,
                 problem_col: str, answer_col: str,
                 system_prompt: str, instruction_prefix: str,
                 hf_dataset: str, hf_split: str) -> List[Dict]:
    """Convert a dataframe to VeRL-style rows with Openr1 (system+user) messages."""
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
            "prompt": messages,                   # <-- system + user messages
            "ability": "reason",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": extra_info,
        })
    return rows

def write_parquet(rows: List[Dict], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

def write_excel_keep_all(df: pd.DataFrame, out_path: Path,
                         problem_col: str, answer_col: str,
                         system_prompt: str, instruction_prefix: str) -> None:
    """Keep all original columns; add system_prompt, user_prompt, ground_truth."""
    df2 = df.copy()
    # Create per-row user prompt (problem with optional instruction prefix)
    df2["system_prompt"] = sanitize_text(system_prompt)
    df2["user_prompt"] = df2[problem_col].map(lambda s: sanitize_text(
        f"{instruction_prefix.strip()}\n\n{s}" if instruction_prefix else s
    ))
    df2["ground_truth"] = df2[answer_col].map(sanitize_text)
    # sanitize all string-like/object columns for Excel safety
    for c in df2.columns:
        try:
            if pd.api.types.is_string_dtype(df2[c]) or df2[c].dtype == object:
                df2[c] = df2[c].map(sanitize_text)
        except Exception:
            pass
    df2.to_excel(out_path, index=False)

# ---------- converters ----------
def load_aime24() -> pd.DataFrame:
    # HuggingFaceH4/aime_2024 → split 'train' (30 rows; id, problem, solution, answer, url, year)
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    return ds.to_pandas()

def load_aime25() -> pd.DataFrame:
    # math-ai/aime25 → split 'test' (30 rows; problem, answer, id)
    ds = load_dataset("math-ai/aime25", split="test")
    return ds.to_pandas()

def convert_one(df: pd.DataFrame, *, out_dir: Path, fname_stub: str,
                data_source: str, split_name: str, hf_dataset: str, hf_split: str,
                problem_col: str = "problem", answer_col: str = "answer",
                system_prompt: str = DEFAULT_SYSTEM_PROMPT, instruction_prefix: str = "") -> Dict[str, Path]:
    out_parquet = out_dir / f"{fname_stub}.parquet"
    out_excel   = out_dir / f"{fname_stub}.xlsx"

    rows = rows_to_verl(
        df, data_source=data_source, split_name=split_name,
        problem_col=problem_col, answer_col=answer_col,
        system_prompt=system_prompt, instruction_prefix=instruction_prefix,
        hf_dataset=hf_dataset, hf_split=hf_split,
    )
    write_parquet(rows, out_parquet)
    write_excel_keep_all(df, out_excel, problem_col, answer_col, system_prompt, instruction_prefix)
    return {"parquet": out_parquet, "excel": out_excel}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    # If both are provided, --system_prompt_path wins.
    ap.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
                    help="System message to use (Openr1 style by default)")
    ap.add_argument("--system_prompt_path", default="",
                    help="Path to a file containing the system message")
    ap.add_argument("--instruction_prefix", default="",
                    help="Optional extra text prepended to the *user* message")
    args = ap.parse_args()

    # Load/override system prompt if a file was given
    system_prompt = args.system_prompt
    if args.system_prompt_path:
        system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both datasets
    df24 = load_aime24()
    df25 = load_aime25()

    # Per-dataset conversions (no splitting)
    paths24 = convert_one(
        df24, out_dir=out_dir, fname_stub="aime_2024",
        data_source="aime_2024", split_name="train",
        hf_dataset="HuggingFaceH4/aime_2024", hf_split="train",
        problem_col="problem", answer_col="answer",
        system_prompt=system_prompt, instruction_prefix=args.instruction_prefix,
    )
    print(f"[AIME-2024] rows={len(df24)} → {paths24['parquet'].name}, {paths24['excel'].name}")

    paths25 = convert_one(
        df25, out_dir=out_dir, fname_stub="aime_2025",
        data_source="aime_2025", split_name="test",
        hf_dataset="math-ai/aime25", hf_split="test",
        problem_col="problem", answer_col="answer",
        system_prompt=system_prompt, instruction_prefix=args.instruction_prefix,
    )
    print(f"[AIME-2025] rows={len(df25)} → {paths25['parquet'].name}, {paths25['excel'].name}")

    # Combined validation = all 60 (keep union of columns in Excel; mark dataset)
    df24_c = df24.copy(); df24_c["dataset_name"] = "aime_2024"
    df25_c = df25.copy(); df25_c["dataset_name"] = "aime_2025"
    df_val = pd.concat([df24_c, df25_c], axis=0, ignore_index=True)

    # Excel (keep union of columns)
    val_excel = out_dir / "aime_validation.xlsx"
    write_excel_keep_all(df_val, val_excel, problem_col="problem", answer_col="answer",
                         system_prompt=system_prompt, instruction_prefix=args.instruction_prefix)

    # Parquet (VeRL-style) — set split_name='validation', with Openr1-style messages
    rows24 = rows_to_verl(
        df24, data_source="aime_validation", split_name="validation",
        problem_col="problem", answer_col="answer",
        system_prompt=system_prompt, instruction_prefix=args.instruction_prefix,
        hf_dataset="HuggingFaceH4/aime_2024", hf_split="train",
    )
    rows25 = rows_to_verl(
        df25, data_source="aime_validation", split_name="validation",
        problem_col="problem", answer_col="answer",
        system_prompt=system_prompt, instruction_prefix=args.instruction_prefix,
        hf_dataset="math-ai/aime25", hf_split="test",
    )
    val_parquet = out_dir / "aime_validation.parquet"
    write_parquet(rows24 + rows25, val_parquet)

    print(f"[AIME-validation] rows={len(df_val)} → {val_parquet.name}, {val_excel.name}")
    print(f"Done. Files saved under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

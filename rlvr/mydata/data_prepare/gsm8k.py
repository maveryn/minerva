# gsm8k_to_verl.py
# Usage:
#   python gsm8k_to_verl.py --out_dir ./out/gsm8k --seed 1337
# Optional:
#   --config main            # or 'socratic'
#   --train_n 3000 --test_n 400 --val_n 200
#   --instruction_prefix ""  # if empty, defaults to PROMPT_STR (from prompt.py)
#
# Requires: pip install datasets pandas pyarrow openpyxl

from __future__ import annotations
import argparse, random, re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from prompt import PROMPT_STR  # your existing prompt string

HF_DATASET = "openai/gsm8k"

# ---------------- helpers ----------------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

# Four (or more) hashes then the final answer on that line.
_FINAL_RE = re.compile(r"#{4,}\s*([^\n]+)")

_BOXED_RE = re.compile(r"\\boxed\\{([^}]+)\\}")  # in case someone uses \boxed{...}

def _clean_final_token(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # remove TeX-y wrappers that might slip in
    m = _BOXED_RE.fullmatch(s) or _BOXED_RE.search(s)
    if m:
        s = m.group(1).strip()
    s = s.replace("$", "").strip()
    # drop trailing period if the rest looks numeric-like
    if s.endswith(".") and re.fullmatch(r"-?\d[\d,]*\.?", s):
        s = s[:-1]
    # remove thousands separators
    if re.fullmatch(r"-?\d[\d,]*", s):
        s = s.replace(",", "")
    return s.strip()

def extract_gsm8k_final(answer_field: str) -> str:
    """Extract only the final answer after #### from GSM8K 'answer'."""
    if not isinstance(answer_field, str):
        return ""
    m = list(_FINAL_RE.finditer(answer_field))
    if m:
        final = m[-1].group(1)
    else:
        # Fallback: try last non-empty line
        lines = [ln.strip() for ln in answer_field.strip().splitlines() if ln.strip()]
        final = lines[-1] if lines else ""
    return _clean_final_token(final)

def build_prompt(question: str, instr_prefix: str) -> str:
    q = sanitize_text(question)
    return f"{instr_prefix.strip()}\n\n{q}" if instr_prefix else q

def rows_to_verl(df: pd.DataFrame, *, data_source: str, split_name: str,
                 instruction_prefix: str, hf_dataset: str, hf_config: str, hf_split: str,
                 question_col: str = "question", answer_col: str = "answer") -> List[Dict]:
    rows: List[Dict] = []
    extra_cols = [c for c in df.columns if c not in (question_col, answer_col)]
    for idx, r in df.reset_index(drop=True).iterrows():
        prompt_msg = [{"role": "user", "content": build_prompt(r[question_col], instruction_prefix)}]
        final_ans  = extract_gsm8k_final(r[answer_col])  # <-- only final answer
        extra_info: Dict = {
            "split": split_name,
            "index": int(idx),
            "hf_dataset": hf_dataset,
            "hf_config": hf_config,
            "hf_split": hf_split,
            "answer_full": sanitize_text(r[answer_col]),  # full CoT for provenance
        }
        for c in extra_cols:
            v = r[c]
            extra_info[c] = v if isinstance(v, (list, dict)) else sanitize_text(v)

        rows.append({
            "data_source": data_source,
            "prompt": prompt_msg,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_ans},
            "extra_info": extra_info,
        })
    return rows

def write_parquet(rows: List[Dict], out_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

def write_excel_minimal(df: pd.DataFrame, out_path: Path,
                        instruction_prefix: str,
                        question_col: str = "question", answer_col: str = "answer") -> None:
    df2 = pd.DataFrame({
        "prompt": df[question_col].map(lambda s: build_prompt(s, instruction_prefix)),
        "ground_truth": df[answer_col].map(extract_gsm8k_final),  # <-- final only
    })
    df2.to_excel(out_path, index=False)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--config", default="main", help="HF config name: 'main' or 'socratic'")
    ap.add_argument("--train_n", type=int, default=3000, help="Train size (from HF train)")
    ap.add_argument("--test_n",  type=int, default=400,  help="Test size (from HF test)")
    ap.add_argument("--val_n",   type=int, default=200,  help="Validation size (from remaining HF test)")
    ap.add_argument("--seed",    type=int, default=1337, help="Random seed")
    ap.add_argument("--instruction_prefix", default="", help="Optional prefix; defaults to PROMPT_STR if empty")
    ap.add_argument("--data_source", default="gsm8k", help="data_source field for VeRL rows")
    args = ap.parse_args()

    if not args.instruction_prefix:
        args.instruction_prefix = PROMPT_STR

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load HF splits
    ds_tr = load_dataset(HF_DATASET, name=args.config, split="train")        # 7473 in 'main'
    df_tr = ds_tr.to_pandas()

    # Public set: prefer 'test'; fall back to 'validation' if needed
    base_split_name = "test"
    try:
        ds_pub = load_dataset(HF_DATASET, name=args.config, split="test")
    except Exception:
        base_split_name = "validation"
        ds_pub = load_dataset(HF_DATASET, name=args.config, split="validation")
    df_pub = ds_pub.to_pandas()

    # ----- sample train (3k from train) -----
    rng = random.Random(args.seed)
    tr_idxs = list(range(len(df_tr))); rng.shuffle(tr_idxs)
    train_df = df_tr.iloc[tr_idxs[: min(args.train_n, len(df_tr))]].reset_index(drop=True)

    # ----- sample test + validation from the same public split (disjoint) -----
    pub_idxs = list(range(len(df_pub))); rng.shuffle(pub_idxs)
    test_take = min(args.test_n, len(df_pub))
    test_idx  = pub_idxs[:test_take]
    remaining = pub_idxs[test_take:]
    val_take  = min(args.val_n, len(remaining))
    val_idx   = remaining[:val_take]

    test_df = df_pub.iloc[test_idx].reset_index(drop=True)
    val_df  = df_pub.iloc[val_idx].reset_index(drop=True)

    assert set(test_idx).isdisjoint(set(val_idx))

    # ----- build VeRL rows -----
    train_rows = rows_to_verl(
        train_df, data_source=args.data_source, split_name="train",
        instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_config=args.config, hf_split="train",
    )
    test_rows = rows_to_verl(
        test_df, data_source=args.data_source, split_name="test",
        instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_config=args.config, hf_split=base_split_name,
    )
    val_rows = rows_to_verl(
        val_df, data_source=args.data_source, split_name="validation",
        instruction_prefix=args.instruction_prefix,
        hf_dataset=HF_DATASET, hf_config=args.config, hf_split=base_split_name,
    )

    # ----- write Parquet -----
    train_pq = out_dir / "gsm8k_train.parquet"
    test_pq  = out_dir / "gsm8k_test.parquet"
    val_pq   = out_dir / "gsm8k_validation.parquet"
    write_parquet(train_rows, train_pq)
    write_parquet(test_rows,  test_pq)
    write_parquet(val_rows,   val_pq)

    # ----- write Excel (prompt + final-only ground_truth) -----
    train_xlsx = out_dir / "gsm8k_train.xlsx"
    test_xlsx  = out_dir / "gsm8k_test.xlsx"
    val_xlsx   = out_dir / "gsm8k_validation.xlsx"
    write_excel_minimal(train_df, train_xlsx, args.instruction_prefix)
    write_excel_minimal(test_df,  test_xlsx,  args.instruction_prefix)
    write_excel_minimal(val_df,   val_xlsx,   args.instruction_prefix)

    # Summary
    print(f"Saved to: {out_dir.resolve()}")
    print(f"GSM8K config: {args.config}")
    print(f"Train rows: {len(train_df)} (from HF train)")
    print(f"Test rows:  {len(test_df)}  (from HF {base_split_name})")
    print(f"Val rows:   {len(val_df)}   (from remaining HF {base_split_name})")
    print("Parquet:", train_pq.name, test_pq.name, val_pq.name)
    print("Excel  :", train_xlsx.name, test_xlsx.name, val_xlsx.name)

if __name__ == "__main__":
    main()

# omni_math_to_verl.py
# Usage:
#   python omni_math_to_verl.py --out_dir .\mydata\omnimath --train_n 2000 --val_n 100 --test_n 400 --seed 1337
# Requires: pip install datasets pandas pyarrow openpyxl

import re
import random
from typing import List, Dict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE  # Excel safety
from prompt import PROMPT_STR

# -------------------- Config (CLI overrides most) --------------------
DEFAULT_DATASET   = "KbsdJames/Omni-MATH"
DEFAULT_HF_SPLIT  = "test"
DEFAULT_TRAIN_N   = 3000
DEFAULT_VAL_N     = 100
DEFAULT_TEST_N    = 200
DEFAULT_SEED      = 1337
DEFAULT_DATA_SRC  = "omni_math"

# -------------------- Excel sanitizer --------------------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

# -------------------- LaTeX/light markup stripper --------------------
_TEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\s*\*)?")  # \text, \boxed, \frac, ...

def strip_tex_like(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("$", " ")
    s = _TEX_CMD_RE.sub(" ", s)
    s = s.replace("{", " ").replace("}", " ")
    s = " ".join(s.split())
    return s

# -------------------- Heuristics --------------------
_MCQ_ANSWER_RE  = re.compile(r"^\s*[\(\[]?\s*[A-Ja-j]\s*[\)\].]?\s*$")
_YESNO_RE       = re.compile(r"^\s*(yes|no|true|false|y|n)\s*$", re.IGNORECASE)
_OPT_PAREN_RE   = re.compile(r"\([A-E]\)")
_OPT_LINED_RE   = re.compile(r"(?m)^[A-Ea-e][\)\.]\s")
_OPT_NUM_LINED  = re.compile(r"(?m)^[1-5][\)\.]\s")
_TEXTUAL_REJECT_REASON_RE = re.compile(r"[A-Za-z]")  # has letters?
_HAS_DIGIT_RE             = re.compile(r"\d")
_HAS_MATH_OP_RE           = re.compile(r"[=+\-*/^_(){}\[\],.;:|<>]")

def looks_like_mcq_by_problem(problem: str) -> bool:
    if not isinstance(problem, str):
        return False
    hits = 0
    hits += len(_OPT_PAREN_RE.findall(problem))
    hits += len(_OPT_LINED_RE.findall(problem))
    hits += len(_OPT_NUM_LINED.findall(problem))
    return hits >= 3

def looks_like_yesno(ans: str) -> bool:
    raw = strip_tex_like(ans)
    return bool(_YESNO_RE.match(raw))

def looks_like_mcq_by_answer(ans: str) -> bool:
    raw = strip_tex_like(ans)
    return bool(_MCQ_ANSWER_RE.match(raw))

def looks_like_textual_answer(ans: str) -> bool:
    raw = strip_tex_like(ans)
    has_letters = bool(_TEXTUAL_REJECT_REASON_RE.search(raw))
    has_digit   = bool(_HAS_DIGIT_RE.search(raw))
    has_op      = bool(_HAS_MATH_OP_RE.search(raw))
    # letters but neither digits nor math operators -> textual statement like "All rectangles", "\text{Yes}"
    return has_letters and not (has_digit or has_op)

def keep_item(problem: str, answer: str) -> bool:
    if answer is None or str(answer).strip() == "":
        return False
    if looks_like_mcq_by_problem(str(problem)):
        return False
    if looks_like_mcq_by_answer(str(answer)):
        return False
    if looks_like_yesno(str(answer)):
        return False
    if looks_like_textual_answer(str(answer)):
        return False
    return True

# -------------------- Builders --------------------
def build_prompt_text(problem: str, instr_prefix: str) -> str:
    core = strip_tex_like(problem)  # keep prompt readable; drop stray TeX commands
    return f"{instr_prefix.strip()}\n\n{core}" if instr_prefix else core

def rows_to_verl(records: List[Dict], split_name: str, *, data_source: str,
                 instruction_prefix: str, hf_dataset: str, hf_split: str) -> List[Dict]:
    rows = []
    for idx, r in enumerate(records):
        problem    = r.get("problem", "")
        answer     = r.get("answer", "")
        solution   = r.get("solution", "")
        domain     = r.get("domain", "")
        difficulty = r.get("difficulty", None)
        source     = r.get("source", "")
        row_uid    = r.get("row_uid", idx)

        prompt = [{"role": "user", "content": build_prompt_text(problem, instruction_prefix)}]
        extra_info = {
            "split": split_name, "index": idx,
            "domain": sanitize_text(domain),
            "difficulty": difficulty,
            "solution": sanitize_text(solution),
            "source": sanitize_text(source),
            "hf_dataset": hf_dataset, "hf_split": hf_split,
            "hf_row_uid": int(row_uid),
        }

        rows.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": sanitize_text(answer)},
            "extra_info": extra_info,
        })
    return rows

def write_parquet(rows: List[Dict], out_path: Path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)

def write_excel_minimal(records: List[Dict], out_path: Path, instruction_prefix: str):
    df = pd.DataFrame({
        "prompt": [sanitize_text(build_prompt_text(r.get("problem", ""), instruction_prefix)) for r in records],
        "ground_truth": [sanitize_text(r.get("answer", "")) for r in records],
    })
    df.to_excel(out_path, index=False)

# -------------------- Main --------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset path")
    ap.add_argument("--hf_split", default=DEFAULT_HF_SPLIT, help="HF split to load")
    ap.add_argument("--train_n", type=int, default=DEFAULT_TRAIN_N, help="Number of training samples")
    ap.add_argument("--val_n",   type=int, default=DEFAULT_VAL_N,   help="Number of validation samples")
    ap.add_argument("--test_n",  type=int, default=DEFAULT_TEST_N,  help="Number of test samples")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed")
    ap.add_argument("--out_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--instruction_prefix", default="", help="Optional instruction prefix for prompts")
    ap.add_argument("--data_source", default=DEFAULT_DATA_SRC, help="data_source field for VeRL rows")
    ap.add_argument("--dump_filtered_all", action="store_true", help="Also save the full filtered pool before sampling")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instruction_prefix is None or args.instruction_prefix == "":
        args.instruction_prefix = PROMPT_STR

    # Output paths
    train_xlsx = out_dir / "omni_math_train.xlsx"
    val_xlsx   = out_dir / "omni_math_val.xlsx"
    test_xlsx  = out_dir / "omni_math_test.xlsx"
    train_pq   = out_dir / "omni_math_train.parquet"
    val_pq     = out_dir / "omni_math_val.parquet"
    test_pq    = out_dir / "omni_math_test.parquet"
    filtered_pq= out_dir / "omni_math_filtered_all.parquet"

    # Load HF dataset
    ds = load_dataset(args.dataset, split=args.hf_split)
    df = ds.to_pandas()

    # Build boolean masks per reason (for quick audit)
    m_empty   = df["answer"].isna() | (df["answer"].astype(str).str.strip() == "")
    m_mcqprob = df["problem"].astype(str).apply(looks_like_mcq_by_problem)
    m_mcqans  = df["answer"].astype(str).apply(looks_like_mcq_by_answer)
    m_yesno   = df["answer"].astype(str).apply(looks_like_yesno)
    m_textual = df["answer"].astype(str).apply(looks_like_textual_answer)

    keep_mask = ~(m_empty | m_mcqprob | m_mcqans | m_yesno | m_textual)
    filt = df[keep_mask].reset_index(drop=True)

    # Shuffle and attach a unique row id for dedup checks
    rng = random.Random(args.seed)
    idxs = list(range(len(filt)))
    rng.shuffle(idxs)
    filt = filt.iloc[idxs].reset_index(drop=True)
    filt["row_uid"] = range(len(filt))

    # Sample sizes (no overlap)
    n_avail = len(filt)
    train_n = min(args.train_n, n_avail)
    rem1    = n_avail - train_n
    val_n   = min(args.val_n, max(0, rem1))
    rem2    = rem1 - val_n
    test_n  = min(args.test_n, max(0, rem2))

    train_df = filt.iloc[:train_n].copy()
    val_df   = filt.iloc[train_n:train_n+val_n].copy()
    test_df  = filt.iloc[train_n+val_n:train_n+val_n+test_n].copy()

    # Optional: dump the full filtered pool (pre-sampling)
    if args.dump_filtered_all:
        all_rows = rows_to_verl(
            filt.to_dict("records"),
            split_name="filtered_all",
            data_source=args.data_source,
            instruction_prefix=args.instruction_prefix,
            hf_dataset=args.dataset, hf_split=args.hf_split,
        )
        write_parquet(all_rows, filtered_pq)

    # Build VeRL rows
    train_rows = rows_to_verl(
        train_df.to_dict("records"),
        split_name="train",
        data_source=args.data_source,
        instruction_prefix=args.instruction_prefix,
        hf_dataset=args.dataset, hf_split=args.hf_split,
    )
    val_rows = rows_to_verl(
        val_df.to_dict("records"),
        split_name="val",
        data_source=args.data_source,
        instruction_prefix=args.instruction_prefix,
        hf_dataset=args.dataset, hf_split=args.hf_split,
    )
    test_rows = rows_to_verl(
        test_df.to_dict("records"),
        split_name="test",
        data_source=args.data_source,
        instruction_prefix=args.instruction_prefix,
        hf_dataset=args.dataset, hf_split=args.hf_split,
    )

    # Write outputs
    write_parquet(train_rows, train_pq)
    write_parquet(val_rows,   val_pq)
    write_parquet(test_rows,  test_pq)
    write_excel_minimal(train_df.to_dict("records"), train_xlsx, args.instruction_prefix)
    write_excel_minimal(val_df.to_dict("records"),   val_xlsx,   args.instruction_prefix)
    write_excel_minimal(test_df.to_dict("records"),  test_xlsx,  args.instruction_prefix)

    # Sanity: ensure no overlap across splits
    set_train = set(train_df["row_uid"])
    set_val   = set(val_df["row_uid"])
    set_test  = set(test_df["row_uid"])
    assert set_train.isdisjoint(set_val)
    assert set_train.isdisjoint(set_test)
    assert set_val.isdisjoint(set_test)

    # Summary
    print(f"Out dir:          {out_dir.resolve()}")
    print(f"HF dataset/split: {args.dataset} / {args.hf_split}")
    print(f"Total HF rows:    {len(df)}")
    print(f"Dropped empty:    {int(m_empty.sum())}")
    print(f"Dropped MCQ prob: {int(m_mcqprob.sum())}")
    print(f"Dropped MCQ ans : {int(m_mcqans.sum())}")
    print(f"Dropped yes/no  : {int(m_yesno.sum())}")
    print(f"Dropped textual : {int(m_textual.sum())}")
    print(f"Filtered kept:    {len(filt)}")
    print(f"Split sizes (train/val/test): {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print("Wrote:")
    print(f"  - {train_pq}")
    print(f"  - {val_pq}")
    print(f"  - {test_pq}")
    print(f"  - {train_xlsx}")
    print(f"  - {val_xlsx}")
    print(f"  - {test_xlsx}")
    if args.dump_filtered_all:
        print(f"  - {filtered_pq}")

if __name__ == "__main__":
    main()

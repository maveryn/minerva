# openr1_mirror_parquet.py
# Usage examples:
#   python openr1_mirror_parquet.py --out_dir ./openr1
#   python openr1_mirror_parquet.py --out_dir ./openr1 --inplace false --patched_subdir patched
#   python openr1_mirror_parquet.py --out_dir ./openr1 --sample_xlsx openr1_samples.xlsx
#
# Requires:
#   pip install -U huggingface_hub pyarrow pandas openpyxl

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

HF_DATASET = "Elliott/Openr1-Math-46k-8192"
REVISION = "refs/convert/parquet"  # viewer conversion branch with parquet files


# ---------- helpers ----------
def sanitize_text(x) -> str:
    s = "" if x is None else str(x)
    s = ILLEGAL_CHARACTERS_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def load_system_prompt(module_name: str, attr_name: str) -> str:
    mod = importlib.import_module(module_name)
    try:
        value = getattr(mod, attr_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' does not define '{attr_name}'."
        ) from e
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{module_name}.{attr_name} must be a non-empty string.")
    return value.strip()


def as_json_str(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return sanitize_text(str(obj))


def parse_if_json_str(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    """
    Ensure we have a list of {role, content} dicts.
    Accepts list[dict] or a JSON string; otherwise returns [].
    """
    messages = parse_if_json_str(messages)
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user"))
        content = "" if m.get("content") is None else str(m.get("content"))
        out.append({"role": role, "content": content})
    return out


def patch_messages(messages: List[Dict[str, Any]], new_sys_prompt: str) -> List[Dict[str, str]]:
    """
    Ensure a 'system' message exists at the top with the new content; otherwise prepend it.
    Keep all other messages as-is.
    """
    msgs = normalize_messages(messages)
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = new_sys_prompt
        return msgs
    return [{"role": "system", "content": new_sys_prompt}] + msgs


def extract_user_text(messages: List[Dict[str, str]]) -> str:
    """Concatenate all USER turns with blank lines, for Excel inspection."""
    users = [m["content"] for m in messages if m.get("role") == "user"]
    return "\n\n".join(sanitize_text(u) for u in users)


def extract_ground_truth(rm_field: Any) -> str:
    """
    reward_model may be a dict or a JSON string; return ground_truth if present.
    """
    rm = parse_if_json_str(rm_field)
    if isinstance(rm, dict):
        gt = rm.get("ground_truth", "")
        return sanitize_text(gt)
    return ""


def patch_parquet_file(
    in_path: Path,
    out_path: Path,
    system_prompt: str,
    prompt_key: str = "prompt",
) -> Tuple[int, int]:
    """
    Patch a single Parquet file, rewriting/ensuring system prompt in each row.
    Returns (#rows, #patched_rows).
    """
    table = pq.read_table(in_path)
    rows: List[Dict[str, Any]] = table.to_pylist()

    changed = 0
    for r in rows:
        if prompt_key in r:
            r[prompt_key] = patch_messages(r[prompt_key], system_prompt)
            changed += 1

    out_table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path)
    return len(rows), changed


def collect_sample_rows(parquet_paths: List[Path], limit: int) -> List[Dict[str, Any]]:
    """
    Deterministically collect the first `limit` rows by file name order, without randomness.
    """
    sample_rows: List[Dict[str, Any]] = []
    for p in parquet_paths:
        if len(sample_rows) >= limit:
            break
        tbl = pq.read_table(p)
        rows = tbl.to_pylist()
        remaining = limit - len(sample_rows)
        sample_rows.extend(rows[:remaining])
    return sample_rows[:limit]


def build_excel_frame(rows: List[Dict[str, Any]], system_prompt: str) -> pd.DataFrame:
    """
    Create a DataFrame with helpful columns:
      - system_prompt (the new one you set)
      - user_prompt   (concatenated USER turns)
      - ground_truth  (from reward_model.ground_truth, if present)
      - plus original top-level fields, with nested fields JSON-stringified
    """
    excel_rows: List[Dict[str, Any]] = []
    for r in rows:
        prompt = normalize_messages(r.get("prompt"))
        # Existing system text may be long; we show the *new* one for quick checks.
        sys_text = sanitize_text(system_prompt)
        user_text = extract_user_text(prompt)
        gt = extract_ground_truth(r.get("reward_model"))

        flat: Dict[str, Any] = {}
        # Keep original top-level fields, stringifying nested ones for visibility.
        for k, v in r.items():
            if k in ("prompt", "reward_model", "extra_info"):
                flat[k] = as_json_str(v)
            else:
                flat[k] = v if isinstance(v, (int, float)) else sanitize_text(v)

        flat["system_prompt"] = sys_text
        flat["user_prompt"] = user_text
        flat["ground_truth"] = gt
        excel_rows.append(flat)

    df = pd.DataFrame(excel_rows)

    # Final sanitation for any object/string columns
    for c in df.columns:
        try:
            if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
                df[c] = df[c].map(sanitize_text)
        except Exception:
            pass
    return df


def write_sample_excel(parquet_root: Path, system_prompt: str, out_xlsx: Path, limit: int = 100) -> None:
    """
    Gather up to `limit` rows from patched parquet files under `parquet_root` and write an Excel file.
    """
    parquets = sorted(parquet_root.rglob("*.parquet"))
    if not parquets:
        print(f"[excel] No parquet files under: {parquet_root}")
        return

    sample_rows = collect_sample_rows(parquets, limit=limit)
    if not sample_rows:
        print("[excel] No rows collected for Excel.")
        return

    df = build_excel_frame(sample_rows, system_prompt=system_prompt)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"[excel] Wrote {len(df)} rows -> {out_xlsx}")


def mirror_and_patch(
    out_dir: Path,
    system_prompt_module: str,
    system_prompt_attr: str,
    inplace: bool,
    patched_subdir: str,
    sample_xlsx: str,
    sample_count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mirror Parquet only (no symlinks) from the viewer conversion branch
    print(f"Mirroring {HF_DATASET} (revision={REVISION}) into: {out_dir.resolve()}")
    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["**/*.parquet"],
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print("Download complete.")

    # 2) Load the new system prompt (e.g., from prompt.DEFAULT_SYSTEM_PROMPT)
    sys_prompt = load_system_prompt(system_prompt_module, system_prompt_attr)
    print(f"Loaded system prompt from {system_prompt_module}.{system_prompt_attr} (length={len(sys_prompt)})")

    # 3) Patch each parquet's prompt[0] system message content (or prepend if missing)
    parquet_files = sorted(out_dir.rglob("*.parquet"))
    if not parquet_files:
        print("No parquet files found — nothing to patch.")
        return

    if inplace:
        print("Patching in-place...")
        total_rows = total_patched = 0
        for p in parquet_files:
            n_rows, n_patched = patch_parquet_file(p, p, sys_prompt)
            total_rows += n_rows
            total_patched += n_patched
        print(f"Patched {total_patched}/{total_rows} rows across {len(parquet_files)} files.")
        source_root = out_dir
    else:
        target_root = out_dir / patched_subdir
        print(f"Patching into: {target_root.resolve()}")
        total_rows = total_patched = 0
        for p in parquet_files:
            rel = p.relative_to(out_dir)
            out_p = target_root / rel
            n_rows, n_patched = patch_parquet_file(p, out_p, sys_prompt)
            total_rows += n_rows
            total_patched += n_patched
        print(f"Patched {total_patched}/{total_rows} rows across {len(parquet_files)} files.")
        source_root = target_root

    # 4) Export a deterministic 100-row sample to Excel for verification
    if sample_xlsx:
        out_xlsx = out_dir / sample_xlsx
        write_sample_excel(source_root, sys_prompt, out_xlsx, limit=sample_count)

    print("Done.")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="openr1", help="Local folder to mirror Parquet into")
    ap.add_argument("--system_prompt_module", default="prompt",
                    help="Python module that defines DEFAULT_SYSTEM_PROMPT (e.g., 'prompt')")
    ap.add_argument("--system_prompt_attr", default="DEFAULT_SYSTEM_PROMPT",
                    help="Attribute name containing the system prompt string")
    ap.add_argument("--inplace", default="true",
                    help="Patch parquet files in-place (true|false). If false, writes to --patched_subdir under out_dir.")
    ap.add_argument("--patched_subdir", default="patched",
                    help="Subdirectory (under out_dir) to write patched parquet when --inplace=false")
    ap.add_argument("--sample_xlsx", default="openr1_samples.xlsx",
                    help="Excel file (under out_dir) to write 100 patched samples for verification, empty to skip")
    ap.add_argument("--sample_count", type=int, default=100,
                    help="How many rows to include in the sample Excel (default 100)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    inplace = str(args.inplace).lower() in ("1", "true", "yes", "y")

    mirror_and_patch(
        out_dir=out_dir,
        system_prompt_module=args.system_prompt_module,
        system_prompt_attr=args.system_prompt_attr,
        inplace=inplace,
        patched_subdir=args.patched_subdir,
        sample_xlsx=args.sample_xlsx.strip(),
        sample_count=int(args.sample_count),
    )


if __name__ == "__main__":
    main()

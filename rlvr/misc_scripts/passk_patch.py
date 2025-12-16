#!/usr/bin/env python3
# passk_patch.py
#
# De-duplicate *_predictions_passk.jsonl by (sample_idx, k) keeping the LAST occurrence,
# robustly re-extract pred_ids / pred_answer, refresh *match flags, and rebuild:
#   *_metrics_per_k.jsonl, *_metrics_summary.json, *_passk_curve.png, *_passk_curve.pdf
#
# It also prints a verification report: original vs post-dedup lines, final K (max k),
# samples-per-k (with outliers), and total unique (k, sample_idx) records.
#
# Usage:
#   python passk_patch.py /path/to/<prefix>_predictions_passk.jsonl
#
# Options:
#   --no-backup       : do not write <file>.bak
#   --preserve-order  : keep original order of the kept (last) records instead of sorting by (k, sample_idx)

import os
import re
import json
import argparse
import shutil
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

# -------- robust extraction regexes ----------
_CLIP_CHARS = 1200
ANS_BRACED_RE = re.compile(r"\\answer\{\s*([+-]?\d+)\s*\}", re.IGNORECASE)
INT_RE        = re.compile(r"[+-]?\d+")
# Strict: prefer a line that *ends* with id/ids{<digits,...>} (backslash optional, 's' optional)
IDS_BRACED_DIGITS_RE = re.compile(r"(?mi)\\?ids?\{\s*(\d+(?:\s*,\s*\d+)*)\s*\}\s*$")
# Fallback: any occurrence of id/ids{...} anywhere (backslash optional, 's' optional)
IDS_BRACED_ANY_RE    = re.compile(r"\\?ids?\{\s*([^}]*)\s*\}", re.IGNORECASE | re.DOTALL)
# For ground truth parsing (simple and reliable)
IDS_GT_RE            = re.compile(r"\\ids\{\s*([^}]*)\s*\}", re.IGNORECASE | re.DOTALL)
ID_INT_RE            = re.compile(r"\d+")
# Backup: last bracketed int list like [1,2,3] or {1, 2, 3} (commas or spaces), across lines
INT_LIST_FALLBACK_RE = re.compile(
    r"[\[\{]\s*(?:\d+(?:\s*[, ]\s*\d+)*)\s*[\]\}]",
    re.DOTALL
)

def _clip_tail(s: str, k: int = _CLIP_CHARS) -> str:
    return s[-k:] if isinstance(s, str) and len(s) > k else (s or "")

def extract_ids_from_text(text: str) -> Optional[List[int]]:
    """Prefer the last strict line match; otherwise fallback to id/ids anywhere; finally try a raw [...] or {...} int list."""
    if not text:
        return None
    strict = IDS_BRACED_DIGITS_RE.findall(text)
    if strict:
        raw = strict[-1]
        return [int(x) for x in ID_INT_RE.findall(raw)]
    anym = IDS_BRACED_ANY_RE.findall(text)
    if anym:
        raw = anym[-1]
        digits = [int(x) for x in ID_INT_RE.findall(raw)]
        if digits:
            return digits
    # NEW: fallback to the last raw bracket/brace int list
    lists = INT_LIST_FALLBACK_RE.findall(text)
    if lists:
        # prefer the last one with >=2 ints to avoid picking code like [1]
        for raw in reversed(lists):
            nums = [int(x) for x in ID_INT_RE.findall(raw)]
            if len(nums) >= 2:
                return nums
        # else accept the very last with >=1 int
        nums = [int(x) for x in ID_INT_RE.findall(lists[-1])]
        return nums if nums else None
    return None

def extract_answer_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = ANS_BRACED_RE.search(text)
    if m:
        return int(m.group(1))
    # tail fallback
    nums = INT_RE.findall(_clip_tail(text))
    return int(nums[-1]) if nums else None

def extract_gt_ids(gt: str) -> Optional[List[int]]:
    m = IDS_GT_RE.search(gt or "")
    if not m:
        return None
    return [int(x) for x in ID_INT_RE.findall(m.group(1))]

def extract_gt_answer(gt: str) -> Optional[int]:
    m = ANS_BRACED_RE.search(gt or "")
    return int(m.group(1)) if m else None

def compute_answer_match(sol: str, gt: str) -> bool:
    gt_m = ANS_BRACED_RE.search(gt or "")
    if not gt_m:
        return False
    gt_ans = int(gt_m.group(1))
    pred = extract_answer_int(sol)
    return (pred is not None) and (pred == gt_ans)

def compute_ids_match(sol: str, gt: str) -> Optional[bool]:
    gt_ids = extract_gt_ids(gt)
    if gt_ids is None:
        return None
    pred_ids = extract_ids_from_text(sol)
    if pred_ids is None:
        return False
    return pred_ids == gt_ids

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def derive_prefix_paths(pred_path: str) -> Tuple[str, str, str, str, str]:
    """
    From ".../<prefix>_predictions_passk.jsonl" return:
      (predictions_path, metrics_per_k_path, summary_json_path, curve_png_path, curve_pdf_path)
    """
    if not pred_path.endswith("_predictions_passk.jsonl"):
        raise ValueError("Input file must end with '_predictions_passk.jsonl'")
    prefix = pred_path[:-len("_predictions_passk.jsonl")]
    return (
        pred_path,
        prefix + "_metrics_per_k.jsonl",
        prefix + "_metrics_summary.json",
        prefix + "_passk_curve.png",
        prefix + "_passk_curve.pdf",
    )

def _mode(counter: Counter, tie_numeric_smallest: bool = True):
    """Deterministic mode with numeric tie-break when possible."""
    if not counter:
        return None
    maxc = max(counter.values())
    cands = [v for v, c in counter.items() if c == maxc]
    if tie_numeric_smallest:
        try:
            return sorted(cands)[0]
        except Exception:
            pass
    return sorted(cands, key=lambda x: str(x))[0]

def recompute_metrics_from_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recreate metrics per-k and curves using the same logic as the original script."""
    if not rows:
        return {
            "per_k": [],
            "summary": {
                "num_samples": 0,
                "num_ids_countable": 0,
                "passk_answer_curve": [],
                "passk_ids_curve": [],
                "sc_answer_curve": [],
                "sc_ids_curve": [],
            }
        }

    # Index by k and sample_idx (last wins assumed already by dedup stage)
    by_k: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    gts_by_sample: Dict[int, str] = {}
    ans_gt_by_sample: Dict[int, Optional[int]] = {}
    ids_gt_by_sample: Dict[int, Optional[List[int]]] = {}
    sample_indices = set()
    max_k = 0

    for r in rows:
        k = int(r.get("k", 1))
        i = int(r.get("sample_idx", 0))
        by_k[k][i] = r
        sample_indices.add(i)
        max_k = max(max_k, k)
        if i not in gts_by_sample:
            gt = r.get("ground_truth", "") or ""
            gts_by_sample[i] = gt
            ans_gt_by_sample[i] = extract_gt_answer(gt)
            ids_gt_by_sample[i] = extract_gt_ids(gt)

    N = len(sample_indices)
    idx_list = sorted(sample_indices)
    idx_to_pos = {idx: pos for pos, idx in enumerate(idx_list)}

    ids_mask = [ids_gt_by_sample.get(idx) is not None for idx in idx_list]
    N_ids = sum(1 for x in ids_mask if x)

    ans_seen = [False] * N
    ids_seen = [False] * N

    ans_counters: List[Counter] = [Counter() for _ in range(N)]
    ids_counters: List[Counter] = [Counter() for _ in range(N)]

    passk_ans_curve: List[float] = []
    passk_ids_curve: List[Optional[float]] = []
    sc_ans_curve:   List[float] = []
    sc_ids_curve:   List[Optional[float]] = []

    per_k_records = []

    for k in range(1, max_k + 1):
        recs_k = by_k.get(k, {})
        inst_ans_hits = 0
        inst_ids_hits = 0

        for idx in idx_list:
            pos = idx_to_pos[idx]
            r = recs_k.get(idx)
            if r is None:
                continue
            am = bool(r.get("answer_match", False))
            im_val = r.get("ids_match", None)
            im = None if im_val is None else bool(im_val)

            if am:
                inst_ans_hits += 1
                ans_seen[pos] = True
            if ids_mask[pos] and (im is True):
                inst_ids_hits += 1
                ids_seen[pos] = True

            pa = r.get("pred_answer", None)
            if pa is not None:
                try:
                    ans_counters[pos][int(pa)] += 1
                except Exception:
                    pass
            pi = r.get("pred_ids", None)
            if isinstance(pi, list) and all(isinstance(x, int) for x in pi):
                ids_counters[pos][tuple(pi)] += 1

        inst_ans_acc = inst_ans_hits / N if N else 0.0
        inst_ids_acc = (inst_ids_hits / N_ids) if N_ids else 0.0

        cum_ans_acc = (sum(1 for s in ans_seen if s) / N) if N else 0.0
        cum_ids_acc = (sum(1 for pos, s in enumerate(ids_seen) if ids_mask[pos] and s) / N_ids) if N_ids else 0.0

        # Self-consistency accuracy @k
        sc_ans_hits = 0
        for pos, idx in enumerate(idx_list):
            mode_ans = _mode(ans_counters[pos], tie_numeric_smallest=True)
            gt_a = ans_gt_by_sample.get(idx)
            if mode_ans is not None and gt_a is not None and int(mode_ans) == int(gt_a):
                sc_ans_hits += 1
        sc_ans_acc = sc_ans_hits / N if N else 0.0

        if N_ids:
            sc_ids_hits = 0
            for pos, idx in enumerate(idx_list):
                if not ids_mask[pos]:
                    continue
                mode_ids = _mode(ids_counters[pos], tie_numeric_smallest=False)
                gt_ids = ids_gt_by_sample.get(idx)
                if mode_ids is not None and gt_ids is not None and list(mode_ids) == list(gt_ids):
                    sc_ids_hits += 1
            sc_ids_acc = sc_ids_hits / N_ids
        else:
            sc_ids_acc = None

        per_k_records.append({
            "k": k,
            "inst_answer_acc": inst_ans_acc,
            "inst_ids_acc": (None if not N_ids else inst_ids_acc),
            "passk_answer": cum_ans_acc,
            "passk_ids": (None if not N_ids else cum_ids_acc),
            "sc_answer_acc": sc_ans_acc,
            "sc_ids_acc": (None if not N_ids else sc_ids_acc),
        })

        passk_ans_curve.append(cum_ans_acc)
        passk_ids_curve.append(cum_ids_acc if N_ids else None)
        sc_ans_curve.append(sc_ans_acc)
        sc_ids_curve.append(sc_ids_acc if N_ids else None)

    summary = {
        "num_samples": N,
        "num_ids_countable": N_ids,
        "passk_answer_curve": passk_ans_curve,
        "passk_ids_curve": passk_ids_curve if N_ids else [],
        "sc_answer_curve": sc_ans_curve,
        "sc_ids_curve": sc_ids_curve if N_ids else [],
        "max_k": max_k,
        "counts_per_k": {int(k): len(v) for k, v in sorted(by_k.items())},
    }
    return {"per_k": per_k_records, "summary": summary}

def save_metrics_and_plots(prefix_paths: Tuple[str, str, str, str, str],
                           per_k: List[Dict[str, Any]],
                           summary: Dict[str, Any],
                           title_hint: Optional[str] = None) -> None:
    pred_path, metrics_path, summary_path, png_path, pdf_path = prefix_paths

    # Write metrics per-k JSONL
    with open(metrics_path, "w", encoding="utf-8") as f:
        for rec in per_k:
            f.write(json.dumps(rec) + "\n")

    base = os.path.basename(pred_path)[:-len("_predictions_passk.jsonl")]
    parts = base.split("_", 1)
    task_hint = parts[0] if parts else None
    model_alias = parts[1] if len(parts) > 1 else None

    full_summary = {
        "task": task_hint,
        "num_samples": summary["num_samples"],
        "num_ids_countable": summary["num_ids_countable"],
        "passk_answer_curve": summary["passk_answer_curve"],
        "passk_ids_curve": summary.get("passk_ids_curve", []),
        "sc_answer_curve": summary["sc_answer_curve"],
        "sc_ids_curve": summary.get("sc_ids_curve", []),
        "config": {
            "patched": True,
            "deduplicated": True,
            "patched_from": pred_path,
            "model_alias": model_alias,
            "k": len(summary["passk_answer_curve"]),
        },
        "verification": {
            "max_k": summary.get("max_k", None),
            "counts_per_k": summary.get("counts_per_k", {}),
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # Plot curves (mirrors your style; seaborn optional)
    try:
        import seaborn as sns  # noqa
        sns.set_theme(style="whitegrid", context="talk")
        have_sns = True
    except Exception:
        have_sns = False

    xs = list(range(1, len(summary["passk_answer_curve"]) + 1))
    plt.figure(figsize=(9.5, 5.5))
    if have_sns:
        import seaborn as sns
        palette = sns.color_palette()
        me = max(1, len(xs) // 12)
        plt.plot(xs, summary["passk_answer_curve"], label="Pass@k (answer)", linewidth=2.2, marker="o", markevery=me, color=palette[0])
        if summary["num_ids_countable"] > 0:
            plt.plot(xs, summary["passk_ids_curve"], label="Pass@k (ids)", linewidth=2.2, marker="o", markevery=me, color=palette[1])
        plt.plot(xs, summary["sc_answer_curve"], label="Self-consistency (answer)", linewidth=2.2, linestyle="--", marker="s", markevery=me, color=palette[2])
        if summary["num_ids_countable"] > 0:
            plt.plot(xs, summary["sc_ids_curve"], label="Self-consistency (ids)", linewidth=2.2, linestyle="--", marker="s", markevery=me, color=palette[3])
    else:
        plt.plot(xs, summary["passk_answer_curve"], label="Pass@k (answer)", linewidth=2.2)
        if summary["num_ids_countable"] > 0:
            plt.plot(xs, summary["passk_ids_curve"], label="Pass@k (ids)", linewidth=2.2)
        plt.plot(xs, summary["sc_answer_curve"], label="Self-consistency (answer)", linewidth=2.2, linestyle="--")
        if summary["num_ids_countable"] > 0:
            plt.plot(xs, summary["sc_ids_curve"], label="Self-consistency (ids)", linewidth=2.2, linestyle="--")

    title = title_hint or f"Pass@k & Self-Consistency — {task_hint or ''}".strip()
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.savefig(pdf_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Dedup + patch *_predictions_passk.jsonl and regenerate metrics/plots with verification output.")
    ap.add_argument("predictions_jsonl", help="Path to <prefix>_predictions_passk.jsonl")
    ap.add_argument("--no-backup", action="store_true", help="Do not write a .bak backup of the original file")
    ap.add_argument("--preserve-order", action="store_true", help="Keep original order of kept records (default: sort by (k, sample_idx))")
    ap.add_argument("--scratch-dir", default=None,help="Directory to write failure cases (default: <predictions_dir>/scratch)")

    args = ap.parse_args()

    pred_path = os.path.abspath(args.predictions_jsonl)
    prefix_paths = derive_prefix_paths(pred_path)

    if args.scratch_dir:
        scratch_dir = os.path.abspath(args.scratch_dir)
    else:
        scratch_dir = os.path.join(os.path.dirname(pred_path), "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    # Backup original file bytes before any changes
    if not args.no_backup:
        bak = pred_path + ".bak"
        try:
            if not os.path.exists(bak):
                shutil.copy2(pred_path, bak)
        except Exception:
            pass  # best effort

    # Load rows
    rows = read_jsonl(pred_path)
    original_total = len(rows)

    # --- De-duplicate by (sample_idx, k), keeping the LAST occurrence ---
    last_idx_for_key: Dict[Tuple[int, int], int] = {}
    for idx, r in enumerate(rows):
        try:
            key = (int(r["sample_idx"]), int(r["k"]))
        except Exception:
            # Skip malformed rows
            continue
        last_idx_for_key[key] = idx  # last wins

    keep_idx = set(last_idx_for_key.values())

    dedup_rows = [r for idx, r in enumerate(rows) if idx in keep_idx]

    # Canonical order: by (k, sample_idx), unless --preserve-order is set
    if not args.preserve_order:
        def _key(r):
            try:
                return (int(r.get("k", 1)), int(r.get("sample_idx", 0)))
            except Exception:
                return (10**9, 10**9)
        dedup_rows.sort(key=_key)

    post_total = len(dedup_rows)
    duplicates_removed = original_total - post_total

    # --- Patch each kept row (robust extraction + refresh matches) ---
    num_newly_extracted = 0
    num_changed = 0
    num_unchanged = 0
    num_still_empty = 0
    no_id_replies: List[str] = []

    for r in dedup_rows:
        prev_ids = r.get("pred_ids", None)
        if isinstance(prev_ids, list):
            prev_list = [int(x) for x in prev_ids if isinstance(x, (int, float))]
        else:
            prev_list = None

        text = (r.get("model_reply") or "").strip()
        gt   = r.get("ground_truth", "") or ""

        new_ids = extract_ids_from_text(text)

        if not new_ids:
            if text:
                no_id_replies.append(text)

        if (prev_list is None or len(prev_list) == 0):
            if new_ids:
                num_newly_extracted += 1
            else:
                num_still_empty += 1
        else:
            if new_ids is None or list(prev_list) != list(new_ids):
                num_changed += 1
            else:
                num_unchanged += 1

        r["pred_ids"] = new_ids
        r["pred_answer"] = extract_answer_int(text)
        r["answer_match"] = compute_answer_match(text, gt)
        im = compute_ids_match(text, gt)
        r["ids_match"] = (None if im is None else bool(im))

    # Overwrite predictions with de-duplicated, patched rows
    write_jsonl(pred_path, dedup_rows)

    base = os.path.basename(pred_path)[:-len("_predictions_passk.jsonl")]
    parts = base.split("_", 1)
    model_alias = parts[1] if len(parts) > 1 else base

    no_ids_path = os.path.join(scratch_dir, f"no_ids_{model_alias}.json")
    with open(no_ids_path, "w", encoding="utf-8") as f:
        json.dump(no_id_replies, f, ensure_ascii=False, indent=2)

    # Recompute metrics and curves from patched predictions
    metrics = recompute_metrics_from_predictions(dedup_rows)
    base = os.path.basename(pred_path)[:-len("_predictions_passk.jsonl")]
    title_hint = f"Pass@k & Self-Consistency — {base}"
    save_metrics_and_plots(prefix_paths, metrics["per_k"], metrics["summary"], title_hint=title_hint)

    # ---- Verification stats ----
    counts_per_k = metrics["summary"]["counts_per_k"]
    max_k = metrics["summary"]["max_k"]
    N = metrics["summary"]["num_samples"]

    # modal count across ks
    if counts_per_k:
        mode_count = Counter(counts_per_k.values()).most_common(1)[0][0]
        outliers = [(k, c) for k, c in sorted(counts_per_k.items()) if c != mode_count]
    else:
        mode_count = 0
        outliers = []

    print("---- passk_patch report ----")
    print(f"File: {pred_path}")
    print(f"Original total lines:         {original_total}")
    print(f"Post-dedup total lines:       {post_total}")
    print(f"Duplicates removed:           {duplicates_removed}")
    print()
    print(f"Newly extracted pred_ids:     {num_newly_extracted}")
    print(f"Changed pred_ids (non-empty): {num_changed}")
    print(f"Unchanged pred_ids:           {num_unchanged}")
    print(f"Still empty (no IDs found):   {num_still_empty}")
    print(f"  Saved {len(no_id_replies)} no-ID model replies to: {no_ids_path}")
    print()
    print("Rebuilt files:")
    print("  metrics per-k JSONL :", prefix_paths[1])
    print("  metrics summary JSON :", prefix_paths[2])
    print("  passk curve PNG      :", prefix_paths[3])
    print("  passk curve PDF      :", prefix_paths[4])
    print()
    print("Verification:")
    print(f"  Final max k (K):            {max_k}")
    print(f"  Unique samples (N):         {N}")
    print(f"  Unique (k, sample_idx):     {post_total}")
    expected = (N * max_k) if N and max_k else 0
    print(f"  Expected N*K:               {expected}")
    if expected and expected != post_total:
        print("  WARNING: N*K does not equal unique records; some (k,sample) pairs are missing.")
    if counts_per_k:
        print(f"  Samples per k — modal count: {mode_count}")
        if outliers:
            print("  Non-uniform k counts (k: count):")
            # show up to 30 outliers
            for k, c in (outliers[:30]):
                print(f"    {k}: {c}")
            if len(outliers) > 30:
                print(f"    ... {len(outliers)-30} more")
        else:
            print("  All k have the same sample count.")
    else:
        print("  No per-k counts available (empty file?).")

if __name__ == "__main__":
    main()

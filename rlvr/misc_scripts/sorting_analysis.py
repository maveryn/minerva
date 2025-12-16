# evaluate_activity_sorting_with_lacs_anchors.py
import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

Row = Tuple[int, int, int]  # (id, start_minutes, end_minutes)

# --------- Regex helpers ---------
ROW_LINE_RE = re.compile(
    r"^\s*\|?\s*(\d+)\s*\|\s*(\d{2}):(\d{2})\s*\|\s*(\d{2}):(\d{2})\s*\|?\s*$",
    re.MULTILINE,
)
TABLE_HEADER_RE = re.compile(r"^\s*\|?\s*ID\s*\|\s*Start\s*\|\s*End\s*\|?\s*$",
                             re.IGNORECASE | re.MULTILINE)
HRULE_RE = re.compile(r"^\s*\|?\s*-{2,}\s*\|\s*-{2,}\s*\|\s*-{2,}\s*\|?\s*$",
                      re.MULTILINE)

ID_TOKEN_RE = re.compile(r"\bID\s*(\d+)\b", re.IGNORECASE)
ID_WITH_TIMES_RE = re.compile(
    r"\bID\s*(\d+)\s*[:\-–—]?\s*(\d{2}):(\d{2})\s*(?:to|–|—|-)\s*(\d{2}):(\d{2})",
    re.IGNORECASE,
)
COMMA_RUN_RE = re.compile(r"(?:\b\d+\b\s*,\s*)+\b\d+\b")
IDS_BRACE_RE = re.compile(r"\\?ids?\{([^}]*)\}", re.IGNORECASE)

SORT_BLOCK_STOP_WORDS = re.compile(
    r"\b(select|greedy|greedily|choose|subset|largest|final answer|so,|thus|therefore|next,)\b",
    re.IGNORECASE,
)


# --------- Basic utils ---------
def hhmm_to_minutes(h: str, m: str) -> int:
    return int(h) * 60 + int(m)


def extract_prompt_text(obj: dict) -> str:
    # Prefer last user message
    if "messages" in obj and isinstance(obj["messages"], list):
        for msg in reversed(obj["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                return str(msg["content"])
        if obj["messages"] and isinstance(obj["messages"][0], dict):
            return str(obj["messages"][0].get("content", ""))
    return str(obj.get("prompt_text", ""))


def get_model_reply(obj: dict) -> str:
    if "model_reply" in obj:
        return str(obj["model_reply"])
    if "assistant" in obj:
        return str(obj["assistant"])
    if "completion" in obj:
        return str(obj["completion"])
    if "messages" in obj and isinstance(obj["messages"], list):
        for msg in reversed(obj["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                return str(msg["content"])
    return ""


# --------- Table extraction ---------

def extract_all_table_blocks_with_line_index(text: str) -> List[Tuple[str, int]]:
    """Return [(table_block_text, header_line_index), ...] for ALL ASCII tables in 'text'."""
    lines = text.splitlines()
    blocks: List[Tuple[str, int]] = []
    i = 0
    while i < len(lines):
        if TABLE_HEADER_RE.match(lines[i]):
            out = [lines[i]]
            j = i + 1
            if j < len(lines) and HRULE_RE.match(lines[j]):
                out.append(lines[j]); j += 1
            # consume row lines
            while j < len(lines) and ROW_LINE_RE.match(lines[j]):
                out.append(lines[j]); j += 1
            blocks.append(("\n".join(out), i))
            i = j
        else:
            i += 1
    return blocks

def extract_sorted_full_ids_from_reply_table(reply: str, id_set: Set[int]) -> Optional[List[int]]:
    """
    Prefer the reply table whose nearby context mentions 'sorted'; fall back to the first
    full-coverage table (if any). Return IDs in the order they appear in that table.
    """
    blocks = extract_all_table_blocks_with_line_index(reply)
    if not blocks:
        return None
    lines = reply.splitlines()

    def _ids_from_block(block: str) -> Optional[List[int]]:
        rows = parse_rows_from_table_block(block)
        ids = [r[0] for r in rows]
        if ids and len(set(ids)) == len(ids) and set(ids) == id_set:
            return ids
        return None

    # 1) context contains 'sorted' -> take it
    for block, idx in blocks:
        ctx = "\n".join(lines[max(0, idx - 3):idx + 1])
        if re.search(r"\bsort(?:ed|ing)?\b", ctx, re.IGNORECASE):
            ids = _ids_from_block(block)
            if ids:
                return ids

    # 2) otherwise first full-coverage table (could be the unsorted prompt copy)
    for block, _ in blocks:
        ids = _ids_from_block(block)
        if ids:
            return ids
    return None


def extract_prompt_table_block(text: str) -> str:
    """
    Return the ASCII table block ("ID | Start | End" + hrule + rows) from the prompt.
    """
    m = TABLE_HEADER_RE.search(text)
    if not m:
        return ""
    lines = text.splitlines()

    hdr_idx = None
    for i, line in enumerate(lines):
        if TABLE_HEADER_RE.match(line):
            hdr_idx = i
            break
    if hdr_idx is None:
        return ""

    out = [lines[hdr_idx]]
    i = hdr_idx + 1
    if i < len(lines) and HRULE_RE.match(lines[i]):
        out.append(lines[i])
        i += 1

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if ROW_LINE_RE.match(line):
            out.append(line)
            i += 1
            continue
        break

    return "\n".join(out)


def parse_rows_from_table_block(block: str) -> List[Row]:
    rows: List[Row] = []
    for m in ROW_LINE_RE.finditer(block):
        _id = int(m.group(1))
        sh, sm, eh, em = m.group(2), m.group(3), m.group(4), m.group(5)
        start = hhmm_to_minutes(sh, sm)
        end = hhmm_to_minutes(eh, em)
        rows.append((_id, start, end))
    return rows


def extract_rows_from_prompt(prompt_text: str) -> List[Row]:
    block = extract_prompt_table_block(prompt_text)
    if block:
        return parse_rows_from_table_block(block)
    rows: List[Row] = []
    for m in ROW_LINE_RE.finditer(prompt_text):
        _id = int(m.group(1))
        sh, sm, eh, em = m.group(2), m.group(3), m.group(4), m.group(5)
        rows.append((_id, hhmm_to_minutes(sh, sm), hhmm_to_minutes(eh, em)))
    return rows


def extract_model_response_table_text(reply: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Recover a table from the assistant's reply.
      (1) Prefer an ASCII table whose nearby context mentions 'sorted' or 'largest subset'.
      (2) Else if ANY ASCII table exists, take the first.
      (3) Else synthesize from lines like 'ID 6: 00:20 to 02:10' (first occurrence per ID).
    Returns (table_text, rows_as_str_triplets).
    """
    blocks = extract_all_table_blocks_with_line_index(reply)
    chosen = None
    if blocks:
        lines = reply.splitlines()
        for block, idx in blocks:
            ctx = "\n".join(lines[max(0, idx - 3):idx + 1])
            if (re.search(r"\bsort(?:ed|ing)?\b", ctx, re.IGNORECASE) or
                re.search(r"\blargest\s+subset\b|\bnon[- ]overlapping\b", ctx, re.IGNORECASE)):
                chosen = block
                break
        if chosen is None:
            chosen = blocks[0][0]

        row_triplets: List[Tuple[int, str, str]] = []
        for m in ROW_LINE_RE.finditer(chosen):
            row_triplets.append((int(m.group(1)),
                                 f"{m.group(2)}:{m.group(3)}",
                                 f"{m.group(4)}:{m.group(5)}"))
        return chosen, row_triplets

    # Fallback: synthesize from "ID x: HH:MM to HH:MM"
    seen: Set[int] = set()
    rows: List[Tuple[int, str, str]] = []
    for m in ID_WITH_TIMES_RE.finditer(reply):
        _id = int(m.group(1))
        if _id in seen:
            continue
        seen.add(_id)
        start = f"{m.group(2)}:{m.group(3)}"
        end = f"{m.group(4)}:{m.group(5)}"
        rows.append((_id, start, end))
    if not rows:
        return "", []
    lines = ["ID | Start | End", "---|-------|-----"]
    for _id, s, e in rows:
        lines.append(f"{_id:>2} | {s} | {e}")
    return "\n".join(lines), rows


# --------- Ground-truth sorting ---------
def true_sorted_order(rows: List[Row]) -> List[int]:
    return [r[0] for r in sorted(rows, key=lambda r: (r[2], r[0]))]


# --------- Exact-metric extraction (full sequence) ---------
def _extract_ids_by_id_token(text: str, id_set: Set[int], n_ids: int) -> Optional[List[int]]:
    seq: List[int] = []
    seen: Set[int] = set()
    for m in ID_TOKEN_RE.finditer(text):
        val = int(m.group(1))
        if val in id_set and val not in seen:
            seq.append(val)
            seen.add(val)
            if len(seq) == n_ids:
                return seq
    return None


def _extract_ids_by_commas(text: str, id_set: Set[int], n_ids: int) -> Optional[List[int]]:
    for m in COMMA_RUN_RE.finditer(text):
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        seq: List[int] = []
        seen: Set[int] = set()
        for x in nums:
            if x in id_set and x not in seen:
                seq.append(x)
                seen.add(x)
                if len(seq) == n_ids:
                    return seq
    return None


def _extract_sorted_block_candidates(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    sorted_like = [p for p in paras if re.search(r"\bsort(?:ed|ing)?\b", p, re.IGNORECASE)]
    return sorted_like if sorted_like else [text]


def extract_full_sorted_ids_for_exact_metric(reply: str, id_set: Set[int]) -> Optional[List[int]]:
    n_ids = len(id_set)

    # NEW: try to read a full-coverage table from the reply (prefer 'sorted' context)
    table_seq = extract_sorted_full_ids_from_reply_table(reply, id_set)
    if table_seq is not None:
        return table_seq

    # Existing heuristics
    for block in _extract_sorted_block_candidates(reply):
        stop = SORT_BLOCK_STOP_WORDS.search(block)
        if stop:
            block = block[:stop.start()]
        seq = _extract_ids_by_id_token(block, id_set, n_ids)
        if seq:
            return seq
        seq = _extract_ids_by_commas(block, id_set, n_ids)
        if seq:
            return seq
    seq = _extract_ids_by_id_token(reply, id_set, n_ids)
    if seq:
        return seq
    seq = _extract_ids_by_commas(reply, id_set, n_ids)
    if seq:
        return seq
    return None


# --------- LACS helpers (with run position in candidate) ---------
def _extract_ids_from_ids_braces(reply: str, id_set: Set[int]) -> List[int]:
    out: List[int] = []
    for m in IDS_BRACE_RE.finditer(reply):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        out.extend([x for x in nums if x in id_set])
    return out


def _extract_id_stream(reply: str, id_set: Set[int]) -> List[int]:
    return [int(m.group(1)) for m in ID_TOKEN_RE.finditer(reply) if int(m.group(1)) in id_set]


def longest_contiguous_block_in_true_order_with_seqpos(
    seq: List[int],
    true_order: List[int],
) -> Tuple[int, List[int], int, int, int]:
    """
    Find the longest contiguous block (by true order) inside 'seq'.
    Returns:
      best_len,
      best_run_ids,
      best_true_start_idx,
      best_seq_start_idx,
      best_seq_end_idx
    """
    if not seq or not true_order:
        return 0, [], -1, -1, -1

    pos = {id_: i for i, id_ in enumerate(true_order)}
    best_len, best_run, best_true_start = 0, [], -1
    best_seq_start, best_seq_end = -1, -1

    cur_len = 0
    cur_seq_start = -1
    cur_true_start = -1
    prev_true_pos = None

    for i, id_ in enumerate(seq):
        p = pos.get(id_)
        if p is None:
            # reset
            cur_len = 0
            cur_seq_start = -1
            cur_true_start = -1
            prev_true_pos = None
            continue

        if prev_true_pos is not None and p == prev_true_pos + 1:
            # extend current run
            cur_len += 1
        else:
            # start new run
            cur_len = 1
            cur_seq_start = i
            cur_true_start = p
        prev_true_pos = p

        if cur_len > best_len:
            best_len = cur_len
            best_true_start = cur_true_start
            best_seq_start = cur_seq_start
            best_seq_end = i
            best_run = seq[cur_seq_start:i + 1]

    return best_len, best_run, best_true_start, best_seq_start, best_seq_end


def compute_lacs_best_of_candidates(
    reply: str,
    id_set: Set[int],
    true_order: List[int],
    sorted_full_seq: Optional[List[int]],
) -> Tuple[int, float, str, List[int], int, int, int]:
    """
    Build multiple candidate ID sequences; pick the one with the largest contiguous block.
    Returns:
      lacs_len,
      lacs_fraction,
      method,
      lacs_run_ids,
      candidate_len,
      run_start_idx_in_candidate,
      run_end_idx_in_candidate
    """
    candidates: List[Tuple[str, List[int]]] = []

    if sorted_full_seq:
        candidates.append(("sorted_block_full", sorted_full_seq))

    ids_brace_seq = _extract_ids_from_ids_braces(reply, id_set)
    if ids_brace_seq:
        candidates.append(("ids_braces", ids_brace_seq))

    id_stream = _extract_id_stream(reply, id_set)
    if id_stream:
        candidates.append(("id_stream", id_stream))

    for block, _ in extract_all_table_blocks_with_line_index(reply):
        ids = [r[0] for r in parse_rows_from_table_block(block)]
        ids = [x for x in ids if x in id_set]
        if ids:
            candidates.append(("reply_table", ids))

    best_comma = []
    for m in COMMA_RUN_RE.finditer(reply):
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        filt = [x for x in nums if x in id_set]
        if len(filt) > len(best_comma):
            best_comma = filt
    if best_comma:
        candidates.append(("comma_run", best_comma))

    best_len, best_frac = 0, 0.0
    best_method, best_run = "", []
    best_cand_len = 0
    best_seq_start, best_seq_end = -1, -1
    denom = len(true_order) if true_order else 1

    for method, seq in candidates:
        length, run, _true_start, seq_start, seq_end = longest_contiguous_block_in_true_order_with_seqpos(seq, true_order)
        frac = length / denom if denom else 0.0
        # Choose the candidate with the largest run; tie-break by earlier candidate in list
        if length > best_len:
            best_len, best_frac = length, frac
            best_method, best_run = method, run
            best_cand_len = len(seq)
            best_seq_start, best_seq_end = seq_start, seq_end

    return best_len, best_frac, best_method, best_run, best_cand_len, best_seq_start, best_seq_end


# --------- Main evaluation ---------
def evaluate_file(
    path: str,
    task_filter: Optional[str] = "activity",
    debug: bool = False,
) -> Tuple[int, int, int, List[dict], int, float, Dict[str, int]]:
    """
    Returns:
      total_items,
      extracted_ok_full,
      sorted_correct_full,
      per_item_records,
      lacs_items_with_seq,
      lacs_mean_fraction,
      lacs_anchor_counts (dict: start, end, both, neither)
    """
    total = 0
    extracted_ok = 0
    sorted_correct = 0
    records: List[dict] = []

    lacs_fracs: List[float] = []
    lacs_nonempty_count = 0
    anchor_counts = {"start": 0, "end": 0, "both": 0, "neither": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                if debug:
                    print(f"[WARN] Line {line_no}: JSON decode error: {e}", file=sys.stderr)
                continue

            if task_filter and str(obj.get("task", "")).lower() != task_filter.lower():
                continue

            total += 1

            prompt_text = extract_prompt_text(obj)
            prompt_table_text = extract_prompt_table_block(prompt_text)

            rows = parse_rows_from_table_block(prompt_table_text) if prompt_table_text else extract_rows_from_prompt(
                prompt_text)
            print(prompt_table_text)
            print(rows)

            if not rows:
                # Hard fail: we must always be able to parse the prompt's table
                print("\n[ERROR] Failed to parse table rows from prompt. Full prompt below:\n", file=sys.stderr)
                print(prompt_text, file=sys.stderr)
                raise RuntimeError(f"Could not parse table rows for index={obj.get('index')} on line {line_no}.")

            if not rows:
                if debug:
                    print(f"[WARN] Item #{total} (line {line_no}): No table rows parsed.", file=sys.stderr)
                records.append({
                    "index": obj.get("index"),
                    "extraction_success": False,
                    "reason": "no_table_rows",
                    "prompt_table_text": prompt_table_text or "",
                    "model_response_table_text": "",
                })
                continue

            id_set = {r[0] for r in rows}
            true_order = true_sorted_order(rows)
            reply = get_model_reply(obj)

            # Exact metric (requires full-length extraction)
            sorted_full_seq = extract_full_sorted_ids_for_exact_metric(reply, id_set)
            success_full = False
            exact_correct = False
            if sorted_full_seq is not None and set(sorted_full_seq) == id_set and len(sorted_full_seq) == len(id_set):
                success_full = True
                extracted_ok += 1
                if sorted_full_seq == true_order:
                    sorted_correct += 1
                    exact_correct = True

            # LACS metric + anchor diagnostics
            (lacs_len, lacs_frac, lacs_method, lacs_run,
             cand_len, run_start, run_end) = compute_lacs_best_of_candidates(
                reply, id_set, true_order, sorted_full_seq if success_full else None
            )

            # Anchor positions (only if we have a non-empty LACS)
            lacs_anchor = "neither"
            is_prefix = False
            is_suffix = False
            if lacs_len > 0 and cand_len > 0 and run_start >= 0 and run_end >= 0:
                is_prefix = (run_start == 0)
                is_suffix = (run_end == cand_len - 1)
                if is_prefix and is_suffix:
                    lacs_anchor = "both"
                elif is_prefix:
                    lacs_anchor = "start"
                elif is_suffix:
                    lacs_anchor = "end"
                else:
                    lacs_anchor = "neither"

                lacs_fracs.append(lacs_frac)
                lacs_nonempty_count += 1
                # Count (non-exclusive) for start/end; neither is exclusive (not start and not end)
                if is_prefix:
                    anchor_counts["start"] += 1
                if is_suffix:
                    anchor_counts["end"] += 1
                if is_prefix and is_suffix:
                    anchor_counts["both"] += 1
                if not is_prefix and not is_suffix:
                    anchor_counts["neither"] += 1

            # Model response table (normalized)
            model_table_text, _model_rows_as_str = extract_model_response_table_text(reply)

            rec = {
                "index": obj.get("index"),
                "num_ids": len(id_set),
                "true_order": true_order,
                # Exact metric
                "sorted_extraction_success": success_full,
                "sorted_correct": exact_correct,
                "sorted_extracted_order": sorted_full_seq if success_full else None,
                # LACS metric
                "lacs_len": lacs_len,
                "lacs_fraction": lacs_frac,
                "lacs_method": lacs_method,
                "lacs_run": lacs_run,
                # Anchor diagnostics (per-item)
                "lacs_candidate_len": cand_len,
                "lacs_run_start_idx_in_candidate": run_start,
                "lacs_run_end_idx_in_candidate": run_end,
                "lacs_is_prefix": is_prefix,
                "lacs_is_suffix": is_suffix,
                "lacs_anchor": lacs_anchor,  # one of: start/end/both/neither
                # Tables-only dumps
                "prompt_table_text": prompt_table_text or "",
                "model_response_table_text": model_table_text or "",
            }
            records.append(rec)

            if debug:
                print(f"[DBG] idx={obj.get('index')} | exact_ok={success_full} "
                      f"| exact_correct={exact_correct} | LACS={lacs_len}/{len(id_set)} ({lacs_frac:.3f}) "
                      f"via {lacs_method} | anchor={lacs_anchor} (start={is_prefix}, end={is_suffix})")

    lacs_mean_fraction = (sum(lacs_fracs) / len(lacs_fracs)) if lacs_fracs else 0.0
    return total, extracted_ok, sorted_correct, records, lacs_nonempty_count, lacs_mean_fraction, anchor_counts


def dump_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate interval-scheduling sorting: exact metric, LACS fraction, and LACS anchor (start/end/neither). Dumps per-item JSONL with tables."
    )
    ap.add_argument("jsonl_path", type=str, help="Path to JSONL with prompts and model replies.")
    ap.add_argument("--task-filter", type=str, default="activity",
                    help="Only evaluate items with this task value (default: 'activity'). Use '' to disable.")
    ap.add_argument("--debug", action="store_true", help="Verbose per-item logs.")
    ap.add_argument("--dump-jsonl", type=str, default=None, help="Path to save per-item results as JSONL.")
    args = ap.parse_args()

    (total, extracted_ok, sorted_correct, records,
     lacs_count, lacs_mean, anchor_counts) = evaluate_file(
        args.jsonl_path,
        task_filter=(args.task_filter or None),
        debug=args.debug,
    )

    if total == 0:
        print("No matching items found. Check --task-filter or the input file.")
        return

    extraction_rate = extracted_ok / total
    conditional_accuracy = (sorted_correct / extracted_ok) if extracted_ok else 0.0
    overall_rate = sorted_correct / total
    lacs_cover_rate = lacs_count / total if total else 0.0

    start_frac = (anchor_counts["start"] / lacs_count) if lacs_count else 0.0
    end_frac = (anchor_counts["end"] / lacs_count) if lacs_count else 0.0
    neither_frac = (anchor_counts["neither"] / lacs_count) if lacs_count else 0.0
    both_frac = (anchor_counts["both"] / lacs_count) if lacs_count else 0.0

    print("=== Activity Sorting Evaluation ===")
    print(f"Total items evaluated        : {total}")
    print(f"Successful full-ID extraction: {extracted_ok} ({extraction_rate:.1%})")
    print(f"Exact sorting (conditional)  : {sorted_correct}/{extracted_ok} ({conditional_accuracy:.1%})")
    print(f"Exact sorting (overall)      : {sorted_correct}/{total} ({overall_rate:.1%})")
    print("---")
    print(f"LACS coverage (any sequence) : {lacs_count}/{total} ({lacs_cover_rate:.1%})")
    print(f"LACS mean fraction           : {lacs_mean:.3f}  (averaged over items with non-empty LACS)")
    print("--- LACS Anchor metrics (fractions among items with non-empty LACS) ---")
    print(f"Start-anchored fraction      : {start_frac:.3f}")
    print(f"End-anchored fraction        : {end_frac:.3f}")
    print(f"Neither-anchored fraction    : {neither_frac:.3f}")
    print(f"(Both start & end) fraction  : {both_frac:.3f}  # note: counted in both start & end")

    if args.dump_jsonl:
        dump_jsonl(args.dump_jsonl, records)
        print(f"Per-item JSONL saved to: {args.dump_jsonl}")


if __name__ == "__main__":
    main()

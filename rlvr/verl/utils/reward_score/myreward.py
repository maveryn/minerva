# verl/utils/reward_score/myreward.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reward_mmr_do.py
# Simple reward for MMR-Gym discrete-optimization tasks (coin_change, interval_scheduling).
# Rewards are the sum of two parts:
#   - Answer match (integer inside \answer{...})
#   - IDs match    (sequence inside \ids{...})
# Each part is scaled by its own multiplier and added.
#
# New: IDs reward can be either:
#   - "exact"  : 1.0 if exact sequence match, else 0
#   - "prefix" : prefix-match fraction (m / len(gt)) minus a length-mismatch penalty
#                (penalty applied if len(pred) != len(gt)), then clamped at >= 0.
#
# Defaults below use the new "prefix" mode with 0.1 penalty.

import json
import re
from typing import Optional

# -------- configuration (edit as needed) --------
ANSWER_MULTIPLIER: float = 1.0
IDS_MULTIPLIER: float = 1.0
IDS_REWARD_MODE: str = "prefix"  # "prefix" (default) or "exact"
IDS_LENGTH_PENALTY: float = 0.1  # penalty (subtracted) if lengths differ in prefix mode
# -----------------------------------------------

_CLIP_CHARS = 1200  # only scan the tail of very long generations for speed

_ANS_BRACED_RE = re.compile(r"\\answer\{\s*([+-]?\d+)\s*\}", flags=re.IGNORECASE)
_INT_RE = re.compile(r"[+-]?\d+")
_IDS_BRACED_RE = re.compile(r"\\ids\{\s*([^}]*)\s*\}", flags=re.IGNORECASE | re.DOTALL)
_ID_INT_RE = re.compile(r"\d+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# -------- regex & helpers for sorting reward --------
# Table parsing
_ROW_LINE_RE = re.compile(r"^\s*\|?\s*(\d+)\s*\|\s*(\d{2}):(\d{2})\s*\|\s*(\d{2}):(\d{2})\s*\|?\s*$", re.MULTILINE)
_TABLE_HEADER_RE = re.compile(r"^\s*\|?\s*ID\s*\|\s*Start\s*\|\s*End\s*\|?\s*$", re.IGNORECASE | re.MULTILINE)
_HRULE_RE = re.compile(r"^\s*\|?\s*-{2,}\s*\|\s*-{2,}\s*\|\s*-{2,}\s*\|?\s*$", re.MULTILINE)

# Reply parsing
_ID_TOKEN_RE = re.compile(r"\bID\s*(\d+)\b", re.IGNORECASE)
_COMMA_RUN_RE = re.compile(r"(?:\b\d+\b\s*,\s*)+\b\d+\b")
_IDS_ANY_BRACED_RE = re.compile(r"\\?ids?\{\s*([^}]*)\s*\}", re.IGNORECASE | re.DOTALL)
_INT_LIST_FALLBACK_RE = re.compile(r"[\[\{]\s*(?:\d+(?:\s*[, ]\s*\d+)*)\s*[\]\}]", re.DOTALL)


# helpers for sorting
def _hhmm_to_minutes(h: str, m: str) -> int:
    return int(h) * 60 + int(m)


def _extract_prompt_text(extra_info) -> str:
    """Best-effort prompt recovery from extra_info."""
    if isinstance(extra_info, dict):
        if isinstance(extra_info.get("prompt_text"), str):
            return extra_info["prompt_text"]
        if isinstance(extra_info.get("prompt"), str):
            return extra_info["prompt"]
        msgs = extra_info.get("messages")
        if isinstance(msgs, list):
            for msg in reversed(msgs):
                if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    return msg["content"]
    return None


def _extract_prompt_table_block(text: str) -> str:
    """Return the ASCII table block from the prompt."""
    if not text:
        return ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _TABLE_HEADER_RE.match(lines[i]):
            out = [lines[i]]
            j = i + 1
            if j < len(lines) and _HRULE_RE.match(lines[j]):
                out.append(lines[j])
                j += 1
            while j < len(lines) and _ROW_LINE_RE.match(lines[j]):
                out.append(lines[j])
                j += 1
            return "\n".join(out)
        i += 1
    return ""


def _parse_rows_from_table_block(block: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for m in _ROW_LINE_RE.finditer(block):
        _id = int(m.group(1))
        start = _hhmm_to_minutes(m.group(2), m.group(3))
        end = _hhmm_to_minutes(m.group(4), m.group(5))
        rows.append((_id, start, end))
    return rows


def _extract_rows_from_prompt(prompt_text: str) -> list[tuple[int, int, int]]:
    block = _extract_prompt_table_block(prompt_text)
    if block:
        return _parse_rows_from_table_block(block)
    # fallback: scan anywhere
    rows: list[tuple[int, int, int]] = []
    for m in _ROW_LINE_RE.finditer(prompt_text or ""):
        _id = int(m.group(1))
        start = _hhmm_to_minutes(m.group(2), m.group(3))
        end = _hhmm_to_minutes(m.group(4), m.group(5))
        rows.append((_id, start, end))
    return rows


def _true_sorted_order(rows: list[tuple[int, int, int]]) -> list[int]:
    """IDs sorted by increasing END time; tie-break by smaller ID."""
    return [r[0] for r in sorted(rows, key=lambda r: (r[2], r[0]))]


def _dedup_keep_first(seq: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_ids_from_reply_braces(reply: str, id_set: set) -> list[int]:
    out: list[int] = []
    for m in _IDS_ANY_BRACED_RE.finditer(reply or ""):
        nums = [int(x) for x in _ID_INT_RE.findall(m.group(1))]
        out.extend([x for x in nums if x in id_set])
    return out


def _extract_ids_from_id_stream(reply: str, id_set: set) -> list[int]:
    return [int(m.group(1)) for m in _ID_TOKEN_RE.finditer(reply or "") if int(m.group(1)) in id_set]


def _best_comma_run_ids(reply: str, id_set: set) -> list[int]:
    best: list[int] = []
    for m in _COMMA_RUN_RE.finditer(reply or ""):
        nums = [int(x) for x in _ID_INT_RE.findall(m.group(0))]
        nums = [x for x in nums if x in id_set]
        if len(nums) > len(best):
            best = nums
    return best


def _int_list_fallback_ids(reply: str, id_set: set) -> list[int]:
    lists = _INT_LIST_FALLBACK_RE.findall(reply or "")
    if not lists:
        return []
    # prefer last with >=2 ints
    for raw in reversed(lists):
        nums = [int(x) for x in _ID_INT_RE.findall(raw)]
        nums = [x for x in nums if x in id_set]
        if len(nums) >= 2:
            return nums
    nums = [int(x) for x in _ID_INT_RE.findall(lists[-1])]
    return [x for x in nums if x in id_set]


def _extract_all_reply_tables_with_line_index(text: str) -> list[tuple[str, int]]:
    """[(table_block_text, header_line_index), ...] for ANY ASCII tables in reply."""
    lines = (text or "").splitlines()
    blocks: list[tuple[str, int]] = []
    i = 0
    while i < len(lines):
        if _TABLE_HEADER_RE.match(lines[i]):
            out = [lines[i]]
            j = i + 1
            if j < len(lines) and _HRULE_RE.match(lines[j]):
                out.append(lines[j])
                j += 1
            while j < len(lines) and _ROW_LINE_RE.match(lines[j]):
                out.append(lines[j])
                j += 1
            blocks.append(("\n".join(out), i))
            i = j
        else:
            i += 1
    return blocks


def _ids_from_sorted_reply_table(reply: str, id_set: set) -> Optional[list[int]]:
    """Pick the reply table whose nearby context mentions 'sorted' or 'largest subset'."""
    blocks = _extract_all_reply_tables_with_line_index(reply or "")
    if not blocks:
        return None
    lines = (reply or "").splitlines()

    def _ids_from_block(block: str) -> Optional[list[int]]:
        ids = [r[0] for r in _parse_rows_from_table_block(block)]
        ids = [x for x in ids if x in id_set]
        return ids if ids else None

    for block, idx in blocks:
        ctx = "\n".join(lines[max(0, idx - 3) : idx + 1])
        if re.search(r"\bsort(?:ed|ing)?\b", ctx, re.IGNORECASE) or re.search(
            r"\blargest\s+subset\b|\bnon[- ]overlapping\b", ctx, re.IGNORECASE
        ):
            ids = _ids_from_block(block)
            if ids:
                return ids
    # fallback: take first table
    ids = _ids_from_block(blocks[0][0])
    return ids


def _extract_sorted_seq_from_reply(reply: str, id_set: set) -> Optional[list[int]]:
    """Robustly extract an ID sequence expressing the model's intended order."""
    # 1) ids{...} (with optional backslash / 's')
    ids = _extract_ids_from_reply_braces(reply, id_set)
    if ids:
        return _dedup_keep_first(ids)
    # 2) Reply ASCII table (prefer 'sorted' context)
    ids = _ids_from_sorted_reply_table(reply, id_set)
    if ids:
        return _dedup_keep_first(ids)
    # 3) Best comma run
    ids = _best_comma_run_ids(reply, id_set)
    if ids:
        return _dedup_keep_first(ids)
    # 4) ID token stream
    ids = _extract_ids_from_id_stream(reply, id_set)
    if ids:
        return _dedup_keep_first(ids)
    # 5) Raw [1,2,3] / {1,2,3}
    ids = _int_list_fallback_ids(reply, id_set)
    if ids:
        return _dedup_keep_first(ids)
    return None


def _rows_from_extra_intervals(extra_info) -> Optional[list[tuple[int, int, int]]]:
    """
    Try to read intervals from extra_info["intervals_json"] (str or list) or extra_info["intervals"].
    Expected item keys: {"id": int, "start": int, "end": int}.
    Returns list of (id, start, end) or None.
    """
    if not isinstance(extra_info, dict):
        return None
    payload = extra_info.get("intervals_json", None)
    if payload is None:
        payload = extra_info.get("intervals", None)
    if payload is None:
        return None
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    rows: list[tuple[int, int, int]] = []
    for it in data:
        try:
            _id = int(it["id"])
            start = int(it["start"])
            end = int(it["end"])
            rows.append((_id, start, end))
        except Exception:
            continue
    return rows or None


# other helpers
def _format_reward(response: str) -> float:
    if not isinstance(response, str):
        return 0.0
    has_think = _THINK_RE.search(response) is not None
    has_ids = _IDS_BRACED_RE.search(response) is not None
    has_answer = _ANS_BRACED_RE.search(response) is not None
    return 1.0 if (has_think and has_ids and has_answer) else 0.0


def _clip_tail(s: str, k: int = _CLIP_CHARS) -> str:
    return s[-k:] if isinstance(s, str) and len(s) > k else (s or "")


def _extract_braced_int(text: str) -> Optional[int]:
    if not text:
        return None
    m = _ANS_BRACED_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_last_int(text: str) -> Optional[int]:
    if not text:
        return None
    matches = _INT_RE.findall(_clip_tail(text))
    return int(matches[-1]) if matches else None


def _extract_ids(text: str) -> Optional[list[int]]:
    """Parse \\ids{...} as a list of ints; returns None if not present."""
    if not text:
        return None
    m = _IDS_BRACED_RE.search(text)
    if not m:
        return None
    return [int(x) for x in _ID_INT_RE.findall(m.group(1))]


def _prefix_match_len(gt: list[int], pred: list[int]) -> int:
    """Count matching elements from the start until the first mismatch."""
    m = 0
    for a, b in zip(gt, pred, strict=False):
        if a != b:
            break
        m += 1
    return m


def _ids_reward_exact(gt_ids: list[int], pred_ids: Optional[list[int]]) -> float:
    if pred_ids is None:
        return 0.0
    return 1.0 if pred_ids == gt_ids else 0.0


def _ids_reward_prefix(gt_ids: list[int], pred_ids: Optional[list[int]]) -> float:
    """
    Prefix reward:
      base = (prefix_match / len(gt))
      penalty = IDS_LENGTH_PENALTY if len(pred) != len(gt) else 0
      reward = max(0, score * (base - penalty))
    """
    if not gt_ids:
        return 0.0  # nothing to match
    if pred_ids is None:
        base = 0.0
        length_penalty = IDS_LENGTH_PENALTY  # pred missing => length differs
    else:
        m = _prefix_match_len(gt_ids, pred_ids)
        base = m / float(len(gt_ids))
        length_penalty = IDS_LENGTH_PENALTY if (len(pred_ids) != len(gt_ids)) else 0.0
    raw = base - length_penalty
    return max(0.0, raw)


def reward_answer_only(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Returns the combined reward:
        ANSWER_MULTIPLIER * (score if predicted answer == GT answer else 0)
      + IDS_MULTIPLIER    * (ids reward per IDS_REWARD_MODE)
    Notes:
      - Ground truth MUST contain \\answer{...}; otherwise ValueError is raised.
      - IDs in GT MUST exist; otherwise ValueError is raised.
      - In 'prefix' mode, reward is clamped at >= 0 after subtracting length penalty.
    """
    # ----- answer component -----
    gt_ans = _extract_braced_int(ground_truth)
    if gt_ans is None:
        raise ValueError("Ground truth missing \\answer{...} integer.")

    pred_ans = _extract_braced_int(solution_str)
    if pred_ans is None:
        pred_ans = _extract_last_int(solution_str)
    ans_reward = 1.0 if (pred_ans is not None and pred_ans == gt_ans) else 0.0

    # ----- combine -----
    total = float(ans_reward)
    return total


def reward_answer_format(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Returns the combined reward:
        ANSWER_MULTIPLIER * (score if predicted answer == GT answer else 0)
      + IDS_MULTIPLIER    * (ids reward per IDS_REWARD_MODE)
    Notes:
      - Ground truth MUST contain \\answer{...}; otherwise ValueError is raised.
      - IDs in GT MUST exist; otherwise ValueError is raised.
      - In 'prefix' mode, reward is clamped at >= 0 after subtracting length penalty.
    """
    # ----- answer component -----
    gt_ans = _extract_braced_int(ground_truth)
    if gt_ans is None:
        raise ValueError("Ground truth missing \\answer{...} integer.")

    format_reward = _format_reward(solution_str)
    pred_ans = _extract_braced_int(solution_str)
    if pred_ans is None:
        pred_ans = _extract_last_int(solution_str)
    ans_reward = 1.0 if (pred_ans is not None and pred_ans == gt_ans) else 0.0

    # ----- combine -----
    format_weight = 0.1
    total = (1 - format_weight) * float(ans_reward) + format_weight * format_reward
    return total


def reward_ids_prefix(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Returns the combined reward:
        ANSWER_MULTIPLIER * (score if predicted answer == GT answer else 0)
      + IDS_MULTIPLIER    * (ids reward per IDS_REWARD_MODE)
    Notes:
      - Ground truth MUST contain \\answer{...}; otherwise ValueError is raised.
      - IDs in GT MUST exist; otherwise ValueError is raised.
      - In 'prefix' mode, reward is clamped at >= 0 after subtracting length penalty.
    """
    # ----- ids component -----
    gt_ids = _extract_ids(ground_truth)
    if gt_ids is None:
        raise ValueError("Ground truth missing \\ids{...}")

    pred_ids = _extract_ids(solution_str)

    ids_reward = _ids_reward_prefix(gt_ids, pred_ids)

    # ----- combine -----
    total = float(ids_reward)
    return total


def reward_ids_exact(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Returns the combined reward:
        ANSWER_MULTIPLIER * (score if predicted answer == GT answer else 0)
      + IDS_MULTIPLIER    * (ids reward per IDS_REWARD_MODE)
    Notes:
      - Ground truth MUST contain \\answer{...}; otherwise ValueError is raised.
      - IDs in GT MUST exist; otherwise ValueError is raised.
      - In 'prefix' mode, reward is clamped at >= 0 after subtracting length penalty.
    """
    # ----- ids component -----
    gt_ids = _extract_ids(ground_truth)
    if gt_ids is None:
        raise ValueError("Ground truth missing \\ids{...}")

    pred_ids = _extract_ids(solution_str)

    ids_reward = _ids_reward_exact(gt_ids, pred_ids)

    # ----- combine -----
    total = float(ids_reward)
    return total


def reward_answer_ids(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Returns the combined reward:
        ANSWER_MULTIPLIER * (score if predicted answer == GT answer else 0)
      + IDS_MULTIPLIER    * (ids reward per IDS_REWARD_MODE)
    Notes:
      - Ground truth MUST contain \\answer{...}; otherwise ValueError is raised.
      - IDs in GT MUST exist; otherwise ValueError is raised.
      - In 'prefix' mode, reward is clamped at >= 0 after subtracting length penalty.
    """
    # ----- answer component -----
    gt_ans = _extract_braced_int(ground_truth)
    if gt_ans is None:
        raise ValueError("Ground truth missing \\answer{...} integer.")

    pred_ans = _extract_braced_int(solution_str)
    if pred_ans is None:
        pred_ans = _extract_last_int(solution_str)
    ans_reward = 1.0 if (pred_ans is not None and pred_ans == gt_ans) else 0.0

    # ----- ids component -----
    gt_ids = _extract_ids(ground_truth)
    if gt_ids is None:
        raise ValueError("Ground truth missing \\ids{...}")

    pred_ids = _extract_ids(solution_str)

    mode = (IDS_REWARD_MODE or "prefix").lower()
    if mode == "exact":
        ids_reward = _ids_reward_exact(gt_ids, pred_ids)
    else:
        # default to 'prefix'
        ids_reward = _ids_reward_prefix(gt_ids, pred_ids)

    # ----- combine -----
    total = ANSWER_MULTIPLIER * float(ans_reward) + IDS_MULTIPLIER * float(ids_reward)
    return total


# -------- reward: sorting prefix (ADD near other reward_* functions) --------
def reward_sort_prefix(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Prefix reward for activity-sorting (interval scheduling) style prompts.

    Steps:
      1) Prefer intervals from extra_info["intervals_json"] (or ["intervals"]) to recover rows (ID, start, end).
         If missing/unparseable, fall back to parsing the prompt table.
      2) Compute the true global sorted order by END time (tie-break by smaller ID).
      3) Extract a candidate ID sequence from the model reply (solution_str).
      4) Reward = prefix_match(true_sorted, pred_seq) / len(true_sorted).

    Notes:
      - Raises ValueError if neither extra_info nor the prompt yields a parseable table.
      - Returns 0.0 if no usable sequence is found in the reply.
      - No length penalty; strictly the prefix fraction in [0, 1].
    """
    # 1) rows from extra_info if available; else fall back to prompt parsing
    rows = _rows_from_extra_intervals(extra_info)
    if rows is None:
        prompt_text = _extract_prompt_text(extra_info)
        if prompt_text is None:
            prompt_text = ground_truth
        rows = _extract_rows_from_prompt(prompt_text)

    if not rows:
        truncated = str(extra_info)[:400] if extra_info is not None else ""
        raise ValueError(
            f"[reward_sort] Failed to obtain intervals from extra_info or prompt. extra_info head:\n{truncated}"
        )

    id_set = {r[0] for r in rows}
    true_sorted = _true_sorted_order(rows)
    if not true_sorted:
        return 0.0

    pred_seq = _extract_sorted_seq_from_reply(solution_str or "", id_set)
    if not pred_seq:
        return 0.0

    m = _prefix_match_len(true_sorted, pred_seq)
    return m / float(len(true_sorted))


# -------- reward: sorting + IDs match + answer --------
def reward_all(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Prefix reward for activity-sorting (interval scheduling) style prompts.

    Steps:
      1) Prefer intervals from extra_info["intervals_json"] (or ["intervals"]) to recover rows (ID, start, end).
         If missing/unparseable, fall back to parsing the prompt table.
      2) Compute the true global sorted order by END time (tie-break by smaller ID).
      3) Extract a candidate ID sequence from the model reply (solution_str).
      4) Reward = prefix_match(true_sorted, pred_seq) / len(true_sorted).

    Notes:
      - Raises ValueError if neither extra_info nor the prompt yields a parseable table.
      - Returns 0.0 if no usable sequence is found in the reply.
      - No length penalty; strictly the prefix fraction in [0, 1].
    """
    # 1) rows from extra_info if available; else fall back to prompt parsing
    rows = _rows_from_extra_intervals(extra_info)
    if rows is None:
        prompt_text = _extract_prompt_text(extra_info)
        if prompt_text is None:
            prompt_text = ground_truth
        rows = _extract_rows_from_prompt(prompt_text)

    if not rows:
        truncated = str(extra_info)[:400] if extra_info is not None else ""
        raise ValueError(
            f"[reward_sort] Failed to obtain intervals from extra_info or prompt. extra_info head:\n{truncated}"
        )

    id_set = {r[0] for r in rows}
    true_sorted = _true_sorted_order(rows)
    if not true_sorted:
        return 0.0

    pred_seq = _extract_sorted_seq_from_reply(solution_str or "", id_set)
    if not pred_seq:
        return 0.0

    m = _prefix_match_len(true_sorted, pred_seq)
    return m / float(len(true_sorted))


def reward_sort_exact(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Exact reward for activity-sorting (interval scheduling):
      1 if the extracted sequence from the reply exactly equals the true
      END-time-sorted order (tie-break by smaller ID); else 0.
    """
    # Prefer intervals from extra_info; fall back to prompt parsing
    rows = _rows_from_extra_intervals(extra_info)
    if rows is None:
        prompt_text = _extract_prompt_text(extra_info)
        if prompt_text is None:
            prompt_text = ground_truth
        rows = _extract_rows_from_prompt(prompt_text)

    if not rows:
        truncated = str(extra_info)[:400] if extra_info is not None else ""
        raise ValueError(
            f"[reward_sort_exact] Failed to obtain intervals from extra_info or prompt. extra_info head:\n{truncated}"
        )

    id_set = {r[0] for r in rows}
    true_sorted = _true_sorted_order(rows)
    if not true_sorted:
        return 0.0

    pred_seq = _extract_sorted_seq_from_reply(solution_str or "", id_set)
    if not pred_seq:
        return 0.0

    return 1.0 if list(pred_seq) == list(true_sorted) else 0.0


def reward_sort_ids_answer_exact(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Combined exact reward (equal weights):
      ( exact-sort + exact-ids + exact-answer ) / 3
    - exact-sort uses reply-extracted order vs true END-time order
    - exact-ids  uses \\ids{...} extraction vs GT \\ids{...}
    - exact-answer uses \\answer{...} vs GT \\answer{...}
    """
    r_sort = reward_sort_exact(data_source, solution_str, ground_truth, extra_info)
    r_ids = reward_ids_exact(data_source, solution_str, ground_truth, extra_info)
    r_ans = reward_answer_only(data_source, solution_str, ground_truth, extra_info)
    reward = (r_sort + r_ids + r_ans) / 3.0
    return {"score": reward, "r_sort": r_sort, "r_ids": r_ids, "r_ans": r_ans}


def reward_sort_prefix_ids_answer(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Combined exact reward (equal weights):
      ( exact-sort + exact-ids + exact-answer ) / 3
    - exact-sort uses reply-extracted order vs true END-time order
    - exact-ids  uses \\ids{...} extraction vs GT \\ids{...}
    - exact-answer uses \\answer{...} vs GT \\answer{...}
    """
    r_sort = reward_sort_exact(data_source, solution_str, ground_truth, extra_info)
    r_ids = reward_ids_exact(data_source, solution_str, ground_truth, extra_info)
    r_ans = reward_answer_only(data_source, solution_str, ground_truth, extra_info)
    reward = (r_sort + r_ids + r_ans) / 3.0
    return {"score": reward, "r_sort": r_sort, "r_ids": r_ids, "r_ans": r_ans}


def reward_all_weighted(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Combined exact reward (equal weights):
      ( exact-sort + exact-ids + exact-answer ) / 3
    - exact-sort uses reply-extracted order vs true END-time order
    - exact-ids  uses \\ids{...} extraction vs GT \\ids{...}
    - exact-answer uses \\answer{...} vs GT \\answer{...}
    """
    r_sort = reward_sort_prefix(data_source, solution_str, ground_truth, extra_info)
    r_ids = reward_ids_prefix(data_source, solution_str, ground_truth, extra_info)
    r_ans = reward_answer_only(data_source, solution_str, ground_truth, extra_info)
    # reward = (0.4 * r_sort + 0.4 * r_ids + 0.2 * r_ans)/1.0
    reward = (0.5 * r_sort + 0.5 * r_ids) / 1.0
    return {"score": reward, "r_sort": r_sort, "r_ids": r_ids, "r_ans": r_ans}

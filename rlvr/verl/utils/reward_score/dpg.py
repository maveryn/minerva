# verl/utils/reward_score/dpg.py
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

import re
from typing import Optional

# -------- configuration (edit as needed) --------
ANSWER_MULTIPLIER: float = 0.0
IDS_MULTIPLIER: float = 1.0
IDS_REWARD_MODE: str = "prefix"  # "prefix" (default) or "exact"
IDS_LENGTH_PENALTY: float = 0.1  # penalty (subtracted) if lengths differ in prefix mode
# -----------------------------------------------

_CLIP_CHARS = 1200  # only scan the tail of very long generations for speed

_ANS_BRACED_RE = re.compile(r"\\answer\{\s*([+-]?\d+)\s*\}", flags=re.IGNORECASE)
_INT_RE = re.compile(r"[+-]?\d+")
_IDS_BRACED_RE = re.compile(r"\\ids\{\s*([^}]*)\s*\}", flags=re.IGNORECASE | re.DOTALL)
_ID_INT_RE = re.compile(r"\d+")


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


def _ids_reward_exact(gt_ids: list[int], pred_ids: Optional[list[int]], score: float) -> float:
    if pred_ids is None:
        return 0.0
    return float(score) if pred_ids == gt_ids else 0.0


def _ids_reward_prefix(gt_ids: list[int], pred_ids: Optional[list[int]], score: float) -> float:
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
    raw = score * (base - length_penalty)
    return max(0.0, raw)


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.0,  # unused in total; kept for API compatibility
    score: float = 1.0,
) -> float:
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
    ans_reward = score if (pred_ans is not None and pred_ans == gt_ans) else 0.0

    # ----- ids component -----
    gt_ids = _extract_ids(ground_truth)
    if gt_ids is None:
        raise ValueError("Ground truth missing \\ids{...}")

    pred_ids = _extract_ids(solution_str)

    mode = (IDS_REWARD_MODE or "prefix").lower()
    if mode == "exact":
        ids_reward = _ids_reward_exact(gt_ids, pred_ids, score)
    else:
        # default to 'prefix'
        ids_reward = _ids_reward_prefix(gt_ids, pred_ids, score)

    # ----- combine -----
    total = ANSWER_MULTIPLIER * float(ans_reward) + IDS_MULTIPLIER * float(ids_reward)
    return total

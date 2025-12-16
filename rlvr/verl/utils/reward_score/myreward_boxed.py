# verl/utils/reward_score/myreward_boxed.py
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

from __future__ import annotations

import re
from typing import Optional

_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-–—]?\s*",
    re.IGNORECASE,
)
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)


def _strip_prefix(s: str) -> str:
    return _PREFIX_RE.sub("", s).strip()


def _extract_from_lines(text: str, pattern: str, transform=lambda x: x) -> str:
    """Return the last regex match when scanning bottom-to-top."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        raw = lines[i]
        line = _strip_prefix(raw)

        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return transform(match.group(1))

        if re.search(r"\banswer\b", raw, re.IGNORECASE):
            if i + 1 < len(lines):
                nxt = _strip_prefix(lines[i + 1])
                match = re.search(pattern, nxt, re.IGNORECASE)
                if match:
                    return transform(match.group(1))
            if i > 0:
                prv = _strip_prefix(lines[i - 1])
                match = re.search(pattern, prv, re.IGNORECASE)
                if match:
                    return transform(match.group(1))
    return ""


def _clean_freeform(s: str) -> str:
    s = s.strip().strip("\"'")
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_last_boxed(text: str) -> Optional[str]:
    """Return the innermost content of the last \\boxed{...} block."""
    matches = list(_BOXED_OPEN_RE.finditer(text or ""))
    for match in reversed(matches):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1]
    return None


def _fallback_answer(text: str) -> str:
    candidate = _extract_from_lines(text, r"(.+)", _clean_freeform)
    return _clean_freeform(candidate)


def reward_answer_only(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
) -> float:
    """
    Compare the boxed answer (if available) against the provided ground truth.
    Falls back to a generic free-form extraction when no boxed answer exists.
    """
    if ground_truth is None:
        return 0.0

    gt_answer = _clean_freeform(str(ground_truth))
    if not gt_answer:
        return 0.0

    pred_answer = _extract_last_boxed(solution_str or "")
    if pred_answer:
        pred_answer = _clean_freeform(pred_answer)
    else:
        pred_answer = _fallback_answer(solution_str or "")

    if not pred_answer:
        return 0.0

    return 1.0 if pred_answer == gt_answer else 0.0


__all__ = ["reward_answer_only"]

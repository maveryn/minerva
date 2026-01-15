"""Shared helpers for building ACR prompts."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def dedupe_labels(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def extract_gold_labels(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple, set)):
        return dedupe_labels(ground_truth)
    if isinstance(ground_truth, dict):
        for key in ("labels", "label", "technique_id", "tactic_ids", "mitigation_ids", "cwe_ids", "capec_id"):
            value = ground_truth.get(key)
            if isinstance(value, (list, tuple, set)):
                return dedupe_labels(value)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
    if isinstance(ground_truth, str) and ground_truth.strip():
        return [ground_truth.strip()]
    return []


def _append_block(messages: List[Dict[str, Any]], block: str) -> bool:
    if not messages:
        return False
    target_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            target_idx = idx
            break
    if target_idx is None:
        target_idx = len(messages) - 1
    content = messages[target_idx].get("content", "")
    if not isinstance(content, str):
        return False
    content = content.rstrip()
    if content:
        messages[target_idx]["content"] = f"{content}\n\n{block}"
    else:
        messages[target_idx]["content"] = block
    return True


def build_acr_block(
    gold_labels: List[str],
    details_text: Optional[str],
    *,
    enforce_no_id: bool,
    reasoning_hint: Optional[str] = None,
) -> str:
    labels_block = "\n".join(f"- {label}" for label in gold_labels)
    reasoning_hint = (reasoning_hint or "").strip()

    lines = [
        "You are generating a reasoning trace for training.",
        "",
        "GROUND_TRUTH_LABELS:",
        labels_block,
        "",
        "Instructions:",
        "- Write a short reasoning that would justify selecting the correct label(s) from the input.",
    ]
    if reasoning_hint:
        lines.append(f"- {reasoning_hint}")
    lines.extend(
        [
            "- Do NOT say or imply that the answer was provided (no phrases like \"given the answer\", \"based on the provided label\", \"ground truth\", etc.).",
            "- End with the final answer in the same format required by the original task.",
        ]
    )
    if details_text is not None:
        if not details_text:
            details_text = "(details omitted)"
        lines.insert(5, "")
        lines.insert(6, "CANONICAL_LABEL_DETAILS:")
        lines.insert(7, details_text)
    return "\n".join(lines)


def try_build_acr_messages(
    base_messages: List[Dict[str, Any]],
    gold_labels: List[str],
    details_text: Optional[str],
    *,
    max_details_chars: int,
    prompt_too_long: Callable[[List[Dict[str, Any]]], bool],
    enforce_no_id: bool,
    reasoning_hint: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    attempts: List[Optional[int]] = []
    if details_text is None:
        attempts.append(None)
    else:
        if max_details_chars and max_details_chars > 0:
            attempts.append(int(max_details_chars))
        attempts.append(0)
    for limit in attempts:
        if limit is None:
            trimmed = None
        else:
            trimmed = details_text or ""
            if limit > 0 and len(trimmed) > limit:
                trimmed = trimmed[:limit].rsplit(" ", 1)[0] or trimmed[:limit]
                trimmed = trimmed.rstrip() + "..."
            elif limit == 0:
                trimmed = ""
        trial_messages = copy.deepcopy(base_messages)
        block = build_acr_block(gold_labels, trimmed, enforce_no_id=enforce_no_id, reasoning_hint=reasoning_hint)
        if _append_block(trial_messages, block):
            if not prompt_too_long(trial_messages):
                return trial_messages, False, trimmed
    return base_messages, True, ""


__all__ = [
    "dedupe_labels",
    "extract_gold_labels",
    "build_acr_block",
    "try_build_acr_messages",
]

"""Helpers for building SQL-R1 answer-conditioned reasoning prompts."""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Optional, Tuple


_ASSISTANT_MARKER = "\n<|im_end|>\n<|im_start|>assistant\n<think>"


def is_sql_r1_preformatted_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) != 1:
        return False
    item = messages[0]
    if not isinstance(item, dict):
        return False
    content = item.get("content")
    if not isinstance(content, str):
        return False
    text = content.strip()
    return text.startswith("<|im_start|>") and "<|im_start|>assistant" in text and "<think>" in text


def _append_block_to_prompt_text(prompt_text: str, block: str) -> Optional[str]:
    if not prompt_text or not block:
        return None
    idx = prompt_text.rfind(_ASSISTANT_MARKER)
    if idx == -1:
        return None
    prefix = prompt_text[:idx].rstrip()
    suffix = prompt_text[idx:]
    return f"{prefix}\n\n{block}{suffix}"


def build_sql_r1_acr_block(gold_sql: str) -> str:
    gold_sql = (gold_sql or "").strip()
    lines = [
        "You are generating a reasoning trace for training.",
        "",
        "GROUND_TRUTH_SQL:",
        "```sql",
        gold_sql,
        "```",
        "",
        "Instructions:",
        "- Write a concise reasoning trace that explains how the question maps to the schema, joins, filters, aggregations, and ordering.",
        "- Use the question and schema as evidence.",
        "- Do NOT say or imply that the SQL was provided (no phrases like \"given SQL\", \"ground truth\", \"reference query\", or similar).",
        "- Do NOT quote or restate the full SQL query inside the reasoning trace.",
        "- End with the final answer in the same format required by the original task, with the SQL inside ```sql``` fences.",
    ]
    return "\n".join(lines)


def try_build_sql_r1_acr_messages(
    base_messages: list[dict[str, Any]],
    gold_sql: str,
    *,
    prompt_too_long: Callable[[list[dict[str, Any]]], bool],
) -> Tuple[list[dict[str, Any]], bool]:
    if not is_sql_r1_preformatted_messages(base_messages):
        return base_messages, True

    gold_sql = re.sub(r"\s+", " ", str(gold_sql or "")).strip()
    if not gold_sql:
        return base_messages, True

    prompt_text = str(base_messages[0].get("content") or "")
    updated_text = _append_block_to_prompt_text(prompt_text, build_sql_r1_acr_block(gold_sql))
    if not updated_text:
        return base_messages, True

    trial_messages = copy.deepcopy(base_messages)
    trial_messages[0]["content"] = updated_text
    if prompt_too_long(trial_messages):
        return base_messages, True
    return trial_messages, False


__all__ = [
    "build_sql_r1_acr_block",
    "is_sql_r1_preformatted_messages",
    "try_build_sql_r1_acr_messages",
]

"""ACR reward for SQL-R1 answer-conditioned reasoning traces."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional


try:
    from rlvr.verl.utils.reward_score.reward_sql_r1 import reward_sql_r1 as base_reward_sql_r1
except Exception:  # pragma: no cover - fallback for script imports
    try:
        from reward_sql_r1 import reward_sql_r1 as base_reward_sql_r1  # type: ignore
    except Exception:  # pragma: no cover
        base_reward_sql_r1 = None


DEFAULT_BANNED_PHRASES = [
    "given sql",
    "provided sql",
    "supplied sql",
    "attached sql",
    "reference sql",
    "reference query",
    "gold sql",
    "gold query",
    "ground truth sql",
    "ground truth query",
    "correct sql",
    "correct query",
    "the query is provided",
    "the sql is provided",
    "the answer is provided",
    "based on the provided sql",
    "based on the reference query",
    "as shown in the provided sql",
    "since the sql is given",
    "copy the provided query",
]

DEFAULT_BANNED_REGEXES = [
    r"\b(given|provided|supplied|attached)\b.{0,40}\b(sql|query)\b",
    r"\b(sql|query)\b.{0,40}\b(given|provided|supplied|attached)\b",
    r"\b(reference|gold|ground truth|correct)\b.{0,40}\b(sql|query)\b",
    r"\b(sql|query)\b.{0,40}\b(reference|gold|ground truth|correct)\b",
]

_QUESTION_RE = re.compile(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _extract_reasoning(solution_str: str) -> str:
    if not solution_str:
        return ""
    if "<think>" not in solution_str and ("</think>" in solution_str or "<answer>" in solution_str):
        solution_str = "<think>" + solution_str
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_question_from_messages(payload: Any) -> str:
    if isinstance(payload, list):
        for item in reversed(payload):
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                hits = _QUESTION_RE.findall(content)
                if hits:
                    return hits[-1].strip()
                return content.strip()
    elif isinstance(payload, str):
        hits = _QUESTION_RE.findall(payload)
        if hits:
            return hits[-1].strip()
        return payload.strip()
    return ""


def _extract_question(extra_info: Optional[dict[str, Any]]) -> str:
    if not isinstance(extra_info, dict):
        return ""
    for key in ("acr_orig_prompt", "raw_prompt", "prompt"):
        question = _extract_question_from_messages(extra_info.get(key))
        if question:
            return question
    return ""


def _contains_banned_phrase(
    text: str,
    phrases: Iterable[str],
    *,
    regexes: Optional[Iterable[str]] = None,
    fuzzy: bool,
    threshold: float,
) -> bool:
    norm = _normalize_text(text)
    for pattern in regexes or []:
        if re.search(pattern, norm, re.IGNORECASE):
            return True
    for phrase in phrases:
        phrase_norm = _normalize_text(phrase)
        if not phrase_norm:
            continue
        if phrase_norm in norm:
            return True
        if fuzzy:
            for sentence in re.split(r"[.!?]+", norm):
                sentence = sentence.strip()
                if sentence and SequenceMatcher(None, sentence, phrase_norm).ratio() >= threshold:
                    return True
    return False


def _word_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in _WORD_RE.findall(text)]


def _jaccard(tokens_a: list[str], tokens_b: list[str]) -> Optional[float]:
    if not tokens_a or not tokens_b:
        return None
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    denom = len(set_a | set_b)
    if denom <= 0:
        return None
    return len(set_a & set_b) / denom


def _longest_common_token_run(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    best = 0
    left_len = len(left)
    right_len = len(right)
    for i in range(left_len):
        for j in range(right_len):
            run = 0
            while i + run < left_len and j + run < right_len and left[i + run] == right[j + run]:
                run += 1
            if run > best:
                best = run
    return best


def _query_copy_leak(reasoning: str, gold_sql: str, *, min_common_run: int, min_ratio: float) -> bool:
    reasoning_norm = _normalize_text(reasoning)
    gold_norm = _normalize_text(gold_sql)
    if not reasoning_norm or not gold_norm:
        return False
    if len(gold_norm) >= 24 and gold_norm in reasoning_norm:
        return True

    reasoning_tokens = _word_tokens(reasoning_norm)
    gold_tokens = _word_tokens(gold_norm)
    if not reasoning_tokens or not gold_tokens:
        return False

    longest_run = _longest_common_token_run(gold_tokens, reasoning_tokens)
    if len(gold_tokens) >= 4 and longest_run >= len(gold_tokens):
        return True
    if longest_run >= max(1, int(min_common_run)):
        return True

    if min_ratio > 0:
        gold_token_set = set(gold_tokens)
        overlap = sum(1 for token in reasoning_tokens if token in gold_token_set)
        ratio = overlap / max(1, len(gold_tokens))
        if ratio >= float(min_ratio):
            return True
    return False


def reward_sql_r1_acr(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    r_correct: float = 0.2,
    leak_penalty: float = 0.5,
    banned_phrases: Optional[list[str]] = None,
    banned_regexes: Optional[list[str]] = None,
    use_fuzzy_leak_check: bool = False,
    fuzzy_threshold: float = 0.85,
    min_reasoning_chars: int = 100,
    min_question_overlap_jaccard: float = 0.0,
    query_copy_min_common_run: int = 12,
    query_copy_min_ratio: float = 0.8,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    **reward_kwargs: Any,
) -> dict[str, Any]:
    if base_reward_sql_r1 is None:
        raise RuntimeError("reward_sql_r1_acr requires reward_sql_r1 to be importable.")

    base_result = base_reward_sql_r1(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        return_dict=True,
        **reward_kwargs,
    )

    pred_sql = str(base_result.get("pred_sql") or "")
    gold_sql = str((ground_truth or {}).get("sql") or "")
    extracted = bool(pred_sql)
    exec_match = float(base_result.get("exec_match", 0.0) or 0.0)
    is_correct = exec_match >= 1.0 - 1e-6

    reasoning = _extract_reasoning(solution_str or "")
    phrases = banned_phrases if banned_phrases is not None else DEFAULT_BANNED_PHRASES
    regexes = banned_regexes if banned_regexes is not None else DEFAULT_BANNED_REGEXES
    banned_hit = _contains_banned_phrase(
        reasoning or solution_str or "",
        phrases,
        regexes=regexes,
        fuzzy=use_fuzzy_leak_check,
        threshold=fuzzy_threshold,
    )

    short_hit = False
    if min_reasoning_chars and len(reasoning.strip()) < int(min_reasoning_chars):
        short_hit = True

    query_copy_hit = _query_copy_leak(
        reasoning,
        gold_sql,
        min_common_run=int(query_copy_min_common_run),
        min_ratio=float(query_copy_min_ratio),
    )

    question_overlap_hit = False
    question_overlap = None
    if min_question_overlap_jaccard and min_question_overlap_jaccard > 0:
        question_text = _extract_question(extra_info)
        question_overlap = _jaccard(_word_tokens(question_text), _word_tokens(reasoning))
        if question_overlap is None or question_overlap < float(min_question_overlap_jaccard):
            question_overlap_hit = True

    leak_hit = bool(banned_hit or short_hit or query_copy_hit or question_overlap_hit)

    score = 0.0
    if extracted and exec_match > 0:
        score += float(r_correct) * exec_match
    if leak_hit:
        score -= float(leak_penalty)

    if score_min is not None:
        score = max(score, float(score_min))
    if score_max is not None:
        score = min(score, float(score_max))

    result = {
        "score": float(score),
        "acr_base_score": float(exec_match),
        "acr_extracted": bool(extracted),
        "acr_is_correct": bool(is_correct),
        "acr_leak_hit": bool(leak_hit),
        "acr_banned_phrase_hit": bool(banned_hit),
        "acr_short_hit": bool(short_hit),
        "acr_query_copy_hit": bool(query_copy_hit),
        "acr_exec_status": str(base_result.get("exec_status") or ""),
        "acr_exec_success": float(base_result.get("exec_success", 0.0) or 0.0),
        "acr_exec_match": float(exec_match),
        "acr_pred_sql": pred_sql,
    }
    if question_overlap is not None:
        result["acr_question_overlap"] = float(question_overlap)
        result["acr_question_overlap_hit"] = bool(question_overlap_hit)
    return result


__all__ = ["reward_sql_r1_acr"]

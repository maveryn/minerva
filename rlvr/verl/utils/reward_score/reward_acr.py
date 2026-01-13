"""Reward function for Answer-Conditioned Reasoning (ACR) outputs."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional


try:
    from rlvr.verl.utils.reward_score import reward_minerva as base_reward
except Exception:  # pragma: no cover - fallback for script imports
    try:
        import reward_minerva as base_reward  # type: ignore
    except Exception:  # pragma: no cover
        base_reward = None

DEFAULT_BANNED_PHRASES = [
    "given the answer",
    "based on the given answer",
    "based on the provided answer",
    "given the label",
    "based on the given label",
    "based on the provided label",
    "as provided in the answer",
    "as given in the answer",
    "as supplied in the answer",
    "as provided in the label",
    "as given in the label",
    "as supplied in the label",
    "the ground truth",
    "the answer is provided",
    "the answer was provided",
    "the label is provided",
    "the label was provided",
    "since you told me",
    "the correct answer is given",
    "the correct label is given",
    "the provided label",
    "the provided answer",
    "the answer you gave",
    "the label you gave",
]

_ID_REGEX = re.compile(r"\b(?:T\d{4}(?:\.\d{3})?|TA\d{4}|M\d{4}|DET-?\d{4}|CWE-\d+|CAPEC-\d+)\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _normalize_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    if _ID_REGEX.match(text):
        text = text.upper().replace("DET-", "DET")
    return text.upper()


def _extract_gold_labels(ground_truth: Any) -> list[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple, set)):
        return [_normalize_label(v) for v in ground_truth if str(v).strip()]
    if isinstance(ground_truth, dict):
        for key in ("labels", "label", "technique_id", "tactic_ids", "mitigation_ids", "cwe_ids", "capec_id"):
            value = ground_truth.get(key)
            if isinstance(value, (list, tuple, set)):
                return [_normalize_label(v) for v in value if str(v).strip()]
            if isinstance(value, str) and value.strip():
                return [_normalize_label(value)]
    if isinstance(ground_truth, str) and ground_truth.strip():
        return [_normalize_label(ground_truth)]
    return []


def _extract_predicted_base(data_source: str, solution_str: str) -> str:
    if base_reward is not None:
        extractor = getattr(base_reward, "_extract_predicted", None)
        if callable(extractor):
            return extractor(data_source, solution_str)
    try:
        from rlvr.verl.utils.reward_score.myreward_boxed import _extract_last_boxed, _fallback_answer
    except ImportError:
        from myreward_boxed import _extract_last_boxed, _fallback_answer  # type: ignore
    boxed = _extract_last_boxed(solution_str or "")
    if boxed:
        return boxed.strip()
    return _fallback_answer(solution_str or "")


def _extract_pred_labels(pred: str) -> list[str]:
    if not pred:
        return []
    hits = _ID_REGEX.findall(pred)
    if hits:
        return [_normalize_label(h) for h in hits]
    return [_normalize_label(pred)]


def _split_reasoning(output: str) -> str:
    if not output:
        return ""
    lines = output.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            return "\n".join(lines[:idx]).strip()
    return output.strip()


def _match_labels(gold: list[str], pred: list[str], policy: str) -> bool:
    if not gold or not pred:
        return False
    gold_set = set(gold)
    pred_set = set(pred)
    policy = (policy or "exact").lower()
    if policy == "subset":
        return pred_set.issubset(gold_set)
    if policy == "superset":
        return gold_set.issubset(pred_set)
    return gold_set == pred_set


def _contains_banned_phrase(text: str, phrases: Iterable[str], *, fuzzy: bool, threshold: float) -> bool:
    norm = _normalize_text(text)
    for phrase in phrases:
        p_norm = _normalize_text(phrase)
        if not p_norm:
            continue
        if p_norm in norm:
            return True
        if fuzzy:
            for sentence in re.split(r"[.!?]+", norm):
                if not sentence:
                    continue
                if SequenceMatcher(None, sentence.strip(), p_norm).ratio() >= threshold:
                    return True
    return False


def _id_leak_in_reasoning(reasoning: str, gold_labels: Iterable[str]) -> bool:
    if not reasoning:
        return False
    reasoning_norm = reasoning.upper()
    for label in gold_labels:
        norm = _normalize_label(label)
        if not norm:
            continue
        if len(norm) <= 2 and not _ID_REGEX.match(norm):
            continue
        if norm in reasoning_norm:
            return True
    return False


def _count_id_mentions(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not text:
        return counts
    for match in _ID_REGEX.findall(text):
        norm = _normalize_label(match)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
    return counts


def reward_acr(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    *,
    r_correct: float = 0.2,
    leak_penalty: float = 0.5,
    multilabel_match: str = "exact",
    banned_phrases: Optional[list[str]] = None,
    use_fuzzy_leak_check: bool = True,
    fuzzy_threshold: float = 0.85,
    enforce_no_id_in_reasoning: bool = True,
    max_id_mentions: Optional[int] = 3,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
) -> dict:
    gold_labels = _extract_gold_labels(ground_truth)
    pred_raw = _extract_predicted_base(data_source, solution_str or "")
    pred_labels = _extract_pred_labels(pred_raw)

    extracted = bool(pred_labels)
    base_score = None
    if base_reward is not None and hasattr(base_reward, "reward_minerva"):
        try:
            base_score = float(base_reward.reward_minerva(data_source, solution_str, ground_truth, extra_info))
        except Exception:
            base_score = None
    if base_score is None:
        is_correct = _match_labels(gold_labels, pred_labels, multilabel_match) if extracted else False
        base_score = 1.0 if is_correct else 0.0
    else:
        is_correct = base_score >= 1.0 - 1e-6

    phrases = banned_phrases if banned_phrases is not None else DEFAULT_BANNED_PHRASES
    leak_hit = _contains_banned_phrase(solution_str or "", phrases, fuzzy=use_fuzzy_leak_check, threshold=fuzzy_threshold)

    id_leak_hit = False
    if enforce_no_id_in_reasoning and gold_labels:
        reasoning = _split_reasoning(solution_str or "")
        id_leak_hit = _id_leak_in_reasoning(reasoning, gold_labels)

    id_overuse_hit = False
    id_overuse_max = 0
    if max_id_mentions is not None and max_id_mentions > 0 and gold_labels:
        counts = _count_id_mentions(solution_str or "")
        for label in gold_labels:
            if not _ID_REGEX.match(label):
                continue
            count = counts.get(_normalize_label(label), 0)
            if count > id_overuse_max:
                id_overuse_max = count
            if count > max_id_mentions:
                id_overuse_hit = True
                break
    if id_overuse_hit:
        leak_hit = True

    score = 0.0
    if extracted and base_score > 0:
        score += float(r_correct) * float(base_score)
    if leak_hit:
        score -= float(leak_penalty)

    if score_min is not None:
        score = max(score, float(score_min))
    if score_max is not None:
        score = min(score, float(score_max))

    return {
        "score": float(score),
        "acr_base_score": float(base_score),
        "acr_extracted": bool(extracted),
        "acr_is_correct": bool(is_correct),
        "acr_leak_hit": bool(leak_hit),
        "acr_id_leak_hit": bool(id_leak_hit),
        "acr_id_overuse_hit": bool(id_overuse_hit),
        "acr_id_overuse_max": int(id_overuse_max),
        "acr_pred_labels": pred_labels,
        "acr_gold_labels": gold_labels,
    }


__all__ = ["reward_acr"]

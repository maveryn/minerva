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
    "label reference",
    "label references",
    "label detail",
    "label details",
    "label text",
    "label description",
    "label reference text",
    "label reference description",
    "label detail text",
    "label detail description",
    "label details text",
    "label details description",
    "reference text",
    "reference description",
    "reference details",
    "reference info",
    "reference information",
    "reference material",
    "reference entry",
    "reference entry text",
    "reference entry description",
    "reference entry details",
    "catalog entry",
    "catalog description",
    "catalog details",
    "catalog text",
    "catalog information",
    "catalog reference",
    "manifestation text",
    "manifestation description",
    "manifestation details",
    "manifestation reference",
    "entry text",
    "entry description",
    "entry details",
    "entry information",
    "entry reference",
    "entry says",
    "entry states",
    "entry notes",
    "entry indicates",
    "entry mentions",
    "entry describes",
    "entry shows",
    "the label reference",
    "the label details",
    "the label description",
    "the reference",
    "the reference text",
    "the reference description",
    "the reference details",
    "the reference entry",
    "the catalog entry",
    "the catalog description",
    "the catalog details",
    "the catalog text",
    "the entry",
    "the entry text",
    "the entry description",
    "the entry details",
    "label says",
    "label states",
    "label notes",
    "label indicates",
    "label mentions",
    "label describes",
    "label shows",
    "label implies",
    "label suggests",
    "reference says",
    "reference states",
    "reference notes",
    "reference indicates",
    "reference mentions",
    "reference describes",
    "reference shows",
    "reference lists",
    "details say",
    "details state",
    "details note",
    "details indicate",
    "details mention",
    "details describe",
    "details show",
    "details list",
    "manifestation says",
    "manifestation states",
    "manifestation notes",
    "manifestation indicates",
    "manifestation mentions",
    "manifestation describes",
    "manifestation shows",
    "according to the label",
    "from the label",
    "according to the reference",
    "from the reference",
    "based on the reference",
    "based on the label",
    "per the reference",
    "per the label",
    "as described in the label",
    "as described in the reference",
    "as described in the details",
    "as stated in the label",
    "as stated in the reference",
    "as stated in the details",
    "from the details",
    "from the manifestation",
    "per the details",
    "per the manifestation",
    "based on the details",
    "as listed in the label",
    "as listed in the reference",
    "as listed in the details",
    "as listed in the catalog",
    "as listed in the entry",
    "cw id manifestation",
    "cwe entry",
    "cwe entry text",
    "cwe entry description",
    "cwe entry details",
    "cwe definition",
    "cwe description",
    "cwe reference",
    "attack technique entry",
    "attack technique description",
    "attack technique definition",
    "technique entry",
    "technique description text",
    "technique definition",
    "tactic entry",
    "tactic description text",
    "tactic definition",
    "mitigation entry",
    "mitigation description text",
    "mitigation definition",
    "attack pattern entry",
    "attack pattern description text",
    "attack pattern definition",
    "given reference",
    "provided reference",
    "supplied reference",
    "attached reference",
    "given details",
    "provided details",
    "supplied details",
    "attached details",
    "given manifestation",
    "provided manifestation",
    "supplied manifestation",
    "attached manifestation",
    "given label description",
    "provided label description",
    "supplied label description",
    "attached label description",
    "provided by the label",
    "provided in the label",
    "provided by the reference",
    "provided in the reference",
    "provided by the details",
    "provided in the details",
    "provided by the catalog",
    "provided in the catalog",
    "provided by the entry",
    "provided in the entry",
    "given by the label",
    "given in the label",
    "given by the reference",
    "given in the reference",
    "given by the details",
    "given in the details",
    "given by the catalog",
    "given in the catalog",
    "given by the entry",
    "given in the entry",
    "supplied by the label",
    "supplied in the label",
    "supplied by the reference",
    "supplied in the reference",
    "supplied by the details",
    "supplied in the details",
    "supplied by the catalog",
    "supplied in the catalog",
    "supplied by the entry",
    "supplied in the entry",
    "attached to the label",
    "attached to the reference",
    "attached to the details",
    "attached to the catalog",
    "attached to the entry",
    "given technique description",
    "provided technique description",
    "supplied technique description",
    "given attack technique description",
    "provided attack technique description",
    "supplied attack technique description",
    "given tactic description",
    "provided tactic description",
    "supplied tactic description",
    "given mitigation description",
    "provided mitigation description",
    "supplied mitigation description",
    "given attack pattern description",
    "provided attack pattern description",
    "supplied attack pattern description",
    "given cwe description",
    "provided cwe description",
    "supplied cwe description",
    "given cvss description",
    "provided cvss description",
    "supplied cvss description",
    "given technique list",
    "provided technique list",
    "supplied technique list",
    "given attack technique list",
    "provided attack technique list",
    "supplied attack technique list",
    "given tactic list",
    "provided tactic list",
    "supplied tactic list",
    "given mitigation list",
    "provided mitigation list",
    "supplied mitigation list",
    "given attack pattern list",
    "provided attack pattern list",
    "supplied attack pattern list",
    "given cwe list",
    "provided cwe list",
    "supplied cwe list",
    "given cvss vector",
    "provided cvss vector",
    "supplied cvss vector",
    "given cvss score",
    "provided cvss score",
    "supplied cvss score",
    "given technique id",
    "provided technique id",
    "supplied technique id",
    "given attack technique id",
    "provided attack technique id",
    "supplied attack technique id",
    "given tactic id",
    "provided tactic id",
    "supplied tactic id",
    "given mitigation id",
    "provided mitigation id",
    "supplied mitigation id",
    "given cwe id",
    "provided cwe id",
    "supplied cwe id",
    "given attack pattern id",
    "provided attack pattern id",
    "supplied attack pattern id",
    "candidate pool",
    "option list",
    "options list",
    "provided options",
    "given options",
    "supplied options",
    "options provided",
    "options given",
    "options supplied",
    "list of options",
    "list of candidates",
    "candidate list",
    "choices list",
    "above options",
    "below options",
    "the options above",
    "the options below",
    "the list above",
    "the list below",
    "the provided list",
    "the given list",
    "the supplied list",
    "the candidate list",
    "the candidates provided",
    "the candidates given",
    "the candidates supplied",
    "from the provided list",
    "from the given list",
    "from the supplied list",
    "based on the provided list",
    "based on the given list",
    "based on the supplied list",
    "using the provided list",
    "using the given list",
    "using the supplied list",
]

DEFAULT_BANNED_REGEXES = [
    r"\blabel\s+(reference|details|description|text)\b",
    r"\b(reference|details|manifestation)\b.{0,80}\b(given|provided|supplied|attached)\b",
    r"\b(given|provided|supplied|attached)\b.{0,80}\b(reference|details|manifestation)\b",
    r"\b(label|answer|option|options|candidate pool|candidate list|choices)\b.{0,80}\b(given|provided|supplied|attached)\b",
    r"\b(given|provided|supplied|attached)\b.{0,80}\b(label|answer|option|options|candidate pool|candidate list|choices)\b",
    r"\b(label|reference|details|manifestation)\b.{0,80}\b(says|states|notes|indicates|mentions|describes|shows|matches|aligns|corresponds)\b",
    r"\b(says|states|notes|indicates|mentions|describes|shows|matches|aligns|corresponds)\b.{0,80}\b(label|reference|details|manifestation)\b",
    r"\b(entry|catalog)\b.{0,80}\b(says|states|notes|indicates|mentions|describes|shows|lists|matches|aligns|corresponds)\b",
    r"\b(says|states|notes|indicates|mentions|describes|shows|lists|matches|aligns|corresponds)\b.{0,80}\b(entry|catalog)\b",
    r"\b(the )?(entry|catalog)\b.{0,80}\b(for|about)\b.{0,40}\b(technique|tactic|mitigation|attack pattern|attack technique|cwe|cvss|capec)\b",
    r"\b(according to|based on|as stated in|as noted in|as described in|per)\b.{0,80}\b(label|reference|details|manifestation)\b",
    r"\b(label|reference|details|manifestation)\b.{0,80}\b(according to|based on|as stated in|as noted in|as described in|per)\b",
    r"\b(this|it)\b.{0,20}\b(matches|aligns with|corresponds to|comes from)\b.{0,40}\b(label|reference|details|manifestation)\b",
    r"\b(label|reference|details|manifestation)\b.{0,40}\b(matches|aligns with|corresponds to|comes from)\b",
    r"\b(as listed in|as shown in|as stated in|as noted in|per)\b.{0,80}\b(entry|catalog|label|reference|details|manifestation)\b",
    r"\b(entry|catalog|label|reference|details|manifestation)\b.{0,80}\b(as listed in|as shown in|as stated in|as noted in|per)\b",
    r"\b(above|below|following)\b.{0,40}\b(options|list|candidates|choices)\b",
    r"\b(options|list|candidates|choices)\b.{0,40}\b(above|below|following)\b",
    r"\b(technique|tactic|mitigation|attack pattern|attack technique|cwe|cvss|capec)\s+(description|list|id|vector|score)\b.{0,80}\b(given|provided|supplied|attached)\b",
    r"\b(given|provided|supplied|attached)\b.{0,80}\b(technique|tactic|mitigation|attack pattern|attack technique|cwe|cvss|capec)\s+(description|list|id|vector|score)\b",
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
    banned_regexes: Optional[list[str]] = None,
    use_fuzzy_leak_check: bool = False,
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
    regexes = banned_regexes if banned_regexes is not None else DEFAULT_BANNED_REGEXES
    leak_hit = _contains_banned_phrase(
        solution_str or "",
        phrases,
        regexes=regexes,
        fuzzy=use_fuzzy_leak_check,
        threshold=fuzzy_threshold,
    )

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

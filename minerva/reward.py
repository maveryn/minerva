import re
from typing import Iterable, List, Optional, Set


def _normalize_id(val: str) -> str:
    return (val or "").strip().upper()


def binary_id(predicted: str, truth: str) -> float:
    """
    Exact-match reward for stable identifiers (CWE, CAPEC, ATT&CK IDs).
    """
    return 1.0 if _normalize_id(predicted) == _normalize_id(truth) else 0.0


def to_id_set(values: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for v in values or []:
        nv = _normalize_id(v)
        if nv:
            out.add(nv)
    return out


def f1_set(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    Multi-label F1 with safe zero handling.
    """
    pset = to_id_set(predicted)
    tset = to_id_set(truth)
    if not pset and not tset:
        return 1.0
    if not pset or not tset:
        return 0.0
    tp = len(pset & tset)
    precision = tp / len(pset) if pset else 0.0
    recall = tp / len(tset) if tset else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_technique_id(predicted: str, truth: str) -> float:
    """
    Technique-aware reward:
    - 1.0 if technique and sub-technique match exactly (case-insensitive).
    - 0.5 if the base technique (e.g., T1059) matches but sub-technique differs
      or is missing on either side.
    - 0.0 otherwise.
    """
    p = _normalize_id(predicted)
    t = _normalize_id(truth)
    if not p or not t:
        return 0.0
    if p == t:
        return 1.0
    p_base = p.split(".")[0]
    t_base = t.split(".")[0]
    p_has_sub = "." in p
    t_has_sub = "." in t
    if p_base == t_base and (p_has_sub or t_has_sub):
        return 0.5
    return 0.0


def reward_technique_id_only(predicted: str, truth: str) -> float:
    """
    Technique-only reward:
    - 1.0 if base technique (e.g., T1059) matches (case-insensitive).
    - 0.0 otherwise.
    """
    p = _normalize_id(predicted)
    t = _normalize_id(truth)
    if not p or not t:
        return 0.0
    return 1.0 if p.split(".")[0] == t.split(".")[0] else 0.0


def reward_technique_sub_id(predicted: str, truth: str) -> float:
    """
    Sub-technique reward:
    - 1.0 if technique and sub-technique match exactly (case-insensitive).
    - 0.0 otherwise.
    """
    return 1.0 if _normalize_id(predicted) == _normalize_id(truth) else 0.0


def reward_tactic_ids(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    Wrapper for tactic-set scoring (TA000x IDs) using multi-label F1.
    """
    return f1_set(predicted, truth)


def reward_mitigation_ids(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    Mitigation-set scoring (M#### IDs) using multi-label F1.
    """
    return f1_set(predicted, truth)


def reward_detection_id(predicted: str, truth: str) -> float:
    """
    Single detection strategy ID exact match.
    """
    return binary_id(predicted, truth)


def reward_cwe_ids(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    CWE-set scoring using multi-label F1.
    """
    return f1_set(predicted, truth)


def _parse_cvss(vector: str, metrics: List[str]) -> dict:
    vals: dict = {}
    if not vector:
        return vals
    for m in metrics:
        match = re.search(rf"{re.escape(m)}:([A-Z0-9]+)", vector, re.IGNORECASE)
        if match:
            vals[m] = match.group(1).upper()
    return vals


_CVSS_V31_RE = re.compile(r"(CVSS:3\.1/[^\s]+)", re.IGNORECASE)


def _extract_cvss_v31(vector: str) -> str:
    if not vector:
        return ""
    match = None
    for found in _CVSS_V31_RE.finditer(vector):
        match = found
    if match:
        return match.group(1).strip()
    return ""


def _cvss_v31_score(vector: str) -> Optional[float]:
    if not vector:
        return None
    try:
        from cvss import CVSS3
    except Exception:
        return None
    try:
        return float(CVSS3(vector).scores()[0])
    except Exception:
        return None


def _cvss_f1(predicted_vector: str, truth_vector: str, metrics: List[str]) -> float:
    if not truth_vector:
        return 0.0
    truth_vals = _parse_cvss(truth_vector, metrics)
    pred_vals = _parse_cvss(predicted_vector, metrics)
    if not truth_vals:
        return 0.0
    tp = 0
    total = len(metrics)
    for m in metrics:
        if pred_vals.get(m) and truth_vals.get(m) and pred_vals[m] == truth_vals[m]:
            tp += 1
    precision = tp / total
    recall = tp / total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_cvss_v31(predicted_vector: str, truth_vector: str, truth_score: float | None = None) -> float:
    """
    Reward for CVSS v3.1 base vectors using score-distance:
    1 - |truth_score - pred_score| / 4, clamped to [0, 1].
    Returns 0 if we cannot extract a valid CVSS v3.1 vector.
    """
    pred_vector = _extract_cvss_v31(str(predicted_vector or ""))
    if not pred_vector:
        return 0.0
    pred_score = _cvss_v31_score(pred_vector)
    if pred_score is None:
        return 0.0

    if truth_score is None:
        if isinstance(truth_vector, dict):
            truth_vector = truth_vector.get("cvss_v31_vector") or truth_vector.get("vector") or ""
        truth_score = _cvss_v31_score(str(truth_vector or ""))
    if truth_score is None:
        return 0.0

    diff = abs(float(truth_score) - float(pred_score))
    reward = 1.0 - (diff / 4.0)
    if reward < 0.0:
        return 0.0
    if reward > 1.0:
        return 1.0
    return float(reward)


def reward_cvss_v40(predicted_vector: str, truth_vector: str, truth_score: float | None = None) -> float:
    """
    Reward for CVSS v4.0 base metrics using per-metric F1 across:
    AV, AC, AT, PR, UI, VC, VI, VA, SC, SI, SA.
    """
    metrics = ["AV", "AC", "AT", "PR", "UI", "VC", "VI", "VA", "SC", "SI", "SA"]
    return _cvss_f1(predicted_vector, truth_vector, metrics)

import math
import re
from typing import Iterable, List, Set


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


def _cvss_score(vector: str) -> float | None:
    """
    Compute CVSS base score if the cvss library is present; otherwise None.
    """
    try:
        from cvss import CVSS3

        return CVSS3(vector).scores()[0]
    except Exception:
        return None


def cvss_reward(predicted_vector: str, truth_vector: str, truth_score: float | None) -> float:
    """
    Reward for CVSS v3.1 vectors. Full 1.0 on exact match; otherwise attempt
    to compute base score difference and return max(0, 1 - |pred - true|).
    """
    if not predicted_vector:
        return 0.0
    if _normalize_id(predicted_vector) == _normalize_id(truth_vector):
        return 1.0
    if truth_score is None:
        truth_score = _cvss_score(truth_vector)
    pred_score = _cvss_score(predicted_vector)
    if pred_score is None or truth_score is None:
        return 0.0
    return max(0.0, 1.0 - abs(pred_score - truth_score))


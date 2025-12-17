"""
Reward functions for Minerva/AthenaBench CTI datasets using data_source routing.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent

# Ensure local imports work when loaded as a standalone file
if __package__ in (None, ""):
    sys.path.append(str(THIS_DIR))
    sys.path.append(str(THIS_DIR.parent.parent.parent.parent))  # project root

try:
    from rlvr.verl.utils.reward_score.myreward_boxed import _clean_freeform, _extract_last_boxed, _fallback_answer
except ImportError:
    from myreward_boxed import _clean_freeform, _extract_last_boxed, _fallback_answer

# Extend path to import minerva reward utils
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    path_objs = [Path(p) for p in sys.path]
    if PROJECT_ROOT not in path_objs:
        sys.path.append(str(PROJECT_ROOT))
    import minerva.reward as minerva_reward
except Exception:  # pragma: no cover - fallback if import fails
    minerva_reward = None


# ---------------------------------------------------------------------------
# AthenaBench helpers (copied from _scratch utilities)

_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-ƒ?\"ƒ?\"]?\s*",
    re.IGNORECASE,
)


def _strip_prefix(s: str) -> str:
    return _PREFIX_RE.sub("", s).strip()


def _extract_from_lines(text: str, pattern: str, transform=lambda x: x) -> str:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
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


def _extract_rcm(text: str) -> str:
    return _extract_from_lines(text, r"(CWE-\d+)", lambda s: s.upper())


def _extract_vsp(text: str) -> str:
    return _extract_from_lines(text, r"(CVSS:3\.1/[^\s]+)", lambda s: s.strip())


def _extract_taa(text: str) -> str:
    return _extract_from_lines(text, r"(.+)", _clean_freeform)


def _extract_rms(text: str) -> str:
    line = _extract_from_lines(text, r"(.+)", _clean_freeform).upper()
    ids = re.findall(r"M\d{4}", line)
    return ", ".join(ids)


def _extract_ate(text: str) -> str:
    tid = _extract_from_lines(text, r"(T\d{4}(?:\.\d{3})?)", lambda s: s.upper())
    return tid


ATHENA_EXTRACTORS = {
    "athena-cti-rcm": _extract_rcm,
    "athena-cti-vsp": _extract_vsp,
    "athena-cti-taa": _extract_taa,
    "athena-cti-rms": _extract_rms,
    "athena-cti-ate": _extract_ate,
}


# ---------------------------------------------------------------------------
# Threat actor alias/related loading

def _load_csv_mapping(path: Path, key_field: str, val_field: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row.get(key_field, "").strip().lower()
            v = row.get(val_field, "").strip().lower()
            if not k or not v:
                continue
            mapping.setdefault(k, []).append(v)
            mapping.setdefault(v, []).append(k)
    return mapping


_ALIAS_DICT = _load_csv_mapping(Path(__file__).parent / "aliases.csv", "ThreatActor", "Alias")
_RELATED_DICT = _load_csv_mapping(Path(__file__).parent / "related_groups.csv", "ThreatActor", "RelatedGroup")


def _is_connected(actor1: str, actor2: str, alias_dict: Dict[str, List[str]], related_dict: Dict[str, List[str]]) -> str:
    """Return 'C' if aliases, 'P' if related, 'I' otherwise."""
    a1 = actor1.strip().lower()
    a2 = actor2.strip().lower()
    if not a1 or not a2:
        return "I"

    def bfs(start: str, target: str, allow_related: bool) -> bool:
        visited = set()
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cur == target:
                return True
            visited.add(cur)
            neighbours = alias_dict.get(cur, [])
            if allow_related:
                neighbours += related_dict.get(cur, [])
            for nxt in neighbours:
                if nxt not in visited:
                    queue.append(nxt)
        return False

    if bfs(a1, a2, allow_related=False):
        return "C"
    if bfs(a1, a2, allow_related=True):
        return "P"
    return "I"


# ---------------------------------------------------------------------------
# Helpers


def _extract_predicted(data_source: str, solution_str: str) -> str:
    # 1) try boxed
    boxed = _extract_last_boxed(solution_str or "")
    if boxed:
        return _clean_freeform(boxed)
    # 2) task-specific extractors
    extractor = ATHENA_EXTRACTORS.get(data_source)
    if extractor:
        candidate = extractor(solution_str or "")
        if candidate:
            return _clean_freeform(candidate)
    # 3) generic fallback
    return _clean_freeform(_fallback_answer(solution_str or ""))


def _normalize_list(pred: str, pattern: str) -> List[str]:
    return [m.upper() for m in re.findall(pattern, pred or "", re.IGNORECASE)]


def _split_candidates(text: str) -> List[str]:
    items: List[str] = []
    for part in re.split(r"[,\n;]+", text or ""):
        cleaned = _clean_freeform(part)
        if cleaned:
            items.append(cleaned.upper())
    return items


def _truth_ids(truth, pattern: str) -> List[str]:
    """
    Normalize truth into a list of IDs matching pattern from strings or iterables.
    """
    if truth is None:
        return []
    if isinstance(truth, (list, tuple, set)):
        joined = " ".join([str(t) for t in truth])
        return _normalize_list(joined, pattern)
    return _normalize_list(str(truth), pattern)


def _extract_detection_ids(text: str) -> List[str]:
    ids = _truth_ids(text, r"DET-?\d{4}")
    if ids:
        return [i.replace("-", "") for i in ids]
    # try split tokens
    tokens = _split_candidates(text)
    out = []
    for tok in tokens:
        m = re.search(r"DET-?\d{4}", tok, re.IGNORECASE)
        if m:
            out.append(m.group(0).replace("-", "").upper())
    return out


def _get_truth(ground_truth, key: Optional[str] = None):
    if key and isinstance(ground_truth, dict):
        return ground_truth.get(key)
    return ground_truth


# ---------------------------------------------------------------------------
# Reward dispatcher


def reward_minerva(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """
    Route reward computation based on data_source.
    """
    pred = _extract_predicted(data_source, solution_str)

    # AthenaBench tasks
    if data_source == "athena-cti-rcm":
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        return 1.0 if pred.lower() == truth.lower() else 0.0
    if data_source == "athena-cti-ate":
        truth = _clean_freeform(str(_get_truth(ground_truth))).upper()
        return 1.0 if pred.split(".")[0].upper() == truth.split(".")[0].upper() and truth else 0.0
    if data_source == "athena-cti-rms":
        truth_vals = set(_truth_ids(_get_truth(ground_truth), r"M\d{4}"))
        pred_vals = set(_truth_ids(pred, r"M\d{4}"))
        if not truth_vals:
            return 0.0
        if not pred_vals:
            return 0.0
        tp = len(pred_vals & truth_vals)
        precision = tp / len(pred_vals) if pred_vals else 0.0
        recall = tp / len(truth_vals) if truth_vals else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    if data_source == "athena-cti-taa":
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        relation = _is_connected(pred, truth, _ALIAS_DICT, _RELATED_DICT)
        if relation == "C":
            return 1.0
        if relation == "P":
            return 0.5
        if pred and truth:
            if pred.strip().lower() == truth.strip().lower():
                return 1.0
            return 0.5
        return 0.0
    if data_source == "athena-cti-vsp":
        # Fallback simple compare of vectors/base metrics
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        return 1.0 if pred == truth and truth else 0.0

    # Minerva tasks: use reward_fn names
    if minerva_reward and data_source:
        fn = getattr(minerva_reward, data_source, None)
        if callable(fn):
            # pick the right truth value for common keys
            key_map = {
                "reward_technique_id": "technique_id",
                "reward_tactic_ids": "tactic_ids",
                "reward_mitigation_ids": "mitigation_ids",
                "reward_detection_id": "detection_id",
                "reward_cwe_ids": "cwe_ids",
                "reward_cvss_v31": "cvss_v31_vector",
                "reward_cvss_v40": "cvss_v4_vector",
                "binary_id": None,
            }
            truth_key = key_map.get(data_source)
            truth_val = _get_truth(ground_truth, truth_key)

            if data_source in {"reward_tactic_ids", "reward_mitigation_ids"}:
                if data_source == "reward_tactic_ids":
                    pred_vals = _truth_ids(pred, r"TA0?\d{4}")
                    if not pred_vals:
                        pred_vals = [p for p in _split_candidates(pred) if p.startswith("TA")]
                    truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"TA0?\d{4}")
                else:
                    pred_vals = _truth_ids(pred, r"M\d{4}")
                    if not pred_vals:
                        pred_vals = [p for p in _split_candidates(pred) if p.startswith("M")]
                    truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"M\d{4}")
                return fn(pred_vals, truth_vals)
            if data_source == "reward_cwe_ids":
                pred_vals = _truth_ids(pred, r"CWE-\d+")
                truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"CWE-\d+")
                return fn(pred_vals, truth_vals)
            if data_source == "reward_technique_id":
                match = re.search(r"T\d{4}(?:\.\d{3})?", pred, re.IGNORECASE)
                pred_val = match.group(0) if match else pred
                return fn(pred_val, truth_val)
            if data_source == "reward_detection_id":
                det_preds = _extract_detection_ids(pred) or _extract_detection_ids(solution_str)
                pred_val = det_preds[0] if det_preds else pred
                truth_candidates = _extract_detection_ids(truth_val) if truth_val else []
                truth_val_norm = truth_candidates[0] if truth_candidates else truth_val
                return fn(pred_val, truth_val_norm)
            if data_source == "reward_cvss_v31":
                return fn(pred, truth_val, None)
            if data_source == "reward_cvss_v40":
                return fn(pred, truth_val, None)
            return fn(pred, truth_val)

    # Fallback: use mathruler if available
    try:
        from mathruler.grader import grade_answer

        return float(grade_answer(pred, _get_truth(ground_truth)))
    except Exception:
        pass

    return 0.0


__all__ = ["reward_answer_only"]

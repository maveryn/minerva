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
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{", re.DOTALL)
_WORD_RE = re.compile(r"\b\w+\b")


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


def _extract_mcq(text: str) -> str:
    return _extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper())


def _normalize_seceval(text: str) -> str:
    candidate = _extract_from_lines(text or "", r"([A-D](?:[^A-D]*[A-D])*)", lambda s: s)
    if not candidate:
        candidate = text or ""
    letters = re.findall(r"[A-D]", candidate.upper())
    if not letters:
        return ""
    return "".join(sorted({letter.upper() for letter in letters}))


ATHENA_EXTRACTORS = {
    "athena-cti-rcm": _extract_rcm,
    "athena-cti-vsp": _extract_vsp,
    "athena-cti-taa": _extract_taa,
    "athena-cti-rms": _extract_rms,
    "athena-cti-ate": _extract_ate,
    "athena-cti-mcq": _extract_mcq,
    "athena-cti-mcq-3k": _extract_mcq,
    "athena-cti-ckt": _extract_mcq,
    "reward_threat_actor_name": _extract_taa,
    "reward_technique_detection_elastic": _extract_ate,
}

DETECTION_TECHNIQUE_MULTI = {
}

DETECTION_TECHNIQUE_SINGLE = {
    "reward_technique_detection_art",
    "reward_technique_detection_sigma",
    "reward_technique_detection_sigma_base",
    "reward_technique_detection_sentinel",
    "reward_technique_detection_splunk",
    "reward_technique_detection_elastic",
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


def _is_connected(actor1: str, actor2: str, alias_dict: Dict[str, List[str]]) -> str:
    """Return 'C' if aliases, 'I' otherwise."""
    a1 = actor1.strip().lower()
    a2 = actor2.strip().lower()
    if not a1 or not a2:
        return "I"

    def bfs(start: str, target: str) -> bool:
        visited = set()
        queue = [start]
        while queue:
            cur = queue.pop(0)
            if cur == target:
                return True
            visited.add(cur)
            neighbours = alias_dict.get(cur, [])
            for nxt in neighbours:
                if nxt not in visited:
                    queue.append(nxt)
        return False

    if bfs(a1, a2):
        return "C"
    return "I"


# ---------------------------------------------------------------------------
# Helpers


def _extract_predicted(data_source: str, solution_str: str, *, allow_fallback: bool = True) -> str:
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
    # 3) generic fallback (optional)
    if allow_fallback:
        return _clean_freeform(_fallback_answer(solution_str or ""))
    return ""


def _is_training_split(extra_info: object) -> bool:
    if not isinstance(extra_info, dict):
        return False
    if extra_info.get("is_train") is True:
        return True
    split = str(extra_info.get("split") or extra_info.get("split_name") or extra_info.get("data_split") or "").lower()
    return split in {"train", "training"}


def _extract_answer_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        raw = lines[i]
        if _PREFIX_RE.match(raw) or re.search(r"\banswer\b", raw, re.IGNORECASE):
            line = _strip_prefix(raw)
            if line and line != raw:
                return line
            if i + 1 < len(lines):
                nxt = _strip_prefix(lines[i + 1])
                if nxt:
                    return nxt
            if i > 0:
                prv = _strip_prefix(lines[i - 1])
                if prv:
                    return prv
    return ""


def _find_last_boxed_span(text: str) -> Optional[tuple[int, int]]:
    matches = list(_BOXED_OPEN_RE.finditer(text or ""))
    for match in reversed(matches):
        start = match.start()
        i = match.end()
        depth = 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return start, i
    return None


def _count_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


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


def reward_minerva(data_source: str, solution_str: str, ground_truth, extra_info=None, return_dict: bool = False):
    """
    Route reward computation based on data_source.
    """
    allow_fallback = not _is_training_split(extra_info)
    def finalize(score):
        if isinstance(score, dict):
            if return_dict:
                return dict(score)
            return score
        if return_dict:
            return {"score": float(score)}
        return float(score)

    if data_source == "reward_instruction_following":
        pred = solution_str or ""
    else:
        pred = _extract_predicted(data_source, solution_str, allow_fallback=allow_fallback)

    # AthenaBench tasks
    if data_source == "athena-cti-rcm":
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        return finalize(1.0 if pred.lower() == truth.lower() else 0.0)
    if data_source == "athena-cti-ate":
        truth = _clean_freeform(str(_get_truth(ground_truth))).upper()
        return finalize(1.0 if pred.split(".")[0].upper() == truth.split(".")[0].upper() and truth else 0.0)
    if data_source == "athena-cti-rms":
        truth_vals = set(_truth_ids(_get_truth(ground_truth), r"M\d{4}"))
        pred_vals = set(_truth_ids(pred, r"M\d{4}"))
        if not truth_vals:
            return finalize(0.0)
        if not pred_vals:
            return finalize(0.0)
        tp = len(pred_vals & truth_vals)
        precision = tp / len(pred_vals) if pred_vals else 0.0
        recall = tp / len(truth_vals) if truth_vals else 0.0
        if precision + recall == 0:
            return finalize(0.0)
        return finalize(2 * precision * recall / (precision + recall))
    if data_source == "athena-cti-taa":
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        relation = _is_connected(pred, truth, _ALIAS_DICT)
        return finalize(1.0 if relation == "C" else 0.0)
    if data_source == "athena-cti-vsp":
        truth = _clean_freeform(str(_get_truth(ground_truth)))
        truth_score = None
        if isinstance(extra_info, dict):
            score_val = extra_info.get("vector_score")
            if isinstance(score_val, (int, float)):
                truth_score = float(score_val)
            elif isinstance(score_val, str):
                try:
                    truth_score = float(score_val.strip())
                except ValueError:
                    truth_score = None
        if not minerva_reward or not hasattr(minerva_reward, "reward_cvss_v31"):
            raise RuntimeError("athena-cti-vsp requires minerva.reward.reward_cvss_v31 for scoring")
        return finalize(minerva_reward.reward_cvss_v31(pred, truth, truth_score, delta=7.7))
    if data_source in {"athena-cti-mcq", "athena-cti-mcq-3k", "athena-cti-ckt"}:
        truth = _clean_freeform(str(_get_truth(ground_truth))).upper()
        return finalize(1.0 if pred.upper() == truth and truth else 0.0)
    if data_source in {"seceval", "seceval-mini"}:
        truth = _normalize_seceval(str(_get_truth(ground_truth)))
        pred_norm = _normalize_seceval(solution_str or "")
        return finalize(1.0 if pred_norm == truth and truth else 0.0)

    # Minerva tasks: use reward_fn names
    if minerva_reward and data_source:
        fn = getattr(minerva_reward, data_source, None)
        if callable(fn):
            # pick the right truth value for common keys
            key_map = {
                "reward_technique_id": "technique_id",
                "reward_technique_id_only": "technique_id",
                "reward_technique_sub_id": "technique_id",
                "reward_technique_ids": "technique_ids",
                "reward_technique_detection_art": "technique_id",
                "reward_technique_detection_sigma": "technique_id",
                "reward_technique_detection_sigma_base": "technique_id",
                "reward_technique_detection_sentinel": "technique_id",
                "reward_technique_detection_splunk": "technique_id",
                "reward_technique_detection_elastic": "technique_id",
                "reward_tactic_ids": "tactic_ids",
                "reward_mitigation_ids": "mitigation_ids",
                "reward_detection_id": "detection_id",
                "reward_cwe_ids": "cwe_ids",
                "reward_cvss_v31": "cvss_v31_vector",
                "reward_cvss_v40": "cvss_v4_vector",
                "reward_capec_id": "capec_id",
                "reward_threat_actor_name": "threat_actor",
                "reward_instruction_following": None,
                "binary_id": None,
            }
            truth_key = key_map.get(data_source)
            truth_val = _get_truth(ground_truth, truth_key)

            if data_source in {"reward_tactic_ids", "reward_mitigation_ids"}:
                if data_source == "reward_tactic_ids":
                    if pred:
                        pred_vals = _truth_ids(pred, r"TA0?\d{4}")
                    elif allow_fallback:
                        pred_vals = _truth_ids(solution_str or "", r"TA0?\d{4}")
                    else:
                        pred_vals = _truth_ids(_extract_answer_line(solution_str), r"TA0?\d{4}")
                    if not pred_vals:
                        source_text = pred or solution_str or ""
                        pred_vals = [p for p in _split_candidates(source_text) if p.startswith("TA")]
                    truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(
                        truth_val, r"TA0?\d{4}"
                    )
                else:
                    if pred:
                        pred_vals = _truth_ids(pred, r"M\d{4}")
                    elif allow_fallback:
                        pred_vals = _truth_ids(solution_str or "", r"M\d{4}")
                    else:
                        pred_vals = _truth_ids(_extract_answer_line(solution_str), r"M\d{4}")
                    if not pred_vals:
                        source_text = pred or solution_str or ""
                        pred_vals = [p for p in _split_candidates(source_text) if p.startswith("M")]
                    truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"M\d{4}")
                return finalize(fn(pred_vals, truth_vals))
            if data_source == "reward_technique_ids" or data_source in DETECTION_TECHNIQUE_MULTI:
                if pred:
                    pred_vals = _truth_ids(pred, r"T\d{4}(?:\.\d{3})?")
                elif allow_fallback:
                    pred_vals = _truth_ids(solution_str or "", r"T\d{4}(?:\.\d{3})?")
                else:
                    pred_vals = _truth_ids(_extract_answer_line(solution_str), r"T\d{4}(?:\.\d{3})?")
                truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"T\d{4}(?:\.\d{3})?")
                return finalize(fn(pred_vals, truth_vals))
            if data_source == "reward_cwe_ids":
                if pred:
                    pred_vals = _truth_ids(pred, r"CWE-\d+(?:\.\d+)?")
                elif allow_fallback:
                    pred_vals = _truth_ids(solution_str or "", r"CWE-\d+(?:\.\d+)?")
                else:
                    pred_vals = _truth_ids(_extract_answer_line(solution_str), r"CWE-\d+(?:\.\d+)?")
                truth_vals = truth_val if isinstance(truth_val, (list, tuple, set)) else _truth_ids(truth_val, r"CWE-\d+")
                score = fn(pred_vals, truth_vals)
                if allow_fallback:
                    expected = len(set(truth_vals or []))
                    if expected and solution_str:
                        mention_source = solution_str
                        mentions = {
                            m.upper() for m in re.findall(r"CWE-\d+(?:\.\d+)?", mention_source or "", re.IGNORECASE)
                        }
                        total_mentions = len(mentions)
                        if total_mentions > expected:
                            score *= expected / total_mentions
                return finalize(score)
            if data_source in {"reward_technique_id", "reward_technique_id_only", "reward_technique_sub_id"} or data_source in DETECTION_TECHNIQUE_SINGLE:
                match = re.search(r"T\d{4}(?:\.\d{3})?", pred, re.IGNORECASE) if pred else None
                if not match and solution_str and allow_fallback:
                    match = re.search(r"T\d{4}(?:\.\d{3})?", solution_str, re.IGNORECASE)
                if not match and solution_str and not allow_fallback:
                    answer_line = _extract_answer_line(solution_str)
                    ids = re.findall(r"T\d{4}(?:\.\d{3})?", answer_line or "", re.IGNORECASE)
                    if len(ids) == 1:
                        match = re.search(r"T\d{4}(?:\.\d{3})?", answer_line, re.IGNORECASE)
                    else:
                        match = None
                if not match and not allow_fallback:
                    pred_val = ""
                else:
                    pred_val = match.group(0) if match else pred
                return finalize(fn(pred_val, truth_val))
            if data_source == "reward_detection_id":
                det_preds = _extract_detection_ids(pred)
                if not det_preds and allow_fallback:
                    det_preds = _extract_detection_ids(solution_str)
                if not det_preds and not allow_fallback:
                    det_preds = _extract_detection_ids(_extract_answer_line(solution_str))
                if len(det_preds) == 1:
                    pred_val = det_preds[0]
                else:
                    pred_val = ""
                truth_candidates = _extract_detection_ids(truth_val) if truth_val else []
                truth_val_norm = truth_candidates[0] if truth_candidates else (truth_val or "")
                return finalize(fn(pred_val, truth_val_norm))
            if data_source == "reward_cvss_v31":
                if not pred and solution_str:
                    if allow_fallback:
                        match = re.search(r"(CVSS:3\\.1/[^\\s]+)", solution_str, re.IGNORECASE)
                        if match:
                            pred = match.group(1).strip()
                    else:
                        answer_line = _extract_answer_line(solution_str)
                        matches = re.findall(r"(CVSS:3\\.1/[^\\s]+)", answer_line or "", re.IGNORECASE)
                        pred = matches[0].strip() if len(matches) == 1 else ""
                truth_score = None
                if isinstance(ground_truth, dict):
                    score_val = ground_truth.get("score")
                    if isinstance(score_val, (int, float)):
                        truth_score = float(score_val)
                return finalize(fn(pred, truth_val, truth_score, delta=10.0))
            if data_source == "reward_cvss_v40":
                if not pred and solution_str:
                    if allow_fallback:
                        match = re.search(r"(CVSS:4\\.0/[^\\s]+)", solution_str, re.IGNORECASE)
                        if match:
                            pred = match.group(1).strip()
                    else:
                        answer_line = _extract_answer_line(solution_str)
                        matches = re.findall(r"(CVSS:4\\.0/[^\\s]+)", answer_line or "", re.IGNORECASE)
                        pred = matches[0].strip() if len(matches) == 1 else ""
                return finalize(fn(pred, truth_val, None))
            if data_source == "reward_capec_id":
                match = re.search(r"CAPEC-\\d+", pred or "", re.IGNORECASE) if pred else None
                if not match and solution_str and allow_fallback:
                    match = re.search(r"CAPEC-\\d+", solution_str, re.IGNORECASE)
                if not match and solution_str and not allow_fallback:
                    answer_line = _extract_answer_line(solution_str)
                    matches = re.findall(r"CAPEC-\\d+", answer_line or "", re.IGNORECASE)
                    match = re.search(r"CAPEC-\\d+", answer_line, re.IGNORECASE) if len(matches) == 1 else None
                if not match and not allow_fallback:
                    pred_val = ""
                else:
                    pred_val = match.group(0).upper() if match else pred
                return finalize(fn(pred_val, truth_val))
            return finalize(fn(pred, truth_val))

    # Fallback: use mathruler if available
    try:
        from mathruler.grader import grade_answer

        return finalize(float(grade_answer(pred, _get_truth(ground_truth))))
    except Exception:
        pass

    return 0.0


__all__ = ["reward_answer_only"]

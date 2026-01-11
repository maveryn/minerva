"""Task specs and label helpers for TARBA retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class TaskSpec:
    label_type: Optional[str]
    is_multilabel: bool
    label_id_regex: Optional[re.Pattern[str]]


_RE_TECHNIQUE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
_RE_TACTIC = re.compile(r"\bTA\d{4}\b", re.IGNORECASE)
_RE_MITIGATION = re.compile(r"\bM\d{4}\b", re.IGNORECASE)
_RE_DETECTION = re.compile(r"\bDET-?\d{4}\b", re.IGNORECASE)
_RE_CWE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)
_RE_CAPEC = re.compile(r"\bCAPEC-\d+\b", re.IGNORECASE)

LABEL_TYPE_REGEX: dict[str, re.Pattern[str]] = {
    "attack_technique_id": _RE_TECHNIQUE,
    "attack_tactic_id": _RE_TACTIC,
    "mitigation_id": _RE_MITIGATION,
    "detection_id": _RE_DETECTION,
    "cwe_id": _RE_CWE,
    "capec_id": _RE_CAPEC,
}


TASK_SPECS: dict[str, TaskSpec] = {
    # Minerva reward_fn keys
    "reward_technique_id": TaskSpec("attack_technique_id", False, _RE_TECHNIQUE),
    "reward_technique_id_only": TaskSpec("attack_technique_id", False, _RE_TECHNIQUE),
    "reward_technique_sub_id": TaskSpec("attack_technique_id", False, _RE_TECHNIQUE),
    "reward_tactic_ids": TaskSpec("attack_tactic_id", True, _RE_TACTIC),
    "reward_mitigation_ids": TaskSpec("mitigation_id", True, _RE_MITIGATION),
    "reward_detection_id": TaskSpec("detection_id", False, _RE_DETECTION),
    "reward_cwe_ids": TaskSpec("cwe_id", True, _RE_CWE),
    "reward_cvss_v31": TaskSpec(None, False, None),
    "reward_cvss_v40": TaskSpec(None, False, None),
    # AthenaBench CTI tasks
    "athena-cti-ate": TaskSpec("attack_technique_id", False, _RE_TECHNIQUE),
    "athena-cti-rcm": TaskSpec("cwe_id", False, _RE_CWE),
    "athena-cti-rms": TaskSpec("mitigation_id", True, _RE_MITIGATION),
    "athena-cti-taa": TaskSpec("threat_actor_name", False, None),
    "athena-cti-vsp": TaskSpec(None, False, None),
}

_RC_TASK_SPECS: Optional[dict[str, dict[str, Any]]] = None


def _load_retrieval_candidate_specs() -> dict[str, dict[str, Any]]:
    global _RC_TASK_SPECS
    if _RC_TASK_SPECS is not None:
        return _RC_TASK_SPECS
    try:
        from minerva.analysis import retrieval_candidates

        _RC_TASK_SPECS = dict(retrieval_candidates.TASK_SPECS)
    except Exception:
        _RC_TASK_SPECS = {}
    return _RC_TASK_SPECS


def get_task_spec(data_source: str, ground_truth: Any | None = None) -> TaskSpec:
    spec = TASK_SPECS.get(str(data_source or "").strip())
    if spec is not None:
        return spec
    rc_specs = _load_retrieval_candidate_specs()
    rc_spec = rc_specs.get(str(data_source or "").strip())
    if isinstance(rc_spec, dict):
        label_type = rc_spec.get("label_type")
        gold_key = str(rc_spec.get("gold_key") or "")
        is_multilabel = gold_key.endswith("ids")
        return TaskSpec(label_type=label_type, is_multilabel=is_multilabel, label_id_regex=regex_for_label_type(label_type))
    is_multilabel = isinstance(ground_truth, (list, tuple, set))
    return TaskSpec(label_type=None, is_multilabel=is_multilabel, label_id_regex=None)


def regex_for_label_type(label_type: Optional[str]) -> Optional[re.Pattern[str]]:
    if not label_type:
        return None
    return LABEL_TYPE_REGEX.get(str(label_type))


def normalize_label(label_type: Optional[str], value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if label_type == "detection_id":
        return text.upper().replace("-", "")
    if label_type in {
        "attack_technique_id",
        "attack_tactic_id",
        "mitigation_id",
        "cwe_id",
        "capec_id",
    }:
        return text.upper()
    return text


def extract_labels_from_truth(ground_truth: Any, spec: TaskSpec) -> list[str]:
    if ground_truth is None:
        return []

    def _extract_from_text(text: str) -> list[str]:
        if spec.label_id_regex is None:
            return [normalize_label(spec.label_type, text)] if text.strip() else []
        return [normalize_label(spec.label_type, m.group(0)) for m in spec.label_id_regex.finditer(text)]

    values: list[str] = []
    if isinstance(ground_truth, (list, tuple, set)):
        for item in ground_truth:
            values.extend(_extract_from_text(str(item)))
    else:
        values.extend(_extract_from_text(str(ground_truth)))

    seen = set()
    out: list[str] = []
    for val in values:
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def extract_label_matches_any(text: str) -> list[tuple[str, str]]:
    """
    Extract all label IDs across known label types, preserving order of appearance.
    """
    if not text:
        return []
    matches = []
    seen = set()
    for label_type, pattern in LABEL_TYPE_REGEX.items():
        for match in pattern.finditer(text):
            norm = normalize_label(label_type, match.group(0))
            key = (label_type, norm)
            if not norm or key in seen:
                continue
            seen.add(key)
            matches.append((match.start(), label_type, norm))
    matches.sort(key=lambda item: item[0])
    return [(label_type, norm) for _, label_type, norm in matches]


__all__ = [
    "TaskSpec",
    "TASK_SPECS",
    "LABEL_TYPE_REGEX",
    "get_task_spec",
    "regex_for_label_type",
    "normalize_label",
    "extract_labels_from_truth",
    "extract_label_matches_any",
]

import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set


def _normalize_id(val: str) -> str:
    return (val or "").strip().upper()


def binary_id(predicted: str, truth: str) -> float:
    """
    Exact-match reward for stable identifiers (CWE, CAPEC, ATT&CK IDs).
    """
    return 1.0 if _normalize_id(predicted) == _normalize_id(truth) else 0.0


def reward_capec_id(predicted: str, truth: str) -> float:
    """
    Exact-match reward for CAPEC IDs.
    """
    return binary_id(predicted, truth)


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


def reward_technique_ids(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    Exact-match reward for technique ID sets.
    """
    return 1.0 if to_id_set(predicted) == to_id_set(truth) else 0.0


def reward_technique_detection_art(predicted: str, truth: str) -> float:
    return reward_technique_id(predicted, truth)


def reward_technique_detection_sigma(predicted: str, truth: str) -> float:
    return reward_technique_id(predicted, truth)


def reward_technique_detection_sigma_base(predicted: str, truth: str) -> float:
    return reward_technique_id_only(predicted, truth)


def reward_technique_detection_sentinel(predicted: str, truth: str) -> float:
    return reward_technique_id(predicted, truth)


def reward_technique_detection_splunk(predicted: str, truth: str) -> float:
    return reward_technique_id(predicted, truth)


def reward_technique_detection_elastic(predicted: str, truth: str) -> float:
    return reward_technique_id_only(predicted, truth)


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


def reward_instruction_following(predicted: str, truth: object) -> dict | float:
    """
    Instruction-following reward using IFEval checks.
    """
    if isinstance(truth, bytes):
        try:
            truth = truth.decode("utf-8")
        except Exception:
            return 0.0
    if isinstance(truth, str):
        try:
            truth = json.loads(truth)
        except Exception:
            return 0.0
    if not isinstance(truth, dict):
        return 0.0
    prompt = str(truth.get("prompt") or "")
    instruction_id_list = truth.get("instruction_id_list") or []
    kwargs_list = truth.get("kwargs") or []
    if hasattr(instruction_id_list, "tolist"):
        instruction_id_list = instruction_id_list.tolist()
    if hasattr(kwargs_list, "tolist"):
        kwargs_list = kwargs_list.tolist()
    if not prompt or not instruction_id_list:
        return 0.0
    if not isinstance(kwargs_list, list):
        kwargs_list = []
    cleaned_kwargs = []
    for kw in kwargs_list:
        if isinstance(kw, dict):
            cleaned_kwargs.append({k: v for k, v in kw.items() if v is not None})
        else:
            cleaned_kwargs.append({})
    kwargs_list = cleaned_kwargs
    if len(kwargs_list) < len(instruction_id_list):
        kwargs_list = list(kwargs_list) + [{} for _ in range(len(instruction_id_list) - len(kwargs_list))]
    try:
        from minerva.instruction_following_eval import evaluation_lib
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "instruction-following evaluation requires the IFEval dependencies; "
            "install absl, langdetect, nltk, immutabledict."
        ) from exc

    inp = evaluation_lib.InputExample(
        key=int(truth.get("key", -1)) if str(truth.get("key", "")).isdigit() else -1,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs_list,
    )
    output = evaluation_lib.test_instruction_following_strict(
        inp,
        {prompt: predicted or ""},
    )
    follow_list = output.follow_instruction_list or []
    instruction_rate = (sum(follow_list) / len(follow_list)) if follow_list else 0.0
    return {
        "score": float(output.follow_all_instructions),
        "ifeval_instruction_rate": float(instruction_rate),
        "ifeval_instruction_count": int(len(follow_list)),
    }


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
_CVSS3_CLASS = None
_CVSS3_IMPORT_ERROR: Exception | None = None


def _extract_cvss_v31(vector: str) -> str:
    if not vector:
        return ""
    match = None
    for found in _CVSS_V31_RE.finditer(vector):
        match = found
    if match:
        return match.group(1).strip()
    return ""


def _require_cvss_v31() -> type:
    global _CVSS3_CLASS, _CVSS3_IMPORT_ERROR
    if _CVSS3_CLASS is not None:
        return _CVSS3_CLASS
    if _CVSS3_IMPORT_ERROR is not None:
        raise RuntimeError("cvss library is required for CVSS v3.1 scoring; install with `pip install cvss`") from _CVSS3_IMPORT_ERROR
    try:
        from cvss import CVSS3
    except Exception as exc:
        _CVSS3_IMPORT_ERROR = exc
        raise RuntimeError("cvss library is required for CVSS v3.1 scoring; install with `pip install cvss`") from exc
    _CVSS3_CLASS = CVSS3
    return CVSS3


def _cvss_v31_score(vector: str) -> Optional[float]:
    if not vector:
        return None
    cvss_cls = _require_cvss_v31()
    try:
        return float(cvss_cls(vector).scores()[0])
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


def reward_cvss_v31(
    predicted_vector: str,
    truth_vector: str,
    truth_score: float | None = None,
    *,
    delta: float = 10.0,
) -> float:
    """
    Reward for CVSS v3.1 base vectors using score-distance:
    1 - |truth_score - pred_score| / delta, clamped to [0, 1].
    Returns 0 if we cannot extract a valid CVSS v3.1 vector.
    """
    _require_cvss_v31()
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
    denom = float(delta) if delta else 0.0
    if denom <= 0.0:
        return 0.0
    reward = 1.0 - (diff / denom)
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


_THREAT_ACTOR_LOOKUP_CACHE: Optional[dict] = None
_THREAT_ACTOR_LOOKUP_PATH: Optional[Path] = None
_THREAT_ACTOR_ALIAS_TO_CANONICAL: Optional[dict[str, str]] = None


def _normalize_actor_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return " ".join(cleaned.split())


def _candidate_lookup_paths() -> List[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        Path(p) for p in [
            root / "rlvr" / "mydata" / "minerva_base" / "threat_actor_lookup.json",
            root / "dataset" / "minerva_base" / "threat_actor_lookup.json",
            root / "dataset" / "minerva" / "threat_actor_lookup.json",
            root / "dataset" / "minerva_base_split" / "threat_actor_lookup.json",
        ]
    ]


def _load_threat_actor_lookup() -> dict:
    global _THREAT_ACTOR_LOOKUP_CACHE, _THREAT_ACTOR_LOOKUP_PATH, _THREAT_ACTOR_ALIAS_TO_CANONICAL
    env_value = os.environ.get("MINERVA_THREAT_ACTOR_LOOKUP", "").strip()
    env_path = Path(env_value) if env_value else None
    if env_path is not None and env_path.exists():
        path = env_path
    else:
        path = None
        for candidate in _candidate_lookup_paths():
            if candidate.exists():
                path = candidate
                break

    if path is None:
        _THREAT_ACTOR_LOOKUP_CACHE = {}
        _THREAT_ACTOR_ALIAS_TO_CANONICAL = {}
        _THREAT_ACTOR_LOOKUP_PATH = None
        return {}

    if _THREAT_ACTOR_LOOKUP_CACHE is not None and _THREAT_ACTOR_LOOKUP_PATH == path:
        return _THREAT_ACTOR_LOOKUP_CACHE

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _THREAT_ACTOR_LOOKUP_CACHE = {}
        _THREAT_ACTOR_ALIAS_TO_CANONICAL = {}
        _THREAT_ACTOR_LOOKUP_PATH = path
        return {}

    lookup: dict[str, set[str]] = {}
    alias_to_canonical: dict[str, str] = {}
    for row in payload.get("actors", []) if isinstance(payload, dict) else []:
        name = row.get("name") if isinstance(row, dict) else None
        if not name:
            continue
        canonical = _normalize_actor_name(name)
        if not canonical:
            continue
        names = [name]
        names.extend(row.get("aliases", []) or [])
        norm_names = {_normalize_actor_name(n) for n in names if n}
        norm_names = {n for n in norm_names if n}
        if not norm_names:
            continue
        lookup[canonical] = norm_names
        for alias in norm_names:
            alias_to_canonical.setdefault(alias, canonical)

    _THREAT_ACTOR_LOOKUP_CACHE = lookup
    _THREAT_ACTOR_ALIAS_TO_CANONICAL = alias_to_canonical
    _THREAT_ACTOR_LOOKUP_PATH = path
    return lookup


def reward_threat_actor_name(predicted: str, truth: str) -> float:
    """
    Reward for threat actor names that accepts aliases.
    """
    pred_norm = _normalize_actor_name(predicted)
    truth_norm = _normalize_actor_name(truth)
    if not pred_norm or not truth_norm:
        return 0.0

    lookup = _load_threat_actor_lookup()
    if not lookup:
        return 1.0 if pred_norm == truth_norm else 0.0

    alias_to_canonical = _THREAT_ACTOR_ALIAS_TO_CANONICAL or {}
    canonical = alias_to_canonical.get(truth_norm, truth_norm)
    allowed = lookup.get(canonical)
    if allowed:
        return 1.0 if pred_norm in allowed else 0.0
    return 1.0 if pred_norm == truth_norm else 0.0

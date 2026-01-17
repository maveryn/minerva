import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from minerva.logger import get_logger
from minerva.utils import dedupe_by_text, normalize_text, parse_date


MITRE_SRC = ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack")


def _external_id(obj: Dict[str, Any], prefixes: Optional[Tuple[str, ...]] = None) -> str:
    for ref in obj.get("external_references", []) or []:
        src = ref.get("source_name") or ""
        if src in MITRE_SRC:
            eid = ref.get("external_id", "")
            if prefixes:
                if any(eid.startswith(p) for p in prefixes):
                    return eid
            else:
                return eid
    return ""


def _first_sentence(text: str, max_len: int = 400) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = parts[0] if parts else text
    return out[:max_len]


def download_bundle(url: str, cache_path: Path, logger=None) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    if logger:
        logger.info("Downloading ATT&CK bundle from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    if logger:
        logger.info("Saved ATT&CK bundle -> %s", cache_path)
    return cache_path


def load_bundle(cfg: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Load ATT&CK bundle and return parsed dictionaries:
      - techniques: tid -> {name, description, tactics, platforms, mitigations, detection_strategies}
      - mitigations: mid -> {name}
      - tactics: tactic_shortname -> {id, name}
      - detection_strategies: det_id -> {name}
    """
    if logger is None:
        logger = get_logger("mitre")
    url = cfg.get("attack_url") or "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache_path = Path(cfg.get("cache_path") or "data/processed/mitre/enterprise-attack.json")
    bundle_path = download_bundle(url, cache_path, logger)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []

    tactics: Dict[str, Dict[str, str]] = {}
    mitigations: Dict[str, Dict[str, str]] = {}
    detection_strategies: Dict[str, Dict[str, str]] = {}
    techniques: Dict[str, Dict[str, Any]] = {}

    # Collect tactic shortname -> TA id
    for obj in objs:
        if obj.get("type") != "x-mitre-tactic":
            continue
        eid = _external_id(obj, prefixes=("TA",))
        if not eid:
            continue
        short = obj.get("x_mitre_shortname") or obj.get("name", "").lower().replace(" ", "-")
        tactics[short] = {"id": eid, "name": obj.get("name", "")}

    # Collect mitigations & detection strategies
    for obj in objs:
        t = obj.get("type")
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if t == "course-of-action":
            mid = _external_id(obj, prefixes=("M",))
            if mid:
                mitigations[mid] = {"name": obj.get("name", "")}
        elif t == "x-mitre-detection-strategy":
            det_id = _external_id(obj, prefixes=("DET",))
            if det_id:
                detection_strategies[det_id] = {"name": obj.get("name", "")}

    # Build technique details
    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = _external_id(obj, prefixes=("T",))
        if not tid:
            continue
        tactics_for_t: List[str] = []
        for phase in obj.get("kill_chain_phases", []) or []:
            short = phase.get("phase_name", "")
            tac = tactics.get(short)
            if tac:
                tactics_for_t.append(tac["id"])
        tactics_for_t = sorted(set(tactics_for_t))
        techniques[tid] = {
            "name": obj.get("name", ""),
            "description": _first_sentence(obj.get("description", "")),
            "tactics": tactics_for_t,
            "platforms": obj.get("x_mitre_platforms", []) or [],
            "mitigations": set(),
            "detection_strategies": set(),
        }

    # Map relationships: mitigates, detects
    id_map = {o["id"]: o for o in objs if o.get("id")}
    for rel in (o for o in objs if o.get("type") == "relationship"):
        if rel.get("relationship_type") not in ("mitigates", "detects"):
            continue
        src = id_map.get(rel.get("source_ref"))
        tgt = id_map.get(rel.get("target_ref"))
        if not src or not tgt:
            continue
        if rel["relationship_type"] == "mitigates":
            mid = _external_id(src, prefixes=("M",))
            tid = _external_id(tgt, prefixes=("T",))
            if mid and tid and tid in techniques:
                techniques[tid]["mitigations"].add(mid)
        elif rel["relationship_type"] == "detects":
            det_id = _external_id(src, prefixes=("DET",))
            tid = _external_id(tgt, prefixes=("T",))
            if det_id and tid and tid in techniques:
                techniques[tid]["detection_strategies"].add(det_id)

    # Freeze sets to lists for downstream JSON serialization
    for tid, obj in techniques.items():
        obj["mitigations"] = sorted(obj["mitigations"])
        obj["detection_strategies"] = sorted(obj["detection_strategies"])

    return {
        "techniques": techniques,
        "mitigations": mitigations,
        "tactics": tactics,
        "detection_strategies": detection_strategies,
    }


# ---------- procedure-scenario builder (RLVR-aligned) ----------

DEFAULT_PLATFORM = "Enterprise"


def _placeholder_for_source(source_ref: str, entities: Dict[str, Dict[str, Any]], logger) -> Tuple[str, str]:
    placeholder = "An entity"
    if source_ref.startswith(("malware--", "tool--")):
        placeholder = "A malware"
    elif source_ref.startswith("intrusion-set--"):
        placeholder = "A threat actor"
    elif source_ref.startswith("campaign--"):
        placeholder = "A campaign"
    else:
        if source_ref:
            logger.debug("Unhandled procedure source_ref prefix: %s", source_ref)
    name = ""
    entity = entities.get(source_ref)
    if entity:
        name = entity.get("name", "") or ""
    return placeholder, name


def _sanitize_procedure(description: str, placeholder: str, entity_name: str) -> str:
    if not description:
        return ""
    text = re.sub(r"\(Citation:[^)]+\)", "", description)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    replaced = False
    if entity_name:
        placeholder_no_article = re.sub(r"^(?:A|An|The)\s+", "", placeholder, flags=re.IGNORECASE).strip()
        pattern = re.compile(rf"(?i)(\b(?:the|a|an)\s+)?\b{re.escape(entity_name)}\b('s)?")
        def _swap_first(match: re.Match) -> str:
            article = match.group(1) or ""
            possessive = match.group(2) or ""
            if possessive:
                if article:
                    return f"{article}{placeholder_no_article}'s"
                return f"{placeholder}'s"
            if article:
                return f"{article}{placeholder_no_article}"
            return placeholder
        text, count = pattern.subn(_swap_first, text, count=1)
        replaced = count > 0
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if not replaced and placeholder not in text:
        text = f"{placeholder} {text}"
    text = re.sub(r"\s+", " ", text).strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _normalize_platforms(obj: Dict[str, Any]) -> List[str]:
    plats = obj.get("x_mitre_platforms", []) or []
    return [p for p in plats if p]


def _normalize_tactics(obj: Dict[str, Any], tactic_map: Dict[str, Dict[str, str]]) -> List[str]:
    phases = obj.get("kill_chain_phases", []) or []
    out: List[str] = []
    for p in phases:
        short = p.get("phase_name", "")
        if short and short in tactic_map:
            out.append(tactic_map[short]["id"])
    return sorted(set(out))


def collect_procedure_scenarios(cfg: Dict[str, Any], logger=None) -> List[Dict[str, Any]]:
    """
    Build scenario records from ATT&CK procedure relationships (relationship_type=='uses')
    rather than technique descriptions. This mirrors RLVR flow and yields richer,
    source-grounded scenario text.
    """
    if logger is None:
        logger = get_logger("mitre-procedures")

    url = cfg.get("attack_url") or "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache_path = Path(cfg.get("cache_path") or "dataset/mitre/enterprise-attack.json")
    min_desc_chars = int(cfg.get("min_desc_chars", 0))
    max_items = int(cfg.get("max_items", 0))
    seed = int(cfg.get("seed", 1337))
    start_time = cfg.get("start_time")
    end_time = cfg.get("end_time")

    rng = random.Random(seed)

    bundle_path = download_bundle(url, cache_path, logger)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objects = data.get("objects", []) or []

    tactics: Dict[str, Dict[str, str]] = {}
    attack_patterns: Dict[str, Dict[str, Any]] = {}
    mitigations: Dict[str, Dict[str, Any]] = {}
    detection_strategies: Dict[str, Dict[str, Any]] = {}
    entities: Dict[str, Dict[str, Any]] = {}
    entity_types = {"intrusion-set", "campaign", "malware", "tool"}

    for obj in objects:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        typ = obj.get("type")
        if typ == "x-mitre-tactic":
            eid = _external_id(obj, prefixes=("TA",))
            short = obj.get("x_mitre_shortname") or obj.get("name", "").lower().replace(" ", "-")
            if eid and short:
                tactics[short] = {"id": eid, "name": obj.get("name", "")}
        elif typ == "attack-pattern":
            eid = _external_id(obj, prefixes=("T",))
            if eid:
                attack_patterns[obj["id"]] = obj
        elif typ == "course-of-action":
            eid = _external_id(obj, prefixes=("M",))
            if eid:
                mitigations[obj["id"]] = obj
        elif typ == "x-mitre-detection-strategy":
            eid = _external_id(obj, prefixes=("DET",))
            if eid:
                detection_strategies[obj["id"]] = obj
        elif typ in entity_types:
            entities[obj["id"]] = obj

    tech_to_mits: Dict[str, Set[str]] = {}
    tech_to_dets: Dict[str, Set[str]] = {}
    for rel in (o for o in objects if o.get("type") == "relationship"):
        rtype = rel.get("relationship_type")
        src = rel.get("source_ref")
        tgt = rel.get("target_ref")
        if rtype == "mitigates" and src in mitigations and tgt in attack_patterns:
            mid = _external_id(mitigations[src], prefixes=("M",))
            tid = _external_id(attack_patterns[tgt], prefixes=("T",))
            if mid and tid:
                tech_to_mits.setdefault(tid, set()).add(mid)
        if rtype == "detects" and src in detection_strategies and tgt in attack_patterns:
            det = _external_id(detection_strategies[src], prefixes=("DET",))
            tid = _external_id(attack_patterns[tgt], prefixes=("T",))
            if det and tid:
                tech_to_dets.setdefault(tid, set()).add(det)

    def normalize(dt_str: str | None) -> datetime | None:
        dt = parse_date(dt_str) if dt_str else None
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    start_dt = normalize(start_time)
    end_dt = normalize(end_time)

    logger.info("Selecting procedure relationships for scenarios")
    candidates: List[Dict[str, Any]] = []
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "uses":
            continue
        target_ref = rel.get("target_ref")
        if not target_ref or target_ref not in attack_patterns:
            continue
        technique = attack_patterns[target_ref]
        tid = _external_id(technique, prefixes=("T",))
        if not tid:
            continue
        mitigation_set = tech_to_mits.get(tid, set())
        detection_set = tech_to_dets.get(tid, set())
        if not mitigation_set:
            # Require at least one mitigation to keep reward verifiable/hard
            continue

        ts_raw = rel.get("modified") or rel.get("created") or ""
        ts_dt = normalize(ts_raw)
        if start_dt and ts_dt and ts_dt < start_dt:
            continue
        if end_dt and ts_dt and ts_dt > end_dt:
            continue

        source_ref = rel.get("source_ref") or ""
        placeholder, entity_name = _placeholder_for_source(source_ref, entities, logger)
        sanitized = _sanitize_procedure(rel.get("description", ""), placeholder, entity_name)
        if not sanitized:
            continue
        if min_desc_chars and len(sanitized) < min_desc_chars:
            continue

        platforms = _normalize_platforms(technique) or [DEFAULT_PLATFORM]
        tactics_for_t = _normalize_tactics(technique, tactics)
        full_desc = (technique.get("description") or "").replace("\\r", " ").replace("\\n", " ").strip()
        short_desc = _first_sentence(full_desc, max_len=400) if full_desc else ""
        ts_iso = ts_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if ts_dt else ts_raw

        candidates.append(
            {
                "technique_id": tid,
                "technique_name": technique.get("name", ""),
                "platforms": platforms,
                "tactics": tactics_for_t,
                "description": short_desc or full_desc,
                "scenario": sanitized,
                "mitigations": sorted(mitigation_set),
                "detection_strategies": sorted(detection_set),
                "timestamp": ts_iso,
                "metadata": {
                    "attack_bundle_source": str(bundle_path),
                    "procedure_source_ref": source_ref,
                    "procedure_relationship_id": rel.get("id") or "",
                },
            }
        )

    logger.info("Procedure candidates after filtering: %d", len(candidates))

    # Deduplicate by scenario text (prefer newer timestamp)
    candidates = dedupe_by_text(
        candidates,
        text_fn=lambda r: r.get("scenario", ""),
        prefer_ts_fn=lambda r: r.get("timestamp"),
    )

    if max_items and len(candidates) > max_items:
        logger.info("Sampling %d of %d procedure scenarios (max_items=%d)", max_items, len(candidates), max_items)
        rng.shuffle(candidates)
        candidates = candidates[:max_items]

    return candidates

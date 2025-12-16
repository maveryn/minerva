import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from minerva.logger import get_logger


CAPEC_URL = "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"
CAPEC_PREFIX = "CAPEC-"


def _external_id(obj: Dict[str, Any], source: str, prefix: str = "") -> Optional[str]:
    for ref in obj.get("external_references", []) or []:
        src = (ref.get("source_name") or "").lower()
        if src != source.lower():
            continue
        eid = ref.get("external_id") or ""
        if prefix and not eid.startswith(prefix):
            continue
        return eid
    return None


def download_bundle(url: str, cache_path: Path, logger=None) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path
    if logger:
        logger.info("Downloading CAPEC bundle from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    if logger:
        logger.info("Saved CAPEC bundle -> %s", cache_path)
    return cache_path


def load_bundle(cfg: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Load CAPEC bundle. Returns:
      - patterns: capec_id -> {name, description, cwe_ids, attack_techniques, example_instances, parent_ids}
      - cwe_to_capec: cwe_id -> set of capec_ids
    """
    if logger is None:
        logger = get_logger("capec")

    url = cfg.get("capec_url") or CAPEC_URL
    cache_path = Path(cfg.get("cache_path") or "data/processed/capec/stix-capec.json")
    bundle_path = download_bundle(url, cache_path, logger)
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []

    patterns: Dict[str, Dict[str, Any]] = {}
    cwe_to_capec: Dict[str, Set[str]] = {}
    tid_regex = re.compile(r"T\d{4}(?:\.\d{3})?")

    stix_to_capec: Dict[str, str] = {}
    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        capec_id = _external_id(obj, "capec", CAPEC_PREFIX)
        if capec_id:
            stix_to_capec[obj.get("id")] = capec_id

    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_capec_status", "").lower() == "deprecated":
            continue
        capec_id = _external_id(obj, "capec", CAPEC_PREFIX)
        if not capec_id:
            continue
        cwe_ids: List[str] = []
        attack_refs: List[str] = []
        for ref in obj.get("external_references", []) or []:
            src = (ref.get("source_name") or "").lower()
            eid = ref.get("external_id") or ""
            if src == "cwe" and eid:
                cwe_ids.append(eid)
            elif tid_regex.match(eid):
                attack_refs.append(eid)
        for cwe in cwe_ids:
            cwe_to_capec.setdefault(cwe, set()).add(capec_id)
        example_instances = []
        for ex in obj.get("x_capec_example_instances") or []:
            text = re.sub(r"<[^>]+>", " ", str(ex))
            text = " ".join(text.split())
            if text:
                example_instances.append(text)
        parent_ids: List[str] = []
        for ref in obj.get("x_capec_child_of_refs") or []:
            pid = stix_to_capec.get(ref)
            if pid:
                parent_ids.append(pid)
        patterns[capec_id] = {
            "name": obj.get("name", ""),
            "description": (obj.get("description") or "").strip(),
            "cwe_ids": sorted(set(cwe_ids)),
            "attack_techniques": sorted(set(attack_refs)),
            "example_instances": example_instances,
            "parent_ids": parent_ids,
        }

    return {
        "patterns": patterns,
        "cwe_to_capec": {k: sorted(v) for k, v in cwe_to_capec.items()},
    }

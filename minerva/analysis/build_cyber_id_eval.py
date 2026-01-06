import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva.utils import load_yaml
from minerva.data_sources.capec import download_bundle as download_capec_bundle
from minerva.data_sources.mitre import (
    MITRE_SRC,
    download_bundle as download_mitre_bundle,
)


ALLOWED_ID_TYPES = {
    "ATTACK_TECHNIQUE",
    "ATTACK_SUBTECHNIQUE",
    "ATTACK_MITIGATION",
    "CWE",
    "CAPEC",
}

CATEGORY_MAP_DESC = {
    "ATTACK_TECHNIQUE": "MITRE ATT&CK Technique ID (Description)",
    "ATTACK_SUBTECHNIQUE": "MITRE ATT&CK Sub-technique ID (Description)",
    "ATTACK_MITIGATION": "MITRE ATT&CK Mitigation ID (Description)",
    "CWE": "CWE ID (Description)",
    "CAPEC": "CAPEC ID (Description)",
}

CATEGORY_MAP_TITLE = {
    "ATTACK_TECHNIQUE": "MITRE ATT&CK Technique ID (Title)",
    "ATTACK_SUBTECHNIQUE": "MITRE ATT&CK Sub-technique ID (Title)",
    "ATTACK_MITIGATION": "MITRE ATT&CK Mitigation ID (Title)",
    "CWE": "CWE ID (Title)",
    "CAPEC": "CAPEC ID (Title)",
}

ID_RE = re.compile(
    r"\b(?:T\d{4}(?:\.\d{3})?|M\d{4}|CWE-\d+|CAPEC-\d+)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of canonical records.")
    return data


def external_id(obj: Dict[str, Any], sources: Tuple[str, ...], prefix: Optional[str] = None) -> str:
    for ref in obj.get("external_references", []) or []:
        src = ref.get("source_name") or ""
        if src not in sources:
            continue
        eid = ref.get("external_id") or ""
        if prefix and not eid.startswith(prefix):
            continue
        return eid
    return ""


def first_paragraph(text: str) -> str:
    cleaned = (text or "").replace("\r", "").strip()
    if not cleaned:
        return ""
    parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
    return parts[0].strip()


def build_mitre_records(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = cfg.get("attack_url") or "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    cache_path = cfg.get("cache_path") or "dataset/mitre/enterprise-attack.json"
    bundle_path = download_mitre_bundle(url, Path(cache_path))
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []

    records: List[Dict[str, Any]] = []
    for obj in objs:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        obj_type = obj.get("type")
        if obj_type == "attack-pattern":
            tid = external_id(obj, MITRE_SRC, prefix="T")
            if not tid:
                continue
            desc = str(obj.get("description") or "").strip()
            if not desc:
                continue
            is_subtech = bool(obj.get("x_mitre_is_subtechnique")) or "." in tid
            id_type = "ATTACK_SUBTECHNIQUE" if is_subtech else "ATTACK_TECHNIQUE"
            records.append(
                {
                    "id_type": id_type,
                    "id": tid,
                    "name": obj.get("name", ""),
                    "official_description": desc,
                    "source_url": url,
                }
            )
        elif obj_type == "course-of-action":
            mid = external_id(obj, MITRE_SRC, prefix="M")
            if not mid:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            if not desc:
                continue
            records.append(
                {
                    "id_type": "ATTACK_MITIGATION",
                    "id": mid,
                    "name": obj.get("name", ""),
                    "official_description": desc,
                    "source_url": url,
                }
            )
    return records


def build_capec_records(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = cfg.get("capec_url") or "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"
    cache_path = cfg.get("cache_path") or "dataset/capec/stix-capec.json"
    bundle_path = download_capec_bundle(url, Path(cache_path))
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []

    records: List[Dict[str, Any]] = []
    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_capec_status", "").lower() == "deprecated":
            continue
        capec_id = external_id(obj, ("capec",), prefix="CAPEC-")
        if not capec_id:
            continue
        desc = str(obj.get("description") or "").strip()
        if not desc:
            continue
        records.append(
            {
                "id_type": "CAPEC",
                "id": capec_id,
                "name": obj.get("name", ""),
                "official_description": desc,
                "source_url": url,
            }
        )
    return records


def build_cwe_records(cwe_path: Path) -> List[Dict[str, Any]]:
    if not cwe_path.exists():
        return []
    tree = ET.parse(cwe_path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0].strip("{")
    ns_prefix = f"{{{namespace}}}" if namespace else ""

    records: List[Dict[str, Any]] = []
    for weakness in root.findall(f".//{ns_prefix}Weakness"):
        cwe_id = weakness.get("ID")
        if not cwe_id:
            continue
        desc_elem = weakness.find(f"{ns_prefix}Description")
        if desc_elem is None:
            continue
        desc_text = " ".join("".join(desc_elem.itertext()).split())
        if not desc_text:
            continue
        records.append(
            {
                "id_type": "CWE",
                "id": f"CWE-{cwe_id}",
                "name": weakness.get("Name", ""),
                "official_description": desc_text,
                "source_url": str(cwe_path),
            }
        )
    return records


def build_records_from_sources(config_path: str) -> List[Dict[str, Any]]:
    cfg = load_yaml(config_path)
    records: List[Dict[str, Any]] = []
    records.extend(build_mitre_records(cfg.get("MITRE_ATTACK", {})))
    records.extend(build_capec_records(cfg.get("CAPEC", {})))
    cwe_cfg = cfg.get("CWE", {}) if isinstance(cfg, dict) else {}
    cwe_path = Path(cwe_cfg.get("cwe_path") or "dataset/cwe/cwec_v4.19.xml")
    records.extend(build_cwe_records(cwe_path))
    return records


def normalize_id(value: Any) -> str:
    return str(value or "").strip().upper()


def find_ids(text: str) -> Set[str]:
    return {match.upper() for match in ID_RE.findall(text or "")}


def sanitize_description(text: str) -> str:
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    cleaned = ID_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def sanitize_title(text: str) -> str:
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    cleaned = ID_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def desc_key(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


PROMPT_TEMPLATES_DESC = {
    "ATTACK_TECHNIQUE": (
        "Given the following description, provide the single most appropriate "
        "MITRE ATT&CK Enterprise technique ID (format: T####).\n"
        "Return EXACTLY one technique ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Description:\n{description}"
    ),
    "ATTACK_SUBTECHNIQUE": (
        "Given the following description, provide the single most appropriate "
        "MITRE ATT&CK Enterprise sub-technique ID (format: T####.###).\n"
        "Return EXACTLY one sub-technique ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Description:\n{description}"
    ),
    "ATTACK_MITIGATION": (
        "Given the following description, provide the single most appropriate "
        "MITRE ATT&CK Enterprise mitigation ID (format: M####).\n"
        "Return EXACTLY one mitigation ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Description:\n{description}"
    ),
    "CWE": (
        "Given the following description, provide the single most appropriate "
        "Common Weakness Enumeration (CWE) ID (format: CWE-<number>).\n"
        "Return EXACTLY one CWE ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Description:\n{description}"
    ),
    "CAPEC": (
        "Given the following description, provide the single most appropriate "
        "Common Attack Pattern Enumeration and Classification (CAPEC) ID (format: CAPEC-<number>).\n"
        "Return EXACTLY one CAPEC ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Description:\n{description}"
    ),
}


PROMPT_TEMPLATES_TITLE = {
    "ATTACK_TECHNIQUE": (
        "Given the following title, provide the single most appropriate "
        "MITRE ATT&CK Enterprise technique ID (format: T####).\n"
        "Return EXACTLY one technique ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Title:\n{title}"
    ),
    "ATTACK_SUBTECHNIQUE": (
        "Given the following title, provide the single most appropriate "
        "MITRE ATT&CK Enterprise sub-technique ID (format: T####.###).\n"
        "Return EXACTLY one sub-technique ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Title:\n{title}"
    ),
    "ATTACK_MITIGATION": (
        "Given the following title, provide the single most appropriate "
        "MITRE ATT&CK Enterprise mitigation ID (format: M####).\n"
        "Return EXACTLY one mitigation ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Title:\n{title}"
    ),
    "CWE": (
        "Given the following title, provide the single most appropriate "
        "Common Weakness Enumeration (CWE) ID (format: CWE-<number>).\n"
        "Return EXACTLY one CWE ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Title:\n{title}"
    ),
    "CAPEC": (
        "Given the following title, provide the single most appropriate "
        "Common Attack Pattern Enumeration and Classification (CAPEC) ID (format: CAPEC-<number>).\n"
        "Return EXACTLY one CAPEC ID and nothing else. Any extra text makes the answer incorrect.\n\n"
        "Title:\n{title}"
    ),
}


def build_prompt_desc(id_type: str, description: str) -> str:
    template = PROMPT_TEMPLATES_DESC.get(id_type)
    if not template:
        return ""
    return template.format(description=description)


def build_prompt_title(id_type: str, title: str) -> str:
    template = PROMPT_TEMPLATES_TITLE.get(id_type)
    if not template:
        return ""
    return template.format(title=title)


def should_skip_id_refs(record_id: str, desc_ids: Set[str]) -> bool:
    if not desc_ids:
        return False
    if len(desc_ids) > 1:
        return True
    return record_id not in desc_ids


def collect_candidates(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    stats = {
        "total": 0,
        "skipped_type": 0,
        "skipped_missing": 0,
        "skipped_ids": 0,
        "skipped_empty": 0,
    }
    for record in records:
        stats["total"] += 1
        id_type = record.get("id_type")
        if id_type not in ALLOWED_ID_TYPES:
            stats["skipped_type"] += 1
            continue
        record_id = normalize_id(record.get("id"))
        description = str(record.get("official_description") or "")
        title = str(record.get("name") or "")
        source_url = str(record.get("source_url") or "")
        if not record_id or not source_url:
            stats["skipped_missing"] += 1
            continue
        desc_ids = find_ids(description)
        if should_skip_id_refs(record_id, desc_ids):
            stats["skipped_ids"] += 1
            continue
        sanitized_desc = sanitize_description(description)
        sanitized_title = sanitize_title(title)
        if not sanitized_desc and not sanitized_title:
            stats["skipped_empty"] += 1
            continue
        if (sanitized_desc and (ID_RE.search(sanitized_desc) or URL_RE.search(sanitized_desc))) or (
            sanitized_title and (ID_RE.search(sanitized_title) or URL_RE.search(sanitized_title))
        ):
            stats["skipped_ids"] += 1
            continue
        candidates.append(
            {
                "id_type": id_type,
                "id": record_id,
                "name": title,
                "sanitized_description": sanitized_desc,
                "sanitized_title": sanitized_title,
                "source_url": source_url,
            }
        )
    return candidates, stats


def filter_ambiguous(candidates: List[Dict[str, Any]], text_fn) -> Tuple[List[Dict[str, Any]], int]:
    desc_map: Dict[str, List[int]] = {}
    for idx, cand in enumerate(candidates):
        key = desc_key(text_fn(cand))
        if not key:
            continue
        desc_map.setdefault(key, []).append(idx)
    ambiguous = {idx for ids in desc_map.values() if len(ids) > 1 for idx in ids}
    if not ambiguous:
        return candidates, 0
    filtered = [cand for idx, cand in enumerate(candidates) if idx not in ambiguous]
    return filtered, len(ambiguous)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSONL dataset of cybersecurity identifier associations."
    )
    parser.add_argument("--input", help="Path to canonical records JSON array.")
    parser.add_argument(
        "--config",
        default="minerva/config.yaml",
        help="Config YAML for data-source loading when --input is not provided.",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL path (default: cyber_id_eval.jsonl next to this script).",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cyber_id_eval.jsonl"
    )

    if args.input:
        try:
            records = load_records(args.input)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    else:
        try:
            records = build_records_from_sources(args.config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc

    candidates, stats = collect_candidates(records)
    desc_candidates = [c for c in candidates if c.get("sanitized_description")]
    title_candidates = [c for c in candidates if c.get("sanitized_title")]

    desc_filtered, ambiguous_desc = filter_ambiguous(
        desc_candidates, lambda c: c.get("sanitized_description", "")
    )
    title_filtered, ambiguous_title = filter_ambiguous(
        title_candidates, lambda c: c.get("sanitized_title", "")
    )

    rows: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}
    uid_idx = 1

    for cand in desc_filtered:
        prompt = build_prompt_desc(cand["id_type"], cand["sanitized_description"])
        if not prompt or ID_RE.search(prompt) or URL_RE.search(prompt):
            continue
        category = CATEGORY_MAP_DESC.get(cand["id_type"], "")
        if not category:
            continue
        rows.append(
            {
                "uid": f"cyber-id-{uid_idx:06d}",
                "id_type": cand["id_type"],
                "category": category,
                "prompt": prompt,
                "answer": cand["id"],
                "evidence": cand["sanitized_description"],
                "source_url": cand["source_url"],
                "notes": "",
            }
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        uid_idx += 1

    for cand in title_filtered:
        prompt = build_prompt_title(cand["id_type"], cand["sanitized_title"])
        if not prompt or ID_RE.search(prompt) or URL_RE.search(prompt):
            continue
        category = CATEGORY_MAP_TITLE.get(cand["id_type"], "")
        if not category:
            continue
        rows.append(
            {
                "uid": f"cyber-id-{uid_idx:06d}",
                "id_type": cand["id_type"],
                "category": category,
                "prompt": prompt,
                "answer": cand["id"],
                "evidence": cand["sanitized_title"],
                "source_url": cand["source_url"],
                "notes": "",
            }
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        uid_idx += 1

    written = write_jsonl(output_path, rows)

    skipped_total = (
        stats["skipped_type"]
        + stats["skipped_missing"]
        + stats["skipped_ids"]
        + stats["skipped_empty"]
        + ambiguous_desc
        + ambiguous_title
    )
    sys.stderr.write(
        "Processed {total} records: wrote {written}, skipped {skipped} "
        "(type={skipped_type}, missing={skipped_missing}, ids={skipped_ids}, "
        "empty={skipped_empty}, ambiguous={ambiguous}).\n".format(
            total=stats["total"],
            written=written,
            skipped=skipped_total,
            skipped_type=stats["skipped_type"],
            skipped_missing=stats["skipped_missing"],
            skipped_ids=stats["skipped_ids"],
            skipped_empty=stats["skipped_empty"],
            ambiguous=ambiguous_desc + ambiguous_title,
        )
    )
    sys.stdout.write(
        "Counts by category: {counts}\n".format(
            counts=json.dumps(category_counts, ensure_ascii=True)
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build canonical label-details JSONL files for ACRD prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List

from minerva.analysis import retrieval_candidates
from minerva.data_sources import mitre as mitre_source
from minerva.retrieval.task_specs import normalize_label


_TECH_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
CAPEC_URL = "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"
CWE_ZIP_URL = "https://cwe.mitre.org/data/xml/cwec_v4.19.xml.zip"


def _default_paths(repo_root: Path) -> Dict[str, Path]:
    return {
        "mitre": repo_root / "dataset" / "mitre" / "enterprise-attack.json",
        "capec": repo_root / "dataset" / "capec" / "stix-capec.json",
        "cwe": repo_root / "dataset" / "cwe" / "cwec_v4.19.xml",
    }


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        path.write_bytes(resp.read())


def _ensure_cwe_xml(path: Path) -> None:
    if path.exists() and not _is_lfs_pointer(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = path.with_suffix(path.suffix + ".zip")
    _download_url(CWE_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        target = None
        for name in zf.namelist():
            if name.endswith(path.name):
                target = name
                break
        if target is None:
            raise RuntimeError("CWE ZIP does not contain expected XML file.")
        with zf.open(target) as src, path.open("wb") as dst:
            dst.write(src.read())
    zip_path.unlink(missing_ok=True)


def _strip_leading_name(text: str, name: str) -> str:
    cleaned = (text or "").strip()
    if not name:
        return cleaned
    lower = cleaned.lower()
    prefix = name.strip().lower()
    if lower.startswith(prefix):
        rest = cleaned[len(name) :].lstrip()
        if rest.startswith("."):
            rest = rest[1:].lstrip()
        if rest:
            return rest
    return cleaned


def _collapse_whitespace(text: str) -> str:
    cleaned = (text or "").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", cleaned).strip()


def _local_name(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _extract_cwe_section(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    parts: List[str] = []
    children = list(elem)
    if children:
        for child in children:
            local = _local_name(child.tag)
            if local == "p":
                text = _collapse_whitespace("".join(child.itertext()))
                if text:
                    parts.append(text)
            elif local in {"ul", "ol"}:
                for li in child.findall(".//{*}li"):
                    text = _collapse_whitespace("".join(li.itertext()))
                    if text:
                        parts.append(f"- {text}")
            elif local == "Background_Detail":
                text = _collapse_whitespace("".join(child.itertext()))
                if text:
                    parts.append(text)
            else:
                text = _collapse_whitespace("".join(child.itertext()))
                if text:
                    parts.append(text)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    return _collapse_whitespace("".join(elem.itertext()))


def _load_cwe_details(path: Path) -> Dict[str, Dict[str, str]]:
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0].strip("{")
    ns_prefix = f"{{{namespace}}}" if namespace else ""
    details: Dict[str, Dict[str, str]] = {}
    for weakness in root.findall(f".//{ns_prefix}Weakness"):
        cwe_id = weakness.get("ID")
        if not cwe_id:
            continue
        ident = f"CWE-{cwe_id}"
        details[ident] = {
            "description": _extract_cwe_section(weakness.find(f"{ns_prefix}Description")),
            "extended_description": _extract_cwe_section(weakness.find(f"{ns_prefix}Extended_Description")),
            "background_details": _extract_cwe_section(weakness.find(f"{ns_prefix}Background_Details")),
        }
    return details


def _format_id_name(values: Iterable[str], id_to_name: Dict[str, str]) -> List[str]:
    items: List[str] = []
    for raw in values or []:
        ident = str(raw).strip()
        if not ident:
            continue
        name = id_to_name.get(ident, "")
        if name:
            items.append(f"{ident} {name}")
        else:
            items.append(ident)
    return items


def _format_id_name_with_dash(values: Iterable[str], id_to_name: Dict[str, str]) -> List[str]:
    items = _format_id_name(values, id_to_name)
    formatted: List[str] = []
    for item in items:
        if " " in item:
            ident, rest = item.split(" ", 1)
            formatted.append(f"{ident} - {rest}".strip())
        else:
            formatted.append(item)
    return formatted


def _format_with_dash(values: Iterable[str]) -> List[str]:
    formatted: List[str] = []
    for item in values or []:
        text = str(item).strip()
        if not text:
            continue
        if " " in text:
            ident, rest = text.split(" ", 1)
            formatted.append(f"{ident} - {rest}".strip())
        else:
            formatted.append(text)
    return formatted


def _extract_technique_ids(text: str) -> List[str]:
    return [m.group(0).upper() for m in _TECH_ID_RE.finditer(text or "")]


def _details_text(
    *,
    canonical_id: str,
    name: str,
    aliases: List[str],
    definition: str,
    metadata: Dict[str, List[str] | str],
    definition_label: str = "Definition",
) -> str:
    lines = [f"ID: {canonical_id}"]
    if name:
        lines.append(f"Name: {name}")
    if aliases:
        lines.append(f"Aliases: {', '.join(aliases)}")

    order = ["tactics", "platforms", "mitigations", "detections", "techniques", "techniques_addressed"]
    labels = {
        "tactics": "Tactics",
        "platforms": "Platforms",
        "mitigations": "Mitigations",
        "detections": "Detections",
        "techniques": "Techniques",
        "techniques_addressed": "Techniques Addressed",
    }
    for key in order:
        value = metadata.get(key)
        if isinstance(value, list) and value:
            lines.append(f"{labels.get(key, key.title())}: {', '.join(value)}")
        elif isinstance(value, str) and value:
            lines.append(f"{labels.get(key, key.title())}: {value}")

    if definition:
        if definition_label:
            lines.append(f"{definition_label}: {definition}")
        else:
            lines.append(definition)
    return "\n".join(lines).strip()


def _details_text_tactic(
    *,
    canonical_id: str,
    name: str,
    definition: str,
    techniques: List[str],
) -> str:
    lines = [f"ID: {canonical_id}"]
    if name:
        lines.append(f"Name: {name}")
    if definition:
        lines.append(definition)
    return "\n".join(lines).strip()


def _details_text_attack_technique(
    *,
    canonical_id: str,
    name: str,
    description: str,
    subtechniques: List[tuple[str, str]],
    mitigations: List[dict],
    detections: List[dict],
) -> str:
    lines: List[str] = []
    lines.append(f"ID: {canonical_id}")
    if name:
        lines.append(f"Name: {name}")
    if subtechniques:
        lines.append(f"Sub-techniques ({len(subtechniques)})")
        lines.append("ID\tName")
        for sub_id, sub_name in subtechniques:
            lines.append(f"{sub_id}\t{sub_name}")
    if description:
        if lines:
            lines.append("")
        lines.append(description.strip())
    return "\n".join([line for line in lines if line is not None]).strip()


def _details_text_cwe(
    *,
    canonical_id: str,
    name: str,
    description: str,
    extended_description: str,
    background_details: str,
) -> str:
    title = f"{canonical_id}: {name}".strip() if name else canonical_id
    lines = [title]
    if description:
        lines.append("Description")
        lines.append(description)
    if extended_description:
        lines.append("")
        lines.append("Extended Description")
        lines.append(extended_description)
    if background_details:
        lines.append("")
        lines.append("Background Details")
        lines.append(background_details)
    return "\n".join([line for line in lines if line is not None]).strip()


def _details_text_mitigation(
    *,
    canonical_id: str,
    name: str,
    definition: str,
    techniques: List[str],
) -> str:
    lines = [f"ID: {canonical_id}"]
    if name:
        lines.append(f"Name: {name}")
    if definition:
        lines.append(f"Definition: {definition}")
    return "\n".join([line for line in lines if line is not None]).strip()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    data = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _build_actor_aliases(bundle: Dict) -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}
    for obj in bundle.get("objects", []) or []:
        if obj.get("type") != "intrusion-set":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        name = str(obj.get("name") or "").strip()
        if not name:
            continue
        aliases = [a for a in (obj.get("aliases") or []) if a and a != name]
        if aliases:
            alias_map[name] = aliases
    return alias_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACRD label details JSONL files.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: dataset/label_details).",
    )
    parser.add_argument("--mitre_path", default=None, help="Path to enterprise-attack.json.")
    parser.add_argument("--capec_path", default=None, help="Path to stix-capec.json.")
    parser.add_argument("--cwe_path", default=None, help="Path to cwec XML.")
    parser.add_argument(
        "--label_types",
        nargs="+",
        default=None,
        help="Restrict label types to build.",
    )
    parser.add_argument(
        "--examples-per-type",
        type=int,
        default=2,
        help="Number of example rows per label type to include in metadata.json.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading source files if missing or LFS pointers.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = _default_paths(repo_root)

    if args.mitre_path:
        paths["mitre"] = Path(args.mitre_path)
    if args.capec_path:
        paths["capec"] = Path(args.capec_path)
    if args.cwe_path:
        paths["cwe"] = Path(args.cwe_path)

    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "dataset" / "label_details")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        if not paths["mitre"].exists() or _is_lfs_pointer(paths["mitre"]):
            _download_url(MITRE_URL, paths["mitre"])
        if not paths["capec"].exists() or _is_lfs_pointer(paths["capec"]):
            _download_url(CAPEC_URL, paths["capec"])
        _ensure_cwe_xml(paths["cwe"])

    mitre_bundle = retrieval_candidates.load_mitre_bundle(paths["mitre"])
    mitre_meta = mitre_source.load_bundle({"cache_path": str(paths["mitre"])}, logger=None)
    actor_aliases = _build_actor_aliases(mitre_bundle)

    tactic_desc_by_id: Dict[str, str] = {}
    technique_info_by_id: Dict[str, Dict[str, str]] = {}
    stix_to_tech_id: Dict[str, str] = {}
    mitigation_by_stix: Dict[str, Dict[str, str]] = {}
    mitigation_details_by_id: Dict[str, str] = {}
    detection_by_stix: Dict[str, Dict[str, str]] = {}
    analytic_by_stix: Dict[str, Dict[str, str]] = {}
    for obj in mitre_bundle.get("objects", []) or []:
        if obj.get("type") != "x-mitre-tactic":
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            if obj.get("type") == "attack-pattern":
                tid = retrieval_candidates.external_id(obj, retrieval_candidates.MITRE_SRC, prefix="T")
                if not tid:
                    continue
                stix_id = obj.get("id", "")
                if stix_id:
                    stix_to_tech_id[stix_id] = tid
                technique_info_by_id[tid] = {
                    "name": str(obj.get("name") or "").strip(),
                    "description": str(obj.get("description") or "").replace("\r", "").strip(),
                    "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False) or "." in tid),
                    "parent_ref": str(obj.get("x_mitre_parent_attack_pattern_ref") or ""),
                }
            elif obj.get("type") == "course-of-action":
                mid = retrieval_candidates.external_id(obj, retrieval_candidates.MITRE_SRC, prefix="M")
                if mid:
                    desc = str(obj.get("description") or "").replace("\r", "").strip()
                    mitigation_details_by_id[mid] = desc
                    mitigation_by_stix[obj.get("id", "")] = {
                        "id": mid,
                        "name": str(obj.get("name") or "").strip(),
                        "description": desc,
                    }
            elif obj.get("type") == "x-mitre-detection-strategy":
                det_id = retrieval_candidates.external_id(obj, retrieval_candidates.MITRE_SRC, prefix="DET")
                if det_id:
                    det_id = det_id.replace("-", "")
                    detection_by_stix[obj.get("id", "")] = {
                        "id": det_id,
                        "name": str(obj.get("name") or "").strip(),
                        "analytic_refs": list(obj.get("x_mitre_analytic_refs") or []),
                    }
            elif obj.get("type") == "x-mitre-analytic":
                analytic_id = ""
                for ref in obj.get("external_references", []) or []:
                    if ref.get("source_name") in retrieval_candidates.MITRE_SRC:
                        analytic_id = str(ref.get("external_id") or "").strip()
                        if analytic_id:
                            break
                analytic_by_stix[obj.get("id", "")] = {
                    "id": analytic_id,
                    "description": str(obj.get("description") or "").replace("\r", "").strip(),
                }
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = retrieval_candidates.external_id(obj, retrieval_candidates.MITRE_SRC, prefix="TA")
        if not tid:
            continue
        desc = str(obj.get("description") or "").replace("\r", "").strip()
        desc = _strip_leading_name(desc, str(obj.get("name") or "").strip())
        tactic_desc_by_id[tid] = desc

    corpora = retrieval_candidates.build_mitre_corpora(mitre_bundle)
    corpora["capec_id"] = retrieval_candidates.load_capec_corpus(paths["capec"])
    corpora["cwe_id"] = retrieval_candidates.load_cwe_corpus(paths["cwe"])
    cwe_details_by_id = _load_cwe_details(paths["cwe"])

    label_types = args.label_types or sorted(corpora.keys())
    manifest = {"files": {}, "counts": {}}
    examples: Dict[str, List[dict]] = {}
    validation: Dict[str, Dict[str, int]] = {}

    tactic_name_by_id = {
        meta.get("id"): meta.get("name", "")
        for meta in (mitre_meta.get("tactics") or {}).values()
        if meta.get("id")
    }
    mitigation_name_by_id = {
        mid: meta.get("name", "") for mid, meta in (mitre_meta.get("mitigations") or {}).items()
    }
    detection_name_by_id = {
        did.replace("-", ""): meta.get("name", "")
        for did, meta in (mitre_meta.get("detection_strategies") or {}).items()
    }

    technique_name_by_id = {
        normalize_label("attack_technique_id", entry.id): entry.name
        for entry in corpora.get("attack_technique_id", retrieval_candidates.Corpus(entries=[])).entries
    }

    tactic_to_techniques: Dict[str, List[str]] = {}
    for tech_id, meta in (mitre_meta.get("techniques") or {}).items():
        for tactic_id in meta.get("tactics", []) or []:
            tactic_to_techniques.setdefault(tactic_id, []).append(tech_id)

    subtechniques_by_parent: Dict[str, List[tuple[str, str]]] = {}
    for tid, info in technique_info_by_id.items():
        if not info.get("is_subtechnique"):
            continue
        parent_ref = info.get("parent_ref")
        parent_tid = stix_to_tech_id.get(parent_ref, "")
        if not parent_tid and "." in tid:
            parent_tid = tid.split(".", 1)[0]
        if not parent_tid:
            continue
        subtechniques_by_parent.setdefault(parent_tid, []).append((tid, info.get("name", "")))

    technique_mitigations: Dict[str, List[dict]] = {}
    technique_detections: Dict[str, List[dict]] = {}
    for obj in mitre_bundle.get("objects", []) or []:
        if obj.get("type") != "relationship":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        rel = obj.get("relationship_type")
        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")
        if rel == "mitigates":
            tech_id = stix_to_tech_id.get(target_ref, "")
            mitigation = mitigation_by_stix.get(source_ref)
            if tech_id and mitigation:
                technique_mitigations.setdefault(tech_id, []).append(mitigation)
        elif rel == "detects":
            tech_id = stix_to_tech_id.get(target_ref, "")
            detection = detection_by_stix.get(source_ref)
            if tech_id and detection:
                technique_detections.setdefault(tech_id, []).append(detection)

    for label_type in label_types:
        corpus = corpora.get(label_type)
        if corpus is None:
            continue
        rows = []
        invalid_id = 0
        missing_name = 0
        for entry in corpus.entries:
            canonical_id = normalize_label(label_type, entry.id)
            name = str(entry.name or "").strip()
            aliases = []
            metadata: Dict[str, List[str] | str] = {}

            if label_type == "threat_actor_name":
                aliases = actor_aliases.get(name, [])

            if label_type == "attack_tactic_id":
                definition = tactic_desc_by_id.get(canonical_id)
                if not definition:
                    definition = str(entry.description or entry.text or "")
                definition = _strip_leading_name(definition, name)
            elif label_type == "attack_technique_id":
                definition = technique_info_by_id.get(canonical_id, {}).get("description") or str(
                    entry.text or entry.description or ""
                )
                definition = _strip_leading_name(definition, name)
            elif label_type == "mitigation_id":
                raw_definition = mitigation_details_by_id.get(canonical_id)
                if raw_definition is None:
                    raw_definition = str(entry.description or entry.text or "")
                definition = _strip_leading_name(raw_definition, name)
            elif label_type == "cwe_id":
                definition = cwe_details_by_id.get(canonical_id, {}).get("description") or str(
                    entry.text or entry.description or ""
                )
            else:
                definition = _strip_leading_name(str(entry.description or entry.text or ""), name)

            if label_type == "attack_technique_id":
                meta = mitre_meta.get("techniques", {}).get(canonical_id)
                if meta:
                    metadata["tactics"] = _format_id_name(meta.get("tactics", []), tactic_name_by_id)
                    metadata["platforms"] = [str(p) for p in meta.get("platforms", []) if p]
                    metadata["mitigations"] = _format_id_name(meta.get("mitigations", []), mitigation_name_by_id)
                    metadata["detections"] = _format_id_name(meta.get("detection_strategies", []), detection_name_by_id)
                subtechs = subtechniques_by_parent.get(canonical_id, [])
                if subtechs:
                    metadata["subtechniques"] = [tid for tid, _ in subtechs]

            if label_type == "attack_tactic_id":
                techniques = sorted(set(tactic_to_techniques.get(canonical_id, [])))
                if techniques:
                    metadata["techniques"] = _format_id_name(techniques, technique_name_by_id)

            if label_type == "mitigation_id":
                techniques = _extract_technique_ids(entry.text)
                if techniques:
                    metadata["techniques_addressed"] = _format_id_name(techniques, technique_name_by_id)

            if label_type == "attack_tactic_id":
                techniques = []
                if isinstance(metadata.get("techniques"), list):
                    techniques = metadata.get("techniques") or []
                details_text = _details_text_tactic(
                    canonical_id=canonical_id,
                    name=name,
                    definition=definition,
                    techniques=techniques,
                )
            elif label_type == "attack_technique_id":
                subtechs = sorted(subtechniques_by_parent.get(canonical_id, []), key=lambda x: x[0])
                mitigations = technique_mitigations.get(canonical_id, [])
                mitigation_rows = []
                if mitigations:
                    seen = set()
                    for item in mitigations:
                        mid = item.get("id", "")
                        if not mid or mid in seen:
                            continue
                        seen.add(mid)
                        mitigation_rows.append(
                            {
                                "id": mid,
                                "name": item.get("name", ""),
                                "description": _strip_leading_name(
                                    _collapse_whitespace(item.get("description", "")),
                                    item.get("name", ""),
                                ),
                            }
                        )
                    mitigation_rows.sort(key=lambda x: x.get("id", ""))

                detections = technique_detections.get(canonical_id, [])
                detection_rows = []
                if detections:
                    seen = set()
                    for item in detections:
                        did = item.get("id", "")
                        if not did or did in seen:
                            continue
                        seen.add(did)
                        detection_rows.append(
                            {
                                "id": did,
                                "name": item.get("name", ""),
                            }
                        )
                    detection_rows.sort(key=lambda x: x.get("id", ""))

                details_text = _details_text_attack_technique(
                    canonical_id=canonical_id,
                    name=name or "",
                    description=definition,
                    subtechniques=subtechs,
                    mitigations=mitigation_rows,
                    detections=detection_rows,
                )
            elif label_type == "mitigation_id":
                techniques = _format_with_dash(metadata.get("techniques_addressed", []))
                details_text = _details_text_mitigation(
                    canonical_id=canonical_id,
                    name=name,
                    definition=definition,
                    techniques=techniques,
                )
            elif label_type == "cwe_id":
                extras = cwe_details_by_id.get(canonical_id, {})
                details_text = _details_text_cwe(
                    canonical_id=canonical_id,
                    name=name,
                    description=definition,
                    extended_description=extras.get("extended_description", ""),
                    background_details=extras.get("background_details", ""),
                )
            else:
                details_text = _details_text(
                    canonical_id=canonical_id,
                    name=name,
                    aliases=aliases,
                    definition=definition,
                    metadata=metadata,
                )

            if label_type == "attack_technique_id":
                if not details_text.startswith(f"ID: {canonical_id}"):
                    invalid_id += 1
                if name and "Name: " not in details_text:
                    missing_name += 1
            elif label_type == "cwe_id":
                if name and name not in details_text.splitlines()[0]:
                    missing_name += 1
                if not details_text.startswith(f"{canonical_id}:"):
                    invalid_id += 1
            else:
                if not details_text.startswith(f"ID: {canonical_id}"):
                    invalid_id += 1
                if name and "Name: " not in details_text:
                    missing_name += 1

            rows.append(
                {
                    "key": f"{label_type}:{canonical_id}",
                    "entity_type": label_type,
                    "canonical_id": canonical_id,
                    "name": name,
                    "aliases": aliases,
                    "definition": definition,
                    "metadata": metadata,
                    "details_text": details_text,
                }
            )

        out_path = out_dir / f"{label_type}.jsonl"
        sha = _write_jsonl(out_path, rows)
        manifest["files"][out_path.name] = sha
        manifest["counts"][label_type] = len(rows)
        validation[label_type] = {
            "total": len(rows),
            "details_missing_id": invalid_id,
            "details_missing_name": missing_name,
        }

        if rows and args.examples_per_type > 0:
            rows_sorted = sorted(rows, key=lambda r: r.get("canonical_id", ""))
            examples[label_type] = rows_sorted[: int(args.examples_per_type)]

    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    metadata = {
        "counts": manifest["counts"],
        "files": manifest["files"],
        "examples": examples,
        "validation": validation,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

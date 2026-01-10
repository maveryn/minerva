"""Build canonical label docs for TARBA retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

from minerva.analysis import retrieval_candidates
from minerva.retrieval.task_specs import normalize_label

MITRE_LABEL_TYPES = {
    "attack_technique_id",
    "attack_tactic_id",
    "mitigation_id",
    "detection_id",
    "threat_actor_name",
}

CAPEC_LABEL_TYPES = {"capec_id"}
CWE_LABEL_TYPES = {"cwe_id"}


def _default_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    return {
        "mitre": root / "dataset" / "mitre" / "enterprise-attack.json",
        "capec": root / "dataset" / "capec" / "stix-capec.json",
        "cwe": root / "dataset" / "cwe" / "cwec_v4.19.xml",
    }


def _apply_entity_types_config(
    cfg: dict,
    paths: Dict[str, Path],
) -> List[str] | None:
    entity_types = cfg.get("entity_types")
    if not isinstance(entity_types, dict):
        return None
    label_types: List[str] = []
    for label_type, spec in entity_types.items():
        label_type = str(label_type)
        label_types.append(label_type)
        if not isinstance(spec, dict):
            continue
        source_paths = spec.get("source_paths")
        if not source_paths:
            continue
        source_path = Path(source_paths[0])
        if label_type in MITRE_LABEL_TYPES:
            paths["mitre"] = source_path
        elif label_type in CAPEC_LABEL_TYPES:
            paths["capec"] = source_path
        elif label_type in CWE_LABEL_TYPES:
            paths["cwe"] = source_path
    return label_types


def _load_sources(paths: Dict[str, Path]) -> Dict[str, retrieval_candidates.Corpus]:
    mitre_bundle = retrieval_candidates.load_mitre_bundle(paths["mitre"])
    corpora = retrieval_candidates.build_mitre_corpora(mitre_bundle)
    corpora["capec_id"] = retrieval_candidates.load_capec_corpus(paths["capec"])
    corpora["cwe_id"] = retrieval_candidates.load_cwe_corpus(paths["cwe"])
    return corpora


def _build_docs(label_type: str, corpus: retrieval_candidates.Corpus) -> List[dict]:
    docs: List[dict] = []
    for entry in corpus.entries:
        canonical_id = normalize_label(label_type, entry.id)
        name = str(entry.name or "").strip()
        desc = str(entry.description or "").strip()
        text = f"{canonical_id} {name}. {desc}".strip()
        doc = {
            "doc_id": f"{label_type}:{canonical_id}",
            "label_type": label_type,
            "canonical_id": canonical_id,
            "name": name,
            "aliases": [],
            "short_definition": desc,
            "metadata": {},
            "text_for_retrieval": text,
        }
        docs.append(doc)
    return docs


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    data = "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TARBA label docs JSONL files.")
    parser.add_argument("--out_dir", required=True, help="Output directory for label docs.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file to override source paths and label types.",
    )
    parser.add_argument(
        "--label_types",
        nargs="+",
        default=None,
        help="Optional label types to build (default: all available types).",
    )
    parser.add_argument(
        "--mitre_path",
        default=None,
        help="Path to enterprise-attack.json (default: dataset/mitre/enterprise-attack.json).",
    )
    parser.add_argument(
        "--capec_path",
        default=None,
        help="Path to stix-capec.json (default: dataset/capec/stix-capec.json).",
    )
    parser.add_argument(
        "--cwe_path",
        default=None,
        help="Path to cwec XML (default: dataset/cwe/cwec_v4.19.xml).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _default_paths()
    label_types = None
    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if isinstance(cfg, dict):
            sources = cfg.get("sources", {})
            if isinstance(sources, dict):
                if sources.get("mitre_path"):
                    paths["mitre"] = Path(sources["mitre_path"])
                if sources.get("capec_path"):
                    paths["capec"] = Path(sources["capec_path"])
                if sources.get("cwe_path"):
                    paths["cwe"] = Path(sources["cwe_path"])
            entity_label_types = _apply_entity_types_config(cfg, paths)
            if entity_label_types:
                label_types = entity_label_types
            cfg_label_types = cfg.get("label_types")
            if cfg_label_types:
                label_types = cfg_label_types
    if args.mitre_path:
        paths["mitre"] = Path(args.mitre_path)
    if args.capec_path:
        paths["capec"] = Path(args.capec_path)
    if args.cwe_path:
        paths["cwe"] = Path(args.cwe_path)

    corpora = _load_sources(paths)

    if args.label_types is not None:
        label_types = args.label_types
    if not label_types:
        label_types = sorted(corpora.keys())
    manifest = {"files": {}, "counts": {}}

    for label_type in label_types:
        corpus = corpora.get(label_type)
        if corpus is None:
            continue
        rows = _build_docs(label_type, corpus)
        out_path = out_dir / f"{label_type}.jsonl"
        sha = _write_jsonl(out_path, rows)
        manifest["files"][out_path.name] = sha
        manifest["counts"][label_type] = len(rows)

    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""
Build candidate pools for LHC using the best retrieval method per label family.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict

import numpy as np
import xml.etree.ElementTree as ET

from minerva.data_sources.mitre import MITRE_SRC

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

ID_RE = re.compile(r"\b(?:T\d{4}(?:\.\d{3})?|TA\d{4}|DET-?\d{4}|M\d{4}|CWE-\d+|CAPEC-\d+)\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class CorpusEntry:
    id: str
    name: str
    description: str
    text: str


@dataclass
class Corpus:
    entries: List[CorpusEntry]


TASK_SPECS = {
    "cve_to_attack_exploitation": {
        "label_type": "attack_technique_id",
        "query_keys": ["cve_description", "description"],
        "gold_key": "technique_id",
    },
    "cve_to_attack_primary_impact": {
        "label_type": "attack_technique_id",
        "query_keys": ["cve_description", "description"],
        "gold_key": "technique_id",
    },
    "cve_to_attack_secondary_impact": {
        "label_type": "attack_technique_id",
        "query_keys": ["cve_description", "description"],
        "gold_key": "technique_id",
    },
    "sigma_to_attack_technique": {
        "label_type": "attack_technique_id",
        "query_keys": ["sigma_rule_excerpt"],
        "gold_key": "technique_id",
    },
    "scenario_to_technique": {
        "label_type": "attack_technique_id",
        "query_keys": ["scenario"],
        "gold_key": "technique_id",
    },
    "capec_example_to_attack": {
        "label_type": "attack_technique_id",
        "query_keys": ["example"],
        "gold_key": "technique_id",
    },
    "sigma_to_attack_tactics": {
        "label_type": "attack_tactic_id",
        "query_keys": ["sigma_rule_excerpt"],
        "gold_key": "tactic_ids",
    },
    "scenario_to_tactics": {
        "label_type": "attack_tactic_id",
        "query_keys": ["scenario"],
        "gold_key": "tactic_ids",
    },
    "scenario_to_detections": {
        "label_type": "detection_id",
        "query_keys": ["scenario"],
        "gold_key": "detection_id",
    },
    "scenario_to_mitigations": {
        "label_type": "mitigation_id",
        "query_keys": ["scenario"],
        "gold_key": "mitigation_ids",
    },
    "cve_to_cwe": {
        "label_type": "cwe_id",
        "query_keys": ["description"],
        "gold_key": "cwe_ids",
    },
    "capec_example_to_cwe": {
        "label_type": "cwe_id",
        "query_keys": ["example"],
        "gold_key": "cwe_ids",
    },
    "capec_example_to_capec": {
        "label_type": "capec_id",
        "query_keys": ["example"],
        "gold_key": "capec_id",
    },
    "threat_actor": {
        "label_type": "threat_actor_name",
        "query_keys": ["procedures"],
        "gold_key": "threat_actor",
    },
    "threat_actor_from_procedures": {
        "label_type": "threat_actor_name",
        "query_keys": ["procedures"],
        "gold_key": "threat_actor",
    },
}


def sanitize_text(text: str, *, keep_ids: bool = False, keep_newlines: bool = False) -> str:
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    if not keep_ids:
        cleaned = ID_RE.sub(" ", cleaned)
    if keep_newlines:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"\s+", " ", line).strip() for line in cleaned.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(text: str) -> List[str]:
    cleaned = TOKEN_CLEAN_RE.sub(" ", str(text or "").lower())
    return [tok for tok in cleaned.split() if tok]


def normalize_id(value: str, *, kind: str) -> str:
    text = str(value or "").strip().upper()
    if kind == "detection_id":
        return text.replace("-", "")
    return text


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


def load_mitre_bundle(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"MITRE bundle not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_mitre_corpora(bundle: Dict[str, Any]) -> Dict[str, Corpus]:
    objs = bundle.get("objects", []) or []
    techniques: List[CorpusEntry] = []
    tactics: List[CorpusEntry] = []
    mitigations: List[CorpusEntry] = []
    detections: List[CorpusEntry] = []
    actors: List[CorpusEntry] = []
    attack_by_stix: Dict[str, CorpusEntry] = {}
    mitigation_by_stix: Dict[str, CorpusEntry] = {}
    detection_by_stix: Dict[str, CorpusEntry] = {}
    mitigations_for_attack: Dict[str, List[CorpusEntry]] = defaultdict(list)
    detections_for_attack: Dict[str, List[CorpusEntry]] = defaultdict(list)
    techniques_for_mitigation: Dict[str, List[CorpusEntry]] = defaultdict(list)

    for obj in objs:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        typ = obj.get("type")
        if typ == "attack-pattern":
            tid = external_id(obj, MITRE_SRC, prefix="T")
            if not tid:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            name = obj.get("name", "")
            text = sanitize_text(f"{name}. {desc}".strip(), keep_ids=True)
            entry = CorpusEntry(id=tid, name=name, description=desc, text=text)
            techniques.append(entry)
            attack_by_stix[obj.get("id", "")] = entry
        elif typ == "x-mitre-tactic":
            tid = external_id(obj, MITRE_SRC, prefix="TA")
            if not tid:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            name = obj.get("name", "")
            text = sanitize_text(f"{name}. {desc}".strip())
            tactics.append(CorpusEntry(id=tid, name=name, description=desc, text=text))
        elif typ == "course-of-action":
            mid = external_id(obj, MITRE_SRC, prefix="M")
            if not mid:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            name = obj.get("name", "")
            text = sanitize_text(f"{name}. {desc}".strip())
            entry = CorpusEntry(id=mid, name=name, description=desc, text=text)
            mitigations.append(entry)
            mitigation_by_stix[obj.get("id", "")] = entry
        elif typ == "x-mitre-detection-strategy":
            det_id = external_id(obj, MITRE_SRC, prefix="DET")
            if not det_id:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            name = obj.get("name", "")
            det_id = det_id.replace("-", "")
            text = sanitize_text(f"{name}. {desc}".strip())
            entry = CorpusEntry(id=det_id, name=name, description=desc, text=text)
            detections.append(entry)
            detection_by_stix[obj.get("id", "")] = entry
        elif typ == "intrusion-set":
            name = obj.get("name", "")
            if not name:
                continue
            aliases = obj.get("aliases", []) or []
            alias_text = ", ".join([a for a in aliases if a])
            desc = first_paragraph(str(obj.get("description") or ""))
            combined = ". ".join([p for p in [name, alias_text, desc] if p])
            text = sanitize_text(combined)
            actors.append(CorpusEntry(id=name, name=name, description=combined, text=text))

    for obj in objs:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if obj.get("type") != "relationship":
            continue
        rel = obj.get("relationship_type")
        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")
        if not source_ref or not target_ref:
            continue
        if rel == "mitigates":
            mitigation = mitigation_by_stix.get(source_ref)
            technique = attack_by_stix.get(target_ref)
            if mitigation and technique:
                mitigations_for_attack[target_ref].append(mitigation)
                techniques_for_mitigation[source_ref].append(technique)
        elif rel == "detects":
            detection = detection_by_stix.get(source_ref)
            if detection and target_ref in attack_by_stix:
                detections_for_attack[target_ref].append(detection)

    def _format_related(entry: CorpusEntry) -> str:
        if not entry:
            return ""
        header_parts = [p for p in [entry.id, entry.name] if p]
        header = " - ".join(header_parts).strip()
        if entry.description:
            if header:
                return f"{header}: {entry.description}".strip()
            return entry.description.strip()
        return header

    def _format_related_compact(entry: CorpusEntry) -> str:
        if not entry:
            return ""
        header_parts = [p for p in [entry.id, entry.name] if p]
        return " - ".join(header_parts).strip()

    for stix_id, entry in attack_by_stix.items():
        mit_entries = mitigations_for_attack.get(stix_id, [])
        det_entries = detections_for_attack.get(stix_id, [])
        if not mit_entries and not det_entries:
            continue
        mit_by_id = {item.id: item for item in mit_entries if item.id}
        det_by_id = {item.id: item for item in det_entries if item.id}
        extra_parts: List[str] = []
        if mit_by_id:
            mitigation_text = "\n".join(
                _format_related_compact(item)
                for item in [mit_by_id[mid] for mid in sorted(mit_by_id)]
                if item.id or item.name or item.description
            ).strip()
            if mitigation_text:
                extra_parts.append(f"Mitigations:\n{mitigation_text}")
        if det_by_id:
            detection_text = "\n".join(
                _format_related(item)
                for item in [det_by_id[did] for did in sorted(det_by_id)]
                if item.id or item.name or item.description
            ).strip()
            if detection_text:
                extra_parts.append(f"Detections:\n{detection_text}")
        if extra_parts:
            merged = "\n".join([entry.text] + extra_parts)
            entry.text = sanitize_text(merged, keep_ids=True, keep_newlines=True)

    def _format_technique(entry: CorpusEntry) -> str:
        if not entry:
            return ""
        if entry.id and entry.name:
            return f"{entry.id} - {entry.name}".strip()
        return entry.id or entry.name or ""

    for stix_id, entry in mitigation_by_stix.items():
        tech_entries = techniques_for_mitigation.get(stix_id, [])
        if not tech_entries:
            continue
        tech_by_id = {item.id: item for item in tech_entries if item.id}
        technique_text = "\n".join(
            _format_technique(item)
            for item in [tech_by_id[tid] for tid in sorted(tech_by_id)]
            if item.id or item.name
        ).strip()
        if not technique_text:
            continue
        merged = "\n".join([entry.text, f"Techniques Addressed by Mitigation:\n{technique_text}"])
        entry.text = sanitize_text(merged, keep_ids=True, keep_newlines=True)

    return {
        "attack_technique_id": Corpus(entries=techniques),
        "attack_tactic_id": Corpus(entries=tactics),
        "mitigation_id": Corpus(entries=mitigations),
        "detection_id": Corpus(entries=detections),
        "threat_actor_name": Corpus(entries=actors),
    }


def load_capec_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(f"CAPEC bundle not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []
    entries: List[CorpusEntry] = []
    for obj in objs:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_capec_status", "").lower() == "deprecated":
            continue
        capec_id = external_id(obj, ("capec",), prefix="CAPEC-")
        if not capec_id:
            continue
        desc = first_paragraph(str(obj.get("description") or ""))
        name = obj.get("name", "")
        text = sanitize_text(f"{name}. {desc}".strip())
        entries.append(CorpusEntry(id=capec_id, name=name, description=desc, text=text))
    return Corpus(entries=entries)


def load_cwe_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(f"CWE XML not found: {path}")
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0].strip("{")
    ns_prefix = f"{{{namespace}}}" if namespace else ""

    entries: List[CorpusEntry] = []
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
        name = weakness.get("Name", "")
        ident = f"CWE-{cwe_id}"
        text = sanitize_text(f"{name}. {desc_text}".strip())
        entries.append(CorpusEntry(id=ident, name=name, description=desc_text, text=text))
    return Corpus(entries=entries)


class BM25Index:
    def __init__(self, corpus_texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lens: List[int] = []
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for text in corpus_texts:
            tokens = tokenize(text)
            self.doc_lens.append(len(tokens))
            counts: Dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, freq in counts.items():
                self.df[tok] = self.df.get(tok, 0) + 1
                self.postings.setdefault(tok, []).append((len(self.doc_lens) - 1, freq))
        self.avgdl = float(sum(self.doc_lens)) / float(len(self.doc_lens)) if self.doc_lens else 0.0
        self.N = len(self.doc_lens)
        self._doc_norm = np.array(
            [
                self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                for dl in self.doc_lens
            ],
            dtype=np.float32,
        )

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        if not query_tokens:
            return scores
        for term in set(query_tokens):
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_idx, freq in self.postings.get(term, []):
                denom = freq + self._doc_norm[doc_idx]
                scores[doc_idx] += idf * (freq * (self.k1 + 1.0)) / denom
        return scores


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    s_min = float(scores.min())
    s_max = float(scores.max())
    if math.isclose(s_min, s_max):
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


class LabelRetriever:
    def __init__(
        self,
        *,
        label_type: str,
        corpus: Corpus,
        method: str,
        alpha: Optional[float],
        dense_model=None,
        batch_size: int = 64,
    ) -> None:
        self.label_type = label_type
        self.corpus = corpus
        self.method = method
        self.alpha = alpha
        self.batch_size = batch_size
        self.entries = corpus.entries
        self.corpus_ids = [normalize_id(e.id, kind=label_type) for e in self.entries]
        self.id_to_entry = {normalize_id(e.id, kind=label_type): e for e in self.entries}
        self.texts = [e.text for e in self.entries]
        self.bm25 = BM25Index(self.texts) if method == "bm25" or method.startswith("hybrid_") else None
        self.dense_model = dense_model if method in {"dense"} or method.startswith("hybrid_") else None
        self.corpus_emb = None
        if self.dense_model is not None:
            self.corpus_emb = self.dense_model.encode(
                self.texts, convert_to_numpy=True, normalize_embeddings=True
            )

    def retrieve_topk(self, queries: List[str], k: int) -> List[List[Dict[str, str]]]:
        if self.method == "bm25":
            return [self._topk_from_scores(self.bm25.get_scores(tokenize(q)), k) for q in queries]
        if self.method == "dense":
            query_emb = self.dense_model.encode(
                queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=self.batch_size
            )
            sim = np.matmul(query_emb, self.corpus_emb.T)
            return [self._topk_from_scores(sim[i], k) for i in range(len(queries))]
        if self.method.startswith("hybrid_"):
            alpha = float(self.method.split("_", 1)[1])
            query_emb = self.dense_model.encode(
                queries, convert_to_numpy=True, normalize_embeddings=True, batch_size=self.batch_size
            )
            dense_sim = np.matmul(query_emb, self.corpus_emb.T)
            pools = []
            for i, q in enumerate(queries):
                bm25_scores = self.bm25.get_scores(tokenize(q))
                dense_scores = dense_sim[i]
                bm25_norm = minmax_normalize(bm25_scores)
                dense_norm = minmax_normalize(dense_scores)
                fused = (1.0 - alpha) * bm25_norm + alpha * dense_norm
                pools.append(self._topk_from_scores(fused, k))
            return pools
        raise ValueError(f"Unknown retrieval method: {self.method}")

    def _topk_from_scores(self, scores: np.ndarray, k: int) -> List[Dict[str, str]]:
        order = np.argsort(-scores, kind="mergesort")
        top = []
        for idx in order[:k]:
            entry = self.entries[int(idx)]
            top.append(
                {
                    "id": normalize_id(entry.id, kind=self.label_type),
                    "name": entry.name,
                    "description": entry.description,
                }
            )
        return top

    def ensure_gold(self, pool: List[Dict[str, str]], gold_ids: List[str], k: int) -> List[Dict[str, str]]:
        seen = {item.get("id") for item in pool}
        for gold in gold_ids:
            if not gold or gold in seen:
                continue
            entry = self.id_to_entry.get(gold)
            if entry:
                pool.append(
                    {
                        "id": gold,
                        "name": entry.name,
                        "description": entry.description,
                    }
                )
            else:
                pool.append({"id": gold, "name": "", "description": ""})
            seen.add(gold)
        if len(pool) > k:
            gold_set = {g for g in gold_ids if g}
            idx = len(pool) - 1
            while len(pool) > k and idx >= 0:
                if pool[idx].get("id") not in gold_set:
                    pool.pop(idx)
                idx -= 1
            if len(pool) > k:
                pool = pool[:k]
        return pool


def extract_query(row: Dict[str, Any], query_keys: List[str]) -> str:
    payload = row.get("input", {}) or {}
    for key in query_keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            return sanitize_text("\n".join(str(v) for v in value))
        return sanitize_text(str(value))
    prompt = payload.get("prompt")
    return sanitize_text(str(prompt or ""))


def extract_gold(row: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    if spec.get("gold_from_metadata"):
        meta = row.get("metadata", {}) or {}
        val = meta.get(spec["gold_key"])
        if val:
            return [normalize_id(str(val), kind=spec["label_type"])]
        return []

    truth = row.get("ground_truth", {}) or {}
    val = truth.get(spec["gold_key"])
    if val is None:
        return []
    if isinstance(val, list):
        return [normalize_id(str(v), kind=spec["label_type"]) for v in val if v]
    return [normalize_id(str(val), kind=spec["label_type"])]


def load_best_methods(selection_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    return data.get("id_types", {})


def build_retrievers(
    *,
    cfg: Dict[str, Any],
    selection_path: Path,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Dict[str, LabelRetriever]:
    mitre_path = Path(cfg.get("MITRE_ATTACK", {}).get("cache_path") or "dataset/mitre/enterprise-attack.json")
    capec_path = Path(cfg.get("CAPEC", {}).get("cache_path") or "dataset/capec/stix-capec.json")
    cwe_path = Path(cfg.get("CWE", {}).get("cwe_path") or "dataset/cwe/cwec_v4.19.xml")

    mitre_bundle = load_mitre_bundle(mitre_path)
    corpora = build_mitre_corpora(mitre_bundle)
    corpora["capec_id"] = load_capec_corpus(capec_path)
    corpora["cwe_id"] = load_cwe_corpus(cwe_path)

    methods = load_best_methods(selection_path)
    needs_dense = any(
        entry.get("best", {}).get("method", "bm25").startswith("hybrid")
        or entry.get("best", {}).get("method", "bm25") == "dense"
        for entry in methods.values()
    )
    dense_model = None
    if needs_dense:
        from sentence_transformers import SentenceTransformer  # type: ignore

        dense_model = SentenceTransformer(dense_model_name)

    retrievers: Dict[str, LabelRetriever] = {}
    for label_type, entry in methods.items():
        corpus = corpora.get(label_type)
        if not corpus:
            continue
        best = entry.get("best", {}) or {}
        method = best.get("method", "bm25")
        alpha = best.get("alpha")
        retrievers[label_type] = LabelRetriever(
            label_type=label_type,
            corpus=corpus,
            method=method,
            alpha=alpha,
            dense_model=dense_model,
            batch_size=batch_size,
        )
    return retrievers


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def augment_task_file(
    *,
    input_path: Path,
    output_path: Path,
    retriever: LabelRetriever,
    spec: Dict[str, Any],
    k: int = 100,
) -> int:
    rows = list(iter_jsonl(input_path))
    if not rows:
        write_jsonl(output_path, [])
        return 0
    queries = [extract_query(row, spec["query_keys"]) for row in rows]
    pools = retriever.retrieve_topk(queries, k)
    out_rows = []
    for row, pool in zip(rows, pools, strict=False):
        gold = extract_gold(row, spec)
        pool = retriever.ensure_gold(pool, gold, k)
        row["candidate_pool_top100"] = pool
        out_rows.append(row)
    write_jsonl(output_path, out_rows)
    return len(out_rows)

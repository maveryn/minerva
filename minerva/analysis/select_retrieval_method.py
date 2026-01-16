"""
Evaluate retrieval methods for Minerva label-hint candidate pools.

Produces MRR for BM25, dense, and hybrid (alpha sweep) per label family and task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva.data_sources.mitre import MITRE_SRC
from minerva.utils import load_yaml

# Avoid loading TensorFlow/Flax backends when using sentence-transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")


ID_RE = re.compile(r"\b(?:T\d{4}(?:\.\d{3})?|TA\d{4}|DET-?\d{4}|M\d{4}|CWE-\d+|CAPEC-\d+)\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class Corpus:
    ids: List[str]
    texts: List[str]


@dataclass
class QueryRecord:
    query: str
    gold: List[str]
    task: str


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = URL_RE.sub(" ", text)
    cleaned = ID_RE.sub(" ", cleaned)
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


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sample_rows(rows: List[Dict[str, Any]], limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    return rng.sample(rows, limit)


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
    techniques: Dict[str, str] = {}
    tactics: Dict[str, str] = {}
    mitigations: Dict[str, str] = {}
    detections: Dict[str, str] = {}
    actors: Dict[str, str] = {}

    for obj in objs:
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        typ = obj.get("type")
        if typ == "attack-pattern":
            tid = external_id(obj, MITRE_SRC, prefix="T")
            if not tid:
                continue
            desc = str(obj.get("description") or "").strip()
            if not desc:
                continue
            name = obj.get("name", "")
            techniques[tid] = sanitize_text(f"{name}. {desc}")
        elif typ == "x-mitre-tactic":
            tid = external_id(obj, MITRE_SRC, prefix="TA")
            if not tid:
                continue
            desc = str(obj.get("description") or "").strip()
            name = obj.get("name", "")
            tactics[tid] = sanitize_text(f"{name}. {desc}".strip())
        elif typ == "course-of-action":
            mid = external_id(obj, MITRE_SRC, prefix="M")
            if not mid:
                continue
            desc = first_paragraph(str(obj.get("description") or ""))
            name = obj.get("name", "")
            mitigations[mid] = sanitize_text(f"{name}. {desc}".strip())
        elif typ == "x-mitre-detection-strategy":
            det_id = external_id(obj, MITRE_SRC, prefix="DET")
            if not det_id:
                continue
            desc = str(obj.get("description") or "").strip()
            name = obj.get("name", "")
            detections[det_id.replace("-", "")] = sanitize_text(f"{name}. {desc}".strip())
        elif typ == "intrusion-set":
            name = obj.get("name", "")
            if not name:
                continue
            aliases = obj.get("aliases", []) or []
            desc = str(obj.get("description") or "").strip()
            alias_text = ", ".join([a for a in aliases if a])
            text = f"{name}. {alias_text}. {desc}".strip()
            actors[name] = sanitize_text(text)

    return {
        "attack_technique_id": Corpus(ids=list(techniques.keys()), texts=list(techniques.values())),
        "attack_tactic_id": Corpus(ids=list(tactics.keys()), texts=list(tactics.values())),
        "mitigation_id": Corpus(ids=list(mitigations.keys()), texts=list(mitigations.values())),
        "detection_id": Corpus(ids=list(detections.keys()), texts=list(detections.values())),
        "threat_actor_name": Corpus(ids=list(actors.keys()), texts=list(actors.values())),
    }


def load_capec_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(f"CAPEC bundle not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    objs = data.get("objects", []) or []
    ids: List[str] = []
    texts: List[str] = []
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
        name = obj.get("name", "")
        ids.append(capec_id)
        texts.append(sanitize_text(f"{name}. {desc}"))
    return Corpus(ids=ids, texts=texts)


def load_cwe_corpus(path: Path) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(f"CWE XML not found: {path}")
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0].strip("{")
    ns_prefix = f"{{{namespace}}}" if namespace else ""

    ids: List[str] = []
    texts: List[str] = []
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
        ids.append(f"CWE-{cwe_id}")
        texts.append(sanitize_text(f"{name}. {desc_text}"))
    return Corpus(ids=ids, texts=texts)


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


def compute_mrr(ranked_ids: List[str], gold_set: set[str]) -> float:
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / float(idx)
    return 0.0


def evaluate_dense_with_sim(
    dense_sim: np.ndarray,
    corpus_ids: List[str],
    gold_labels: List[List[str]],
) -> Tuple[float, int]:
    missing = 0
    scores = []
    for idx in range(len(gold_labels)):
        gold_set = set(gold_labels[idx])
        if not gold_set:
            missing += 1
            scores.append(0.0)
            continue
        order = np.argsort(-dense_sim[idx], kind="mergesort")
        ranked_ids = [corpus_ids[i] for i in order.tolist()]
        scores.append(compute_mrr(ranked_ids, gold_set))
    return float(np.mean(scores)) if scores else 0.0, missing


def evaluate_bm25(
    bm25: BM25Index,
    corpus_ids: List[str],
    queries: List[str],
    gold_labels: List[List[str]],
) -> Tuple[float, int]:
    scores = []
    missing = 0
    for query, gold in zip(queries, gold_labels, strict=False):
        gold_set = set(gold)
        if not gold_set:
            missing += 1
            scores.append(0.0)
            continue
        q_tokens = tokenize(query)
        score_vec = bm25.get_scores(q_tokens)
        order = np.argsort(-score_vec, kind="mergesort")
        ranked_ids = [corpus_ids[i] for i in order.tolist()]
        scores.append(compute_mrr(ranked_ids, gold_set))
    return float(np.mean(scores)) if scores else 0.0, missing


def evaluate_dense(
    corpus_emb: np.ndarray,
    corpus_ids: List[str],
    queries: List[str],
    gold_labels: List[List[str]],
    model,
    query_emb: Optional[np.ndarray] = None,
) -> Tuple[float, int, np.ndarray]:
    missing = 0
    scores = []
    if query_emb is None:
        query_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    sim = np.matmul(query_emb, corpus_emb.T)
    for idx in range(len(queries)):
        gold_set = set(gold_labels[idx])
        if not gold_set:
            missing += 1
            scores.append(0.0)
            continue
        order = np.argsort(-sim[idx], kind="mergesort")
        ranked_ids = [corpus_ids[i] for i in order.tolist()]
        scores.append(compute_mrr(ranked_ids, gold_set))
    return float(np.mean(scores)) if scores else 0.0, missing, sim


def evaluate_hybrid(
    bm25: BM25Index,
    corpus_ids: List[str],
    corpus_emb: np.ndarray,
    queries: List[str],
    gold_labels: List[List[str]],
    model,
    alphas: List[float],
    query_emb: Optional[np.ndarray] = None,
    dense_sim: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    missing_counts: Dict[str, int] = {}
    mrr_scores: Dict[str, float] = {}

    if dense_sim is None:
        if query_emb is None:
            query_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        dense_sim = np.matmul(query_emb, corpus_emb.T)

    for alpha in alphas:
        scores = []
        missing = 0
        for q_idx, query in enumerate(queries):
            gold_set = set(gold_labels[q_idx])
            if not gold_set:
                missing += 1
                scores.append(0.0)
                continue
            bm25_scores = bm25.get_scores(tokenize(query))
            dense_scores = dense_sim[q_idx]
            bm25_norm = minmax_normalize(bm25_scores)
            dense_norm = minmax_normalize(dense_scores)
            fused = (1.0 - alpha) * bm25_norm + alpha * dense_norm
            order = np.argsort(-fused, kind="mergesort")
            ranked_ids = [corpus_ids[i] for i in order.tolist()]
            scores.append(compute_mrr(ranked_ids, gold_set))
        mrr_scores[f"hybrid_{alpha:.1f}"] = float(np.mean(scores)) if scores else 0.0
        missing_counts[f"hybrid_{alpha:.1f}"] = missing
    return mrr_scores, missing_counts


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


def load_task_rows(path: Path, spec: Dict[str, Any], sample_size: int, rng: random.Random, task_name: str) -> List[QueryRecord]:
    rows = list(read_jsonl(path))
    rows = sample_rows(rows, sample_size, rng)
    records: List[QueryRecord] = []
    for row in rows:
        query = extract_query(row, spec["query_keys"])
        if not query:
            continue
        gold = extract_gold(row, spec)
        records.append(QueryRecord(query=query, gold=gold, task=task_name))
    return records


def load_dense_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sentence-transformers not installed") from exc
    return SentenceTransformer(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select retrieval method per label family using MRR.")
    parser.add_argument("--config", default="minerva/config.yaml")
    parser.add_argument("--input-dir", default="dataset/minerva")
    parser.add_argument("--out-dir", default="minerva/analysis/retrieval_selection")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense/hybrid evaluation.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]

    cfg = load_yaml(args.config)
    mitre_path = Path(cfg.get("MITRE_ATTACK", {}).get("cache_path") or "dataset/mitre/enterprise-attack.json")
    capec_path = Path(cfg.get("CAPEC", {}).get("cache_path") or "dataset/capec/stix-capec.json")
    cwe_path = Path(cfg.get("CWE", {}).get("cwe_path") or "dataset/cwe/cwec_v4.19.xml")

    mitre_bundle = load_mitre_bundle(mitre_path)
    corpora = build_mitre_corpora(mitre_bundle)
    corpora["capec_id"] = load_capec_corpus(capec_path)
    corpora["cwe_id"] = load_cwe_corpus(cwe_path)

    task_results: Dict[str, Any] = {}
    id_results: Dict[str, Any] = {}
    id_agg: Dict[str, Dict[str, Any]] = {}
    id_records: Dict[str, List[QueryRecord]] = {}
    task_indices: Dict[str, Dict[str, Any]] = {}

    input_dir = Path(args.input_dir)
    for task_name, spec in TASK_SPECS.items():
        path = input_dir / f"{task_name}.jsonl"
        if not path.exists():
            continue
        records = load_task_rows(path, spec, args.sample_size, rng, task_name)
        if not records:
            continue
        id_type = spec["label_type"]
        start = len(id_records.get(id_type, []))
        id_records.setdefault(id_type, []).extend(records)
        end = len(id_records[id_type])
        task_indices[task_name] = {"id_type": id_type, "start": start, "end": end}
        task_results[task_name] = {
            "label_type": id_type,
            "num_samples": len(records),
        }

    dense_model = None
    dense_error = None
    if not args.skip_dense:
        try:
            dense_model = load_dense_model(args.dense_model)
        except Exception as exc:
            dense_error = str(exc)

    corpus_cache: Dict[str, Dict[str, Any]] = {}

    for task_name, meta in task_indices.items():
        id_type = meta["id_type"]
        records = id_records.get(id_type, [])
        if not records:
            continue
        start = meta["start"]
        end = meta["end"]
        task_recs = records[start:end]
        queries = [rec.query for rec in task_recs]
        labels = [rec.gold for rec in task_recs]
        corpus = corpora.get(id_type)
        if corpus is None or not corpus.ids:
            continue

        if id_type not in corpus_cache:
            corpus_ids = [normalize_id(cid, kind=id_type) for cid in corpus.ids]
            bm25 = BM25Index(corpus.texts)
            corpus_emb = None
            if dense_model is not None:
                corpus_emb = dense_model.encode(corpus.texts, convert_to_numpy=True, normalize_embeddings=True)
            corpus_cache[id_type] = {
                "corpus_ids": corpus_ids,
                "bm25": bm25,
                "corpus_emb": corpus_emb,
            }
        cache = corpus_cache[id_type]

        bm25_mrr, bm25_missing = evaluate_bm25(cache["bm25"], cache["corpus_ids"], queries, labels)
        metrics = {"bm25": bm25_mrr}
        missing = {"bm25": bm25_missing}

        if dense_model is not None and cache["corpus_emb"] is not None:
            # Reuse dense similarities from aggregated id_type if available.
            dense_sim_all = cache.get("dense_sim")
            if dense_sim_all is None:
                all_queries = [rec.query for rec in records]
                query_emb_all = dense_model.encode(all_queries, convert_to_numpy=True, normalize_embeddings=True)
                dense_sim_all = np.matmul(query_emb_all, cache["corpus_emb"].T)
                cache["dense_sim"] = dense_sim_all
            dense_sim = dense_sim_all[start:end]
            dense_mrr, dense_miss = evaluate_dense_with_sim(
                dense_sim,
                cache["corpus_ids"],
                labels,
            )
            metrics["dense"] = dense_mrr
            missing["dense"] = dense_miss
            hybrid_metrics, hybrid_missing = evaluate_hybrid(
                cache["bm25"],
                cache["corpus_ids"],
                cache["corpus_emb"],
                queries,
                labels,
                dense_model,
                alphas,
                dense_sim=dense_sim,
            )
            metrics.update(hybrid_metrics)
            missing.update(hybrid_missing)

        best_method = max(metrics.items(), key=lambda kv: kv[1])
        best_name = best_method[0]
        best_entry = {"method": best_name, "mrr": best_method[1]}
        if best_name.startswith("hybrid_"):
            best_entry["alpha"] = float(best_name.split("_", 1)[1])

        task_results[task_name].update(
            {
                "metrics_mrr": metrics,
                "missing_labels": missing,
                "best": best_entry,
            }
        )

        agg = id_agg.setdefault(
            id_type,
            {"total_samples": 0, "sum_mrr": {}, "missing": {}},
        )
        agg["total_samples"] += len(queries)
        for key, val in metrics.items():
            agg["sum_mrr"][key] = agg["sum_mrr"].get(key, 0.0) + val * len(queries)
        for key, val in missing.items():
            agg["missing"][key] = agg["missing"].get(key, 0) + int(val)

    for id_type, agg in id_agg.items():
        total = agg["total_samples"]
        if total <= 0:
            continue
        metrics = {k: v / total for k, v in agg["sum_mrr"].items()}
        best_method = max(metrics.items(), key=lambda kv: kv[1])
        best_name = best_method[0]
        best_entry = {"method": best_name, "mrr": best_method[1]}
        if best_name.startswith("hybrid_"):
            best_entry["alpha"] = float(best_name.split("_", 1)[1])
        id_results[id_type] = {
            "num_samples": total,
            "metrics_mrr": metrics,
            "missing_labels": agg["missing"],
            "best": best_entry,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_selection.json"

    examples_path = out_dir / "retrieval_examples_top5.jsonl"

    # Save retrieval examples (top-5) for each id_type.
    example_rng = random.Random(args.seed + 7)
    example_lines: List[str] = []
    for id_type, records in id_records.items():
        if not records:
            continue
        best = id_results.get(id_type, {}).get("best", {})
        method = best.get("method", "bm25")
        alpha = best.get("alpha")
        cache = corpus_cache.get(id_type)
        corpus = corpora.get(id_type)
        if cache is None or corpus is None:
            continue
        indices = list(range(len(records)))
        sample_k = min(5, len(indices))
        picked = example_rng.sample(indices, sample_k)
        bm25 = cache["bm25"]
        corpus_ids = cache["corpus_ids"]
        dense_sim_all = cache.get("dense_sim")
        for idx in picked:
            rec = records[idx]
            gold_set = set(rec.gold)
            if method == "bm25":
                scores = bm25.get_scores(tokenize(rec.query))
            elif method == "dense":
                if dense_sim_all is None:
                    continue
                scores = dense_sim_all[idx]
            elif method.startswith("hybrid_"):
                if dense_sim_all is None:
                    continue
                alpha_val = float(method.split("_", 1)[1])
                bm25_scores = bm25.get_scores(tokenize(rec.query))
                dense_scores = dense_sim_all[idx]
                bm25_norm = minmax_normalize(bm25_scores)
                dense_norm = minmax_normalize(dense_scores)
                scores = (1.0 - alpha_val) * bm25_norm + alpha_val * dense_norm
            else:
                scores = bm25.get_scores(tokenize(rec.query))

            order = np.argsort(-scores, kind="mergesort")
            top5 = []
            for rank, doc_idx in enumerate(order[:5], start=1):
                doc_id = corpus_ids[doc_idx]
                doc_text = corpus.texts[doc_idx] if doc_idx < len(corpus.texts) else ""
                top5.append(
                    {
                        "rank": rank,
                        "id": doc_id,
                        "score": float(scores[doc_idx]),
                        "is_gold": doc_id in gold_set,
                        "text": doc_text,
                    }
                )
            payload = {
                "id_type": id_type,
                "task": rec.task,
                "query": rec.query,
                "gold_labels": rec.gold,
                "method": method,
                "alpha": alpha,
                "top5": top5,
            }
            example_lines.append(json.dumps(payload, ensure_ascii=True))
    if example_lines:
        examples_path.write_text("\n".join(example_lines) + "\n", encoding="utf-8")

    payload = {
        "config": {
            "input_dir": str(input_dir),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "dense_model": args.dense_model,
            "alphas": alphas,
        },
        "dense_available": dense_model is not None,
        "dense_error": dense_error,
        "id_types": id_results,
        "tasks": task_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote retrieval selection metrics -> {out_path}")
    if example_lines:
        print(f"Wrote retrieval examples -> {examples_path}")


if __name__ == "__main__":
    main()

"""Retrieval engine for TARBA label-doc search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from minerva.retrieval.index_bm25 import BM25Index, tokenize
from minerva.retrieval.task_specs import normalize_label, regex_for_label_type


@dataclass(frozen=True)
class LabelDoc:
    doc_id: str
    label_type: str
    canonical_id: str
    name: str
    snippet: str
    text_for_retrieval: str


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    canonical_id: str
    title: str
    snippet: str
    score: float


class RetrievalEngine:
    def __init__(
        self,
        *,
        label_docs_dir: str | Path,
        index_dir: str | Path | None = None,
        topk_cap: int = 8,
        max_query_chars: int = 256,
        max_snippet_chars: int = 800,
    ) -> None:
        self.label_docs_dir = Path(label_docs_dir)
        self.index_dir = Path(index_dir) if index_dir else None
        self.topk_cap = int(topk_cap)
        self.max_query_chars = int(max_query_chars)
        self.max_snippet_chars = int(max_snippet_chars)

        self._docs: Dict[str, List[LabelDoc]] = {}
        self._id_map: Dict[str, Dict[str, LabelDoc]] = {}
        self._bm25: Dict[str, BM25Index] = {}
        self._load_label_docs()

    def _load_label_docs(self) -> None:
        if not self.label_docs_dir.exists():
            raise FileNotFoundError(f"Label docs dir not found: {self.label_docs_dir}")
        for path in sorted(self.label_docs_dir.glob("*.jsonl")):
            label_type = path.stem
            docs = self._load_label_docs_file(path)
            if not docs:
                continue
            self._docs[label_type] = docs
            self._id_map[label_type] = {doc.canonical_id: doc for doc in docs if doc.canonical_id}
            index = self._load_index(label_type)
            if index is None:
                index = BM25Index([doc.text_for_retrieval for doc in docs])
            self._bm25[label_type] = index

    def _load_index(self, label_type: str) -> Optional[BM25Index]:
        if self.index_dir is None:
            return None
        index_path = self.index_dir / f"{label_type}.pkl"
        if not index_path.exists():
            return None
        try:
            import pickle

            with index_path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict) and "bm25" in payload:
                return payload["bm25"]
        except Exception:
            return None
        return None

    def _load_label_docs_file(self, path: Path) -> List[LabelDoc]:
        docs: List[LabelDoc] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label_type = str(raw.get("label_type") or path.stem)
                canonical_id = str(raw.get("canonical_id") or "").strip()
                name = str(raw.get("name") or "").strip()
                text_for_retrieval = str(raw.get("text_for_retrieval") or "").strip()
                snippet = str(raw.get("short_definition") or raw.get("text_for_retrieval") or "").strip()
                doc_id = str(raw.get("doc_id") or f"{label_type}:{canonical_id}").strip()
                docs.append(
                    LabelDoc(
                        doc_id=doc_id,
                        label_type=label_type,
                        canonical_id=canonical_id,
                        name=name,
                        snippet=snippet,
                        text_for_retrieval=text_for_retrieval or snippet,
                    )
                )
        return docs

    def _truncate_snippet(self, text: str) -> str:
        if self.max_snippet_chars <= 0:
            return text
        if len(text) <= self.max_snippet_chars:
            return text
        return text[: self.max_snippet_chars].rstrip() + "..."

    def _match_exact_id(self, label_type: str, query: str) -> Optional[str]:
        pattern = regex_for_label_type(label_type)
        if pattern is None:
            return None
        match = pattern.search(query)
        if not match:
            return None
        return normalize_label(label_type, match.group(0))

    def retrieve(self, label_type: str, query: str, topk: int) -> List[RetrievalResult]:
        label_type = str(label_type or "").strip()
        if not label_type or label_type not in self._docs:
            return []
        query = str(query or "")[: self.max_query_chars]
        topk = min(max(int(topk or 0), 1), self.topk_cap)

        docs = self._docs[label_type]
        id_map = self._id_map[label_type]
        bm25 = self._bm25[label_type]

        results: List[RetrievalResult] = []
        exact_id = self._match_exact_id(label_type, query)
        if exact_id:
            doc = id_map.get(exact_id)
            if doc is not None:
                results.append(
                    RetrievalResult(
                        doc_id=doc.doc_id,
                        canonical_id=doc.canonical_id,
                        title=doc.name,
                        snippet=self._truncate_snippet(doc.snippet or doc.text_for_retrieval),
                        score=float("inf"),
                    )
                )

        scores = bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="mergesort")
        for idx in order:
            if len(results) >= topk:
                break
            doc = docs[int(idx)]
            if exact_id and doc.canonical_id == exact_id:
                continue
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    canonical_id=doc.canonical_id,
                    title=doc.name,
                    snippet=self._truncate_snippet(doc.snippet or doc.text_for_retrieval),
                    score=float(scores[int(idx)]),
                )
            )

        return results


__all__ = ["RetrievalEngine", "RetrievalResult"]

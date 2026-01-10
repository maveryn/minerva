"""Lightweight BM25 index for TARBA retrieval."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> List[str]:
    cleaned = _TOKEN_RE.sub(" ", str(text or "").lower())
    return [tok for tok in cleaned.split() if tok]


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

    def get_scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        query_tokens = list(query_tokens)
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


__all__ = ["BM25Index", "tokenize"]

"""FastAPI server exposing TARBA retrieval."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from minerva.retrieval.engine import RetrievalEngine


DEFAULT_LABEL_DOCS_DIR = Path(__file__).resolve().parents[2] / "dataset" / "retrieval" / "label_docs"

app = FastAPI()
_ENGINE: Optional[RetrievalEngine] = None


class RetrieveRequest(BaseModel):
    label_type: Optional[str] = None
    query: str
    topk: int = 5
    return_scores: bool = True


class RetrievalItem(BaseModel):
    doc_id: str
    canonical_id: str
    title: str
    snippet: str
    score: float | None = None


class RetrieveResponse(BaseModel):
    results: List[RetrievalItem]


def _get_engine() -> RetrievalEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    label_docs_dir = os.environ.get("TARBA_LABEL_DOCS_DIR") or str(DEFAULT_LABEL_DOCS_DIR)
    index_dir = os.environ.get("TARBA_INDEX_DIR")
    topk_cap = int(os.environ.get("TARBA_TOPK_CAP", "8"))
    max_query_chars = int(os.environ.get("TARBA_MAX_QUERY_CHARS", "128"))
    max_snippet_chars = int(os.environ.get("TARBA_MAX_SNIPPET_CHARS", "1536"))
    retrieval_mode = os.environ.get("TARBA_RETRIEVAL_MODE", "per_type")
    _ENGINE = RetrievalEngine(
        label_docs_dir=Path(label_docs_dir),
        index_dir=Path(index_dir) if index_dir else None,
        topk_cap=topk_cap,
        max_query_chars=max_query_chars,
        max_snippet_chars=max_snippet_chars,
        retrieval_mode=retrieval_mode,
    )
    return _ENGINE


@app.on_event("startup")
def _startup() -> None:
    _get_engine()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    engine = _get_engine()
    if engine.retrieval_mode != "global" and not (req.label_type and req.label_type.strip()):
        raise HTTPException(status_code=400, detail="label_type is required for per_type retrieval mode.")
    results = engine.retrieve(req.label_type or "", req.query, req.topk)
    items = []
    for res in results:
        score = res.score if req.return_scores else None
        if score is not None and not math.isfinite(float(score)):
            score = None
        item = RetrievalItem(
            doc_id=res.doc_id,
            canonical_id=res.canonical_id,
            title=res.title,
            snippet=res.snippet,
            score=score,
        )
        items.append(item)
    return RetrieveResponse(results=items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TARBA retrieval via FastAPI.")
    parser.add_argument(
        "--label_docs_dir",
        default=None,
        help="Directory with label docs JSONL files (default: dataset/retrieval/label_docs).",
    )
    parser.add_argument("--index_dir", default=None, help="Directory with BM25 index files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk_cap", type=int, default=8)
    parser.add_argument("--max_query_chars", type=int, default=128)
    parser.add_argument("--max_snippet_chars", type=int, default=1536)
    parser.add_argument("--retrieval_mode", choices=["per_type", "global"], default="per_type")
    args = parser.parse_args()

    label_docs_dir = Path(args.label_docs_dir) if args.label_docs_dir else DEFAULT_LABEL_DOCS_DIR
    os.environ["TARBA_LABEL_DOCS_DIR"] = str(label_docs_dir)
    if args.index_dir:
        os.environ["TARBA_INDEX_DIR"] = str(Path(args.index_dir))
    os.environ["TARBA_TOPK_CAP"] = str(args.topk_cap)
    os.environ["TARBA_MAX_QUERY_CHARS"] = str(args.max_query_chars)
    os.environ["TARBA_MAX_SNIPPET_CHARS"] = str(args.max_snippet_chars)
    os.environ["TARBA_RETRIEVAL_MODE"] = str(args.retrieval_mode)

    uvicorn.run("minerva.retrieval.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

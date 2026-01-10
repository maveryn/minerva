"""Build BM25 indexes for TARBA retrieval."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

from minerva.retrieval.index_bm25 import BM25Index


def _load_label_docs(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TARBA BM25 indexes.")
    parser.add_argument("--label_docs_dir", required=True, help="Directory with label docs JSONL files.")
    parser.add_argument("--out_dir", required=True, help="Output directory for indexes.")
    args = parser.parse_args()

    label_docs_dir = Path(args.label_docs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(label_docs_dir.glob("*.jsonl")):
        label_type = path.stem
        rows = _load_label_docs(path)
        texts = [str(row.get("text_for_retrieval") or "") for row in rows]
        bm25 = BM25Index(texts)
        payload: Dict[str, object] = {"bm25": bm25}
        out_path = out_dir / f"{label_type}.pkl"
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)


if __name__ == "__main__":
    main()

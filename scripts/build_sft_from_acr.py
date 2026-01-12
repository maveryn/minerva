#!/usr/bin/env python3
"""Build a MultiTurn SFT parquet dataset from accepted ACR traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _dedup_by_uid(records: List[dict]) -> List[dict]:
    deduped: Dict[str, dict] = {}
    for record in records:
        uid = record.get("uid")
        if not uid:
            continue
        reward = float(record.get("reward", 0.0))
        prev = deduped.get(uid)
        if prev is None or reward > float(prev.get("reward", 0.0)) + 1e-6:
            deduped[uid] = record
    return list(deduped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT parquet from ACR trace buffer.")
    parser.add_argument("--input", required=True, help="Path to ACR trace JSONL buffer.")
    parser.add_argument("--out", required=True, help="Output parquet path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples.")
    parser.add_argument("--dedup-by-uid", action="store_true", help="Deduplicate by uid keeping best reward.")
    args = parser.parse_args()

    in_path = Path(args.input)
    records = _load_jsonl(in_path)
    if args.dedup_by_uid:
        records = _dedup_by_uid(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    rows = []
    for record in records:
        orig_prompt = record.get("orig_prompt") or record.get("acr_orig_prompt")
        target = record.get("target_completion")
        if not isinstance(orig_prompt, list) or not isinstance(target, str):
            continue
        messages = list(orig_prompt) + [{"role": "assistant", "content": target}]
        rows.append(
            {
                "uid": record.get("uid"),
                "data_source": record.get("data_source"),
                "messages": messages,
                "reward": record.get("reward"),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
    print(f"Wrote {len(rows)} SFT rows to {out_path}")


if __name__ == "__main__":
    main()

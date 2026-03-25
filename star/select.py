from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

from star.common import iter_jsonl, write_jsonl


def select_rows(
    original_rows: Iterable[Dict],
    rationalization_rows: Iterable[Dict],
) -> List[Dict]:
    original_by_uid = {str(row.get("uid")): row for row in original_rows}
    rationalization_by_uid = {str(row.get("uid")): row for row in rationalization_rows}
    selected: List[Dict] = []

    for uid, original in original_by_uid.items():
        chosen = None
        if bool(original.get("verifier_success")):
            chosen = dict(original)
            chosen["selected_from"] = "original"
        else:
            rationalized = rationalization_by_uid.get(uid)
            if rationalized and bool(rationalized.get("verifier_success")):
                chosen = dict(rationalized)
                chosen["selected_from"] = "rationalization"
        if chosen is not None:
            selected.append(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select faithful STaR traces")
    parser.add_argument("--original", required=True)
    parser.add_argument("--rationalization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = select_rows(iter_jsonl(args.original), iter_jsonl(args.rationalization))
    write_jsonl(args.output, selected)


if __name__ == "__main__":
    main()


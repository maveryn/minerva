"""Append-only JSONL buffer for accepted ACR traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


class AcrTraceBuffer:
    """Simple JSONL writer for accepted ACR traces."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def extend(self, records: Iterable[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")


def is_accepted(
    reward_info: dict,
    *,
    require_no_id_leak: bool = True,
) -> bool:
    if not isinstance(reward_info, dict):
        return False
    if not reward_info.get("acr_extracted", False):
        return False
    if not reward_info.get("acr_is_correct", False):
        return False
    if reward_info.get("acr_leak_hit", False):
        return False
    if require_no_id_leak and reward_info.get("acr_id_leak_hit", False):
        return False
    return True


__all__ = ["AcrTraceBuffer", "is_accepted"]

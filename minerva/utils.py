import json
import os
import re
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


ISO_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def parse_date(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    s = str(val).strip()
    if not s:
        return None
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def within_inclusive(ts: Any, start: Optional[datetime], end: Optional[datetime]) -> bool:
    if start is None and end is None:
        return True
    dt = parse_date(ts)
    if dt is None:
        return False
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def word_count(text: str) -> int:
    return len((text or "").split())


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    return value if value else default


# ---------- text helpers ----------

def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def dedupe_by_text(
    records: List[Dict[str, Any]],
    *,
    text_fn,
    prefer_ts_fn=None,
) -> List[Dict[str, Any]]:
    """
    Deduplicate records by normalized text value. If duplicates exist and
    prefer_ts_fn is provided, keep the record with the latest timestamp.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = normalize_text(text_fn(rec))
        if not key:
            continue
        if key not in best:
            best[key] = rec
            continue
        if prefer_ts_fn is None:
            continue
        ts_new = parse_date(prefer_ts_fn(rec))
        ts_old = parse_date(prefer_ts_fn(best[key]))
        # Normalize timezones to avoid naive/aware comparison errors
        if ts_new and ts_new.tzinfo is None:
            ts_new = ts_new.replace(tzinfo=timezone.utc)
        if ts_old and ts_old.tzinfo is None:
            ts_old = ts_old.replace(tzinfo=timezone.utc)
        if ts_new and (not ts_old or ts_new >= ts_old):
            best[key] = rec
    return list(best.values())


def balanced_sample(
    records: List[Dict[str, Any]],
    *,
    label_fn,
    max_items: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    The goal is to balance over labels while respecting max_items.
    - Compute unique labels.
    - Take up to floor(max_items / num_labels) per label (at least 1 if max_items > 0).
    - If total < max_items, fill remaining from leftover items.
    - If total > max_items, downsample.
    """
    if max_items and max_items <= 0:
        return []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        label = str(label_fn(rec))
        groups.setdefault(label, []).append(rec)

    if not groups:
        return []
    if max_items == 0:
        # No cap requested; shuffle for randomness
        all_items = []
        for items in groups.values():
            rng.shuffle(items)
            all_items.extend(items)
        return all_items

    num_labels = len(groups)
    per_label = max(1, max_items // num_labels) if max_items else 1

    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for label, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        take = min(len(items), per_label)
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    if len(selected) < max_items and leftovers:
        rng.shuffle(leftovers)
        need = max_items - len(selected)
        selected.extend(leftovers[:need])

    if len(selected) > max_items:
        selected = rng.sample(selected, max_items)

    return selected

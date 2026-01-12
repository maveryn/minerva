"""Load and query canonical label details for ACRD prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from minerva.retrieval.task_specs import normalize_label


DEFAULT_DETAILS_DIR = Path(__file__).resolve().parents[1] / "dataset" / "label_details"


class LabelDetailsStore:
    """In-memory store for label details built by scripts/build_label_details.py."""

    def __init__(self, root_dir: Optional[str | Path] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else DEFAULT_DETAILS_DIR
        self._details: Optional[Dict[str, str]] = None
        self._alias_lookup: Optional[Dict[str, Dict[str, str]]] = None

    @staticmethod
    def _alias_key(entity_type: Optional[str], value: str) -> str:
        norm = normalize_label(entity_type, value)
        return norm.casefold() if norm else ""

    def _load(self) -> Dict[str, str]:
        if self._details is not None:
            return self._details
        details: Dict[str, str] = {}
        alias_lookup: Dict[str, Dict[str, str]] = {}
        if not self.root_dir.exists():
            self._details = details
            self._alias_lookup = alias_lookup
            return details
        for path in sorted(self.root_dir.glob("*.jsonl")):
            if path.name.startswith("_"):
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = row.get("key")
                    text = row.get("details_text") or ""
                    entity_type = row.get("entity_type") or ""
                    canonical_id = row.get("canonical_id") or ""
                    if isinstance(key, str) and isinstance(text, str) and key:
                        details[key] = text
                    if entity_type and canonical_id:
                        alias_map = alias_lookup.setdefault(entity_type, {})
                        for alias in [canonical_id, row.get("name")] + list(row.get("aliases") or []):
                            if not alias:
                                continue
                            alias_key = self._alias_key(entity_type, str(alias))
                            if alias_key and alias_key not in alias_map:
                                alias_map[alias_key] = str(canonical_id)
        self._details = details
        self._alias_lookup = alias_lookup
        return details

    def get_details(
        self,
        entity_type: Optional[str],
        label_ids: Iterable[str],
        *,
        include_missing: bool = False,
    ) -> str:
        """Return concatenated details text for the given labels."""
        if not entity_type:
            return ""
        details = self._load()
        blocks = []
        for raw in label_ids or []:
            label = normalize_label(entity_type, raw)
            if not label:
                continue
            key = f"{entity_type}:{label}"
            text = details.get(key, "")
            if not text and self._alias_lookup is not None:
                alias_map = self._alias_lookup.get(entity_type, {})
                alias_key = self._alias_key(entity_type, label)
                canonical_id = alias_map.get(alias_key, "")
                if canonical_id:
                    text = details.get(f"{entity_type}:{canonical_id}", "")
            if not text and include_missing:
                text = f"ID: {label}"
            if text:
                blocks.append(text)
        return "\n\n---\n\n".join(blocks)


_STORE: Optional[LabelDetailsStore] = None


def get_details(entity_type: Optional[str], label_ids: Iterable[str], *, root_dir: Optional[str | Path] = None) -> str:
    """Module-level helper to fetch details with a shared store."""
    global _STORE
    if _STORE is None or (root_dir is not None and _STORE.root_dir != Path(root_dir)):
        _STORE = LabelDetailsStore(root_dir=root_dir)
    return _STORE.get_details(entity_type, label_ids)


__all__ = ["LabelDetailsStore", "DEFAULT_DETAILS_DIR", "get_details"]

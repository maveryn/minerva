import random
from typing import Any, Dict, List

from minerva.utils import balanced_sample


def _balanced_cap(
    records: List[Dict[str, Any]],
    max_items: int,
    *,
    label_fn,
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
        all_items: List[Dict[str, Any]] = []
        for items in groups.values():
            rng.shuffle(items)
            all_items.extend(items)
        return all_items

    per_label = max(1, max_items // len(groups)) if max_items else 0
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []
    for items in groups.values():
        rng.shuffle(items)
        take = items[:per_label] if per_label else items
        selected.extend(take)
        leftovers.extend(items[len(take) :])

    if max_items and len(selected) < max_items and leftovers:
        rng.shuffle(leftovers)
        needed = max_items - len(selected)
        selected.extend(leftovers[:needed])
    if max_items and len(selected) > max_items:
        rng.shuffle(selected)
        selected = selected[:max_items]
    return selected

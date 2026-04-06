from __future__ import annotations

from typing import Any, Iterable, Optional


def coerce_numeric_sequence(values: Any, expected_len: int) -> Optional[list[float]]:
    """Convert a list/array-like payload to floats.

    Returns ``None`` when the payload is missing, length-mismatched, or contains
    values that cannot be interpreted numerically.
    """

    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        try:
            values = list(values)
        except TypeError:
            return None
    if len(values) != expected_len:
        return None

    out: list[float] = []
    for value in values:
        if value is None:
            return None
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            return None
    return out


def select_hard_uids(
    uid_list: Iterable[Any],
    scores: list[float],
    *,
    hard_mode: str,
    threshold: float,
) -> set[str]:
    """Select UIDs that remain hard under the configured score aggregation.

    ``hard_mode="mean"`` marks a UID hard when the mean score across its rollouts
    is below ``threshold``.

    ``hard_mode="max"`` marks a UID hard when its best rollout score is below
    ``threshold``.
    """

    reward_sum_by_uid: dict[str, float] = {}
    reward_count_by_uid: dict[str, int] = {}
    reward_max_by_uid: dict[str, float] = {}

    for uid, score in zip(uid_list, scores, strict=False):
        key = str(uid)
        reward_sum_by_uid[key] = reward_sum_by_uid.get(key, 0.0) + float(score)
        reward_count_by_uid[key] = reward_count_by_uid.get(key, 0) + 1
        prev = reward_max_by_uid.get(key)
        if prev is None or float(score) > prev:
            reward_max_by_uid[key] = float(score)

    mode = str(hard_mode or "max").lower().strip()
    if mode == "mean":
        mean_reward_by_uid = {
            uid: reward_sum_by_uid[uid] / max(1, reward_count_by_uid.get(uid, 0)) for uid in reward_sum_by_uid
        }
        return {uid for uid, score in mean_reward_by_uid.items() if score < threshold}

    return {uid for uid, score in reward_max_by_uid.items() if score < (threshold - 1e-6)}


__all__ = ["coerce_numeric_sequence", "select_hard_uids"]

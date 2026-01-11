"""Reward function for TARBA (answer + retrieval shaping)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from minerva.retrieval.engine import RetrievalEngine
from minerva.retrieval.task_specs import extract_labels_from_truth, get_task_spec
from verl.utils.reward_score import reward_minerva


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DOC_IDS_RE = re.compile(r"DOC_IDS:\s*(\[.*?\])", re.DOTALL)

_ENGINE: Optional[RetrievalEngine] = None


def _get_engine(label_docs_dir: Optional[str] = None, index_dir: Optional[str] = None) -> RetrievalEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    if label_docs_dir is None:
        label_docs_dir = str(PROJECT_ROOT / "dataset" / "retrieval" / "label_docs")
    _ENGINE = RetrievalEngine(label_docs_dir=label_docs_dir, index_dir=index_dir)
    return _ENGINE


def _parse_tool_calls(solution_str: str) -> list[dict]:
    calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(solution_str or ""):
        payload = match.group(1)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            calls.append(data)
    return calls


def _parse_doc_ids(solution_str: str) -> list[str]:
    match = _DOC_IDS_RE.search(solution_str or "")
    if not match:
        return []
    raw = match.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(x) for x in data if x]
    return []


def _extract_tool_args(tool_call: dict) -> tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    query = args.get("query")
    label_type = args.get("label_type")
    topk = args.get("topk")
    try:
        topk_val = int(topk) if topk is not None else None
    except (TypeError, ValueError):
        topk_val = None
    return name, query, label_type, topk_val


def _rank_from_doc_ids(doc_ids: list[str], gold_doc_ids: list[str]) -> Dict[str, Optional[int]]:
    ranks: Dict[str, Optional[int]] = {gid: None for gid in gold_doc_ids}
    if not doc_ids:
        return ranks
    for idx, doc_id in enumerate(doc_ids, start=1):
        for gid in gold_doc_ids:
            if doc_id == gid:
                ranks[gid] = idx
    return ranks


def _rank_from_retrieval(
    query: str,
    label_type: str,
    topk: int,
    gold_doc_ids: list[str],
    label_docs_dir: Optional[str],
    index_dir: Optional[str],
) -> Dict[str, Optional[int]]:
    try:
        engine = _get_engine(label_docs_dir=label_docs_dir, index_dir=index_dir)
    except Exception:
        return {gid: None for gid in gold_doc_ids}
    results = engine.retrieve(label_type, query, topk)
    doc_ids = [res.doc_id for res in results]
    return _rank_from_doc_ids(doc_ids, gold_doc_ids)


def _compute_retrieval_reward(
    ranks: Dict[str, Optional[int]],
    *,
    floor: float,
    topk: int,
    policy: str,
) -> float:
    if not ranks:
        return 0.0
    if topk <= 0:
        return 0.0
    if topk <= 1:
        slope = 0.0
    else:
        slope = (1.0 - float(floor)) / float(topk - 1)
    values = []
    for rank in ranks.values():
        if rank is None:
            values.append(0.0)
        else:
            score = 1.0 - slope * (rank - 1)
            values.append(max(float(floor), score))
    if not values:
        return 0.0
    policy = (policy or "mean").lower()
    if policy == "min":
        return min(values)
    if policy == "all_or_nothing":
        return 1.0 if all(v > 0 for v in values) else 0.0
    return sum(values) / len(values)


def reward_tarba(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    *,
    lambda_ret: float = 0.2,
    floor: float = 0.2,
    tool_call_bonus: float = 0.1,
    penalty_tool_call: float = 0.02,
    penalty_illegal_tool: float = 0.5,
    multilabel_policy: str = "mean",
    score_threshold: float = 0.5,
    label_docs_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    topk_cap: int = 8,
) -> dict:
    extra_info = extra_info or {}
    r_ans = float(reward_minerva.reward_minerva(data_source, solution_str, ground_truth, extra_info))

    spec = get_task_spec(data_source, ground_truth)
    gold_ids = extract_labels_from_truth(ground_truth, spec)
    label_type = extra_info.get("tarba_label_type") or spec.label_type or ""
    gold_doc_ids = extra_info.get("tarba_gold_doc_ids")
    if not isinstance(gold_doc_ids, list):
        if label_type:
            gold_doc_ids = [f"{label_type}:{gid}" for gid in gold_ids]
        else:
            gold_doc_ids = []

    allow_retrieval = bool(extra_info.get("tarba_allow_retrieval", False))
    budget_B = extra_info.get("tarba_budget_B")
    if budget_B is not None:
        try:
            budget_B = int(budget_B)
        except (TypeError, ValueError):
            budget_B = None
    if budget_B is not None:
        topk_cap = int(budget_B)

    tool_calls = _parse_tool_calls(solution_str)
    valid_calls = []
    for call in tool_calls:
        name, *_ = _extract_tool_args(call)
        if name in {"cti_retrieve", "search"}:
            valid_calls.append(call)
    tool_call_count = len(valid_calls)
    tool_called = False
    ranks: Dict[str, Optional[int]] = {gid: None for gid in gold_doc_ids}

    tool_doc_ids = extra_info.get("tarba_tool_doc_ids")
    tool_call_count_meta = extra_info.get("tarba_tool_call_count")
    if tool_call_count_meta is not None:
        try:
            tool_call_count = int(tool_call_count_meta)
        except (TypeError, ValueError):
            tool_call_count = len(valid_calls)
        tool_called = tool_call_count > 0
    if isinstance(tool_doc_ids, list):
        if tool_call_count_meta is None and tool_doc_ids:
            tool_call_count = max(tool_call_count, 1)
            tool_called = True
        if tool_doc_ids and gold_doc_ids:
            ranks = _rank_from_doc_ids(tool_doc_ids, list(gold_doc_ids))

    chosen_call = valid_calls[0] if valid_calls else None
    if chosen_call is not None:
        tool_called = True
        name, query, call_label_type, topk = _extract_tool_args(chosen_call)
        if not label_type and call_label_type:
            label_type = str(call_label_type)
        if not isinstance(tool_doc_ids, list) and query and label_type and gold_doc_ids:
            effective_topk = int(topk or len(gold_doc_ids) or 1)
            if budget_B is not None:
                if budget_B <= 0:
                    effective_topk = 0
                else:
                    effective_topk = min(effective_topk, budget_B)
            if topk_cap is not None:
                effective_topk = min(effective_topk, int(topk_cap))
            if effective_topk <= 0:
                ranks = {gid: None for gid in gold_doc_ids}
            else:
                ranks = _rank_from_retrieval(
                    query=str(query),
                    label_type=label_type,
                    topk=effective_topk,
                    gold_doc_ids=list(gold_doc_ids),
                    label_docs_dir=label_docs_dir,
                    index_dir=index_dir,
                )
    else:
        doc_ids = _parse_doc_ids(solution_str)
        if doc_ids:
            tool_called = True
            ranks = _rank_from_doc_ids(doc_ids, list(gold_doc_ids))

    illegal_tool = False
    if tool_called and not allow_retrieval:
        illegal_tool = True
    if tool_call_count > 1:
        illegal_tool = True
    valid_tool = tool_called and allow_retrieval and tool_call_count == 1

    r_ret = 0.0
    if allow_retrieval and label_type:
        r_ret = _compute_retrieval_reward(ranks, floor=floor, topk=topk_cap, policy=multilabel_policy)
    if valid_tool and tool_call_bonus > 0:
        r_ret += float(tool_call_bonus)
    # Do not gate retrieval shaping on answer score.
    score = r_ans + lambda_ret * r_ret
    if tool_called:
        score -= penalty_tool_call
    if illegal_tool:
        score -= penalty_illegal_tool

    is_correct = r_ans >= score_threshold

    out = {
        "score": float(score),
        "r_ans": float(r_ans),
        "r_ret": float(r_ret),
        "is_correct": bool(is_correct),
    }
    if penalty_tool_call > 0 or penalty_illegal_tool > 0:
        out["tool_called"] = int(tool_called)
        out["tool_call_count"] = int(tool_call_count)
        out["illegal_tool"] = int(illegal_tool)
    return out


__all__ = ["reward_tarba"]

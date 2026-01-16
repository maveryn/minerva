"""Task specs and helpers for ACRD (data_source -> label metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional

from minerva.analysis import retrieval_candidates
from minerva.retrieval import task_specs as retrieval_task_specs


@dataclass(frozen=True)
class CTITaskSpec:
    """Resolved task metadata for ACRD prompt construction."""

    entity_type: Optional[str]
    is_multilabel: bool
    label_id_regex: Optional[re.Pattern[str]]
    task_key: Optional[str] = None


# Direct mapping for data_source-based specs (reward_fn or Athena datasets).
TASK_SPECS: dict[str, CTITaskSpec] = {
    key: CTITaskSpec(
        entity_type=spec.label_type,
        is_multilabel=spec.is_multilabel,
        label_id_regex=spec.label_id_regex,
        task_key=None,
    )
    for key, spec in retrieval_task_specs.TASK_SPECS.items()
}

# Task-name aliases seen in Minerva JSONL rows.
TASK_ALIASES = {
    "scenario_to_attack_technique": "scenario_to_technique",
    "scenario_to_attack_tactics": "scenario_to_tactics",
    "scenario_to_attack_mitigations": "scenario_to_mitigations",
    "scenario_to_attack_detection": "scenario_to_detections",
    "capec_example_to_attack_technique": "capec_example_to_attack",
    "threat_actor_from_procedures": "threat_actor",
    "threat_actor_from_procedures_mcq": "threat_actor",
}


def _normalize_task_key(task_key: str) -> str:
    task_key = str(task_key or "").strip()
    if not task_key:
        return ""
    return TASK_ALIASES.get(task_key, task_key)


def resolve_task_key(data_source: str | None, extra_info: Optional[dict]) -> Optional[str]:
    """Resolve a canonical task key from extra_info or data_source."""
    if isinstance(extra_info, dict):
        source_file = extra_info.get("source_file")
        if source_file:
            stem = Path(str(source_file)).stem
            stem = _normalize_task_key(stem)
            if stem in retrieval_candidates.TASK_SPECS:
                return stem

        task_name = extra_info.get("task")
        if task_name:
            task_name = _normalize_task_key(task_name)
            if task_name in retrieval_candidates.TASK_SPECS:
                return task_name

    if data_source:
        task_key = _normalize_task_key(data_source)
        if task_key in retrieval_candidates.TASK_SPECS:
            return task_key
    return None


def get_task_spec(
    data_source: str | None,
    ground_truth: Any | None = None,
    extra_info: Optional[dict] = None,
) -> CTITaskSpec:
    """Resolve a CTITaskSpec for the current sample."""
    task_key = resolve_task_key(data_source, extra_info)
    if task_key and task_key in retrieval_candidates.TASK_SPECS:
        spec = retrieval_candidates.TASK_SPECS[task_key]
        label_type = spec.get("label_type")
        gold_key = str(spec.get("gold_key") or "")
        is_multilabel = gold_key.endswith("ids")
        regex = retrieval_task_specs.regex_for_label_type(label_type)
        return CTITaskSpec(
            entity_type=label_type,
            is_multilabel=is_multilabel,
            label_id_regex=regex,
            task_key=task_key,
        )

    data_source = str(data_source or "").strip()
    direct = TASK_SPECS.get(data_source)
    if direct is not None:
        return direct

    fallback = retrieval_task_specs.get_task_spec(data_source, ground_truth)
    return CTITaskSpec(
        entity_type=fallback.label_type,
        is_multilabel=fallback.is_multilabel,
        label_id_regex=fallback.label_id_regex,
        task_key=task_key,
    )


__all__ = ["CTITaskSpec", "TASK_SPECS", "TASK_ALIASES", "resolve_task_key", "get_task_spec"]

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from minerva.acr_prompt import dedupe_labels, extract_gold_labels, try_build_acr_messages
from minerva.cti_task_specs import get_task_spec
from minerva.label_details_store import LabelDetailsStore
from minerva.retrieval.task_specs import TaskSpec as RetrievalTaskSpec
from minerva.retrieval.task_specs import extract_labels_from_truth, normalize_label

try:
    from rlvr.verl.utils.dataset.minerva_acr_dataset import (
        DEFAULT_ENTITY_REASONING_HINTS,
        DEFAULT_TASK_REASONING_HINTS,
    )
except Exception:
    DEFAULT_TASK_REASONING_HINTS = {}
    DEFAULT_ENTITY_REASONING_HINTS = {}


def _apply_reasoning_hints(defaults: dict[str, str], override: Any) -> dict[str, str]:
    if not isinstance(defaults, dict):
        defaults = {}
    merged = dict(defaults)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        hint_key = str(key or "").strip()
        if not hint_key:
            continue
        if value is None:
            merged.pop(hint_key, None)
            continue
        hint_text = str(value).strip()
        if not hint_text:
            merged.pop(hint_key, None)
            continue
        merged[hint_key] = hint_text
    return merged


class DartPromptBuilder:
    def __init__(
        self,
        tokenizer,
        *,
        label_details_dir: str | None = None,
        max_details_chars: int = 8096,
        max_prompt_length: int = 4096,
        enforce_no_id: bool = False,
        task_reasoning_hints: Optional[dict[str, str]] = None,
        entity_reasoning_hints: Optional[dict[str, str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.details_store = LabelDetailsStore(label_details_dir)
        self.max_details_chars = int(max_details_chars)
        self.max_prompt_length = int(max_prompt_length)
        self.enforce_no_id = bool(enforce_no_id)
        self.task_reasoning_hints = _apply_reasoning_hints(
            DEFAULT_TASK_REASONING_HINTS, task_reasoning_hints
        )
        self.entity_reasoning_hints = _apply_reasoning_hints(
            DEFAULT_ENTITY_REASONING_HINTS, entity_reasoning_hints
        )

    def build_plain_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return copy.deepcopy(messages)

    def build_direct_answer_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        built = copy.deepcopy(messages)
        direct_answer_suffix = (
            "Return only the final answer in the exact format required by the original task. "
            "Do not include reasoning, explanation, or any extra text."
        )
        for msg in reversed(built):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                msg["content"] = msg["content"].rstrip() + "\n\n" + direct_answer_suffix
                return built
        built.append({"role": "user", "content": direct_answer_suffix})
        return built

    def _prompt_too_long(self, messages: List[Dict[str, Any]]) -> bool:
        if self.max_prompt_length <= 0:
            return False
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            return len(input_ids) > self.max_prompt_length
        except Exception:
            return False

    def _resolve_reasoning_hint(self, task_key: str | None, entity_type: str | None) -> str | None:
        if task_key:
            hint = self.task_reasoning_hints.get(task_key)
            if hint:
                return hint
        if entity_type:
            return self.entity_reasoning_hints.get(entity_type)
        return None

    def _extract_option_text(self, messages: List[Dict[str, str]], label_letter: str) -> str:
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                user_text = msg["content"]
                break
        if "Options:" not in user_text:
            return ""
        tail = user_text.split("Options:", 1)[1]
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([A-Z])\s*[\.\)]\s*(.+)$", line)
            if match and match.group(1).upper() == label_letter.upper():
                return match.group(2).strip()
        return ""

    def build_guided_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        data_source: str,
        ground_truth: Any,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[Dict[str, str]]], Dict[str, Any]]:
        built = copy.deepcopy(messages)
        extra_info = extra_info if isinstance(extra_info, dict) else {}
        spec = get_task_spec(data_source, ground_truth, extra_info)
        entity_type = spec.entity_type
        reasoning_hint = self._resolve_reasoning_hint(spec.task_key, entity_type)

        gold_norm: List[str] = []
        try:
            retrieval_spec = RetrievalTaskSpec(
                label_type=entity_type,
                is_multilabel=bool(spec.is_multilabel),
                label_id_regex=spec.label_id_regex,
            )
            gold_norm = extract_labels_from_truth(ground_truth, retrieval_spec)
        except Exception:
            gold_norm = []
        if not gold_norm:
            gold_labels = extract_gold_labels(ground_truth)
            gold_norm = [
                normalize_label(entity_type, label) if entity_type else str(label).strip()
                for label in gold_labels
            ]
            gold_norm = dedupe_labels(gold_norm)

        meta: Dict[str, Any] = {
            "guided_prompt_skipped": False,
            "guided_prompt_skip_reason": "",
            "acr_entity_type": entity_type or "",
            "acr_gold_labels": gold_norm,
            "acr_gold_keys": [f"{entity_type}:{label}" if entity_type else label for label in gold_norm],
            "acr_task_key": spec.task_key or "",
            "acr_reasoning_hint": reasoning_hint or "",
        }

        if not gold_norm:
            meta["guided_prompt_skipped"] = True
            meta["guided_prompt_skip_reason"] = "missing_labels"
            return None, meta

        details_labels = list(gold_norm)
        option_text = ""
        if entity_type and len(gold_norm) == 1 and len(gold_norm[0]) == 1:
            option_text = self._extract_option_text(built, gold_norm[0])
            if option_text:
                try:
                    retrieval_spec = RetrievalTaskSpec(
                        label_type=entity_type,
                        is_multilabel=bool(spec.is_multilabel),
                        label_id_regex=spec.label_id_regex,
                    )
                    extracted = extract_labels_from_truth(option_text, retrieval_spec)
                    if extracted:
                        details_labels = extracted
                    else:
                        details_labels = [option_text]
                except Exception:
                    details_labels = [option_text]

        details_text = None
        if entity_type:
            details_text = self.details_store.get_details(entity_type, details_labels)

        updated_messages, skipped, used_details = try_build_acr_messages(
            built,
            gold_norm,
            details_text,
            max_details_chars=self.max_details_chars,
            prompt_too_long=self._prompt_too_long,
            enforce_no_id=self.enforce_no_id,
            reasoning_hint=reasoning_hint,
        )
        meta["acr_option_text"] = option_text
        meta["acr_details_labels"] = details_labels
        meta["acr_details_used"] = used_details
        meta["acr_details_omitted"] = used_details == ""
        if skipped:
            meta["guided_prompt_skipped"] = True
            meta["guided_prompt_skip_reason"] = "prompt_too_long"
            return None, meta
        return updated_messages, meta

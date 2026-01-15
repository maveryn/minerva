"""Answer-conditioned reasoning dataset wrapper for Minerva ACRD."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from minerva.acr_prompt import dedupe_labels, extract_gold_labels, try_build_acr_messages
from minerva.cti_task_specs import get_task_spec
from minerva.label_details_store import LabelDetailsStore
from minerva.retrieval.task_specs import TaskSpec as RetrievalTaskSpec
from minerva.retrieval.task_specs import extract_labels_from_truth, normalize_label
from verl.utils.dataset.rl_dataset import RLHFDataset

DEFAULT_TASK_REASONING_HINTS = {
    "cve_to_attack_exploitation": (
        "Be specific about how the CVE description indicates the exploitation behavior and why that technique ID fits."
    ),
    "cve_to_attack_primary_impact": (
        "Be specific about how the CVE description indicates the primary impact behavior and why that technique ID fits."
    ),
    "cve_to_attack_secondary_impact": (
        "Be specific about how the CVE description indicates the secondary impact behavior and why that technique ID fits."
    ),
    "sigma_to_attack_technique": "Tie the Sigma rule signals to the technique behavior and justify the technique ID.",
    "scenario_to_technique": "Tie the scenario actions to the technique behavior and justify the technique ID.",
    "capec_example_to_attack": "Use the CAPEC example to justify the ATT&CK technique ID.",
    "sigma_to_attack_tactics": "Tie the Sigma rule signals to the tactic intent and justify the tactic ID(s).",
    "scenario_to_tactics": "Explain the adversary intent in the scenario and why it matches the tactic ID(s).",
    "scenario_to_detections": "Point to the observable signals in the scenario that this detection ID covers.",
    "scenario_to_mitigations": "Explain how each mitigation directly reduces or blocks the scenario behavior.",
    "cve_to_cwe": "Explain the weakness pattern in the CVE description and why it matches the CWE ID(s).",
    "capec_example_to_cwe": "Explain the weakness pattern in the CAPEC example and why it matches the CWE ID(s).",
    "capec_example_to_capec": "Explain how the example matches the CAPEC pattern for that ID.",
    "threat_actor_mcq": "Cite procedures or TTPs in the input that uniquely indicate the threat actor.",
}

DEFAULT_ENTITY_REASONING_HINTS = {
    "attack_technique_id": (
        "Be specific about which behaviors or artifacts in the input match the technique definition and why that ID fits."
    ),
    "attack_tactic_id": "Explain the adversary intent in the input and why it matches the tactic ID(s).",
    "mitigation_id": "Explain how each mitigation addresses the described behavior.",
    "detection_id": "Point to the observable signals that this detection would fire on.",
    "cwe_id": "Explain the weakness pattern and why it matches the CWE ID(s).",
    "capec_id": "Explain the attack pattern characteristics that match the CAPEC ID.",
    "threat_actor_name": "Cite TTPs, targets, or aliases that uniquely indicate the threat actor.",
}


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


class ACRRLHFDataset(RLHFDataset):
    """RLHFDataset that injects answer-conditioned reasoning prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ) -> None:
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

        acr_cfg = config.get("acr", {}) if config is not None else {}
        self.acr_enabled = bool(acr_cfg.get("enabled", True))
        self.max_details_chars = int(acr_cfg.get("max_details_chars", 4048))
        self.enforce_no_id = bool(acr_cfg.get("enforce_no_id_in_reasoning", True))
        self.label_details_dir = acr_cfg.get("label_details_dir")
        self.details_store = LabelDetailsStore(self.label_details_dir)
        self.task_reasoning_hints = _apply_reasoning_hints(
            DEFAULT_TASK_REASONING_HINTS, acr_cfg.get("task_reasoning_hints")
        )
        self.entity_reasoning_hints = _apply_reasoning_hints(
            DEFAULT_ENTITY_REASONING_HINTS, acr_cfg.get("entity_reasoning_hints")
        )

    def _prompt_too_long(self, messages: List[Dict[str, Any]]) -> bool:
        try:
            if self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                input_ids = self.processor(text=[raw_prompt], return_tensors="pt")["input_ids"][0]
                return len(input_ids) > self.max_prompt_length
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            input_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            return len(input_ids) > self.max_prompt_length
        except Exception:
            return False

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.acr_enabled:
            return super()._build_messages(example)

        messages = example.get(self.prompt_key)
        if not isinstance(messages, list):
            return super()._build_messages(example)

        extra_info = example.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
            example["extra_info"] = extra_info

        if "acr_orig_prompt" not in extra_info:
            extra_info["acr_orig_prompt"] = copy.deepcopy(messages)

        data_source = str(example.get("data_source") or example.get("reward_fn") or "")
        ground_truth = example.get("reward_model", {}).get("ground_truth")
        spec = get_task_spec(data_source, ground_truth, extra_info)
        entity_type = spec.entity_type
        reasoning_hint = None
        if spec.task_key:
            reasoning_hint = self.task_reasoning_hints.get(spec.task_key)
        if not reasoning_hint and entity_type:
            reasoning_hint = self.entity_reasoning_hints.get(entity_type)

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
        gold_keys = [f"{entity_type}:{label}" if entity_type else label for label in gold_norm]

        if not gold_norm:
            extra_info["acr_skipped"] = True
            extra_info["acr_skip_reason"] = "missing_labels"
            extra_info["acr_entity_type"] = entity_type or ""
            extra_info["acr_gold_labels"] = []
            extra_info["acr_gold_keys"] = []
            extra_info["acr_mode"] = "acr"
            example[self.prompt_key] = messages
            return super()._build_messages(example)

        details_labels = gold_norm
        option_text = ""
        if entity_type and len(gold_norm) == 1 and len(gold_norm[0]) == 1:
            user_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    user_text = msg["content"]
                    break
            if "Options:" in user_text:
                tail = user_text.split("Options:", 1)[1]
                for line in tail.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r"^([A-Z])\s*[\.\)]\s*(.+)$", line)
                    if not match:
                        continue
                    if match.group(1).upper() == gold_norm[0].upper():
                        option_text = match.group(2).strip()
                        break
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
                except Exception:
                    pass
                if details_labels == gold_norm:
                    details_labels = [option_text]

        details_text = None
        if entity_type:
            details_text = self.details_store.get_details(entity_type, details_labels)
        updated_messages, skipped, used_details = try_build_acr_messages(
            messages,
            gold_norm,
            details_text,
            max_details_chars=self.max_details_chars,
            prompt_too_long=self._prompt_too_long,
            enforce_no_id=self.enforce_no_id,
            reasoning_hint=reasoning_hint,
        )
        if skipped:
            extra_info["acr_skipped"] = True
            extra_info["acr_skip_reason"] = "prompt_too_long"
        else:
            extra_info["acr_skipped"] = False

        extra_info["acr_entity_type"] = entity_type or ""
        extra_info["acr_gold_labels"] = gold_norm
        extra_info["acr_gold_keys"] = gold_keys
        extra_info["acr_mode"] = "acr"
        if option_text:
            extra_info["acr_option_text"] = option_text
        if details_labels != gold_norm:
            extra_info["acr_details_labels"] = details_labels
        if spec.task_key:
            extra_info["acr_task_key"] = spec.task_key
        if used_details:
            extra_info["acr_details_chars"] = int(len(used_details))

        example[self.prompt_key] = updated_messages
        return super()._build_messages(example)


__all__ = ["ACRRLHFDataset"]

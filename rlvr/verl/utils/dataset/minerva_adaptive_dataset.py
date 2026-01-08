"""
Adaptive label-hint curriculum dataset for Minerva RLVR (LHC).
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


@dataclass
class OptionCurriculumState:
    """Per-task curriculum state."""

    k: int
    p_drop: float
    ema_acc: float
    initialized: bool = False


class OptionCurriculumController:
    """Adaptive controller for label-hint curriculum per task key."""

    def __init__(
        self,
        *,
        target_acc: float = 0.5,
        tol: float = 0.05,
        warmup_steps: int = 50,
        ema_beta: float = 0.9,
        k_min: int = 2,
        k_max: int = 30,
        k_step: int = 2,
        p_drop_init: float = 0.0,
        p_drop_step: float = 0.05,
        p_drop_max: float = 1.0,
    ) -> None:
        self.target_acc = float(target_acc)
        self.tol = float(tol)
        self.warmup_steps = int(warmup_steps)
        self.ema_beta = float(ema_beta)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.k_step = int(k_step)
        self.p_drop_init = float(p_drop_init)
        self.p_drop_step = float(p_drop_step)
        self.p_drop_max = float(p_drop_max)
        self._states: Dict[str, OptionCurriculumState] = {}

    def _get_state(self, task_key: str) -> OptionCurriculumState:
        if task_key not in self._states:
            self._states[task_key] = OptionCurriculumState(
                k=0,
                p_drop=self.p_drop_init,
                ema_acc=0.0,
                initialized=False,
            )
        return self._states[task_key]

    def get_params(self, task_key: str, global_step: int) -> Tuple[int, float]:
        """Return (K, p_drop) for the task at the current step."""
        state = self._get_state(task_key)
        if global_step <= self.warmup_steps and not state.initialized:
            return 0, state.p_drop
        return state.k, state.p_drop

    def update(self, task_key: str, batch_acc: float, global_step: int) -> OptionCurriculumState:
        """Update EMA accuracy and adapt K/p_drop for a task."""
        state = self._get_state(task_key)
        acc = float(batch_acc)
        state.ema_acc = self.ema_beta * state.ema_acc + (1.0 - self.ema_beta) * acc

        if global_step <= self.warmup_steps:
            return state

        if not state.initialized:
            if state.ema_acc >= self.target_acc:
                state.k = 0
            else:
                state.k = self.k_min
            state.initialized = True
            return state

        if state.ema_acc > self.target_acc + self.tol:
            if state.k < self.k_max:
                state.k = min(self.k_max, state.k + self.k_step)
            else:
                state.p_drop = min(self.p_drop_max, state.p_drop + self.p_drop_step)
        elif state.ema_acc < self.target_acc - self.tol:
            if state.k == 0:
                state.k = self.k_min
                state.p_drop = 0.0
            else:
                state.k = max(self.k_min, state.k - self.k_step)
                state.p_drop = max(0.0, state.p_drop - self.p_drop_step)

        return state

    def get_metrics(self, prefix: str = "option_curriculum/") -> Dict[str, float]:
        """Return a flat metrics dict of K/p_drop/ema_acc for all tasks."""
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        metrics: Dict[str, float] = {}
        for task_key, state in self._states.items():
            metrics[f"{prefix}K/{task_key}"] = float(state.k)
            metrics[f"{prefix}p_drop/{task_key}"] = float(state.p_drop)
            metrics[f"{prefix}ema_acc/{task_key}"] = float(state.ema_acc)
        return metrics


class AdaptiveOptionRLHFDataset(RLHFDataset):
    """RLHFDataset that injects adaptive label-hint options into prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ) -> None:
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

        adaptive_cfg = config.get("adaptive_options", {}) if config is not None else {}
        self.adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
        self.candidate_pool_key = adaptive_cfg.get("candidate_pool_key", "candidate_pool_top100")
        self.buffer = int(adaptive_cfg.get("buffer", 10))
        self.base_seed = int(adaptive_cfg.get("base_seed", adaptive_cfg.get("seed", 1337)))
        self.score_threshold = float(adaptive_cfg.get("score_threshold", 0.5))
        self.option_desc_max_chars = int(adaptive_cfg.get("option_desc_max_chars", 400))
        self.option_include_descriptions = bool(adaptive_cfg.get("include_descriptions", False))
        self._global_step = 0

        self.option_controller: Optional[OptionCurriculumController] = None
        if self.adaptive_enabled:
            self.option_controller = OptionCurriculumController(
                target_acc=adaptive_cfg.get("target_acc", 0.5),
                tol=adaptive_cfg.get("tol", 0.05),
                warmup_steps=adaptive_cfg.get("warmup_steps", 50),
                ema_beta=adaptive_cfg.get("ema_beta", 0.9),
                k_min=adaptive_cfg.get("k_min", 2),
                k_max=adaptive_cfg.get("k_max", 30),
                k_step=adaptive_cfg.get("k_step", 2),
                p_drop_init=adaptive_cfg.get("p_drop_init", 0.0),
                p_drop_step=adaptive_cfg.get("p_drop_step", 0.05),
                p_drop_max=adaptive_cfg.get("p_drop_max", 1.0),
            )

    def _normalize_pool(self, pool: Any) -> List[Dict[str, str]]:
        if not isinstance(pool, list):
            return []
        seen = set()
        normalized: List[Dict[str, str]] = []
        for item in pool:
            if item is None:
                continue
            if isinstance(item, dict):
                val = None
                for key in ("id", "label", "value", "name"):
                    if key in item:
                        val = item.get(key)
                        break
                if val is None:
                    continue
                text = str(val).strip()
                name = str(item.get("name") or "").strip()
                desc = str(item.get("description") or item.get("desc") or "").strip()
            else:
                text = str(item).strip()
                name = ""
                desc = ""
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append({"id": text, "name": name, "description": desc})
        return normalized

    def _extract_single_label(self, ground_truth: Any) -> Optional[str]:
        if ground_truth is None:
            return None
        if isinstance(ground_truth, str):
            return ground_truth.strip()
        if isinstance(ground_truth, (list, tuple)) and len(ground_truth) == 1:
            return str(ground_truth[0]).strip()
        if isinstance(ground_truth, dict):
            key_order = (
                "technique_id",
                "detection_id",
                "mitigation_id",
                "cwe_id",
                "capec_id",
                "attack_id",
                "tactic_id",
                "id",
                "label",
            )
            list_key_order = (
                "technique_ids",
                "tactic_ids",
                "mitigation_ids",
                "cwe_ids",
                "capec_ids",
                "attack_ids",
                "ids",
                "labels",
            )
            for key in key_order:
                if key in ground_truth:
                    value = ground_truth.get(key)
                    if isinstance(value, str):
                        return value.strip()
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        return str(value[0]).strip()
            for key in list_key_order:
                if key in ground_truth:
                    value = ground_truth.get(key)
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        return str(value[0]).strip()
        return None

    def _format_option_item(self, item: Dict[str, str]) -> str:
        ident = item.get("id", "").strip()
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        if not self.option_include_descriptions:
            desc = ""
        elif desc and self.option_desc_max_chars > 0 and len(desc) > self.option_desc_max_chars:
            desc = desc[: self.option_desc_max_chars].rsplit(" ", 1)[0] + "..."
        parts = [p for p in (ident, name, desc) if p]
        return " | ".join(parts) if parts else ident

    def _format_options(self, options: List[Dict[str, str]]) -> str:
        lines = ["Candidate IDs (choose one):"]
        for idx, item in enumerate(options, start=1):
            lines.append(f"{idx}) {self._format_option_item(item)}")
        lines.append(
            "The correct ID is one of the candidates above. You may use them as a reference, but do not mention the list. "
            "Reason step by step to arrive at the answer."
        )
        return "\n".join(lines)

    def _append_options(self, messages: List[Dict[str, Any]], options: List[str]) -> bool:
        if not messages:
            return False
        target_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                target_idx = idx
                break
        if target_idx is None:
            target_idx = len(messages) - 1
        content = messages[target_idx].get("content", "")
        if not isinstance(content, str):
            return False
        options_text = self._format_options(options)
        content = content.rstrip()
        if content:
            content = f"{content}\n\n{options_text}"
        else:
            content = options_text
        messages[target_idx]["content"] = content
        return True

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
        if not self.adaptive_enabled or self.option_controller is None:
            return super()._build_messages(example)

        messages = example.get(self.prompt_key)
        if not isinstance(messages, list):
            return super()._build_messages(example)

        task_key = str(example.get("data_source") or example.get("reward_fn") or "unknown")
        pool_raw = example.get(self.candidate_pool_key)
        if pool_raw is None:
            pool_raw = (example.get("extra_info") or {}).get(self.candidate_pool_key)
        pool = self._normalize_pool(pool_raw)
        ground_truth = example.get("reward_model", {}).get("ground_truth")
        gold = self._extract_single_label(ground_truth)

        extra_info = example.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
            example["extra_info"] = extra_info

        hint_used = False
        hint_k = 0
        hint_seed = None
        if pool and gold:
            if gold not in {item["id"] for item in pool}:
                pool = [{"id": gold, "name": "", "description": ""}] + pool
            sample_index = extra_info.get("index", 0)
            try:
                sample_index_int = int(sample_index)
            except (TypeError, ValueError):
                sample_index_int = 0
            hint_seed = self.base_seed ^ (int(self._global_step) << 16) ^ sample_index_int
            k, p_drop = self.option_controller.get_params(task_key, self._global_step)
            if k > 0 and random.Random(hint_seed).random() >= p_drop:
                effective_k = min(k, len(pool))
                base_messages = copy.deepcopy(messages)
                while effective_k > 0:
                    rng = random.Random(hint_seed)
                    top_n = min(len(pool), effective_k + self.buffer)
                    top_pool = pool[:top_n]
                    distractors = [item for item in top_pool if item.get("id") != gold]
                    if len(distractors) < max(0, effective_k - 1):
                        extras = [
                            item
                            for item in pool
                            if item.get("id") != gold and item not in distractors
                        ]
                        distractors.extend(extras)
                    take = min(max(0, effective_k - 1), len(distractors))
                    sampled = rng.sample(distractors, take) if take > 0 else []
                    options = [{"id": gold, "name": "", "description": ""}] + sampled
                    rng.shuffle(options)
                    trial_messages = copy.deepcopy(base_messages)
                    if self._append_options(trial_messages, options):
                        if not self._prompt_too_long(trial_messages):
                            messages = trial_messages
                            hint_used = True
                            hint_k = len(options)
                            break
                    effective_k -= 1

        extra_info["hint_used"] = bool(hint_used)
        extra_info["hint_K"] = int(hint_k)
        extra_info["hint_pool_size"] = int(len(pool))
        if hint_seed is not None:
            extra_info["hint_seed"] = int(hint_seed)

        example[self.prompt_key] = messages
        return super()._build_messages(example)

    def update_option_curriculum_from_rollout(
        self,
        data_source_arr: Iterable[Any],
        uid_arr: Iterable[Any],
        is_correct_arr: Iterable[Any],
        global_step: int,
    ) -> Dict[str, float]:
        if not self.adaptive_enabled or self.option_controller is None:
            return {}
        self._global_step = int(global_step)

        data_source_np = np.asarray(list(data_source_arr))
        uid_np = np.asarray(list(uid_arr))
        is_correct_np = np.asarray(list(is_correct_arr))

        if len(data_source_np) == 0:
            return {}

        grouped: Dict[Tuple[str, str], List[float]] = {}
        for ds, uid, correct in zip(data_source_np, uid_np, is_correct_np, strict=False):
            task_key = str(ds)
            uid_key = str(uid)
            grouped.setdefault((task_key, uid_key), []).append(float(bool(correct)))

        task_accs: Dict[str, List[float]] = {}
        for (task_key, _), vals in grouped.items():
            if not vals:
                continue
            acc = float(sum(vals) / len(vals))
            task_accs.setdefault(task_key, []).append(acc)

        metrics: Dict[str, float] = {}
        for task_key, accs in task_accs.items():
            batch_acc = float(sum(accs) / len(accs))
            self.option_controller.update(task_key, batch_acc, self._global_step)
            metrics[f"option_curriculum/batch_acc/{task_key}"] = batch_acc
            metrics[f"option_curriculum/batch_count/{task_key}"] = float(len(accs))

        return metrics


__all__ = [
    "AdaptiveOptionRLHFDataset",
    "OptionCurriculumController",
]

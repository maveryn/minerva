"""
Stochastic label-hint curriculum dataset for Minerva RLVR (SLHC).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


@dataclass
class SLHCTaskState:
    """Per-task controller state."""

    mu_d: float
    ema_acc: float
    ema_zero: float
    ema_all: float


class StochasticSLHCController:
    """Stochastic SLHC controller keyed by task key (data_source)."""

    def __init__(self, *, config: DictConfig) -> None:
        slhc_cfg = config.get("stochastic_slhc", {}) if config is not None else {}
        self.control_mode = str(slhc_cfg.get("control_mode", "acc_target"))
        self.success_threshold = float(slhc_cfg.get("success_threshold", 0.1))
        self.candidate_pool_key = slhc_cfg.get("candidate_pool_key", "candidate_pool_top100")
        self.buffer = int(slhc_cfg.get("buffer", 10))
        self.k_max_total = int(slhc_cfg.get("K_max_total", 30))
        self.d_min = int(slhc_cfg.get("d_min", 1))
        self.sigma_init = float(slhc_cfg.get("sigma_init", 2.0))
        self.p_nohint_start = float(slhc_cfg.get("p_nohint_start", 0.05))
        self.p_nohint_end = float(slhc_cfg.get("p_nohint_end", 0.50))
        self.total_steps = int(slhc_cfg.get("total_steps", 0))

        self.ema_beta = float(slhc_cfg.get("ema_beta", 0.90))
        self.lr_mu = float(slhc_cfg.get("lr_mu", 0.50))
        self.target_acc = float(slhc_cfg.get("target_acc", 0.50))
        self.zero_target = float(slhc_cfg.get("zero_target", 0.15))
        self.all_target = float(slhc_cfg.get("all_target", 0.15))

        self.drift_delta = float(slhc_cfg.get("drift_delta", 0.25))
        self.alpha_zero = float(slhc_cfg.get("alpha_zero", 1.0))
        self.alpha_all = float(slhc_cfg.get("alpha_all", 0.7))
        self.temp_zero = float(slhc_cfg.get("temp_zero", 0.05))
        self.temp_all = float(slhc_cfg.get("temp_all", 0.05))

        base_seed = slhc_cfg.get("seed", config.get("seed", 1337) if config is not None else 1337)
        self.base_seed = int(base_seed)
        self.rng = random.Random(self.base_seed)

        self.global_step = 0
        self._states: Dict[str, SLHCTaskState] = {}

    def _get_state(self, task_key: str) -> SLHCTaskState:
        if task_key not in self._states:
            self._states[task_key] = SLHCTaskState(
                mu_d=1.0,
                ema_acc=0.0,
                ema_zero=0.0,
                ema_all=0.0,
            )
        return self._states[task_key]

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def get_forced_nohint_p(self, global_step: Optional[int] = None, total_steps: Optional[int] = None) -> float:
        step = self.global_step if global_step is None else int(global_step)
        steps_total = self.total_steps if total_steps is None else int(total_steps)
        if steps_total <= 0:
            return self.p_nohint_end
        ratio = min(max(step / steps_total, 0.0), 1.0)
        return self.p_nohint_start + (self.p_nohint_end - self.p_nohint_start) * ratio

    def _sample_gaussian(self, values: List[int], mu: float, sigma: float) -> int:
        if not values:
            return 0
        sigma = max(float(sigma), 1e-6)
        weights = [math.exp(-((val - mu) ** 2) / (2.0 * sigma * sigma)) for val in values]
        total = sum(weights)
        if total <= 0:
            return values[0]
        pick = self.rng.random() * total
        acc = 0.0
        for val, weight in zip(values, weights, strict=False):
            acc += weight
            if pick <= acc:
                return val
        return values[-1]

    def sample_decision(self, task_key: str, c: int, pool_size: int, uid: Any) -> Dict[str, Any]:
        seed = self.rng.randint(0, 2**31 - 1)
        if c <= 0 or pool_size <= 0:
            return {
                "ctrl_active": True,
                "hint_used": False,
                "nohint_reason": "unavailable",
                "d": None,
                "seed": seed,
                "K_total": None,
            }

        p_forced = self.get_forced_nohint_p(self.global_step, self.total_steps)
        if self.rng.random() < p_forced:
            return {
                "ctrl_active": False,
                "hint_used": False,
                "nohint_reason": "forced_schedule",
                "d": None,
                "seed": seed,
                "K_total": None,
            }

        d_max_sample = min(self.k_max_total - c, pool_size - c)
        if d_max_sample < self.d_min:
            return {
                "ctrl_active": True,
                "hint_used": False,
                "nohint_reason": "unavailable",
                "d": None,
                "seed": seed,
                "K_total": None,
            }

        state = self._get_state(task_key)
        domain = list(range(self.d_min, d_max_sample + 2))
        sampled_d = self._sample_gaussian(domain, state.mu_d, self.sigma_init)
        if sampled_d == d_max_sample + 1:
            return {
                "ctrl_active": True,
                "hint_used": False,
                "nohint_reason": "sentinel",
                "d": sampled_d,
                "seed": seed,
                "K_total": None,
            }
        return {
            "ctrl_active": True,
            "hint_used": True,
            "nohint_reason": None,
            "d": sampled_d,
            "seed": seed,
            "K_total": c + sampled_d,
        }

    def update_from_groups(
        self,
        group_summaries: Iterable[Dict[str, Any]],
        global_step: int,
        total_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        self.global_step = int(global_step)
        if total_steps is not None:
            self.total_steps = int(total_steps)

        control_mode_id = {"acc_target": 0.0, "zero_constraint": 1.0, "two_tail_smooth": 2.0}
        base_metrics = {
            "slhc/p_forced_nohint": float(self.get_forced_nohint_p(self.global_step, self.total_steps)),
            "slhc/control_mode": float(control_mode_id.get(self.control_mode, -1.0)),
        }

        controlled = [g for g in group_summaries if g.get("ctrl_active") is True]
        if not controlled:
            return base_metrics

        agg: Dict[str, Dict[str, float]] = {}
        for group in controlled:
            task_key = str(group.get("task_key", "unknown"))
            slot = agg.setdefault(task_key, {"acc": 0.0, "zero": 0.0, "all": 0.0, "count": 0.0})
            slot["acc"] += float(group.get("acc_g", 0.0))
            slot["zero"] += float(group.get("zero_g", 0.0))
            slot["all"] += float(group.get("all_g", 0.0))
            slot["count"] += 1.0

        metrics: Dict[str, float] = dict(base_metrics)

        for task_key, stats in agg.items():
            count = max(stats["count"], 1.0)
            acc_mean = stats["acc"] / count
            zero_mean = stats["zero"] / count
            all_mean = stats["all"] / count

            state = self._get_state(task_key)
            state.ema_acc = self.ema_beta * state.ema_acc + (1.0 - self.ema_beta) * acc_mean
            state.ema_zero = self.ema_beta * state.ema_zero + (1.0 - self.ema_beta) * zero_mean
            state.ema_all = self.ema_beta * state.ema_all + (1.0 - self.ema_beta) * all_mean

            if self.control_mode == "acc_target":
                state.mu_d += self.lr_mu * (state.ema_acc - self.target_acc)
            elif self.control_mode == "zero_constraint":
                state.mu_d += self.lr_mu * (self.zero_target - state.ema_zero)
            elif self.control_mode == "two_tail_smooth":
                dz = self._sigmoid((state.ema_zero - self.zero_target) / self.temp_zero)
                du = self._sigmoid((state.ema_all - self.all_target) / self.temp_all)
                state.mu_d += self.drift_delta + (self.alpha_all * du) - (self.alpha_zero * dz)

            state.mu_d = max(self.d_min, min(self.k_max_total, state.mu_d))

            metrics[f"slhc/mu_d/{task_key}"] = float(state.mu_d)
            metrics[f"slhc/ema_acc/{task_key}"] = float(state.ema_acc)
            metrics[f"slhc/ema_zero/{task_key}"] = float(state.ema_zero)
            metrics[f"slhc/ema_all/{task_key}"] = float(state.ema_all)

        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "total_steps": int(self.total_steps),
            "control_mode": self.control_mode,
            "states": {
                task_key: {
                    "mu_d": float(state.mu_d),
                    "ema_acc": float(state.ema_acc),
                    "ema_zero": float(state.ema_zero),
                    "ema_all": float(state.ema_all),
                }
                for task_key, state in self._states.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        self.global_step = int(state_dict.get("global_step", self.global_step))
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.control_mode = str(state_dict.get("control_mode", self.control_mode))
        states = state_dict.get("states", {})
        if not isinstance(states, dict):
            return
        restored: Dict[str, SLHCTaskState] = {}
        for task_key, payload in states.items():
            if not isinstance(payload, dict):
                continue
            restored[str(task_key)] = SLHCTaskState(
                mu_d=float(payload.get("mu_d", 1.0)),
                ema_acc=float(payload.get("ema_acc", 0.0)),
                ema_zero=float(payload.get("ema_zero", 0.0)),
                ema_all=float(payload.get("ema_all", 0.0)),
            )
        self._states = restored


class StochasticSLHCRLHFDataset(RLHFDataset):
    """RLHFDataset that injects stochastic SLHC label hints into prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ) -> None:
        self.slhc_enabled = False
        self.candidate_pool_key = "candidate_pool_top100"
        self.buffer = 10
        self.slhc_controller = None
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)
        slhc_cfg = config.get("stochastic_slhc", {}) if config is not None else {}
        self.slhc_enabled = bool(slhc_cfg.get("enabled", False))
        self.candidate_pool_key = slhc_cfg.get("candidate_pool_key", "candidate_pool_top100")
        self.buffer = int(slhc_cfg.get("buffer", 10))

        if self.slhc_enabled:
            self.slhc_controller = StochasticSLHCController(config=config)

    def _normalize_pool(self, pool: Any) -> List[str]:
        if not isinstance(pool, list):
            return []
        seen = set()
        normalized: List[str] = []
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
            else:
                text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def _dedupe_labels(self, values: Iterable[Any]) -> List[str]:
        seen = set()
        out: List[str] = []
        for raw in values or []:
            val = str(raw).strip()
            if not val or val in seen:
                continue
            seen.add(val)
            out.append(val)
        return out

    def _extract_label_list(self, ground_truth: Any) -> List[str]:
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            val = ground_truth.strip()
            return [val] if val else []
        if isinstance(ground_truth, (list, tuple)):
            return self._dedupe_labels(ground_truth)
        if isinstance(ground_truth, dict):
            list_key_order = (
                "technique_ids",
                "tactic_ids",
                "mitigation_ids",
                "detection_ids",
                "cwe_ids",
                "capec_ids",
                "attack_ids",
                "ids",
                "labels",
            )
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
            for key in list_key_order:
                if key in ground_truth:
                    value = ground_truth.get(key)
                    if isinstance(value, str):
                        val = value.strip()
                        return [val] if val else []
                    if isinstance(value, (list, tuple)):
                        return self._dedupe_labels(value)
            for key in key_order:
                if key in ground_truth:
                    value = ground_truth.get(key)
                    if isinstance(value, str):
                        val = value.strip()
                        return [val] if val else []
                    if isinstance(value, (list, tuple)):
                        return self._dedupe_labels(value)
        return []

    def _format_options(self, options: List[str], required_count: int) -> str:
        count = max(1, int(required_count))
        label = "ID" if count == 1 else "IDs"
        lines = [f"Candidate {label} (choose EXACTLY {count}):"]
        for idx, opt in enumerate(options, start=1):
            lines.append(f"{idx}) {opt}")
        if count == 1:
            tail = "The correct ID is one of the candidates above. "
        else:
            tail = "The correct IDs are among the candidates above. "
        lines.append(
            f"{tail}You may use them as a reference, but do not mention the list. "
            "Reason step by step to arrive at the answer."
        )
        return "\n".join(lines)

    def _append_options(
        self,
        messages: List[Dict[str, Any]],
        options: List[str],
        required_count: int,
    ) -> bool:
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
        options_text = self._format_options(options, required_count)
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
        if not self.slhc_enabled or self.slhc_controller is None:
            return super()._build_messages(example)

        messages = example.get(self.prompt_key)
        if not isinstance(messages, list):
            return super()._build_messages(example)

        extra_info = example.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        if "slhc_prompt_nohint" not in extra_info:
            extra_info["slhc_prompt_nohint"] = copy.deepcopy(messages)
        example["extra_info"] = extra_info

        split_label = extra_info.get("split")
        if split_label is not None and str(split_label).lower() not in ("train",):
            return super()._build_messages(example)

        task_key = str(example.get("data_source") or example.get("reward_fn") or "unknown")
        uid = example.get("uid")
        if uid is None:
            sample_index = extra_info.get("index", 0)
            uid = f"{task_key}::{sample_index}"
            example["uid"] = uid

        pool_raw = example.get(self.candidate_pool_key)
        if pool_raw is None:
            pool_raw = extra_info.get(self.candidate_pool_key)
        pool = self._normalize_pool(pool_raw)

        ground_truth = example.get("reward_model", {}).get("ground_truth")
        gold_ids = self._extract_label_list(ground_truth)
        c = len(gold_ids)
        pool_size = len(pool)

        decision = self.slhc_controller.sample_decision(task_key=task_key, c=c, pool_size=pool_size, uid=uid)
        extra_info["slhc_ctrl_active"] = bool(decision["ctrl_active"])
        extra_info["slhc_hint_used"] = bool(decision["hint_used"])
        extra_info["slhc_nohint_reason"] = decision.get("nohint_reason")
        extra_info["slhc_hint_seed"] = int(decision.get("seed", 0))
        extra_info["slhc_hint_pool_size"] = int(pool_size)

        if not decision["hint_used"]:
            return super()._build_messages(example)

        if pool and gold_ids:
            pool_ids = set(pool)
            missing = [gid for gid in gold_ids if gid not in pool_ids]
            if missing:
                pool = missing + pool
                pool_ids.update(missing)

        hint_seed = decision.get("seed", 0)
        requested_d = decision.get("d", 0) or 0
        effective_d = int(requested_d)
        base_messages = copy.deepcopy(messages)
        hint_used = False
        hint_k_total = None
        hint_d_used = None

        while effective_d >= 0:
            total_k = c + effective_d
            top_n = min(len(pool), total_k + self.buffer) if pool else 0
            pool_top = pool[:top_n] if top_n > 0 else []
            gold_set = set(gold_ids)
            negatives = [item for item in pool_top if item not in gold_set]
            if len(negatives) < effective_d:
                effective_d = len(negatives)
                total_k = c + effective_d
            rng = random.Random(hint_seed)
            distractors = rng.sample(negatives, effective_d) if effective_d > 0 else []
            options = list(gold_ids) + distractors
            rng.shuffle(options)
            trial_messages = copy.deepcopy(base_messages)
            if self._append_options(trial_messages, options, required_count=c):
                if not self._prompt_too_long(trial_messages):
                    messages = trial_messages
                    hint_used = True
                    hint_k_total = total_k
                    hint_d_used = effective_d
                    break
            effective_d -= 1

        if not hint_used:
            extra_info["slhc_hint_used"] = False
            extra_info["slhc_nohint_reason"] = "unavailable"
            return super()._build_messages(example)

        extra_info["slhc_hint_used"] = True
        extra_info["slhc_hint_d"] = int(hint_d_used or 0)
        extra_info["slhc_hint_K_total"] = int(hint_k_total or c)

        example[self.prompt_key] = messages
        return super()._build_messages(example)

    def update_slhc_controller_from_groups(
        self,
        group_summaries: Iterable[Dict[str, Any]],
        global_step: int,
        total_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        if not self.slhc_enabled or self.slhc_controller is None:
            return {}
        return self.slhc_controller.update_from_groups(
            group_summaries=group_summaries,
            global_step=global_step,
            total_steps=total_steps,
        )

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "slhc_enabled": self.slhc_enabled,
        }
        if self.slhc_controller is not None:
            state["slhc_controller"] = self.slhc_controller.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        controller_state = state_dict.get("slhc_controller")
        if self.slhc_controller is not None and isinstance(controller_state, dict):
            self.slhc_controller.load_state_dict(controller_state)


__all__ = [
    "StochasticSLHCRLHFDataset",
    "StochasticSLHCController",
]

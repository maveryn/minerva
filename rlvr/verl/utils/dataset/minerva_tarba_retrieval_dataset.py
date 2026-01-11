"""TARBA retrieval curriculum dataset for Minerva RLVR."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from minerva.retrieval.task_specs import extract_labels_from_truth, get_task_spec, normalize_label


@dataclass
class TarbaTaskState:
    budget_B: int
    p_noret: float
    ema_acc_ret: float = 0.0
    ema_acc_noret: float = 0.0
    steps_ret: int = 0
    steps_noret: int = 0


class RetrievalBudgetController:
    """Per-task TARBA controller for retrieval budgets."""

    def __init__(self, config: DictConfig | dict | None = None) -> None:
        cfg = config.get("tarba", {}) if config is not None else {}
        self.ema_beta = float(cfg.get("ema_beta", 0.90))
        self.target_acc_noret = float(cfg.get("target_acc_noret", 0.60))
        self.tol = float(cfg.get("tol", 0.05))
        self.p_noret_init = float(cfg.get("p_noret_init", 0.10))
        self.p_noret_max = float(cfg.get("p_noret_max", 0.90))
        self.p_step = float(cfg.get("p_step", 0.05))
        self.b_min = int(cfg.get("B_min", 0))
        self.default_b_max = int(cfg.get("default_B_max", 8))
        self.b_step = int(cfg.get("B_step", 1))
        self.min_steps_before_anneal = int(cfg.get("min_steps_before_anneal", 50))
        self.b_max_by_task = cfg.get("B_max_by_task", {}) or {}
        seed = cfg.get("seed", config.get("seed", 1337) if config is not None else 1337)
        self.rng = random.Random(int(seed))
        self._states: Dict[str, TarbaTaskState] = {}

    def _get_state(self, task_key: str) -> TarbaTaskState:
        if task_key not in self._states:
            b_max = int(self.b_max_by_task.get(task_key, self.default_b_max))
            self._states[task_key] = TarbaTaskState(budget_B=b_max, p_noret=self.p_noret_init)
        return self._states[task_key]

    def sample(self, task_key: str) -> Dict[str, Any]:
        state = self._get_state(task_key)
        allow_retrieval = self.rng.random() >= state.p_noret
        budget_B = state.budget_B if allow_retrieval else 0
        seed = self.rng.randint(0, 2**31 - 1)
        return {
            "allow_retrieval": allow_retrieval,
            "budget_B": int(budget_B),
            "seed": int(seed),
        }

    def update_from_groups(self, group_summaries: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        per_task: Dict[str, Dict[str, list[float]]] = {}
        for summary in group_summaries:
            task_key = str(summary.get("task_key", ""))
            allow_retrieval = bool(summary.get("allow_retrieval", False))
            acc_g = float(summary.get("acc_g", 0.0))
            buckets = per_task.setdefault(task_key, {"ret": [], "noret": []})
            if allow_retrieval:
                buckets["ret"].append(acc_g)
            else:
                buckets["noret"].append(acc_g)

        for task_key, buckets in per_task.items():
            state = self._get_state(task_key)
            ret_vals = buckets.get("ret", [])
            noret_vals = buckets.get("noret", [])

            if ret_vals:
                state.steps_ret += 1
                state.ema_acc_ret = self._ema(state.ema_acc_ret, sum(ret_vals) / len(ret_vals))
            if noret_vals:
                state.steps_noret += 1
                state.ema_acc_noret = self._ema(state.ema_acc_noret, sum(noret_vals) / len(noret_vals))

            if state.steps_noret >= self.min_steps_before_anneal:
                acc_mix = (1.0 - state.p_noret) * state.ema_acc_ret + state.p_noret * state.ema_acc_noret
                if acc_mix > self.target_acc_noret + self.tol:
                    state.p_noret = min(self.p_noret_max, state.p_noret + self.p_step)

        return self.get_metrics(prefix="tarba_ctrl/")

    def _ema(self, prev: float, value: float) -> float:
        return self.ema_beta * prev + (1.0 - self.ema_beta) * value

    def get_metrics(self, prefix: str = "") -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for task_key, state in self._states.items():
            metrics[f"{prefix}B/{task_key}"] = float(state.budget_B)
            metrics[f"{prefix}p_noret/{task_key}"] = float(state.p_noret)
            metrics[f"{prefix}ema_ret/{task_key}"] = float(state.ema_acc_ret)
            metrics[f"{prefix}ema_noret/{task_key}"] = float(state.ema_acc_noret)
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "states": {
                task_key: {
                    "budget_B": state.budget_B,
                    "p_noret": state.p_noret,
                    "ema_acc_ret": state.ema_acc_ret,
                    "ema_acc_noret": state.ema_acc_noret,
                    "steps_ret": state.steps_ret,
                    "steps_noret": state.steps_noret,
                }
                for task_key, state in self._states.items()
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        states = state_dict.get("states", {}) if isinstance(state_dict, dict) else {}
        restored: Dict[str, TarbaTaskState] = {}
        for task_key, payload in states.items():
            if not isinstance(payload, dict):
                continue
            restored[str(task_key)] = TarbaTaskState(
                budget_B=int(payload.get("budget_B", self.default_b_max)),
                p_noret=float(payload.get("p_noret", self.p_noret_init)),
                ema_acc_ret=float(payload.get("ema_acc_ret", 0.0)),
                ema_acc_noret=float(payload.get("ema_acc_noret", 0.0)),
                steps_ret=int(payload.get("steps_ret", 0)),
                steps_noret=int(payload.get("steps_noret", 0)),
            )
        self._states = restored


class TarbaRLHFDataset(RLHFDataset):
    """RLHFDataset that injects TARBA retrieval tool instructions into prompts."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ) -> None:
        tarba_cfg = config.get("tarba", {}) if config is not None else {}
        self._tarba_cfg = tarba_cfg
        self.tarba_enabled = bool(tarba_cfg.get("enabled", False))
        self.tarba_controller: Optional[RetrievalBudgetController] = None
        self.eval_mode = tarba_cfg.get("eval_mode")
        self.eval_budget_B = tarba_cfg.get("eval_budget_B", None)
        self.train_label_type = tarba_cfg.get("train_label_type", "all")
        self.eval_label_type = tarba_cfg.get("eval_label_type", "all")
        super().__init__(data_files=data_files, tokenizer=tokenizer, config=config, processor=processor)

        if self.tarba_enabled:
            self.tarba_controller = RetrievalBudgetController(config=config)

    def _append_tool_instructions(self, messages: list[dict], allow_retrieval: bool, budget_B: int) -> bool:
        if not messages:
            return False
        if not allow_retrieval:
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
        content = content.rstrip()
        block = (
            "Retrieval tool:\n"
            "Use short keyword queries to retrieve label docs that support the answer.\n"
            "You may request retrieval at most once.\n"
            f"Request at most B={budget_B} docs.\n"
            "Limit the query to 128 characters; do not paste long lists.\n"
            "Output exactly one JSON object with `query` and `topk` fields only; no extra text.\n"
            "Examples:\n"
            "1) {\"query\":\"cwe-79 xss sql inj\",\"topk\":3}\n"
            "2) {\"query\":\"credential dumping dcsync process injection\",\"topk\":5}\n"
            "3) {\"query\":\"APT28 Lazarus FIN7 spearphishing banking malware\",\"topk\":2}"
        )
        messages[target_idx]["content"] = f"{content}\n\n{block}" if content else block
        return True

    def _resolve_label_type(self, raw_value: Any, label_type: Optional[str]) -> str:
        raw = str(raw_value or "").strip().lower()
        if raw in {"", "task", "per_type", "label"}:
            return str(label_type or "")
        if raw in {"all", "global"}:
            return "all"
        return str(raw_value)

    def _resolve_eval_label_type(self, label_type: Optional[str]) -> str:
        return self._resolve_label_type(self.eval_label_type, label_type)

    def _resolve_train_label_type(self, label_type: Optional[str]) -> str:
        return self._resolve_label_type(self.train_label_type, label_type)

    def _update_tools_kwargs(
        self,
        extra_info: Dict[str, Any],
        allow_retrieval: bool,
        budget_B: int,
        label_type: Optional[str] = None,
    ) -> None:
        tools_kwargs = extra_info.get("tools_kwargs")
        if not isinstance(tools_kwargs, dict):
            tools_kwargs = {}
        if allow_retrieval:
            cti_kwargs = tools_kwargs.get("cti_retrieve")
            if not isinstance(cti_kwargs, dict):
                cti_kwargs = {}
            execute_kwargs = cti_kwargs.get("execute_kwargs")
            if not isinstance(execute_kwargs, dict):
                execute_kwargs = {}
            execute_kwargs["budget_B"] = int(budget_B)
            if label_type:
                execute_kwargs["label_type"] = str(label_type)
            cti_kwargs["execute_kwargs"] = execute_kwargs
            tools_kwargs["cti_retrieve"] = cti_kwargs
        else:
            tools_kwargs.pop("cti_retrieve", None)
        extra_info["tools_kwargs"] = tools_kwargs

    def _max_budget_for_filter(self) -> int:
        cfg = self._tarba_cfg or {}
        default_b_max = int(cfg.get("default_B_max", 8))
        b_max_by_task = cfg.get("B_max_by_task", {}) or {}
        max_values = []
        for val in b_max_by_task.values():
            try:
                max_values.append(int(val))
            except (TypeError, ValueError):
                continue
        return max([default_b_max] + max_values) if max_values else default_b_max

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if not self.filter_overlong_prompts or not self.tarba_enabled:
            return super().maybe_filter_out_long_prompts(dataframe)

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key
        max_budget = self._max_budget_for_filter()

        def build_messages(doc):
            messages = doc.get(prompt_key)
            if not isinstance(messages, list):
                return messages
            messages = copy.deepcopy(messages)
            self._append_tool_instructions(messages, allow_retrieval=True, budget_B=max_budget)
            return messages

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc2len(doc) -> int:
                messages = build_messages(doc)
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

        else:

            def doc2len(doc) -> int:
                messages = build_messages(doc)
                return len(
                    tokenizer.apply_chat_template(messages, add_generation_prompt=True, **self.apply_chat_template_kwargs)
                )

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def _build_messages(self, example: Dict[str, Any]) -> list[dict]:
        if not self.tarba_enabled or self.tarba_controller is None:
            return super()._build_messages(example)

        messages = example.get(self.prompt_key)
        if not isinstance(messages, list):
            return super()._build_messages(example)

        extra_info = example.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        if "tarba_prompt_no_tool" not in extra_info:
            extra_info["tarba_prompt_no_tool"] = copy.deepcopy(messages)
        example["extra_info"] = extra_info

        task_key = str(example.get("data_source") or example.get("reward_fn") or "unknown")
        uid = example.get("uid")
        if uid is None:
            sample_index = extra_info.get("index", 0)
            uid = f"{task_key}::{sample_index}"
            example["uid"] = uid

        ground_truth = example.get("reward_model", {}).get("ground_truth")
        spec = get_task_spec(task_key, ground_truth)
        label_type = spec.label_type
        gold_ids = extract_labels_from_truth(ground_truth, spec)
        gold_doc_ids = [f"{label_type}:{normalize_label(label_type, gid)}" for gid in gold_ids] if label_type else []

        split_label = str(extra_info.get("split", "")).lower()
        allow_retrieval = False
        budget_B = 0
        seed = 0

        retrieval_label_type = label_type or ""
        if label_type:
            if split_label and split_label not in {"train"}:
                if str(self.eval_mode or "").lower() == "ret_on":
                    allow_retrieval = True
                    if self.eval_budget_B is not None:
                        budget_B = int(self.eval_budget_B)
                    else:
                        budget_B = int(self.tarba_controller._get_state(task_key).budget_B)
                    retrieval_label_type = self._resolve_eval_label_type(label_type)
                else:
                    allow_retrieval = False
                    budget_B = 0
            else:
                decision = self.tarba_controller.sample(task_key)
                allow_retrieval = bool(decision["allow_retrieval"])
                budget_B = int(decision["budget_B"])
                seed = int(decision["seed"])
                if allow_retrieval:
                    retrieval_label_type = self._resolve_train_label_type(label_type)
        if allow_retrieval and budget_B <= 0:
            allow_retrieval = False
            budget_B = 0

        extra_info["tarba_allow_retrieval"] = bool(allow_retrieval)
        extra_info["tarba_budget_B"] = int(budget_B)
        extra_info["tarba_label_type"] = label_type or ""
        extra_info["tarba_gold_doc_ids"] = gold_doc_ids
        extra_info["tarba_seed"] = int(seed)
        self._update_tools_kwargs(
            extra_info,
            allow_retrieval=allow_retrieval,
            budget_B=budget_B,
            label_type=retrieval_label_type or label_type,
        )

        messages = copy.deepcopy(messages)
        self._append_tool_instructions(messages, allow_retrieval=allow_retrieval, budget_B=budget_B)
        example[self.prompt_key] = messages
        return super()._build_messages(example)

    def update_tarba_controller_from_groups(
        self,
        group_summaries: Iterable[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not self.tarba_enabled or self.tarba_controller is None:
            return {}
        return self.tarba_controller.update_from_groups(group_summaries)

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"tarba_enabled": self.tarba_enabled}
        if self.tarba_controller is not None:
            state["tarba_controller"] = self.tarba_controller.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        controller_state = state_dict.get("tarba_controller")
        if self.tarba_controller is not None and isinstance(controller_state, dict):
            self.tarba_controller.load_state_dict(controller_state)


__all__ = ["TarbaRLHFDataset", "RetrievalBudgetController"]

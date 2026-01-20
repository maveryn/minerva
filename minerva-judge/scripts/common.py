"""Shared helpers for the minerva-judge pipeline."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT / "rlvr") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "rlvr"))

from athena_eval.models import GeminiModel, HuggingFaceModel, OpenAIModel, VLLMModel  # noqa: E402
from rlvr.verl.utils.reward_score import reward_minerva as reward_minerva_mod  # noqa: E402

from minerva.acr_prompt import build_acr_block, dedupe_labels, extract_gold_labels  # noqa: E402
from minerva.cti_task_specs import get_task_spec  # noqa: E402
from minerva.label_details_store import LabelDetailsStore  # noqa: E402
from minerva.retrieval.task_specs import (  # noqa: E402
    TaskSpec as RetrievalTaskSpec,
    extract_labels_from_truth,
    normalize_label,
)


@dataclass
class ModelConfig:
    name: str
    backend: str
    max_new_tokens: int
    batch_size: int = 1
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cti_system_prompt() -> str:
    import importlib.util

    prompt_path = PROJECT_ROOT / "rlvr" / "mydata" / "data_prepare" / "prompt.py"
    spec = importlib.util.spec_from_file_location("minerva_cti_prompt", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prompt module from {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "CTI_SYSTEM_PROMPT")


def format_answer(answer: Any) -> str:
    if isinstance(answer, (list, tuple)):
        return ", ".join(str(a) for a in answer)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def extract_option_map(prompt: str) -> dict[str, str]:
    marker = "Options:"
    if marker not in prompt:
        return {}
    tail = prompt.split(marker, 1)[1]
    options: dict[str, str] = {}
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Z])\s*[\.\)]\s*(.+)$", line)
        if not match:
            continue
        options[match.group(1).upper()] = match.group(2).strip()
    return options


def build_acr_prompt_text(
    prompt: str,
    answer: Any,
    data_source: str,
    task: str | None,
    *,
    details_store: LabelDetailsStore,
    max_details_chars: int = 4048,
    enforce_no_id: bool = True,
    max_prompt_chars: int = 0,
) -> tuple[str | None, dict[str, Any]]:
    extra_info = {"task": task} if task else {}
    spec = get_task_spec(data_source, answer, extra_info)
    entity_type = spec.entity_type

    gold_norm: list[str] = []
    if entity_type:
        try:
            retrieval_spec = RetrievalTaskSpec(
                label_type=entity_type,
                is_multilabel=bool(spec.is_multilabel),
                label_id_regex=spec.label_id_regex,
            )
            gold_norm = extract_labels_from_truth(answer, retrieval_spec)
        except Exception:
            gold_norm = []
    if not gold_norm:
        gold_labels = extract_gold_labels(answer)
        gold_norm = [
            normalize_label(entity_type, label) if entity_type else str(label).strip() for label in gold_labels
        ]
        gold_norm = dedupe_labels(gold_norm)
    if not gold_norm:
        return None, {"acr_skipped": "missing_labels"}

    details_labels = list(gold_norm)
    option_text = ""
    if entity_type and len(gold_norm) == 1 and len(gold_norm[0]) == 1:
        options = extract_option_map(prompt)
        option_text = options.get(gold_norm[0].upper(), "")
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
        details_text = details_store.get_details(entity_type, details_labels)
    details_truncated = False
    if details_text and max_details_chars > 0 and len(details_text) > max_details_chars:
        details_text = details_text[:max_details_chars].rsplit(" ", 1)[0] or details_text[:max_details_chars]
        details_text = details_text.rstrip() + "..."
        details_truncated = True

    block = build_acr_block(gold_norm, details_text, enforce_no_id=enforce_no_id)
    acr_prompt = prompt.rstrip() + "\n\n" + block
    if max_prompt_chars and len(acr_prompt) > max_prompt_chars:
        block = build_acr_block(gold_norm, "", enforce_no_id=enforce_no_id)
        acr_prompt = prompt.rstrip() + "\n\n" + block
        if len(acr_prompt) > max_prompt_chars:
            return None, {"acr_skipped": "prompt_too_long"}

    meta: dict[str, Any] = {
        "acr_gold_labels": gold_norm,
        "acr_entity_type": entity_type or "",
        "acr_option_text": option_text or "",
        "acr_details_truncated": details_truncated,
    }
    if details_labels != gold_norm:
        meta["acr_details_labels"] = details_labels
    return acr_prompt, meta


def resolve_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    lowered = model_name.lower()
    if lowered.startswith("gpt-") or lowered.startswith("o1") or lowered.startswith("o3"):
        return "openai"
    if lowered.startswith("gemini"):
        return "gemini"
    return "hf"


def load_model(config: ModelConfig):
    if config.backend == "openai":
        return OpenAIModel(config.name)
    if config.backend == "gemini":
        return GeminiModel(config.name)
    if config.backend in {"hf", "vllm"}:
        try:
            return VLLMModel(
                config.name,
                max_new_tokens=config.max_new_tokens,
                vllm_kwargs=config.vllm_kwargs or {},
                batch_size=config.batch_size,
            )
        except Exception:
            if config.backend == "vllm":
                raise
        return HuggingFaceModel(config.name, max_new_tokens=config.max_new_tokens)
    return HuggingFaceModel(config.name, max_new_tokens=config.max_new_tokens)


def load_rubric_prompt() -> str:
    prompt_path = PROJECT_ROOT / "minerva-judge" / "prompts" / "judge_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_judge_prompt(template: str, question: str, response: str) -> str:
    return template.replace("{QUESTION}", question).replace("{RESPONSE}", response)


def build_chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_text = (system_prompt or "").strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def render_messages_for_model(model: object, messages: list[dict[str, str]]) -> tuple[str, bool]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True
    if len(messages) == 2 and messages[0].get("role") == "system":
        return f"{messages[0]['content']}\n\n{messages[1]['content']}", False
    return messages[-1]["content"], False


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


_BAD_CATEGORIES = {
    1: "Leakage",
    2: "Incoherent",
    3: "Ungrounded",
    4: "Mismatch",
    5: "Unsupported",
    6: "Other",
}


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_title(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _find_category_id(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\b([1-6])\b", text)
    if not match:
        return None
    return int(match.group(1))


def parse_rubric(obj: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not obj:
        return None
    label = _normalize_label(obj.get("label"))
    if label not in {"GOOD", "BAD"}:
        return None
    parsed: dict[str, Any] = {"label": label}
    if label == "GOOD":
        return parsed

    raw_id = obj.get("category_id")
    cat_id = None
    if raw_id is not None:
        try:
            cat_id = int(str(raw_id).strip())
        except Exception:
            cat_id = _find_category_id(str(raw_id))
    if cat_id is None:
        cat_id = _find_category_id(_normalize_title(obj.get("category")))

    raw_title = obj.get("category_title") or obj.get("category")
    cat_title = _normalize_title(raw_title)
    if cat_id is None and cat_title:
        for cand_id, cand_title in _BAD_CATEGORIES.items():
            if cat_title.casefold() == cand_title.casefold():
                cat_id = cand_id
                break
    if not cat_title:
        cat_title = _BAD_CATEGORIES.get(cat_id, "")

    if cat_id not in _BAD_CATEGORIES:
        return None

    expected = _BAD_CATEGORIES.get(cat_id, "")
    if expected and cat_title and cat_title.casefold() != expected.casefold():
        return None

    parsed["category_id"] = cat_id
    parsed["category_title"] = cat_title or expected
    return parsed


def compute_rubric_score(rubric: dict[str, Any]) -> float:
    label = _normalize_label(rubric.get("label"))
    if label == "GOOD":
        return 1.0
    if label == "BAD":
        return 0.0
    return 0.0


def extract_predicted(data_source: str, response: str) -> str:
    extract_pred = getattr(reward_minerva_mod, "_extract_predicted")
    return extract_pred(data_source, response)


def reward_minerva(data_source: str, response: str, answer: Any) -> float:
    return float(reward_minerva_mod.reward_minerva(data_source, response, answer))

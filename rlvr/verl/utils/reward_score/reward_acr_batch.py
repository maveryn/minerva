"""Batch reward function for Answer-Conditioned Reasoning (ACR) outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score.reward_acr import reward_acr


_RUBRIC_KEYS = (
    "Q1_no_leakage",
    "Q2_clarity",
    "Q3_groundedness",
    "Q4_alignment",
)
_RUBRIC_WEIGHTS = (0.30, 0.30, 0.20, 0.20)
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "acr_rubric_prompt.txt"
_PROMPT_TEMPLATE: Optional[str] = None


@dataclass
class _JudgeState:
    model_name: Optional[str] = None
    tokenizer: Any = None
    model: Any = None


_JUDGE_STATE = _JudgeState()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _chunked(values: List[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        yield values
        return
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _render_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").upper()
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _load_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is not None:
        return _PROMPT_TEMPLATE
    try:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        _PROMPT_TEMPLATE = (
            "QUESTION:\n{QUESTION}\n\nRESPONSE:\n{RESPONSE}\n\n"
            "Return JSON with Q1_no_leakage, Q2_clarity, Q3_groundedness, Q4_alignment (1-4)."
        )
    return _PROMPT_TEMPLATE


def _build_judge_prompt(*, question_text: str, response_text: str) -> str:
    template = _load_prompt_template()
    return template.replace("{QUESTION}", question_text).replace("{RESPONSE}", response_text)


def _ensure_judge(
    *,
    model_name: str,
    device: Optional[str],
    device_map: Optional[str],
    dtype: Optional[str],
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    global _JUDGE_STATE
    if _JUDGE_STATE.model_name == model_name and _JUDGE_STATE.model is not None:
        return _JUDGE_STATE.model, _JUDGE_STATE.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = None
    if dtype and dtype != "auto":
        torch_dtype = getattr(torch, dtype, None)

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not device_map or device_map == "none":
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    model.eval()
    _JUDGE_STATE = _JudgeState(model_name=model_name, tokenizer=tokenizer, model=model)
    return model, tokenizer


def _judge_batch(
    prompts: List[str],
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    device: Optional[str],
    device_map: Optional[str],
    dtype: Optional[str],
    trust_remote_code: bool,
) -> List[str]:
    if not prompts:
        return []
    model, tokenizer = _ensure_judge(
        model_name=model_name,
        device=device,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    outputs: List[str] = []
    for chunk in _chunked(prompts, batch_size):
        inputs = tokenizer(chunk, return_tensors="pt", padding=True)
        if not device_map or device_map == "none":
            if device is None or device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()
        for idx, seq in enumerate(generated):
            start = int(prompt_lens[idx])
            outputs.append(tokenizer.decode(seq[start:], skip_special_tokens=True))
    return outputs


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _normalize_key(key: str) -> Optional[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(key).casefold()))
    if not tokens:
        return None
    if "q1" in tokens:
        return "Q1_no_leakage"
    if "q2" in tokens:
        return "Q2_clarity"
    if "q3" in tokens:
        return "Q3_groundedness"
    if "q4" in tokens:
        return "Q4_alignment"

    if tokens & {"leak", "leakage", "hint", "label", "labels", "option", "options"}:
        return "Q1_no_leakage"
    if tokens & {"clarity", "concise", "conciseness", "redundancy", "readability"}:
        return "Q2_clarity"
    if tokens & {"grounded", "groundedness", "evidence", "hallucination", "hallucinated"}:
        return "Q3_groundedness"
    if tokens & {"alignment", "aligned", "linkage", "support", "supports"}:
        return "Q4_alignment"
    return None


def _collect_scores(data: dict) -> dict[str, int]:
    scores: dict[str, int] = {}
    for raw_key, raw_val in data.items():
        axis = _normalize_key(str(raw_key))
        if axis is None or axis in scores:
            continue
        value = _to_int(raw_val)
        if value is None:
            continue
        scores[axis] = value
    return scores


def _score_from_rubric(data: Optional[dict]) -> float:
    if not data:
        return 0.0
    normalized = _collect_scores(data)
    scores: List[int] = []
    for key in _RUBRIC_KEYS:
        value = normalized.get(key)
        if value is None or value < 1 or value > 4:
            return 0.0
        scores.append(value)
    weighted = float(sum(w * s for w, s in zip(_RUBRIC_WEIGHTS, scores, strict=False)))
    normalized = (weighted - 1.0) / 3.0
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def reward_acr_batch(
    *,
    data_sources: Iterable[str],
    solution_strs: Iterable[str],
    ground_truths: Iterable[Any],
    extra_infos: Optional[Iterable[dict]] = None,
    r_correct: float = 0.2,
    leak_penalty: float = 0.5,
    multilabel_match: str = "exact",
    banned_phrases: Optional[list[str]] = None,
    use_fuzzy_leak_check: bool = True,
    fuzzy_threshold: float = 0.85,
    enforce_no_id_in_reasoning: bool = True,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    judge_enabled: bool | str = False,
    judge_model: str = "openai/gpt-oss-20b",
    judge_batch_size: int = 8,
    judge_max_new_tokens: int = 8,
    judge_temperature: float = 0.0,
    judge_top_p: float = 1.0,
    judge_device: Optional[str] = "auto",
    judge_device_map: Optional[str] = "auto",
    judge_dtype: Optional[str] = "auto",
    judge_trust_remote_code: bool = True,
) -> List[dict]:
    data_sources = list(data_sources)
    solution_strs = list(solution_strs)
    ground_truths = list(ground_truths)
    if extra_infos is None:
        extra_infos = [{} for _ in range(len(solution_strs))]
    else:
        extra_infos = list(extra_infos)

    results: List[dict] = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=False
    ):
        result = reward_acr(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            r_correct=r_correct,
            leak_penalty=leak_penalty,
            multilabel_match=multilabel_match,
            banned_phrases=banned_phrases,
            use_fuzzy_leak_check=use_fuzzy_leak_check,
            fuzzy_threshold=fuzzy_threshold,
            enforce_no_id_in_reasoning=enforce_no_id_in_reasoning,
            score_min=score_min,
            score_max=score_max,
        )
        results.append(result)

    if not _as_bool(judge_enabled):
        return results

    prompts: List[str] = []
    for solution_str, extra_info in zip(solution_strs, extra_infos, strict=False):
        question_text = ""
        if isinstance(extra_info, dict):
            question_text = (
                _render_messages(extra_info.get("acr_orig_prompt"))
                or _render_messages(extra_info.get("orig_prompt"))
                or _render_messages(extra_info.get("raw_prompt"))
                or _render_messages(extra_info.get("prompt"))
                or _render_messages(extra_info.get("acr_prompt"))
            )
        prompts.append(_build_judge_prompt(question_text=question_text, response_text=solution_str))

    judge_outputs = _judge_batch(
        prompts,
        model_name=judge_model,
        max_new_tokens=int(judge_max_new_tokens),
        temperature=float(judge_temperature),
        top_p=float(judge_top_p),
        batch_size=int(judge_batch_size),
        device=judge_device,
        device_map=judge_device_map,
        dtype=judge_dtype,
        trust_remote_code=bool(judge_trust_remote_code),
    )

    for result, judge_text in zip(results, judge_outputs, strict=False):
        result["acr_rubric_score"] = _score_from_rubric(_extract_json(judge_text))
    return results


__all__ = ["reward_acr_batch"]

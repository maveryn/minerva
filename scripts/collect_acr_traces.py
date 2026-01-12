#!/usr/bin/env python3
"""Generate ACR traces offline and write accepted ones to a JSONL buffer."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from minerva.acr_prompt import build_acr_block
from minerva.acr_trace_buffer import AcrTraceBuffer, is_accepted
from minerva.cti_task_specs import get_task_spec
from minerva.label_details_store import LabelDetailsStore
from minerva.retrieval.task_specs import normalize_label
from rlvr.verl.utils.reward_score.reward_acr import reward_acr


def _dedupe(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        val = str(raw).strip()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _extract_gold_labels(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, (list, tuple, set)):
        return _dedupe(ground_truth)
    if isinstance(ground_truth, dict):
        for key in ("labels", "label", "technique_id", "tactic_ids", "mitigation_ids", "cwe_ids", "capec_id"):
            value = ground_truth.get(key)
            if isinstance(value, (list, tuple, set)):
                return _dedupe(value)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
    if isinstance(ground_truth, str) and ground_truth.strip():
        return [ground_truth.strip()]
    return []


def _append_block(messages: List[Dict[str, Any]], block: str) -> List[Dict[str, Any]]:
    if not messages:
        return messages
    target_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            target_idx = idx
            break
    if target_idx is None:
        target_idx = len(messages) - 1
    content = messages[target_idx].get("content", "")
    if not isinstance(content, str):
        return messages
    content = content.rstrip()
    messages[target_idx]["content"] = f"{content}\n\n{block}" if content else block
    return messages


def _build_acr_messages(
    base_messages: List[Dict[str, Any]],
    gold_labels: List[str],
    details_text: Optional[str],
    *,
    enforce_no_id: bool,
) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(base_messages)
    block = build_acr_block(gold_labels, details_text, enforce_no_id=enforce_no_id)
    return _append_block(messages, block)


def _generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str,
) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    prompt_lens = inputs["attention_mask"].sum(dim=-1).tolist()
    results = []
    for idx, seq in enumerate(outputs):
        start = int(prompt_lens[idx])
        results.append(tokenizer.decode(seq[start:], skip_special_tokens=True))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect offline ACR traces from a checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint/path for generation.")
    parser.add_argument("--dataset", required=True, help="Parquet dataset file for ACR prompts.")
    parser.add_argument("--out", required=True, help="Output JSONL buffer path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling.")
    parser.add_argument("--device", default=None, help="Device to run on (cuda or cpu).")
    parser.add_argument("--device-map", default=None, help="Optional device_map for transformers.")
    parser.add_argument("--max-details-chars", type=int, default=1200, help="Max chars for label details.")
    parser.add_argument("--enforce-no-id", action="store_true", help="Enforce no-ID-in-reasoning instruction.")
    parser.add_argument("--require-no-id-leak", action="store_true", help="Filter out ID leaks when accepting traces.")
    parser.add_argument("--label-details-dir", default=None, help="Optional label-details directory override.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"trust_remote_code": True}
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, **model_kwargs)
    if not args.device_map:
        model.to(device)
    model.eval()

    dataset = load_dataset("parquet", data_files=args.dataset, split="train")
    buffer = AcrTraceBuffer(args.out)
    details_store = LabelDetailsStore(args.label_details_dir)

    pending_prompts: List[str] = []
    pending_rows: List[Dict[str, Any]] = []
    accepted = 0
    processed = 0

    for idx, row in enumerate(dataset):
        if args.max_samples is not None and processed >= args.max_samples:
            break
        processed += 1

        base_messages = row.get("prompt")
        if not isinstance(base_messages, list):
            continue

        data_source = str(row.get("data_source") or row.get("reward_fn") or "")
        reward_model = row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth")
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}

        spec = get_task_spec(data_source, ground_truth, extra_info)
        gold_labels = _extract_gold_labels(ground_truth)
        gold_norm = [
            normalize_label(spec.entity_type, label) if spec.entity_type else str(label).strip()
            for label in gold_labels
        ]
        gold_norm = _dedupe(gold_norm)
        details_text: Optional[str] = None
        if spec.entity_type:
            details_text = details_store.get_details(spec.entity_type, gold_norm)
            if details_text and args.max_details_chars > 0 and len(details_text) > args.max_details_chars:
                details_text = (
                    details_text[: args.max_details_chars].rsplit(" ", 1)[0]
                    or details_text[: args.max_details_chars]
                )
                details_text = details_text.rstrip() + "..."

        acr_messages = _build_acr_messages(
            base_messages,
            gold_norm,
            details_text,
            enforce_no_id=args.enforce_no_id,
        )
        prompt_str = tokenizer.apply_chat_template(acr_messages, add_generation_prompt=True, tokenize=False)

        pending_prompts.append(prompt_str)
        pending_rows.append(
            {
                "uid": row.get("uid") or extra_info.get("index") or f"{data_source}:{idx}",
                "data_source": data_source,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
                "orig_prompt": copy.deepcopy(base_messages),
                "acr_prompt": copy.deepcopy(acr_messages),
            }
        )

        if len(pending_prompts) >= args.batch_size:
            outputs = _generate_batch(
                model,
                tokenizer,
                pending_prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            for meta, output in zip(pending_rows, outputs, strict=False):
                reward_info = reward_acr(
                    data_source=meta["data_source"],
                    solution_str=output,
                    ground_truth=meta["ground_truth"],
                    extra_info=meta.get("extra_info", {}),
                )
                if is_accepted(reward_info, require_no_id_leak=args.require_no_id_leak):
                    buffer.append(
                        {
                            "uid": meta["uid"],
                            "data_source": meta["data_source"],
                            "orig_prompt": meta["orig_prompt"],
                            "acr_prompt": meta.get("acr_prompt"),
                            "target_completion": output,
                            "reward": reward_info.get("score"),
                            "reward_info": reward_info,
                        }
                    )
                    accepted += 1
            pending_prompts = []
            pending_rows = []

    if pending_prompts:
        outputs = _generate_batch(
            model,
            tokenizer,
            pending_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        for meta, output in zip(pending_rows, outputs, strict=False):
            reward_info = reward_acr(
                data_source=meta["data_source"],
                solution_str=output,
                ground_truth=meta["ground_truth"],
                extra_info=meta.get("extra_info", {}),
            )
            if is_accepted(reward_info, require_no_id_leak=args.require_no_id_leak):
                buffer.append(
                    {
                        "uid": meta["uid"],
                        "data_source": meta["data_source"],
                        "orig_prompt": meta["orig_prompt"],
                        "acr_prompt": meta.get("acr_prompt"),
                        "target_completion": output,
                        "reward": reward_info.get("score"),
                        "reward_info": reward_info,
                    }
                )
                accepted += 1

    print(f"Processed {processed} samples, accepted {accepted} traces.")


if __name__ == "__main__":
    main()

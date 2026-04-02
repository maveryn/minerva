from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dart.common import (
    compose_training_response,
    first_present_field,
    get_user_prompt,
    load_parquet_rows,
    make_uid,
    normalize_messages,
    split_reasoning_response,
    to_jsonable,
    write_jsonl,
)
from dart.prompting import DartPromptBuilder


class DartGenerator:
    def __init__(
        self,
        *,
        model_path: str,
        backend: str = "hf",
        batch_size: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_prompt_length: int = 0,
        trust_remote_code: bool = False,
        gpu_memory_utilization: float = 0.9,
        vllm_kwargs: Optional[Dict[str, Any]] = None,
        guided_label_details_dir: Optional[str] = None,
        guided_max_details_chars: int = 8096,
        guided_max_prompt_length: int = 4096,
        guided_enforce_no_id: bool = False,
        task_reasoning_hints: Optional[dict[str, str]] = None,
        entity_reasoning_hints: Optional[dict[str, str]] = None,
    ) -> None:
        self.model_path = model_path
        self.backend = str(backend).lower()
        if self.backend not in {"hf", "vllm"}:
            raise ValueError(f"Unsupported generation backend: {backend}")
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_prompt_length = int(max_prompt_length)
        self.trust_remote_code = bool(trust_remote_code)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.vllm_kwargs = dict(vllm_kwargs or {})

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.prompt_builder = DartPromptBuilder(
            self.tokenizer,
            label_details_dir=guided_label_details_dir,
            max_details_chars=guided_max_details_chars,
            max_prompt_length=guided_max_prompt_length,
            enforce_no_id=guided_enforce_no_id,
            task_reasoning_hints=task_reasoning_hints,
            entity_reasoning_hints=entity_reasoning_hints,
        )

        self.model = None
        self.llm = None
        self.device = None
        if self.backend == "hf":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
            )
            self.model.eval()
            self.device = next(self.model.parameters()).device
        else:
            from vllm import LLM

            engine_kwargs = {
                "model": model_path,
                "trust_remote_code": trust_remote_code,
                "gpu_memory_utilization": gpu_memory_utilization,
            }
            engine_kwargs.update(self.vllm_kwargs)
            self.llm = LLM(**engine_kwargs)

    def _prepare_row(
        self,
        row: Dict[str, Any],
        *,
        mode: str,
        fallback_index: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        base_messages = normalize_messages(
            first_present_field(row, "prompt", "base_messages", "prompt_nohint_messages", default=[])
        )
        if not base_messages:
            skipped = {
                "uid": make_uid(row, fallback_index=fallback_index),
                "mode": mode,
                "generation_skipped": True,
                "generation_skip_reason": "missing_messages",
                "source_index": fallback_index,
            }
            return None, skipped

        data_source = str(row.get("data_source") or row.get("reward_fn") or "")
        reward_model = row.get("reward_model") or {}
        ground_truth = row.get("ground_truth", reward_model.get("ground_truth"))
        extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
        extra_info = dict(extra_info)
        uid = make_uid(row, fallback_index=fallback_index)

        prompt_meta: Dict[str, Any] = {}
        if mode == "plain":
            prompt_messages = self.prompt_builder.build_plain_messages(base_messages)
        elif mode == "direct_answer":
            prompt_messages = self.prompt_builder.build_direct_answer_messages(base_messages)
        elif mode == "guided_fill":
            prompt_messages, prompt_meta = self.prompt_builder.build_guided_messages(
                base_messages,
                data_source=data_source,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if prompt_messages is None:
                skipped = {
                    "uid": uid,
                    "mode": mode,
                    "generation_skipped": True,
                    "generation_skip_reason": prompt_meta.get("guided_prompt_skip_reason", "guided_prompt_failed"),
                    "data_source": data_source,
                    "reward_fn": extra_info.get("reward_fn", data_source),
                    "task": extra_info.get("task"),
                    "source_file": row.get("source_file"),
                    "source_index": extra_info.get("index", fallback_index),
                    "prompt_meta": prompt_meta,
                    "ground_truth": ground_truth,
                }
                return None, skipped
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        if mode == "guided_fill":
            extra_info.setdefault("acr_orig_prompt", copy.deepcopy(base_messages))
            extra_info["acr_prompt"] = copy.deepcopy(prompt_messages)

        prepared = {
            "uid": uid,
            "mode": mode,
            "data_source": data_source,
            "reward_fn": extra_info.get("reward_fn", data_source),
            "task": extra_info.get("task"),
            "source_file": row.get("source_file"),
            "source_index": extra_info.get("index", fallback_index),
            "prompt_nohint": get_user_prompt(base_messages),
            "base_messages": base_messages,
            "prompt_used_messages": prompt_messages,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "attempt_index": row.get("attempt_index"),
            "generation_config": {
                "model_path": self.model_path,
                "backend": self.backend,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_prompt_length": self.max_prompt_length,
            },
            "prompt_meta": prompt_meta,
        }
        return prepared, None

    def _generate_texts(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        if self.backend == "hf":
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "add_special_tokens": False,
            }
            if self.max_prompt_length > 0:
                tokenizer_kwargs.update({"truncation": True, "max_length": self.max_prompt_length})
            else:
                tokenizer_kwargs["truncation"] = False
            encoded = self.tokenizer(texts, **tokenizer_kwargs)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p
            with torch.no_grad():
                generated = self.model.generate(**generate_kwargs)
            input_lengths = attention_mask.sum(dim=1).tolist()
            outputs = []
            for generated_ids, input_length in zip(generated, input_lengths):
                response_ids = generated_ids[int(input_length) :]
                outputs.append(self.tokenizer.decode(response_ids, skip_special_tokens=True).strip())
            return outputs

        from vllm import SamplingParams

        prompt_texts = []
        for text in texts:
            if self.max_prompt_length > 0:
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                )
                prompt_texts.append(self.tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False))
            else:
                prompt_texts.append(text)
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        generated = self.llm.generate(prompt_texts, sampling_params=sampling_params)
        outputs = []
        for output in generated:
            if not output.outputs:
                outputs.append("")
            else:
                outputs.append((output.outputs[0].text or "").strip())
        return outputs

    def generate_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        mode: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prepared_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            prepared, skipped = self._prepare_row(row, mode=mode, fallback_index=idx)
            if prepared is not None:
                prepared_rows.append(prepared)
            elif skipped is not None:
                skipped_rows.append(skipped)

        outputs: List[Dict[str, Any]] = []
        for start in range(0, len(prepared_rows), self.batch_size):
            batch = prepared_rows[start : start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    item["prompt_used_messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]
            responses = self._generate_texts(texts)
            for item, response in zip(batch, responses, strict=True):
                row = dict(item)
                raw_response = (response or "").strip()
                response_analysis, response_final = split_reasoning_response(raw_response)
                row["response_text_raw"] = raw_response
                row["response_analysis"] = response_analysis
                row["response_final"] = response_final
                row["response_text"] = compose_training_response(
                    analysis=response_analysis,
                    final=response_final,
                )
                outputs.append(row)
        return outputs, skipped_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DART traces from parquet rows")
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["plain", "guided_fill", "direct_answer"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skipped-output")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-length", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--guided-label-details-dir")
    parser.add_argument("--guided-max-details-chars", type=int, default=8096)
    parser.add_argument("--guided-max-prompt-length", type=int, default=4096)
    parser.add_argument("--guided-enforce-no-id", action="store_true")
    args = parser.parse_args()

    rows = load_parquet_rows(args.parquet, limit=args.limit)
    generator = DartGenerator(
        model_path=args.model_path,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=args.max_prompt_length,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        guided_label_details_dir=args.guided_label_details_dir,
        guided_max_details_chars=args.guided_max_details_chars,
        guided_max_prompt_length=args.guided_max_prompt_length,
        guided_enforce_no_id=args.guided_enforce_no_id,
    )
    generated, skipped = generator.generate_rows(rows, mode=args.mode)
    write_jsonl(args.output, generated)
    if args.skipped_output:
        write_jsonl(args.skipped_output, skipped)
    print(json.dumps({"output": args.output, "rows": len(generated), "skipped": len(skipped)}))


if __name__ == "__main__":
    main()

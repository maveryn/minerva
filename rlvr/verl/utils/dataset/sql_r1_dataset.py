from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask


class SQLR1RLHFDataset(RLHFDataset):
    """Dataset loader for SQL-R1 parquet files with preformatted prompt strings.

    SQL-R1 stores each prompt as a single flattened chat transcript in
    ``prompt[0]["content"]`` instead of a structured list of messages that should
    be passed through ``tokenizer.apply_chat_template`` again. This dataset class
    preserves that format and tokenizes the prompt text directly.
    """

    @staticmethod
    def _normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()

        if isinstance(raw_prompt, list):
            messages: list[dict[str, str]] = []
            for item in raw_prompt:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                if not isinstance(content, str):
                    continue
                role = str(item.get("role") or "user")
                messages.append({"role": role, "content": content})
            if messages:
                return messages

        if isinstance(raw_prompt, str):
            return [{"role": "user", "content": raw_prompt}]

        raise TypeError(f"Unsupported SQL-R1 prompt payload: {type(raw_prompt)!r}")

    @classmethod
    def _extract_prompt_text(cls, raw_prompt: Any) -> tuple[list[dict[str, str]], str]:
        messages = cls._normalize_messages(raw_prompt)
        prompt_text = messages[0].get("content", "")
        if not isinstance(prompt_text, str) or not prompt_text:
            raise ValueError("SQL-R1 prompt is missing preformatted text content.")
        return messages, prompt_text

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer

        def doc2len(doc) -> int:
            _, prompt_text = self._extract_prompt_text(doc[self.prompt_key])
            return len(tokenizer.encode(prompt_text, add_special_tokens=False))

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering SQL-R1 prompts longer than {self.max_prompt_length} tokens",
        )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict[str, Any] = copy.deepcopy(self.dataframe[item])
        raw_prompt = row_dict.pop(self.prompt_key)
        messages, prompt_text = self._extract_prompt_text(raw_prompt)

        model_inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = prompt_text

        extra_info = row_dict.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info = dict(extra_info)
        extra_info.setdefault("prompt_format", "sql_r1_preformatted")
        row_dict["extra_info"] = extra_info

        index = extra_info.get("index", 0)
        row_dict["index"] = index
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        return row_dict


__all__ = ["SQLR1RLHFDataset"]

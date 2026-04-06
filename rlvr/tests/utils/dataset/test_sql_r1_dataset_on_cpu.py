from __future__ import annotations

from pathlib import Path

import pandas as pd

from verl.utils.dataset.sql_r1_dataset import SQLR1RLHFDataset


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [idx + 2 for idx, _ in enumerate(text.split())]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        import torch

        token_ids = self.encode(text, add_special_tokens=add_special_tokens)
        if not token_ids:
            token_ids = [self.eos_token_id]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


def test_sql_r1_dataset_uses_preformatted_prompt(tmp_path: Path):
    parquet_path = tmp_path / "train.parquet"
    prompt_text = "<|im_start|>system\nSQL expert\n<|im_end|>\n<|im_start|>user\nQuestion\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    dataframe = pd.DataFrame(
        [
            {
                "data_source": "synsql",
                "prompt": [{"role": "user", "content": prompt_text}],
                "reward_model": {"style": "rule", "ground_truth": {"db_id": "toy", "sql": "SELECT 1;"}},
                "extra_info": {"index": 7, "split": "train"},
            }
        ]
    )
    dataframe.to_parquet(parquet_path)

    dataset = SQLR1RLHFDataset(
        data_files=str(parquet_path),
        tokenizer=DummyTokenizer(),
        config={
            "prompt_key": "prompt",
            "max_prompt_length": 128,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "return_raw_chat": True,
            "return_full_prompt": True,
        },
    )

    row = dataset[0]
    assert row["index"] == 7
    assert row["full_prompts"] == prompt_text
    assert row["raw_prompt"][0]["content"] == prompt_text
    assert row["extra_info"]["prompt_format"] == "sql_r1_preformatted"
    assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["position_ids"])

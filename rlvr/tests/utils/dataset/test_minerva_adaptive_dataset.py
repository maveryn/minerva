# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from verl.utils.dataset.minerva_adaptive_dataset import AdaptiveOptionRLHFDataset


class DummyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        parts = [f"{m.get('role', '')}: {m.get('content', '')}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones(1, len(ids), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text.split()) + 1))


def _write_parquet(tmp_path: Path) -> Path:
    row = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Pick the most appropriate ID."},
        ],
        "data_source": "cti_technique_id",
        "reward_model": {"ground_truth": "T1059"},
        "extra_info": {"index": 1},
        "candidate_pool_top100": ["T1059", "T1059.003", "T1047", "T1204"],
    }
    path = tmp_path / "minerva_test.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    return path


def test_lhc_injects_hints(tmp_path: Path) -> None:
    parquet_path = _write_parquet(tmp_path)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": False,
            "return_raw_chat": True,
            "adaptive_options": {
                "enabled": True,
                "candidate_pool_key": "candidate_pool_top100",
                "target_acc": 0.5,
                "tol": 0.05,
                "warmup_steps": 0,
                "ema_beta": 0.0,
                "k_min": 2,
                "k_max": 4,
                "k_step": 2,
                "buffer": 10,
                "p_drop_init": 0.0,
                "p_drop_step": 0.1,
                "p_drop_max": 1.0,
                "score_threshold": 0.5,
            },
        }
    )
    dataset = AdaptiveOptionRLHFDataset(
        data_files=str(parquet_path),
        tokenizer=DummyTokenizer(),
        config=config,
    )

    dataset.update_option_curriculum_from_rollout(
        data_source_arr=["cti_technique_id", "cti_technique_id"],
        uid_arr=["u1", "u1"],
        is_correct_arr=[0, 0],
        global_step=1,
    )

    item = dataset[0]
    raw_prompt = item["raw_prompt"]
    user_content = raw_prompt[-1]["content"]
    assert "Candidate IDs (choose one):" in user_content
    assert "Return ONLY the ID." in user_content

    extra_info = item["extra_info"]
    assert extra_info["hint_used"] is True
    assert extra_info["hint_K"] == 2
    assert extra_info["hint_pool_size"] == 4


def test_lhc_curriculum_updates_k(tmp_path: Path) -> None:
    parquet_path = _write_parquet(tmp_path)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": False,
            "adaptive_options": {
                "enabled": True,
                "candidate_pool_key": "candidate_pool_top100",
                "target_acc": 0.5,
                "tol": 0.05,
                "warmup_steps": 0,
                "ema_beta": 0.0,
                "k_min": 2,
                "k_max": 4,
                "k_step": 2,
                "buffer": 10,
                "p_drop_init": 0.0,
                "p_drop_step": 0.1,
                "p_drop_max": 1.0,
                "score_threshold": 0.5,
            },
        }
    )
    dataset = AdaptiveOptionRLHFDataset(
        data_files=str(parquet_path),
        tokenizer=DummyTokenizer(),
        config=config,
    )

    dataset.update_option_curriculum_from_rollout(
        data_source_arr=["cti_technique_id", "cti_technique_id"],
        uid_arr=["u1", "u1"],
        is_correct_arr=[0, 0],
        global_step=1,
    )
    metrics = dataset.option_controller.get_metrics(prefix="option_curriculum/")
    assert int(metrics["option_curriculum/K/cti_technique_id"]) == 2

    dataset.update_option_curriculum_from_rollout(
        data_source_arr=["cti_technique_id", "cti_technique_id"],
        uid_arr=["u1", "u1"],
        is_correct_arr=[1, 1],
        global_step=2,
    )
    metrics = dataset.option_controller.get_metrics(prefix="option_curriculum/")
    assert int(metrics["option_curriculum/K/cti_technique_id"]) == 4

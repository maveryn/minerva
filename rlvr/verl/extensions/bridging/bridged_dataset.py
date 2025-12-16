# verl/extensions/bridging/bridged_dataset.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import random
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset


class BridgedRLHFDataset(Dataset):
    """
    Wrap a base dataframe (Parquet) that already contains:
      - prompt: chat list [{"role":"user","content":...}] (the *current* prompt used for rollout)
      - extra_info.orig_prompt: the original wording
      - extra_info.paraphrases: list[str] paraphrase candidates
      - reward_model.ground_truth: string

    Policy: Start with original wordings.
            If a uid's GRPO group has pass@k=0 (all scores 0) in the LAST batch,
            switch its 'prompt' to a sampled paraphrase for the NEXT epoch step.
            If later pass@k>0 on original, switch back to original.
    """

    def __init__(self, parquet_path: str, seed: int = 0):
        self.df = pd.read_parquet(parquet_path)

        # Normalize nested JSON columns
        def normalize_extra(row):
            extra = row.get("extra_info", {}) or {}
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            row["orig_prompt"] = extra.get("orig_prompt", None)
            row["paraphrases"] = extra.get("paraphrases", [])
            return row

        self.df = self.df.apply(normalize_extra, axis=1)
        self._rng = random.Random(seed)
        self._fail_uids = set()  # uids that failed in last batch
        # Pre-assign stable uids to each row
        if "uid" not in self.df.columns:
            self.df["uid"] = [f"uid-{i}" for i in range(len(self.df))]
        # Start from original prompts
        self._use_paraphrase = {uid: False for uid in self.df["uid"]}

    def __len__(self):
        return len(self.df)

    def _make_prompt(self, text: str):
        return [{"role": "user", "content": text}]

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        uid = row["uid"]
        orig = row.get("orig_prompt", None)
        paras = row.get("paraphrases", [])

        # choose current wording
        if self._use_paraphrase.get(uid, False) and paras:
            chosen = self._rng.choice(paras)
            current_prompt = self._make_prompt(chosen)
        else:
            # fall back to the stored 'prompt' or orig
            if isinstance(row["prompt"], str):
                current_prompt = json.loads(row["prompt"])
            else:
                current_prompt = row["prompt"]
            if not current_prompt and orig:
                current_prompt = self._make_prompt(orig)

        row["prompt"] = current_prompt
        # Always keep original wording around for distillation
        if "extra_info" not in row or row["extra_info"] in (None, ""):
            row["extra_info"] = {}
        if isinstance(row["extra_info"], str):
            try:
                row["extra_info"] = json.loads(row["extra_info"])
            except Exception:
                row["extra_info"] = {}
        row["extra_info"]["orig_prompt"] = orig or (current_prompt[0]["content"] if current_prompt else None)
        return row

    # Hook called by trainer after each batch (if present)
    def on_batch_end(self, batch):
        """
        batch.batch["token_level_scores"]: (B, T) tensor
        batch.non_tensor_batch["uid"]: array of uid strings, repeated n times by VeRL
        """
        # Find pass@k=0 per uid for the *original wording* attempts
        # Here we look at per-uid maximum score across repeats.
        scores = batch.batch["token_level_scores"].sum(-1)  # (B,)
        uids = batch.non_tensor_batch["uid"]  # (B,) object array
        uid2max = defaultdict(float)
        for s, u in zip(scores.detach().cpu().tolist(), uids.tolist(), strict=False):
            uid2max[u] = max(uid2max[u], s)

        self._fail_uids = {u for u, mx in uid2max.items() if mx <= 0.0}
        # Update sampling policy for next step
        for u in self._use_paraphrase.keys():
            self._use_paraphrase[u] = u in self._fail_uids

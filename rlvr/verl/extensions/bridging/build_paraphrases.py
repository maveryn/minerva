# verl/extensions/bridging/build_paraphrases.py
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

"""Utilities for building paraphrase pools for VeRL datasets."""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "You are a paraphrase generator. Rephrase the math question without changing "
    "its meaning or numerical values. Preserve all conditions. Output ONLY the "
    "rewritten question."
)


def generate_paraphrases(model, tok, text, m=3, max_new_tokens=128, temperature=0.8, top_p=0.95, device="cuda"):
    prompts = [f"{PROMPT}\n\nQuestion:\n{text}\n\nParaphrase:" for _ in range(m)]
    inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    dec = tok.batch_decode(out, skip_special_tokens=True)
    paras = []
    for p, full in zip(prompts, dec, strict=False):
        # keep only model continuation
        paras.append(full.split("Paraphrase:")[-1].strip())
    return paras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    # Each row has columns including: prompt (chat list), reward_model.ground_truth, data_source, etc.
    # We will add extra_info.orig_prompt and extra_info.paraphrases (JSON).
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)

    new_rows = []
    for i, row in df.iterrows():
        # Extract the user message as raw question text
        # VeRL preprocessors put prompt as a list of chat turns [{"role":"user","content":"..."}]
        chat = row["prompt"]
        if isinstance(chat, str):
            chat = json.loads(chat)
        assert chat and chat[0]["role"] == "user"
        orig_q = chat[0]["content"]

        paras = generate_paraphrases(model, tok, orig_q, m=args.m, device=args.device)
        row = row.to_dict()
        extra = row.get("extra_info", {}) or {}
        extra["orig_prompt"] = orig_q
        extra["paraphrases"] = paras
        row["extra_info"] = extra
        new_rows.append(row)

    out = pd.DataFrame(new_rows)
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    out.to_parquet(args.out_parquet, index=False)
    print("Wrote:", args.out_parquet)


if __name__ == "__main__":
    main()

# Rebuttal Artifact Package - 2026-07-04

This note records the artifacts prepared for the Eval12 rebuttal additions.

## Hugging Face Model

- Answer-only in-loop distillation ablation model:
  - `xashru/minerva_grpo_answer_sft_llama8b_ablation`
  - Local HF-format copy: `/dev/shm/minerva/hf_models/minerva_grpo_answer_sft_llama31_8b_best320_hf`
  - Remote files verified: tokenizer/config plus four `safetensors` shards and `model.safetensors.index.json`
  - `README.md` model card uploaded

Weights should stay on Hugging Face. Do not add the 15 GB HF-format model or the 180 GB VeRL checkpoint tree to GitHub/Git LFS.

## Minerva GitHub Changes

- `rlvr/cti-scripts/train_minerva_grpo_answer_sft_llama8b.sh`
  - default GPUs changed from 4 to 8
  - default checkpoint root set to `/dev/shm/minerva/checkpoints/$ANSWER_SFT_EXPERIMENT_NAME`
- `.gitignore`
  - ignores local `hf-token.txt`

## LLMBench GitHub/LFS Artifacts

Staged in `/home/shadeform/llmbench` under the existing `runs/**` LFS rule:

- `runs/minerva_llama8b_answer_sft_best320_local/`
- `runs/eval12_ablation_grpo12_llama8b_65536/`
- `runs/eval12_ablation_sft_answer_llama8b_65536/`
- `runs/eval12_ablation_emaoff_llama8b_65536/`
- `runs/eval12_ablation_filteroff_llama8b_65536/`
- `runs/eval12_ablation_mloff_llama8b_65536/`
- `runs/passk_eval12_t0.7_p0.95_k8/`

Staged LLMBench code adds:

- Eval CLI support for `--temperature`, `--top-p`, and `--seed`
- vLLM sampling support for temperature/top-p/seed
- pass@k launch scripts and oracle best-of-k summarizer
- Llama-8B ablation batch launch scripts

Approximate new LFS payload: 9 GB. Local LLMBench `.git/lfs/objects` after staging: about 14 GB.

## Canonical Summary Files

Use these committed LLMBench files as the starting point for tables or follow-up CI scripts.

Pass@8 oracle summaries:

- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/llama3b_grpo/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/llama3b_minerva/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/llama8b_grpo/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/llama8b_minerva/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/qwen4b_grpo/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/qwen4b_minerva/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/qwen8b_grpo/passk-oracle-summary.json`
- `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/qwen8b_minerva/passk-oracle-summary.json`

Ablation Eval12 summaries:

- `/home/shadeform/llmbench/runs/eval12_ablation_grpo12_llama8b_65536/eval12-summary.json`
- `/home/shadeform/llmbench/runs/eval12_ablation_sft_answer_llama8b_65536/eval12-summary.json`
- `/home/shadeform/llmbench/runs/eval12_ablation_emaoff_llama8b_65536/eval12-summary.json`
- `/home/shadeform/llmbench/runs/eval12_ablation_filteroff_llama8b_65536/eval12-summary.json`
- `/home/shadeform/llmbench/runs/eval12_ablation_mloff_llama8b_65536/eval12-summary.json`
- `/home/shadeform/llmbench/runs/minerva_llama8b_answer_sft_best320_local/eval12-summary.json`

Raw CI inputs:

- Pass@8: `/home/shadeform/llmbench/runs/passk_eval12_t0.7_p0.95_k8/*/sample_*/*/*-scored.jsonl`
- Ablations: `/home/shadeform/llmbench/runs/eval12_ablation_*_llama8b_65536/*-scored.jsonl`
- Answer-matched ablation: `/home/shadeform/llmbench/runs/minerva_llama8b_answer_sft_best320_local/*-scored.jsonl`

Useful commands:

```bash
cd /home/shadeform/llmbench
python scripts/summarize_eval12_passk_oracle.py --run-root runs/passk_eval12_t0.7_p0.95_k8 --k 8
```

## Answer-Matched Ablation Eval12

`runs/minerva_llama8b_answer_sft_best320_local/eval12-summary.json`

| Model | Avg |
| --- | ---: |
| answer-matched GRPO + answer-only SFT | 57.8 |

## Llama-8B Final Ablation Eval12 Runs

| Run | Avg |
| --- | ---: |
| GRPO with 12 rollouts | 53.9 |
| answer-only SFT | 47.2 |
| no EMA teacher | 57.2 |
| no filtering | 57.8 |
| no ML filter | 57.7 |
| answer-matched GRPO + answer-only SFT | 57.8 |

Existing MinervaRL Llama-8B Eval12 result is not rerun here; earlier run average was approximately 60.2.

## Pass@8 Oracle Eval12

Settings:

- `k=8`
- temperature `0.7`
- top-p `0.95`
- seed base `20260704`
- max response length `max_new_tokens=2048`
- vLLM context length: Llama `65536`, Qwen `32768`
- metric: row-level oracle/best-of-k from scored JSONL rows, followed by Eval12 metric recomputation

| Model | Avg |
| --- | ---: |
| Llama-3B GRPO | 53.2 |
| Llama-3B MinervaRL | 60.5 |
| Llama-8B GRPO | 62.4 |
| Llama-8B MinervaRL | 70.9 |
| Qwen-4B GRPO | 62.1 |
| Qwen-4B MinervaRL | 65.0 |
| Qwen-8B GRPO | 62.8 |
| Qwen-8B MinervaRL | 67.5 |

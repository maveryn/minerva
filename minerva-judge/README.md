# Minerva Judge SFT Pipeline

This folder contains a small pipeline to build a judge SFT dataset from Minerva
train prompts. The flow is:

1) use the Minerva train split (optionally sample for quick tests)
2) generate responses for every prompt, both with ACR-style answer hints and without
3) score (prompt, response) pairs with a GPT-5 judge
4) export an SFT dataset to train a smaller judge model (e.g., Qwen 4B/8B)

The judge prompt uses the GOOD/BAD rubric in `minerva-judge/prompts/judge_prompt.txt`
(derived from `docs/acr-judge-prompt.md`).

## Folder layout

- `minerva-judge/scripts/` pipeline scripts
- `minerva-judge/data/` generated JSONL + parquet outputs

## Dependencies

- Python deps already used in the repo: `transformers`, `torch`, `pyarrow`, `tqdm`
- OpenAI judge: set `OPENAI_API_KEY` (uses `athena_eval.models.OpenAIModel`)
- Hugging Face model generation: set `HF_TOKEN` if the model is gated

## Step 1: choose prompts

For full runs, use the full train split directly:

```bash
INPUT=dataset/minerva_base_split/minerva-base-train.jsonl
```

Optional: sample a smaller subset for quick tests.

```bash
python minerva-judge/scripts/sample_minerva_prompts.py \
  --input dataset/minerva_base_split/minerva-base-train.jsonl \
  --output minerva-judge/data/prompts_1k.jsonl \
  --sample-size 1000 \
  --seed 1337
```

## Step 2: generate responses (hinted + plain)

The script iterates each prompt once and emits two responses:
`prompt_variant=hinted` (ACR answer hints) and `prompt_variant=plain` (no hints).
It records the rule-based reward and correctness flag for each response.
For HF models, it renders a system+user chat template using the model tokenizer
(matching ACRD’s chat-formatting behavior).

Example (repeat for each base model; use `--backend vllm` to enable batched vLLM inference):

```bash
python minerva-judge/scripts/generate_answer_guided.py \
  --input "$INPUT" \
  --output minerva-judge/data/responses_llama3_8b.jsonl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --backend vllm \
  --batch-size 32 \
  --max-new-tokens 1024 \
  --temperature 0.7
```

Use `--variants plain|hinted|both` (default `both`) to control which prompt
variants are generated. Add `--limit` for a quick sanity check. For vLLM
engine tuning (e.g., tensor parallelism), pass `--vllm-args '{"tensor_parallel_size": 2}'`.

Suggested model set:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-8B`
- `Qwen/Qwen3-4B-Instruct`
- `Qwen/Qwen3-8B-Instruct`
- `openai/gpt-oss-20b`

Helper scripts:
- `minerva-judge/run_generate_models.sh`: generate responses for the full train split across the 5 target models.
- `minerva-judge/run_judge_folder.sh`: run the judge over all `responses_*.jsonl` files in `minerva-judge/data`.
  Writes `judged_*.jsonl` into `minerva-judge/data` by default (replaces the
  `responses_` prefix) and supports `MINERVA_JUDGE_JUDGED_OUTPUT` for a combined file.

Note: vLLM uses multiprocessing; the helper scripts set
`VLLM_WORKER_MULTIPROC_METHOD=spawn` to avoid CUDA re-init errors.

## Step 3: score with the judge model (e.g., GPT-OSS 120B)

```bash
python minerva-judge/scripts/score_with_judge.py \
  --input minerva-judge/data/responses_llama3_8b.jsonl \
  --output minerva-judge/data/judged_llama3_8b.jsonl \
  --judge-model openai/gpt-oss-120b \
  --backend vllm \
  --batch-size 8 \
  --max-new-tokens 64
```

By default the judge only scores responses marked correct. Use
`--include-incorrect` to score all responses (recommended for BAD labels).
If you want the judge to see the CTI system prompt as well, pass `--include-system`.
You can also pass `--output minerva-judge/data` to write one `judged_*.jsonl`
file per input.

## Step 4: build the SFT dataset

This produces a parquet with `messages` suitable for `MultiTurnSFTDataset`.
You can pass multiple `--input` files to combine models.

```bash
python minerva-judge/scripts/build_sft_dataset.py \
  --input minerva-judge/data/judged_llama3_8b.jsonl \
  --input minerva-judge/data/judged_qwen3_8b.jsonl \
  --output minerva-judge/data/judge_sft.parquet
```

## Output schema (high level)

- `prompts_*.jsonl`: `uid`, `prompt`, `answer`, `reward_fn`, `task`
- `responses_*.jsonl`: adds `model`, `prompt_variant`, `prompt` (the user prompt given to the model),
  `response`, `prediction`, `reward`, `correct`
- `judged_*.jsonl`: adds `judge_response`, `rubric_valid`, `rubric_score`
  (1 for GOOD, 0 for BAD), plus `judge_label`, `judge_category_id`,
  `judge_category_title` when present. Stores `source_file` + `source_line`
  references instead of the original prompt/response and judge prompt.
- `judge_sft.parquet`: `messages` for SFT (`user` = judge prompt, `assistant` = judge response)

# Minerva Judge SFT Pipeline

This folder contains a small pipeline to build a judge SFT dataset from Minerva
train prompts. The flow is:

1) sample prompts from the Minerva train split
2) generate ACRD-style answer-conditioned responses from several base models
3) keep only responses verified as correct by `reward_minerva`
4) score those (prompt, response) pairs with a GPT-5 judge
5) export an SFT dataset to train a smaller judge model (e.g., Qwen 4B/8B)

The judge prompt uses the ACR rubric (`rlvr/verl/utils/reward_score/prompts/acr_rubric_prompt.txt`).

## Folder layout

- `minerva-judge/scripts/` pipeline scripts
- `minerva-judge/data/` generated JSONL + parquet outputs

## Dependencies

- Python deps already used in the repo: `transformers`, `torch`, `pyarrow`, `tqdm`
- OpenAI judge: set `OPENAI_API_KEY` (uses `athena_eval.models.OpenAIModel`)
- Hugging Face model generation: set `HF_TOKEN` if the model is gated

## Step 1: sample prompts

```bash
python minerva-judge/scripts/sample_minerva_prompts.py \
  --input dataset/minerva_base_split/minerva-base-train.jsonl \
  --output minerva-judge/data/prompts_1k.jsonl \
  --sample-size 1000 \
  --seed 1337
```

## Step 2: generate answer-guided responses

The script keeps sampling prompts (with replacement) until it collects
`--target-correct` verified-correct responses. It uses the same ACRD prompt
template (GROUND_TRUTH_LABELS + optional CANONICAL_LABEL_DETAILS block) to
condition the model on the correct answer while instructing it not to mention
the label in reasoning. For HF models, it renders a system+user chat template
using the model tokenizer (matching ACRD’s chat-formatting behavior).

Example (repeat for each base model):

```bash
python minerva-judge/scripts/generate_answer_guided.py \
  --input minerva-judge/data/prompts_1k.jsonl \
  --output minerva-judge/data/responses_llama3_8b.jsonl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --target-correct 1024 \
  --max-attempts 20000 \
  --temperature 0.2
```

Suggested model set:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-8B`
- `Qwen/Qwen3-4B-Instruct`
- `Qwen/Qwen3-8B-Instruct`

## Step 3: score with the GPT-5 judge

```bash
python minerva-judge/scripts/score_with_judge.py \
  --input minerva-judge/data/responses_llama3_8b.jsonl \
  --output minerva-judge/data/judged_llama3_8b.jsonl \
  --judge-model gpt-5 \
  --max-new-tokens 64
```

By default the judge only scores responses marked correct. Use
`--include-incorrect` to score all responses.

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
- `responses_*.jsonl`: adds `model`, `acr_prompt`, `response`,
  `prediction`, `reward`, `correct`, `attempt`
- `judged_*.jsonl`: adds `judge_prompt`, `judge_response`, `rubric_valid`,
  `rubric_score`, and parsed rubric fields when possible
- `judge_sft.parquet`: `messages` for SFT (`user` = judge prompt, `assistant` = judge response)

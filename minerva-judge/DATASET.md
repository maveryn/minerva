# Minerva Judge Reasoning-Quality Dataset

This document describes how we build the reasoning-quality dataset used to train
the GOOD/BAD classifier for Minerva responses. The dataset is designed to judge
reasoning quality independently from answer correctness, so we filter to
reward-correct responses before applying the judge.

## Overview

1. Source prompts from the Minerva base train split.
2. Generate model responses in two variants:
   - plain: original prompt only
   - hinted: prompt + ACR block with ground-truth labels and label references
3. Score each response with the Minerva reward function and keep only correct ones.
4. Judge remaining responses with a rubric-based LLM judge (GOOD/BAD + category).
5. Augment BAD examples using synthetic responses from gpt-oss-120b.
6. Balance classes and split into train/val (80/20).

## Source prompts

- Input JSONL: `dataset/minerva_base_split/minerva-base-train.jsonl`
- Fields used: `prompt`, `answer` or `ground_truth`, `reward_fn`, `task`
- Each row is assigned a stable `uid` by hashing the prompt if missing.

## Response generation (plain + hinted)

Script: `minerva-judge/scripts/generate_answer_guided.py`

Variants:
- plain: uses the prompt as-is.
- hinted: appends an ACR block that includes:
  - `GROUND_TRUTH_LABELS`: gold labels extracted from the answer
  - `LABEL_REFERENCE`: label details pulled from `dataset/label_details/`
  - marker text: `You are generating a reasoning trace for training.`

The ACR block is built via `minerva.acr_prompt.build_acr_block`, with label
details supplied by `LabelDetailsStore`. The label reference is truncated to
`4048` chars if needed. If the prompt becomes too long, the reference is omitted.

Default system prompt (unless `--no-system-prompt`):
- `CTI_SYSTEM_PROMPT` from `rlvr/mydata/data_prepare/prompt.py`

Models used to create the dataset responses:
- `meta-llama/Llama-3.1-8B-Instruct` (slug `llama3_1_8b`)
- `meta-llama/Llama-3.2-3B-Instruct` (slug `llama3_2_3b`)
- `Qwen/Qwen3-4B-Instruct-2507` (slug `qwen3_4b`)
- `Qwen/Qwen3-8B` (slug `qwen3_8b`)
- `openai/gpt-oss-20b` (slug `gpt_oss_20b`)

Typical generation settings (see `minerva-judge/run_generate_models.sh`):
- backend: vLLM
- temperature: `0.7`
- max_new_tokens: `1024`
- variants: `both` (plain + hinted)

Output files:
- `minerva-judge/data/responses_<model>.jsonl`
- Each row includes `prompt`, `prompt_variant`, `response`, `reward`, `correct`,
  plus ACR metadata for hinted prompts.

## Reward filtering

For each response, we compute:
- predicted label(s): via `extract_predicted(...)`
- reward score: via `reward_minerva(...)`

We mark `correct = (reward >= 0.999)` and only keep `correct == true` for
judge labeling and classifier training. This isolates reasoning quality from
answer correctness.

## LLM judge labeling

Script: `minerva-judge/scripts/score_with_judge.py`

Judge model:
- `openai/gpt-oss-120b` (vLLM backend)
- no system prompt by default

Prompt (from `minerva-judge/prompts/judge_prompt.txt`):

```
You are an expert CTI evaluator. You will be given a Question and a Response.
The Question may include answer/reference text as hints.

Label the Response as GOOD or BAD. Mark BAD if any single criterion below is present;
otherwise mark GOOD. Do not grade correctness beyond these checks.

Return exactly one JSON object on a single line and nothing else.
- GOOD: {"label": "GOOD"}
- BAD: {"label": "BAD", "category_id": <number>, "category_title": "<title>"}

If multiple BAD criteria apply, choose the lowest-numbered category.

Criteria (one per bullet):
1. Leakage: says or implies the answer/label/options/reference were provided,
   or quotes/paraphrases provided reference text instead of reasoning from the prompt.
2. Incoherent: loops, repeated phrases/lines, templated filler, or gibberish.
3. Ungrounded: invents concrete details not in the question (extra CVEs,
   vendors, malware names, IOCs, dates, techniques, etc.).
4. Mismatch: reasoning supports a different label than the final answer,
   or directly contradicts it.
5. Unsupported: provides reasoning but does not use evidence from the prompt to
   justify the answer (hand-wavy or irrelevant justification), OR provides an
   answer with no reasoning at all (answer-only).
6. Other: refusals, policy/meta artifacts, generic CTI tutorials, or prompt copying.

Example outputs:
{"label": "GOOD"}
{"label": "BAD", "category_id": 1, "category_title": "Leakage"}


QUESTION:
{QUESTION}

RESPONSE:
{RESPONSE}
```

Outputs:
- One judged file per responses file:
  `minerva-judge/judge_data/judged_<model>.jsonl`
- Each row includes `judge_label`, `judge_category_id`, `judge_category_title`,
  and the original response metadata.

We exclude `judged_gpt_oss_120b.jsonl` from classifier training to avoid
overlap with the judge model.

## Synthetic BAD augmentation

Goal: increase BAD coverage with controlled rubric violations.

Script: `minerva-judge/scripts/generate_synthetic_bad.py`

Source candidates:
- hinted prompts only (must include label reference)

Model:
- `openai/gpt-oss-120b` (vLLM)
- no system prompt by default

Prompt template (from `minerva-judge/prompts/synthetic_bad_prompt.txt`):

```
You are given a question, reference text, and the correct answer. Write a response
that arrives at the correct answer while following the criterion below. The response
MUST satisfy the criterion. Do NOT output only the boxed answer unless the criterion
explicitly allows answer-only. If you include reasoning, keep it brief. Put the
final answer inside \boxed{...}, and follow the question's required output format
for the content inside the box. Output ONLY the response text.
Do NOT add meta commentary or mention instruction conflicts or hierarchy.
Do NOT mention the criterion or the instructions themselves.

CRITERION_INSTRUCTION:
{CRITERION_INSTRUCTION}

CORRECT_ANSWER:
{CORRECT_ANSWER}

QUESTION:
{QUESTION}

REFERENCE:
{REFERENCE}
```

Criteria (from `minerva-judge/prompts/synthetic_bad_criteria.json`):
- Leakage
- Incoherent
- Ungrounded
- Mismatch
- Unsupported

Each synthetic sample:
- picks a random hinted prompt
- picks a random criterion
- generates a response that violates the criterion
- labels the row as BAD with the selected category

We then run `minerva-judge/scripts/score_synthetic_bad.py` to re-score and keep
only synthetic rows where the extracted answer is reward-correct.

Output:
- `minerva-judge/judge_data/synthetic_bad.jsonl` (correct-only rows are used)

## Dataset assembly and split

Script: `minerva-judge/classifier/build_dataset.py`

Steps:
1. Join judged rows back to response rows using `judge_key` and `source_file`.
2. Keep only `correct == true` and `judge_label in {GOOD, BAD}`.
3. Add synthetic BAD rows with `correct == true`.
4. Balance by downsampling to 32k GOOD + 32k BAD (seed 1337).
5. Split 80/20 by class into train/val.

Outputs:
- `minerva-judge/classifier/data/train.jsonl`
- `minerva-judge/classifier/data/val.jsonl`

Training tokenization:
- We keep the full response and truncate only the prompt when needed.
- Implemented as `tokenizer(prompt, response, truncation="only_first", max_length=...)`.

## Repro commands

```
# Generate responses (plain + hinted)
bash minerva-judge/run_generate_models.sh

# Judge correct responses only
bash minerva-judge/run_judge_folder.sh

# Synthetic BAD generation + correctness scoring
bash minerva-judge/run_generate_synthetic_bad.sh
python minerva-judge/scripts/score_synthetic_bad.py \
  --input minerva-judge/judge_data/synthetic_bad.jsonl \
  --mismatch-output minerva-judge/judge_data/synthetic_bad_mismatches.jsonl

# Build balanced classifier dataset
python minerva-judge/classifier/build_dataset.py \
  --judged-dir minerva-judge/judge_data \
  --responses-dir minerva-judge/data \
  --synthetic minerva-judge/judge_data/synthetic_bad.jsonl \
  --sample-per-class 32000 \
  --train-ratio 0.8
```

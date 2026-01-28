# Pairwise CTI Preference Evaluation

This folder contains a script that runs pairwise preference judging over prompts where **all five models answered correctly**. The judge compares two LLM responses at a time and selects which is preferable for **cyber threat intelligence (CTI) analysis**.

## Input

- `all_models_correct_samples.json` (JSON list)
  - `task`, `subtask`, `prompt`
- `responses`: map of model name -> response text

The current file includes five models, so there are **10 unordered pairs** per prompt.

## What the script does

`run_pairwise_judge.py`:

1. Loads `all_models_correct_samples.json`.
2. Enumerates all 10 model pairs per prompt.
3. Randomizes the A/B order per pair **deterministically** using a hash of `(seed, sample_key, model_a, model_b)` so runs are reproducible.
4. Builds a judge prompt using the template below.
5. Calls the judge LLM (default: **gpt-5.2**).
6. Saves the raw judge response and a parsed winner (A/B/tie/parse_error).

### Judge prompt template

```text
You are a cyber threat intelligence (CTI) analyst expert. You are given a question and two LLM responses. Both responses are verifiably correct. Select the response you would prefer to receive for CTI analysis.

Question:
{question}

Response A:
{response_a}

Response B:
{response_b}

Return JSON only:
{"winner": "A"|"B"|"tie", "rationale": "<short rationale>"}
```

## Output

The script writes a JSONL file (`pairwise_judge_results.jsonl` by default). Each line contains:

- `pair_key`: unique key for (prompt, model pair)
- `sample_index`, `sample_key`, `task`, `subtask`, `prompt`
- `model_a`, `model_b`, `response_a`, `response_b`
- `judge_model`, `judge_prompt`, `judge_response`, `parsed`
- `winner`: `A` / `B` / `tie` / `parse_error`
- `winner_model`: model name / `tie` / `parse_error`
- `rationale`: short text from the judge (if present)

The script appends to the output file and skips any `pair_key` already present unless `--overwrite` is used.

## How to run

From this folder:

```bash
python run_pairwise_judge.py \
  --input all_models_correct_samples.json \
  --output pairwise_judge_results.jsonl \
  --judge-model gpt-5.2
```

### Environment

- Set `OPENAI_API_KEY` for OpenAI judge models.
- The script uses the OpenAI Responses API for `gpt-5.*` models and does not require `transformers` when `--judge-model` is OpenAI.

## Options

- `--judge-model`: change the judge model (default `gpt-5.2`).
- `--batch-size`: number of pairs per batch (default 1).
- `--temperature`: judge sampling temperature (default 0.0).
- `--seed`: controls A/B order shuffling (default 42).
- `--limit`: cap the total number of pairs judged.
- `--overwrite`: overwrite the output instead of resuming.

## Summarize and plot

Use `summarize_pairwise_judge.py` to compute per-pair win/tie rates and render
the triangular comparison plot:

```bash
python summarize_pairwise_judge.py \
  --input pairwise_judge_results.jsonl
```

Outputs:
- `pairwise_summary.json`
- `pairwise_summary.csv`
- `pairwise_heatmap.png`
- `pairwise_heatmap.pdf`

Notes:
- For the seaborn-styled heatmap, install `seaborn` (otherwise it falls back to matplotlib defaults).
- The heatmap encodes P(Model A wins) with ties excluded from the denominator.

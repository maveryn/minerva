# Response Quality Judge Sampling

This folder contains a script that samples prompts where five specified models all answered correctly on a filtered set of evalall tasks.

## What it does
- Uses scored JSONL files in `runs/` for:
  - llama-3-8B
  - llama3-primus
  - llama3-sec
  - minerva_llama8b_grpo
  - minerva_llama8b_noctua
- Tasks included (12 total):
  - MCQ3k, CyberMetric, ElasticToAttack, ATE, RCM, RMS, VSP, ThreatIntelReasoning, APTNER
  - Prism (meta; combines PrismIP/PrismURL/PrismDomain/PrismHash)
  - AZERG (meta; combines AZERG-T1..T4)
  - ANNOCTR (meta; combines ANNOCTR-T1..T4)
- Tasks ignored: CISSP, TAA-MCQ, CyNER.

## Correctness criteria
- VSP: MAD <= 1.0 (from `*-scored.jsonl`)
- ThreatIntelReasoning: `answered_correctly == true`
- APTNER: `fp == 0` and `fn == 0`
- AZERG/ANNOCTR: `predicted == truth` (T1) and non-empty truth for T2..T4
- All other base tasks: `score == 1`
- Prism meta: correctness is computed at the report level (majority-vote IoC labeling), then any non-empty prompt from those correct reports is eligible.

## Sampling
- For each of the 12 tasks, select up to 100 examples at random.
- If fewer than 100 are available, take all.
- Random seed: 42.

## Output
The script writes:
- `response-quality-judge/all_models_correct_samples.json`

Each entry contains:
- `task`: one of the 12 task groups
- `subtask`: populated for meta tasks (e.g., `PrismIP`, `AZERG-T2`), otherwise `null`
- `prompt`
- `responses`: a map of model name to response text

## Sample set size (current)

As of January 28, 2026, `all_models_correct_samples.json` contains:
- 752 total questions

## Pairwise judge prompt template (exact)

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

## Run
From repo root:

```
python response-quality-judge/generate_all_correct_samples.py
```

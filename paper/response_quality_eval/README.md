# Response Quality Eval Workspace

This folder is the clean paper-side workspace for the response-quality study.

It points to the current canonical datasets and rubric under `minerva-judge/human-judge`, but keeps the paper-facing assets grouped in one place.

## Contents

- [full_gpt_eval_dataset](/home/jovyan/work/minerva/paper/response_quality_eval/full_gpt_eval_dataset)
  - canonical 7-model full evaluation set
  - `350` prompts total
  - `50` prompts for each of the 7 tasks
  - `2450` response rows total

- [full_gpt_eval_dataset_3b5models](/home/jovyan/work/minerva/paper/response_quality_eval/full_gpt_eval_dataset_3b5models)
  - canonical 3B-family full evaluation set on the same `350` benchmark questions
  - `350` prompts total
  - `50` prompts for each of the 7 tasks
  - `1750` response rows total

- [human_agreement_subset](/home/jovyan/work/minerva/paper/response_quality_eval/human_agreement_subset)
  - balanced human agreement subset
  - `168` single-response items total
  - built from `84` underlying prompts

- [human_annotator_bundle.zip](/home/jovyan/work/minerva/paper/response_quality_eval/human_annotator_bundle.zip)
  - standalone human annotation bundle for the `168`-item subset

- [POINTWISE_RUBRIC.md](/home/jovyan/work/minerva/paper/response_quality_eval/POINTWISE_RUBRIC.md)
  - current rubric used for human and GPT scoring

- [response_quality_subset_design.md](/home/jovyan/work/minerva/paper/response_quality_eval/response_quality_subset_design.md)
  - design note describing exactly how the full and human subsets were constructed

- [judge_outputs](/home/jovyan/work/minerva/paper/response_quality_eval/judge_outputs)
  - reserved for paper-side summaries or copied judge outputs

## Canonical Sources

The source-of-truth dataset directories still live under:

- `minerva-judge/human-judge/shared_prompt_350_seven_tasks_seven_models`
- `minerva-judge/human-judge/shared_prompt_350_seven_tasks_five_3b_models`
- `minerva-judge/human-judge/pointwise_168_seven_tasks_seven_models`

This folder uses symlinks so the paper workspace stays clean without duplicating the data.

## Current Model Set

- `llama-3-8B`
- `llama3-sec`
- `llama3-sec-reasoning`
- `minerva_llama8b_grpo`
- `minerva_llama8b_noctua`
- `llama3-8b-dart`
- `llama3-8b-star`

## Current 3B Model Set

- `llama-3-3B`
- `minerva_llama3b_grpo`
- `llama3-3b-star`
- `llama3-3b-dart`
- `minerva_llama3b_noctua`

## Current Task Set

- `MCQ3k`
- `CyberMetric`
- `RCM`
- `VSP`
- `ATE`
- `RMS`
- `ElasticToAttack`

## Run GPT Evaluation Here

The scorer now supports a separate output directory, so you can keep judge
outputs under this paper workspace.

Example:

```bash
cd /home/jovyan/work/minerva/minerva-judge/human-judge

python score_pointwise_with_gpt.py \
  --subset shared_prompt_350_seven_tasks_seven_models \
  --input /home/jovyan/work/minerva/paper/response_quality_eval/full_gpt_eval_dataset/blind_responses.jsonl \
  --output-dir /home/jovyan/work/minerva/paper/response_quality_eval/judge_outputs/full_gpt_eval_dataset
```

This will write:

```text
/home/jovyan/work/minerva/paper/response_quality_eval/judge_outputs/full_gpt_eval_dataset/gpt52_scores.jsonl
```

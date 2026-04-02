# Response Quality Evaluation Subset Design

This note documents the two subset constructions used for the response-quality study:

- the full shared-prompt set used for judge-model evaluation
- the smaller human annotation set used for human vs GPT agreement

## Tasks

We use these 7 tasks:

- `MCQ3k` (`CKT`)
- `CyberMetric`
- `RCM`
- `VSP`
- `ATE`
- `RMS`
- `ElasticToAttack` (`ElasticRule`)

These were chosen because they either:

- explicitly request brief justification in the prompt, or
- require nontrivial CTI reasoning / explanation

We excluded the more structured extraction / tagging tasks (`SOCEval`, `APTNER`, `LANCE`, `AnnoCTR`, `AZERG`) from the main study because they are dominated by schema-following or label output, which makes the response-quality rubric less meaningful.

## Models

We use these 7 model runs:

- `llama-3-8B` -> `runs/meta-llama/Llama-3.1-8B-Instruct`
- `llama3-sec` -> `runs/fdtn-ai/Foundation-Sec-8B-Instruct`
- `llama3-sec-reasoning` -> `runs/foundation-sec-8b-reasoning`
- `minerva_llama8b_grpo` -> `runs/xashru/minerva_grpo_llama8b_500_490`
- `minerva_llama8b_noctua` -> `runs/xashru/minerva_noctua_llama8b_ema_0.05`
- `llama3-8b-dart` -> `runs/dart_llama31_8b_dart_best470`
- `llama3-8b-star` -> `runs/star_llama31_8b_round0`

## Full GPT Evaluation Set

Builder:

- `minerva-judge/human-judge/build_shared_prompt_350_seven_tasks_seven_models.py`

Output:

- `minerva-judge/human-judge/shared_prompt_350_seven_tasks_seven_models`

Design:

- require prompt presence in all 7 model runs
- do not filter by correctness
- do not filter by response length
- sample `50` shared prompts per task
- total = `350` prompts, `2450` responses

Sampling procedure:

1. For each task, build the set of prompt keys shared across all 7 model runs.
2. Shuffle the shared prompt pool with a fixed task-specific seed.
3. Take the first `50` prompts.
4. Expand each selected prompt to all 7 model responses.

Why this design:

- It keeps the evaluation prompt-aligned across models.
- It does not bias the sample toward jointly correct or jointly verbose responses.
- It supports later pairwise comparison by deriving preference from pointwise scores on the same prompt.
- It is large enough for judge-model scoring because GPT/Claude can score it automatically.

Available shared-prompt pool sizes before sampling:

- `MCQ3k`: `3000`
- `CyberMetric`: `2000`
- `RCM`: `2000`
- `VSP`: `2000`
- `ATE`: `500`
- `RMS`: `500`
- `ElasticToAttack`: `431`

Selected counts:

- `50` prompts per task
- `350` prompts total
- `350` responses per model

## Human Agreement Set

Final human pointwise builder:

- `minerva-judge/human-judge/build_pointwise_168_seven_tasks_seven_models.py`

Final output:

- `minerva-judge/human-judge/pointwise_168_seven_tasks_seven_models`

Human annotator bundle:

- `minerva-judge/human-judge/dist/human-annotator-pointwise-168-seven-tasks-seven-models.zip`

### Why We Did Not Use All 350 Prompts For Humans

The full `350`-prompt set is feasible for GPT/Claude, but too large for repeated human annotation.

We also wanted the human subset to preserve fair comparison coverage across:

- all `7` tasks
- all `7` models
- all `21` unordered model pairs

At the same time, we wanted humans to score responses independently with the rubric, not do direct pairwise preference judgments.

### Pair-Coverage Logic

With `7` models there are `21` unordered model pairs.

If we required exact pair coverage within every task, the minimum pointwise human set would be:

- `21` prompts per task
- `147` prompts overall
- `294` single-response items after expanding each prompt to two model responses

That was too large for the human study. Instead, we used a smaller design that keeps:

- exact task balance
- exact overall model balance
- exact overall pair balance

but does not require every pair to appear inside every task.

### Construction

We build the human set directly from the 7-model full shared-prompt set.

1. For each task, start from the `50` prompts already selected for the full GPT evaluation.
2. Compute a prompt-level difficulty proxy using the mean objective task score across the `7` model responses on that prompt.
3. Sort prompts by that difficulty score and split the `50` prompts into `3` buckets.
4. Sample `4` prompts from each bucket, giving `12` prompts per task and `84` prompts total.
5. Build the `21` unordered model pairs from the `7` models.
6. Assign pairs so that:
   - each pair appears exactly `4` times overall
   - each task gets exactly `12` prompt-pair assignments
7. Expand each paired prompt into its two model responses as separate single-response items.
8. Shuffle globally for annotation order while preventing adjacent duplicate prompts.

This gives a smaller human subset that remains representative across task difficulty and balanced across models and pairs overall.

### Final Human Set Properties

- `168` single-response items total
- `84` underlying prompts
- `24` items per task
- `24` items per model overall
- each of the `21` model pairs appears exactly `4` times overall
- `0` adjacent duplicate prompts after shuffle

Important:

- Annotators see one prompt and one blind response at a time.
- Some prompts reappear later with a different blind model response.
- The annotator does not see the two responses side by side.
- Pairwise preference is derived later from the independent scores on the matched prompt.

## Rubric

Current rubric file:

- `minerva-judge/human-judge/POINTWISE_RUBRIC.md`

We use a 3-criterion pointwise rubric on a `1` to `4` scale:

- `Writing Quality`
- `Evidence Use`
- `CTI Concept Use`

The criterion question text is shown inline in the annotation app so annotators do not need to open the rubric separately while scoring.

## Resulting Workflow

1. Run GPT / Claude on the full `350`-prompt shared set.
2. Use the smaller `168`-item human set for double annotation and human vs GPT agreement.
3. Derive pairwise model preferences afterward by comparing independent pointwise totals on matched prompts.
# Response Quality Evaluation Subset Design

This note documents the two subset constructions used for the response-quality study:

- the full shared-prompt set used for judge-model evaluation
- the smaller human annotation set used for human vs GPT agreement

## Tasks

We use these 7 tasks:

- `MCQ3k` (`CKT`)
- `CyberMetric`
- `RCM`
- `VSP`
- `ATE`
- `RMS`
- `ElasticToAttack` (`ElasticRule`)

These were chosen because they either:

- explicitly request brief justification in the prompt, or
- require nontrivial CTI reasoning / explanation

We excluded the more structured extraction / tagging tasks (`SOCEval`, `APTNER`, `LANCE`, `AnnoCTR`, `AZERG`) from the main study because they are dominated by schema-following or label output, which makes the response-quality rubric less meaningful.

## Models

We use these 7 model runs:

- `llama-3-8B` -> `runs/meta-llama/Llama-3.1-8B-Instruct`
- `llama3-sec` -> `runs/fdtn-ai/Foundation-Sec-8B-Instruct`
- `llama3-sec-reasoning` -> `runs/foundation-sec-8b-reasoning`
- `minerva_llama8b_grpo` -> `runs/xashru/minerva_grpo_llama8b_500_490`
- `minerva_llama8b_noctua` -> `runs/xashru/minerva_noctua_llama8b_ema_0.05`
- `llama3-8b-dart` -> `runs/dart_llama31_8b_dart_best470`
- `llama3-8b-star` -> `runs/star_llama31_8b_round0`

## Full GPT Evaluation Set

Builder:

- `minerva-judge/human-judge/build_shared_prompt_350_seven_tasks_seven_models.py`

Output:

- `minerva-judge/human-judge/shared_prompt_350_seven_tasks_seven_models`

Design:

- require prompt presence in all 7 model runs
- do not filter by correctness
- do not filter by response length
- sample `50` shared prompts per task
- total = `350` prompts, `2450` responses

Sampling procedure:

1. For each task, build the set of prompt keys shared across all 7 model runs.
2. Shuffle the shared prompt pool with a fixed task-specific seed.
3. Take the first `50` prompts.
4. Expand each selected prompt to all 7 model responses.

Why this design:

- It keeps the evaluation prompt-aligned across models.
- It does not bias the sample toward jointly correct or jointly verbose responses.
- It supports later pairwise comparison by deriving preference from pointwise scores on the same prompt.
- It is large enough for judge-model scoring because GPT/Claude can score it automatically.

Available shared-prompt pool sizes before sampling:

- `MCQ3k`: `3000`
- `CyberMetric`: `2000`
- `RCM`: `2000`
- `VSP`: `2000`
- `ATE`: `500`
- `RMS`: `500`
- `ElasticToAttack`: `431`

Selected counts:

- `50` prompts per task
- `350` prompts total
- `350` responses per model

## Human Agreement Set

Final human pointwise builder:

- `minerva-judge/human-judge/build_pointwise_168_seven_tasks_seven_models.py`

Final output:

- `minerva-judge/human-judge/pointwise_168_seven_tasks_seven_models`

Human annotator bundle:

- `minerva-judge/human-judge/dist/human-annotator-pointwise-168-seven-tasks-seven-models.zip`

### Why We Did Not Use All 350 Prompts For Humans

The full `350`-prompt set is feasible for GPT/Claude, but too large for repeated human annotation.

We also wanted the human subset to preserve fair comparison coverage across:

- all `7` tasks
- all `7` models
- all `21` unordered model pairs

At the same time, we wanted humans to score responses independently with the rubric, not do direct pairwise preference judgments.

### Pair-Coverage Logic

With `7` models there are `21` unordered model pairs.

If we required exact pair coverage within every task, the minimum pointwise human set would be:

- `21` prompts per task
- `147` prompts overall
- `294` single-response items after expanding each prompt to two model responses

That was too large for the human study. Instead, we used a smaller design that keeps:

- exact task balance
- exact overall model balance
- exact overall pair balance

but does not require every pair to appear inside every task.

### Construction

We build the human set directly from the 7-model full shared-prompt set.

1. For each task, start from the `50` prompts already selected for the full GPT evaluation.
2. Compute a prompt-level difficulty proxy using the mean objective task score across the `7` model responses on that prompt.
3. Sort prompts by that difficulty score and split the `50` prompts into `3` buckets.
4. Sample `4` prompts from each bucket, giving `12` prompts per task and `84` prompts total.
5. Build the `21` unordered model pairs from the `7` models.
6. Assign pairs so that:
   - each pair appears exactly `4` times overall
   - each task gets exactly `12` prompt-pair assignments
7. Expand each paired prompt into its two model responses as separate single-response items.
8. Shuffle globally for annotation order while preventing adjacent duplicate prompts.

This gives a smaller human subset that remains representative across task difficulty and balanced across models and pairs overall.

### Final Human Set Properties

- `168` single-response items total
- `84` underlying prompts
- `24` items per task
- `24` items per model overall
- each of the `21` model pairs appears exactly `4` times overall
- `0` adjacent duplicate prompts after shuffle

Important:

- Annotators see one prompt and one blind response at a time.
- Some prompts reappear later with a different blind model response.
- The annotator does not see the two responses side by side.
- Pairwise preference is derived later from the independent scores on the matched prompt.

## Rubric

Current rubric file:

- `minerva-judge/human-judge/POINTWISE_RUBRIC.md`

We use a 3-criterion pointwise rubric on a `1` to `4` scale:

- `Writing Quality`
- `Evidence Use`
- `CTI Concept Use`

The criterion question text is shown inline in the annotation app so annotators do not need to open the rubric separately while scoring.

## Resulting Workflow

1. Run GPT / Claude on the full `350`-prompt shared set.
2. Use the smaller `168`-item human set for double annotation and human vs GPT agreement.
3. Derive pairwise model preferences afterward by comparing independent pointwise totals on matched prompts.

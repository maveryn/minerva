# Human Judge Subsets

This folder contains human-annotation subsets derived from the pairwise GPT-5.2
CTI preference judgments in
`../judge-pairwise-quality-eval/pairwise_judge_results.jsonl`.

## Design

- Sampling is done at the prompt level from
  `../judge-pairwise-quality-eval/all_models_correct_samples.json`.
- The 100-item subset is a strict subset of the 200-item subset.
- Task coverage is proportional to the full 752-prompt pool.
- Model-pair coverage is exactly uniform within each subset:
  - `subset_100`: 10 items per unordered model pair
  - `subset_200`: 20 items per unordered model pair
- Each item includes only one model pair, so human annotators never see the same
  prompt repeated across multiple pairs within a subset.

## Files

For each subset:

- `blind_master.csv`: shared blind annotation item file
- `blind_master.jsonl`: same content as JSONL, used by the web app
- `key.jsonl`: hidden answer key with model identities and GPT-5.2 judgment
- `summary.json`: task, pair, and model-appearance counts

Top level:

- `build_human_judge_subsets.py`: reproducible generator
- `build_pointwise_dataset.py`: reproducible pointwise single-response dataset builder
- `build_pointwise_pilot_50.py`: reproducible 50-item pilot sampler
- `build_pointwise_pilot_50_correct_only.py`: reproducible 50-item independent-per-model correct-only smoke-test sampler
- `app.py`: local web app for human annotation
- `analyze_agreement.py`: compares human-human and human-GPT agreement
- `score_pointwise_with_gpt.py`: scores pointwise items with an OpenAI or Anthropic judge model using the rubric
- `analyze_pointwise_agreement.py`: compares pointwise human rubric scores with a judge file
- `annotator_app.py`: dependency-free app for the annotator handoff zip
- `build_annotator_bundle.py`: builds the distributable annotator zip
- `POINTWISE_RUBRIC.md`: human scoring rubric for single-response evaluation
- `summary.json`: nested subset overview

## Pointwise Dataset

`pointwise_subset_238/` is the current single-response rubric dataset. It is
built from the shared-correct llmbench pool with these filters:

- tasks: `VSP`, `RCM`, `ATE`, `MCQ3k`, `CyberMetric`
- all 6 models must be correct
- at least 4 of 6 model responses must have `20+` words
- cap at 50 prompts per task

This yields:

- `238` prompts
- `1428` blind response items

Files in `pointwise_subset_238/`:

- `blind_prompts.jsonl`: prompt-level blind view with the six shuffled responses
- `blind_responses.jsonl`: one blind response item per row for annotation
- `blind_responses.csv`: CSV version of the blind response rows
- `annotation_template.csv`: same rows with empty rubric columns
- `annotation_template.jsonl`: JSONL version of the blank rubric template
- `key.jsonl`: hidden response-item to model mapping
- `selected_prompts_with_models.jsonl`: internal prompt-level file with model names
- `summary.json`: counts, task mix, model list, and filter settings

Regenerate:

```bash
python build_pointwise_dataset.py
```

## Pointwise Pilot 50

`pointwise_pilot_50/` is a small alignment-check subset drawn from
`pointwise_subset_238/`.

Design:

- `50` total items
- `10` items per task across `VSP`, `RCM`, `ATE`, `MCQ3k`, `CyberMetric`
- one blind response per prompt
- model exposure balanced as evenly as possible across the 6 models

Files in `pointwise_pilot_50/`:

- `blind_responses.jsonl`: annotation order for the pilot
- `blind_responses.csv`: CSV version of the pilot items
- `annotation_template.csv`: blank rubric sheet
- `annotation_template.jsonl`: JSONL blank rubric sheet
- `key.jsonl`: hidden model mapping
- `summary.json`: task/model counts and seeds

Regenerate:

```bash
python build_pointwise_pilot_50.py
```

Independent per-model correct-only smoke test:

```bash
python build_pointwise_pilot_50_correct_only.py
```

Score the correct-only smoke test with GPT-5.2:

```bash
python score_pointwise_with_gpt.py --subset pointwise_pilot_50_correct_only
```

Score the pilot with Claude Sonnet 4.6:

```bash
python score_pointwise_with_gpt.py --subset pointwise_pilot_50_correct_only --judge-model claude-sonnet-4-6
```

Compare one or two human runs with a judge file:

```bash
python analyze_pointwise_agreement.py --subset pointwise_pilot_50_correct_only --annotator-a alice
python analyze_pointwise_agreement.py --subset pointwise_pilot_50_correct_only --annotator-a alice --annotator-b bob
python analyze_pointwise_agreement.py --subset pointwise_pilot_50_correct_only --annotator-a alice --judge-file pointwise_pilot_50_correct_only/claude_sonnet_4_6_scores.jsonl --judge-name claude-sonnet-4-6
python analyze_pointwise_agreement.py --subset pointwise_pilot_50_correct_only --annotator-a alice --annotator-a-file dist/alice.jsonl --judge-file pointwise_pilot_50_correct_only/gpt52_scores.jsonl
```

## Annotation labels

Use the same label space as the GPT judge:

- `A`
- `B`
- `TIE`

## Regenerate

From this folder:

```bash
python build_human_judge_subsets.py
```

## Run the annotation app

From this folder:

```bash
python app.py
```

Then open `http://127.0.0.1:8787/`, choose a dataset, and enter an annotator ID.
The app auto-detects pairwise vs pointwise subsets. Annotations are saved to:

```text
annotations/<subset>/<annotator>.jsonl
```

## Compare agreement

After both annotators finish:

```bash
python analyze_agreement.py --subset subset_100 --annotator-a alice --annotator-b bob
```

## Build Annotator Zip

To create the handoff bundle for annotators:

```bash
python build_annotator_bundle.py
```

This writes:

- `dist/human-annotator-bundle/`
- `dist/human-annotator-bundle.zip`

The zip includes only the blind subset files and the dependency-free
`annotator_app.py`. It does not include `key.jsonl`.

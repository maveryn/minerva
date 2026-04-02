# Release Artifact Strategy

This note records what should live in the main `minerva` git history versus what
should be published as external artifacts.

## Goal

Keep the repository focused on:

- source code
- configs
- reproducibility scripts
- lineage / methodology docs
- small preview samples when needed

Avoid putting large generated datasets and judge outputs directly into the main
git history.

## Current artifact sizes

At the time of writing, local artifact sizes are approximately:

- `dart-artifacts/`: `50G`
- `minerva-judge/human-judge/`: `144M`
- `minerva-judge/data/`: `1.6G`
- `minerva-judge/judge_data/`: `393M`
- `paper/`: `63M`

These numbers are too large and too generated to treat as normal source files.

## What to commit

### DART

Commit:

- `dart/*.py`
- `dart/configs/`
- `dart/*.sh`
- DART docs such as `README.md`, `IMPLEMENTATION_PLAN.md`, `dataset_lineage.md`,
  and `paper.md`

Do not commit:

- `dart-artifacts/`
- full generated attempts / accepted JSONL files
- large parquet outputs

Recommended in-repo metadata:

- dataset lineage docs
- small preview samples only
- checksums / manifests for released datasets

### LLM judge code

Commit:

- pairwise judge code
- pointwise judge code
- prompts and rubrics
- subset builders
- agreement / analysis scripts

Do not commit new large generated judge outputs by default.

Note:

- `minerva-judge/data/` and `minerva-judge/judge_data/` already contain tracked
  data in this repo. We should avoid expanding these directories further in the
  main history.

### Human annotation tooling

Commit:

- `minerva-judge/human-judge/*.py`
- `minerva-judge/human-judge/*.md`
- start scripts

Do not commit by default:

- generated subset folders
- `dist/` zip bundles
- local `annotations/`

### Paper workspace

Commit only source assets by default:

- plotting scripts
- build scripts
- markdown notes that are actually used

Keep generated figures, tables, JSON summaries, CSV exports, and zip bundles out
of the main history unless we explicitly decide a particular final artifact
belongs in the repo.

## Recommended dataset publication path

For DART and large judge outputs, publish the datasets outside the main git
history.

Preferred options:

1. Hugging Face dataset repositories for the canonical released datasets
2. Release tarballs plus checksums if a separate dataset repo is not desired

In this repo we should keep:

- the generation code
- the scoring code
- the annotation tooling
- lineage docs
- download instructions
- manifests / checksums

## Practical commit split

1. DART code + docs
2. judge / annotation code + docs
3. optional paper scripts
4. dataset manifests and download instructions

This keeps the main repo reviewable while still making the datasets
reproducible and publishable.

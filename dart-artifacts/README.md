# DART Artifacts

This folder stores the generated artifacts for the Llama-3.1-8B DART-CTI teacher run. We do not track the large raw attempt logs in git, but we do track the final train/validation parquets and the stage-level `summary*.json` files that describe how those datasets were built.

Teacher model used throughout:

- `meta-llama/Llama-3.1-8B-Instruct`

## Tracked final parquets

Train:

- [dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet)
  - final DART SFT train set
  - `64,000` traces = `32,000` prompts x `2` traces each

Minerva synthetic validation:

- [dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet)
  - final synthetic Minerva dev set
  - `1,200` prompts with `1` accepted trace each

Athena synthetic validation:

- [dart_cti_llama8b/datasets/athena_bench/athena_bench_v2_from_athena_bench_v3_from_v2.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v2_from_athena_bench_v3_from_v2.parquet)
  - final synthetic Athena validation set
  - `950` prompts with `1` accepted trace each

## Tracked summary files

Train:

- [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v1.json)
- [summary_llama8b_v2_cap32_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v2_cap32_seed.json)
- [summary_llama8b_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v3_cap200_seed.json)
- [summary_v2_from_llama8b_direct_from_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v2_from_llama8b_direct_from_v3_cap200_seed.json)

Minerva synthetic validation:

- [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v1.json)
- [summary_v2_from_accepted_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_accepted_v1.json)
- [summary_v2_from_minerva_dev_v3_from_v2.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_minerva_dev_v3_from_v2.json)

Athena synthetic validation:

- [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v1.json)
- [summary_v2_from_accepted_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v2_from_accepted_v1.json)
- [summary_v2_from_athena_bench_v3_from_v2.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v2_from_athena_bench_v3_from_v2.json)

## Full train-set generation pipeline

Target:

- `32,000` train prompts
- `2` accepted traces per prompt
- final target size `64,000`

Stage T1: plain generation only

- source summary: [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v1.json)
- settings: original prompt only, target `2`, max `32` attempts per prompt
- result:
  - `880,782` generated attempts
  - `16,807` accepted plain traces
  - coverage after stage: `6,987` complete, `2,833` one-trace, `22,180` zero-trace

Stage T2: answer-guided fill with `ml+heuristic` filter

- source summary: [summary_llama8b_v2_cap32_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v2_cap32_seed.json)
- settings: answer-conditioned prompting, verifier + heuristic/TextCNN filter, cap `32`
- result:
  - seed traces entering stage: `16,807`
  - `38,868` new guided traces accepted
  - cumulative traces: `55,675`
  - coverage after stage: `26,640` complete, `2,395` one-trace, `8,325` missing traces

Stage T3: answer-guided fill with filter disabled

- source summary: [summary_llama8b_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v3_cap200_seed.json)
- settings: answer-conditioned prompting, verifier only, cap `200`
- result:
  - seed traces entering stage: `55,675`
  - `8,296` new guided traces accepted
  - cumulative traces: `63,971`
  - coverage after stage: `31,981` complete, `9` one-trace, `29` missing traces

Stage T4: deterministic direct-answer tail fill

- source summary: [summary_v2_from_llama8b_direct_from_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v2_from_llama8b_direct_from_v3_cap200_seed.json)
- settings: boxed gold answer, verifier checked
- result:
  - `29` direct-answer traces added
  - final total: `64,000`
  - final coverage: `32,000 / 32,000` complete

Final train composition:

- plain: `16,807`
- guided fill: `47,164`
- direct-answer tail fill: `29`

## Synthetic validation generation

Minerva dev target:

- `1,200` prompts
- `1` accepted trace per prompt

Minerva dev stages:

- V1 plain: [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v1.json)
  - `401` accepted plain traces, `799` prompts still missing
- V2 guided fill with `ml+heuristic`: [summary_v2_from_accepted_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_accepted_v1.json)
  - cumulative `1,120` accepted traces, `80` prompts still missing
- V3 guided fill with verifier only: [summary_v2_from_minerva_dev_v3_from_v2.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_minerva_dev_v3_from_v2.json)
  - final `1,200 / 1,200` coverage

Athena target:

- `950` prompts
- `1` accepted trace per prompt

Athena stages:

- A1 plain: [summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v1.json)
  - `595` accepted plain traces, `355` prompts still missing
- A2 guided fill with `ml+heuristic`: [summary_v2_from_accepted_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v2_from_accepted_v1.json)
  - cumulative `890` accepted traces, `60` prompts still missing
- A3 guided fill with verifier only: [summary_v2_from_athena_bench_v3_from_v2.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v2_from_athena_bench_v3_from_v2.json)
  - final `950 / 950` coverage

## Notes

- The `datasets_capec_bug_20260327_180344/` subtree is an archived buggy snapshot and is intentionally not tracked here.
- The large raw attempt logs and accepted-trace JSONLs remain generated artifacts outside git; this folder tracks only the final parquets plus compact stage summaries and documentation.

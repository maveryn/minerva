# DART Llama 8B Dataset Lineage

This note records how the current Llama 8B DART train and validation datasets were built, stage by stage.

Teacher model used throughout:

- `meta-llama/Llama-3.1-8B-Instruct`

## Current artifact status

### Final train dataset

- parquet: [dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet)
- accepted jsonl: [dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v2_from_llama8b_direct_from_v3_cap200_seed.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v2_from_llama8b_direct_from_v3_cap200_seed.jsonl)
- size: `64,000` traces
- coverage: `32,000 / 32,000` questions complete

### Final Minerva dev synthetic validation dataset

- parquet: [dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet)
- accepted jsonl: [dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v2_from_minerva_dev_v3_from_v2.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v2_from_minerva_dev_v3_from_v2.jsonl)
- size: `1,200` traces
- coverage: `1,200 / 1,200` prompts complete

### Athena synthetic validation dataset

- current parquet: [dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet)
- current accepted jsonl: [dart-artifacts/dart_cti_llama8b/datasets/athena_bench/accepted_v1.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/accepted_v1.jsonl)
- current size: `595` traces
- current coverage: `595 / 950` prompts complete
- status: partial `v1` only, `v2/v3` not built yet

## Train set lineage

Target:

- `32,000` train prompts
- `2` accepted traces per prompt
- final target size `64,000`

### Stage T1: plain `v1`

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/train/summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v1.json)

Settings:

- mode: plain original prompt only
- target accepted per question: `2`
- max attempts per question: `32`

Result:

- generated attempts: `880,782`
- accepted plain traces: `16,807`
- complete questions with 2 traces: `6,987`
- one-trace questions: `2,833`
- zero-trace questions: `22,180`
- timing: `13,990.265s`

Artifacts:

- [accepted_v1.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v1.jsonl)
- [train_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v1.parquet)

### Stage T2: guided fill with `ml+heuristic` filter, capped at 32

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v2_cap32_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v2_cap32_seed.json)

Settings:

- mode: answer-guided ACR prompt
- acceptance: verifier + `ml+heuristic`
- cap used for recovered dataset: `32` fill attempts per question

Result:

- seed traces entering stage: `16,807`
- guided traces added at this stage: `38,868`
- cumulative traces after stage: `55,675`
- complete questions with 2 traces: `26,640`
- one-trace questions: `2,395`
- missing traces after stage: `8,325`

Artifacts:

- [accepted_llama8b_v2_cap32_seed.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_llama8b_v2_cap32_seed.jsonl)
- [train_v2_cap32_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_cap32_seed.parquet)

Note:

- this dataset was reconstructed from the interrupted `v2` attempts log, so there is no official stage timing json for the capped recovery

### Stage T3: guided fill with filter disabled, capped at 200

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_llama8b_v3_cap200_seed.json)

Settings:

- mode: answer-guided ACR prompt
- acceptance: verifier only
- cap used for recovered dataset: `200` fill attempts per question

Result:

- seed traces entering stage: `55,675`
- guided traces added at this stage: `8,296`
- cumulative traces after stage: `63,971`
- complete questions with 2 traces: `31,981`
- one-trace questions: `9`
- missing traces after stage: `29`

Artifacts:

- [accepted_llama8b_v3_cap200_seed.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_llama8b_v3_cap200_seed.jsonl)
- [train_v3_cap200_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v3_cap200_seed.parquet)

Note:

- this dataset was also reconstructed from a stopped run, so there is no official live-run timing summary for the capped recovery

### Stage T4: direct boxed-answer tail fill

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/train/summary_v2_from_llama8b_direct_from_v3_cap200_seed.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/summary_v2_from_llama8b_direct_from_v3_cap200_seed.json)

Settings:

- mode: deterministic boxed gold answer
- acceptance: normal verifier only

Result:

- direct-answer traces added: `29`
- cumulative traces after stage: `64,000`
- complete questions with 2 traces: `32,000`
- zero remaining questions
- timing: `1.107s`

Artifacts:

- [accepted_v2_from_llama8b_direct_from_v3_cap200_seed.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/accepted_v2_from_llama8b_direct_from_v3_cap200_seed.jsonl)
- [train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet)

### Final train-stage composition

The final `64,000` train traces break down as:

- plain `v1`: `16,807`
- guided fill with filter on: `38,868`
- guided fill with filter off: `8,296`
- deterministic direct boxed answers: `29`

The final accepted jsonl itself stores only:

- `plain`
- `guided_fill`
- `direct_answer_gold`

So the filter-on vs filter-off split is tracked by stage lineage, not by a dedicated column in the final parquet.

## Minerva dev validation lineage

Target:

- `1,200` prompts
- `1` accepted trace per prompt

### Stage V1: plain `v1`

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v1.json)

Settings:

- mode: plain original prompt only
- target accepted per prompt: `1`
- max attempts per prompt: `32`

Result:

- generated attempts: `28,922`
- accepted plain traces: `401`
- remaining prompts after stage: `799`
- timing: `527.504s`

Artifacts:

- [accepted_v1.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v1.jsonl)
- [minerva_dev_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v1.parquet)

### Stage V2: guided fill with `ml+heuristic` filter

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_accepted_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_accepted_v1.json)

Settings:

- mode: answer-guided ACR prompt
- acceptance: verifier + `ml+heuristic`
- max attempts per prompt: `32`

Result:

- seed traces entering stage: `401`
- guided traces added at this stage: `719`
- cumulative traces after stage: `1,120`
- remaining prompts after stage: `80`
- timing: `375.597s`

Artifacts:

- [accepted_v2_from_accepted_v1.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v2_from_accepted_v1.jsonl)
- [minerva_dev_v2_from_accepted_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_accepted_v1.parquet)

### Stage V3: guided fill with filter disabled

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_minerva_dev_v3_from_v2.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/summary_v2_from_minerva_dev_v3_from_v2.json)

Settings:

- mode: answer-guided ACR prompt
- acceptance: verifier only
- live run was uncapped

Result:

- seed traces entering stage: `1,120`
- additional traces added at this stage: `80`
- cumulative traces after stage: `1,200`
- complete prompts after stage: `1,200 / 1,200`
- max fill attempt index observed: `303`
- timing: `554.686s`

Artifacts:

- [accepted_v2_from_minerva_dev_v3_from_v2.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/accepted_v2_from_minerva_dev_v3_from_v2.jsonl)
- [minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet)

Note:

- the summary file reports `guided_fill` totals after the stage, not a clean stage-only delta
- the actual stage delta is `1,200 - 1,120 = 80`

## Athena validation lineage

Target:

- `950` prompts
- `1` accepted trace per prompt

### Stage A1: plain `v1`

Source summary:

- [dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v1.json](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/summary_v1.json)

Settings:

- mode: plain original prompt only
- target accepted per prompt: `1`
- max attempts per prompt: `32`

Result:

- generated attempts: `13,783`
- accepted plain traces: `595`
- remaining prompts after stage: `355`
- timing: `480.774s`

Artifacts:

- [accepted_v1.jsonl](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/accepted_v1.jsonl)
- [athena_bench_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet)

Status:

- Athena synthetic validation is still partial
- `v2` and `v3` Athena stages have not been built yet
- the SFT launcher should not use Athena val until the full synthetic Athena set exists

## Recommended parquets for SFT

Train:

- [train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet)

Minerva synthetic val:

- [minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet)

Athena synthetic val:

- not ready yet; current partial artifact is [athena_bench_v1.parquet](/home/jovyan/work/minerva/dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet)

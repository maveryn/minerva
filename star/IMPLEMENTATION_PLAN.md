# STaR-CTI Implementation Plan

## Objective

Implement an offline STaR baseline that:

- reads the existing Minerva CTI training parquet
- generates one original reasoning trace per example
- generates one gold-conditioned rationalization trace per example
- scores them with the existing verifier
- keeps the original trace if correct, otherwise keeps the rationalization trace
  only if the original failed and the rationalization is correct
- builds or updates an SFT dataset from the kept traces
- fine-tunes with the existing SFT trainer
- repeats for multiple rounds

The implementation should stay outside the current PPO trainer. STaR is a separate
baseline, not another branch of ACRD.

## Design choice

Use an offline pipeline, not an in-trainer loop.

Why:

- the repo already has a clean SFT path in `rlvr/cti-scripts/train_minerva_sft.sh`
- the verifier is already reusable from `reward_minerva.py`
- STaR does not require PPO-specific state
- an offline loop is easier to audit, resume, and compute-match against Noctua

## Existing components to reuse

- Train/dev parquet:
  - `rlvr/mydata/minerva_base/minerva_base_train.parquet`
  - `rlvr/mydata/minerva_base/minerva_base_dev.parquet`
- Reward parser/verifier:
  - `rlvr/verl/utils/reward_score/reward_minerva.py`
  - `minerva/reward.py`
- Task metadata:
  - `minerva/cti_task_specs.py`
- SFT trainer:
  - `verl.trainer.fsdp_sft_trainer`
- Existing SFT launcher pattern:
  - `rlvr/cti-scripts/train_minerva_sft.sh`

## Proposed implementation layout

### 1. Config

Add a STaR config module or YAML with:

- base model path
- train parquet path
- validation parquet paths
- `k_original=1`
- `k_rationalization=1`
- number of STaR rounds
- batch sizes for generation and scoring
- max prompt / response lengths
- sampling params for original and rationalization generation
- output root
- whether to require boxed final answer

Suggested location:

- `star/configs/star_cti.yaml`

### 2. Prompt builders

Create prompt builders for:

- original prompt generation
- rationalization prompt generation

Original generation should preserve the current CTI prompt and only add a light
format instruction such as:

"Reason step by step, then give the final answer in \\boxed{}."

Rationalization generation should append:

- the correct answer `y*`
- an instruction to produce a reasoning trace that ends in that answer
- no "reference text" or ACR label-details block

Suggested file:

- `star/prompting.py`

Functions:

- `build_star_original_prompt(messages_or_prompt, data_source) -> prompt`
- `build_star_rationalization_prompt(messages_or_prompt, gold_answer, data_source) -> prompt`

### 3. Generation runner

Implement a generation script that:

- loads parquet rows
- runs batched generation from a checkpoint/model
- samples exactly one original completion per example
- stores raw text outputs and metadata

Suggested file:

- `star/generate.py`

Output format per completion:

- uid
- round_id
- source
- mode: `original` or `rationalization`
- prompt_nohint
- prompt_used
- gold_answer
- response_text
- response_index
- generation params

Suggested artifact:

- `star-artifacts/<exp>/round_<n>/original_samples.jsonl`

### 4. Verifier-scoring bridge

Implement an offline scorer that applies the same verifier used in RL runs.

Suggested file:

- `star/score.py`

This module should:

- map each sampled response to the row's `data_source` and `reward_model.ground_truth`
- call `reward_minerva`
- store:
  - extracted answer
  - score
  - per-task reward dict if available
  - success flag

Important rule:

- do not introduce a new heuristic correctness check
- use the same extraction + reward path as MinervaRL

### 5. Rationalization stage

For every training example:

- build a rationalization prompt using the gold answer
- generate exactly one rationalization trace
- score it with the same offline verifier

Suggested file:

- `star/rationalize.py`

### 6. Trace selection policy

Implement a minimal, auditable selection policy for successful traces.

Recommended default:

- per example, keep the original trace if it is correct
- otherwise keep the rationalization trace if:
  - the original trace is incorrect, and
  - the rationalization trace is correct
- otherwise keep nothing

Suggested file:

- `star/select.py`

Rationale:

- this matches the faithful STaR keep rule the baseline is supposed to test
- one kept trace per example keeps dataset growth controlled and easy to audit

### 7. SFT dataset builder

Build a parquet compatible with the existing SFT trainer.

Suggested file:

- `star/build_sft_dataset.py`

Expected output schema:

- `prompt`: original prompt text only
- `response`: selected successful rationale + final answer
- optional metadata:
  - `uid`
  - `round_id`
  - `mode`
  - `data_source`
  - `selected_from`

Suggested artifact:

- `star-artifacts/<exp>/round_<n>/star_sft_train.parquet`

### 8. Round trainer wrapper

Create a script that launches SFT for one STaR round, using the current SFT
trainer with STaR-generated parquet.

Suggested file:

- `star/train_round.sh`

It should mirror:

- `rlvr/cti-scripts/train_minerva_sft.sh`

but with STaR-specific train parquet paths and checkpoint naming.

### 9. Multi-round controller

Create a controller that runs:

1. original generation
2. scoring
3. rationalization generation
4. rationalization scoring
5. trace selection
6. SFT dataset build or update
7. one SFT training run or continued training
8. save checkpoint
9. evaluate on the combined validation set
10. advance to next round using the newly trained checkpoint

Suggested file:

- `star/run_star_cti.py`

This is the main entrypoint for the baseline.

### 10. Evaluation wrapper

Add an evaluation script that scores each round checkpoint on the same validation
suite used by MinervaRL.

Suggested file:

- `star/eval.py`

Use the same reward-eval setup already used by:

- `rlvr/cti-scripts/train_minerva_sft.sh`
- `rlvr/cti-scripts/train_minerva_grpo.sh`
- `rlvr/cti-scripts/train_minerva_noctua.sh`

## Minimal file map

Recommended first-pass file set:

- `star/README.md`
- `star/IMPLEMENTATION_PLAN.md`
- `star/EXPERIMENT_PLAN.md`
- `star/prompting.py`
- `star/generate.py`
- `star/score.py`
- `star/rationalize.py`
- `star/select.py`
- `star/build_sft_dataset.py`
- `star/run_star_cti.py`
- `star/train_round.sh`
- `star/configs/star_cti.yaml`

## Recommended implementation order

1. Build offline scoring first
2. Build original generation path
3. Build rationalization prompting and rationalization generation
4. Build SFT parquet writer
5. Build one-round training wrapper
6. Build multi-round controller
7. Add checkpoint selection and round summaries

This order de-risks the verifier integration before training infrastructure.

## Data contracts

### Input

Source examples come from Minerva parquet rows with fields like:

- `prompt`
- `data_source`
- `reward_model.ground_truth`
- `extra_info`

### Intermediate sampled-trace JSONL

Each sampled trace should contain:

- `uid`
- `round_id`
- `mode`
- `data_source`
- `prompt_nohint`
- `prompt_used`
- `ground_truth`
- `response_text`
- `verifier_score`
- `verifier_success`
- `reward_info`

### Output SFT parquet

Should be directly consumable by `verl.trainer.fsdp_sft_trainer`.

## Key fairness decisions

### Same verifier

All success/failure decisions must be based on the existing Minerva verifier only.

### Same training data

Use the same Minerva base train split, not a custom subset.

### Same eval data

Use Minerva dev and the existing Athena CTI reward-eval suite.

## Checkpointing and model selection

STaR should be run iteratively across rounds.

After each round:

- save the trained checkpoint
- evaluate on the combined validation set
- record the validation score in a round summary

For reporting:

- report the best checkpoint based on validation
- optionally also report which round produced the best checkpoint

## Risks

- Generation formatting may drift from the verifier's extraction logic.
- Rationalization prompts may become too leading and produce low-quality reasoning.
- Multi-label tasks may need stricter formatting guidance in the final answer.
- Offline generation throughput may become the bottleneck.

## First implementation milestone

Milestone 1 should prove the end-to-end path on one round:

1. sample original traces for a small shard of train data
2. score with `reward_minerva`
3. generate rationalization traces
4. build one SFT parquet
5. train one small SFT run
6. evaluate on Minerva dev

Do not start with full multi-round orchestration before this works.

### Same eval data

Use Minerva dev and the existing Athena CTI reward-eval suite.

## Open decisions

1. Whether STaR should always demand boxed answers.
2. Whether to keep all successful traces or cap to one per prompt.
3. Whether repaired traces should be mixed equally with original successful traces.
4. Whether to run a fixed number of STaR rounds or early-stop on dev reward.
5. Whether to exclude CVSS from repair prompts if rationale quality is poor there.

## Recommended default answers

1. Yes, require boxed answers for easier extraction consistency.
2. Cap to one selected trace per example.
3. Prefer original-success traces; fall back to repaired-success traces.
4. Run a fixed small number of rounds, e.g. 2 to 4.
5. Keep CVSS in the baseline unless it breaks extraction materially.

## Risks

- Generation formatting may drift from the verifier's extraction logic.
- Repaired prompts may become too leading and produce low-quality reasoning.
- Keeping all successful traces could overweight easy tasks.
- Multi-label tasks may need stricter formatting guidance in the final answer.
- Offline generation throughput may become the bottleneck.

## First implementation milestone

Milestone 1 should prove the end-to-end path on one round:

1. sample original traces for a small shard of train data
2. score with `reward_minerva`
3. repair failures
4. build one SFT parquet
5. train one small SFT run
6. evaluate on Minerva dev

Do not start with full multi-round orchestration before this works.

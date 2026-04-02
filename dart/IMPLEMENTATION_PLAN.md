# DART-CTI Implementation Plan

## Objective

Implement a DART-style offline rejection-tuning baseline for Minerva-CTI that:

- uses a stronger teacher model to generate candidate reasoning traces
- scores each trace with the existing Minerva verifier
- retains only verifier-correct traces
- constructs a fixed Uniform dataset with `k_u = 2`
- supports two dataset variants:
  - `v1`: plain original-prompt rejection tuning with a safety cap
  - `v2`: `v1` plus answer-guided fill for missing traces
- reuses that same fixed dataset to fine-tune multiple student models
- selects student checkpoints by synthetic validation loss
- evaluates final student checkpoints on the same RL-matched validation target used elsewhere in this repo

The implementation should stay separate from PPO / GRPO / Noctua. This is a standalone baseline.

## High-level method

The CTI adaptation should follow a target-correct-count workflow for Uniform:

1. Choose a teacher model.
2. Generate candidate reasoning traces from the teacher on the Minerva train set.
3. Score each candidate with the existing Minerva reward / verifier.
4. Keep only verifier-correct traces.
5. For each question, keep generating until it reaches `k_u = 2` accepted traces or hits the safety cap.
6. Build `v1` from these accepted traces.
7. Optionally build `v2` by filling any missing accepted traces with answer-guided generation.
8. Reuse the same fixed dataset variant to fine-tune each target student model.
9. Select checkpoints by synthetic validation loss.
10. Evaluate each student model on the RL-matched validation set.

## Important adaptation rule

Do not use answer-conditioned rescue prompts in the main DART baseline.

Reason:

- answer-conditioned rescue changes the method materially
- that would make the baseline closer to STaR-style repair than pure rejection tuning
- the primary DART-CTI baseline should stay faithful to rejection-tuning behavior

If answer-conditioned rescue is ever added, it should be treated as a separate variant, not the main DART baseline.

This is now the intended interpretation for CTI:

- `v1` is the main DART-style baseline
- `v2` is a separate follow-up variant that fills missing traces using answer guidance

## Proposed teacher model

Use the same GPT-OSS model used in `minerva-judge/`:

- `openai/gpt-oss-120b`

Reason:

- already used in this repo as a stronger generator / judge-family model
- matches prior repo practice better than introducing a new external teacher

## Proposed student models

Use the same four target models already used in Minerva baselines:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-4B-Base`
- `Qwen/Qwen3-8B-Base`

Important rule:

- the teacher-generated DART dataset should be built once
- that same fixed dataset should then be reused across all four student models
- do not regenerate a separate teacher dataset per student model

## Data source

Use the existing Minerva training parquet:

- `rlvr/mydata/minerva_base/minerva_base_train.parquet`

Final evaluation should use the same RL-matched target used by Minerva RL:

- `rlvr/mydata/minerva_base/minerva_base_dev.parquet`
- `rlvr/mydata/athena/athena_cti_ate.parquet`
- `rlvr/mydata/athena/athena_cti_ckt.parquet`
- `rlvr/mydata/athena/athena_cti_rcm.parquet`
- `rlvr/mydata/athena/athena_cti_rms.parquet`
- `rlvr/mydata/athena/athena_cti_taa.parquet`
- `rlvr/mydata/athena/athena_cti_vsp.parquet`

Checkpoint selection during SFT should not use online reward evaluation. Instead, build synthetic validation trace sets from:

- Minerva dev
- Athena bench

and select the best checkpoint by the mean validation loss across those two held-out synthetic datasets, evaluated every 10 training steps.

## Core implementation decisions

### 1. Generation policy

Generate traces from the original prompt only.

Each trace should include:

- reasoning
- final answer in the standard format expected by the verifier

Generation should reuse the same prompt style and answer format conventions already used in Minerva STaR / SFT generation.

### 2. Verification policy

Use the same extraction and reward path already used in Minerva RL / STaR:

- `rlvr/verl/utils/reward_score/reward_minerva.py`

Do not introduce a new correctness heuristic.

### 3. Uniform target and safety cap

Uniform mode should be target-correct-count driven:

- target accepted traces per question: `k_u = 2`
- generation uses the original prompt only
- accepted traces are verifier-correct outputs only

Use a bounded active-queue loop as an engineering guard:

- maintain an active queue of questions that have not yet reached 2 accepted traces
- generate batched traces only for active questions
- remove a question once:
  - it reaches 2 accepted traces, or
  - it reaches the safety cap

Chosen safety cap:

- `B_max = 100`

The cap is not the main definition of the algorithm. It is a practical stop condition and must be logged and reported.

### 4. Dataset variants

The implementation should produce two explicit dataset versions.

`v1`:

- plain original-prompt rejection tuning only
- stop each question at `accepted == 2` or `attempts == 100`
- keep whatever accepted traces were found
- resulting dataset may have fewer than 2 traces for some questions

`v2`:

- start from `v1`
- for questions with fewer than 2 accepted traces, fill only the missing slots
- use answer-guided generation with the exact same ACR prompt block already used in Minerva
- keep generating guided traces until each question has exactly 2 accepted traces
- do not replace plain accepted traces with guided ones

Important rule:

- `v2` is a separate variant, not the main DART-style baseline
- each accepted trace should record whether it came from the plain stage or the guided fill stage

### 5. Validation policy for SFT

Checkpoint selection should use validation loss, not online reward evaluation.

Implementation plan:

- synthesize a held-out DART validation set from Minerva dev
- synthesize a held-out DART validation set from Athena bench
- evaluate every 10 training steps
- compute validation loss on both synthetic validation datasets
- select the checkpoint using the mean of the two validation losses

This keeps SFT checkpointing lightweight while preserving a Minerva-dev / Athena-bench split that is comparable to RLVR at a high level.

## Step-by-step implementation plan

### Step 1. Add a `dart/` config

Create a YAML config that specifies:

- teacher model path
- train parquet path
- synthetic validation source parquet paths:
  - Minerva dev
  - Athena bench
- dataset variant:
  - `v1_plain_cap100`
  - `v2_plain_cap100_plus_guided_fill`
- target accepted traces:
  - `k_u = 2`
- max attempts:
  - `B_max = 100`
- generation batch size
- max response length
- prompt truncation settings
- output root
- dataset-build settings
- validation-build settings
- student model list
- per-model SFT settings
- W&B settings

Suggested location:

- `dart/configs/dart_cti.yaml`

### Step 2. Add teacher-generation prompt handling

Define prompt construction for DART generation.

Requirements:

- use the original CTI prompt
- request reasoning plus a final answer in the verifier-compatible format
- do not use answer-conditioned prompting
- separately expose the existing ACR answer-guided block for `v2` fill generation only

Suggested file:

- `dart/prompting.py`

### Step 3. Add teacher-generation runner

Create a generation runner that:

- loads Minerva train parquet rows
- accepts a list of active questions
- generates candidate traces from the teacher model
- stores raw outputs and metadata

Suggested file:

- `dart/generate.py`

Artifacts should record:

- uid
- source row metadata
- attempt index
- question id
- prompt text / messages
- response text
- generation config

For the fill stage, the same runner should also support answer-guided generation using the existing Minerva ACR prompt block.

### Step 4. Add verifier scoring

Create an offline scorer that:

- applies `reward_minerva`
- records:
  - extracted answer
  - verifier score
  - verifier success

Suggested file:

- `dart/score.py`

### Step 5. Add bounded Uniform collector

Implement the queue logic for accepted-trace collection for `v1`.

Uniform mode:

- initialize all questions with:
  - `accepted = 0`
  - `attempts = 0`
- keep generating for active questions
- update accepted / attempts after each batch
- remove a question once:
  - `accepted >= 2`, or
  - `attempts >= B_max`

Suggested file:

- `dart/collect.py`

This should be the core of the `v1` pipeline.

### Step 6. Add guided fill collector for `v2`

Implement a second pass that fills only missing accepted traces.

Rules:

- input is the `v1` accepted-trace state
- identify questions with `accepted < 2`
- generate only the missing number of traces
- use the exact same answer-guided ACR prompt block already used in Minerva
- keep generating guided traces until each question reaches exactly 2 accepted traces
- record fill-stage metadata separately from plain-stage metadata

Suggested file:

- `dart/fill.py`

### Step 7. Add accepted-trace dataset builder

Build the accepted traces into fixed SFT parquets that will be reused for all target student models.

Suggested file:

- `dart/build_sft_dataset.py`

Expected contents:

- original prompt
- accepted reasoning trace
- final answer
- metadata:
  - uid
  - source question
  - dataset version
  - trace source:
    - `plain`
    - `guided_fill`
  - attempt count
  - teacher model

Expected outputs:

- `dart_train_v1.parquet`
- `dart_train_v2.parquet`
- `dart_val_minerva_v1.parquet`
- `dart_val_athena_v1.parquet`
- `dart_val_minerva_v2.parquet`
- `dart_val_athena_v2.parquet`

### Step 8. Add SFT training wrapper

Reuse the existing Minerva SFT path for student training on the fixed DART dataset.

Suggested file:

- `dart/train_sft.sh`

Requirements:

- standard full fine-tuning
- validate every 10 training steps
- use synthetic DART validation trace datasets, not online reward evaluation
- select the best checkpoint by the mean validation loss across:
  - Minerva dev synthetic validation set
  - Athena bench synthetic validation set
- W&B enabled
- save final student checkpoint

The training wrapper should support running the same fixed DART dataset against:

- Llama 3B
- Llama 8B
- Qwen 4B
- Qwen 8B

### Step 9. Add evaluation wrapper

Add an evaluation script for the student checkpoint.

Suggested file:

- `dart/eval.py`

It should:

- use the same reward parser / extractor
- evaluate on the RL-matched validation target
- compute:
  - Minerva dev mean
  - Athena bench mean
  - RL global validation mean

### Step 10. Add top-level controllers

Create:

- one controller for teacher-side dataset construction
- one driver for student-model training on the fixed DART dataset

Suggested file:

- `dart/run_dart_cti.py`

The teacher-side controller should:

1. Load config.
2. Load train parquet.
3. Run teacher candidate generation for plain original-prompt rejection tuning.
4. Score all candidates.
5. Collect accepted traces for `v1` with `k_u = 2` and `B_max = 100`.
6. Build `v1`.
7. Optionally run guided fill to build `v2`.
8. Build synthetic validation trace datasets for Minerva dev and Athena bench.
9. Save dataset-level summaries and coverage statistics.

Then the student-side driver should:

1. Load a fixed DART dataset variant.
2. Train each target student model on that same dataset.
3. Select the checkpoint by mean validation loss on the two synthetic validation datasets.
4. Evaluate the selected checkpoint on the RL-matched final evaluation target.
5. Save per-model summaries.

## Metrics to log

At minimum log:

- total teacher samples generated
- accepted traces total
- accepted traces per question
- coverage after the plain stage
- coverage after the guided fill stage
- fraction of questions with:
  - 0 accepted traces
  - 1 accepted trace
  - 2 accepted traces
- average attempts per question
- attempts distribution
- plain-stage accepted traces
- guided-fill accepted traces
- mean validation loss on synthetic Minerva dev
- mean validation loss on synthetic Athena bench
- mean of the two synthetic validation losses
- Minerva dev score
- Athena bench score
- RL global validation score

## Recommended first baseline order

Implement in this order:

1. Uniform `v1` DART-CTI
2. Fixed `v1` dataset build and dataset-level reporting
3. Uniform `v2` guided-fill dataset build
4. Student training for the four Minerva baseline models
5. Full evaluation and logging

Reason:

- Uniform is simpler and cleaner
- `v1` provides the plain DART-style baseline
- `v2` provides a separate completed-coverage variant
- both provide direct comparisons against STaR and plain SFT

## Open decisions to confirm before implementation

The following should be confirmed before coding:

1. Teacher model:
   - use `openai/gpt-oss-120b` as the default?
2. Student model list:
   - confirm the exact four target models:
     - Llama 3B
     - Llama 8B
     - Qwen 4B
     - Qwen 8B
3. Validation trace construction:
   - use one held-out synthetic Minerva dev set and one held-out synthetic Athena bench set?
4. Total target dataset size:
   - keep approximately `64k` accepted traces for the first Uniform run?

## Non-goals for the first version

Do not add these in the first implementation:

- Prop2Diff
- DPO or PPO updates
- mixed-method hybrid baselines
- retrieval augmentation

For the first version:

- `v1` should be the main plain DART-style baseline
- `v2` may use answer-guided fill, but only as a separate variant built after `v1`

First make the core DART-CTI baseline correct and auditable.

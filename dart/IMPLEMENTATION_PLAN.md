# DART-CTI Implementation Plan

## Objective

Implement a DART-style offline rejection-tuning baseline for Minerva-CTI that:

- uses a stronger teacher model to generate candidate reasoning traces
- scores each trace with the existing Minerva verifier
- retains only verifier-correct traces
- constructs one fixed SFT dataset using either:
  - Uniform allocation
  - Prop2Diff allocation
- reuses that same fixed dataset to fine-tune multiple student models
- evaluates student checkpoints on the same RL-matched validation target used elsewhere in this repo

The implementation should stay separate from PPO / GRPO / Noctua. This is a standalone baseline.

## High-level method

The CTI adaptation should follow a bounded rejection-tuning workflow:

1. Choose a teacher model.
2. Generate candidate reasoning traces from the teacher on the Minerva train set.
3. Score each candidate with the existing Minerva reward / verifier.
4. Keep only verifier-correct traces.
5. Allocate accepted traces per question using either:
   - Uniform
   - Prop2Diff
6. Build one fixed SFT dataset from the accepted traces.
7. Reuse that same dataset to fine-tune each target student model.
8. Evaluate each student model on the RL-matched validation set.

## Important adaptation rule

Do not use answer-conditioned rescue prompts in the main DART baseline.

Reason:

- answer-conditioned rescue changes the method materially
- that would make the baseline closer to STaR-style repair than pure rejection tuning
- the primary DART-CTI baseline should stay faithful to rejection-tuning behavior

If answer-conditioned rescue is ever added, it should be treated as a separate variant, not the main DART baseline.

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
- Qwen 4B baseline model used in Minerva RL runs
- Qwen 8B baseline model used in Minerva RL runs

Important rule:

- the teacher-generated DART dataset should be built once
- that same fixed dataset should then be reused across all four student models
- do not regenerate a separate teacher dataset per student model

## Data source

Use the existing Minerva training parquet:

- `rlvr/mydata/minerva_base/minerva_base_train.parquet`

Validation should use the same RL-matched checkpoint-selection target:

- `rlvr/mydata/minerva_base/minerva_base_dev.parquet`
- `rlvr/mydata/athena/athena_cti_ate.parquet`
- `rlvr/mydata/athena/athena_cti_ckt.parquet`
- `rlvr/mydata/athena/athena_cti_rcm.parquet`
- `rlvr/mydata/athena/athena_cti_rms.parquet`
- `rlvr/mydata/athena/athena_cti_taa.parquet`
- `rlvr/mydata/athena/athena_cti_vsp.parquet`

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

### 3. Bounded generation budget

Do not generate forever until success.

Use a bounded queue-based rejection-tuning loop:

- maintain an active queue of questions that have not yet reached their accepted-trace target
- generate batched traces only for active questions
- remove a question once:
  - it has enough accepted traces, or
  - it reaches its max-attempt cap

Recommended initial cap:

- `B_max = 50`

Optional higher-cap fallback:

- `B_max = 100`

The cap must be logged and reported.

### 4. Uniform mode

Uniform mode should use:

- target accepted traces per question: `k = 2`
- bounded queue-based sampling
- accepted traces only from verifier-correct outputs

For each question:

- stop when accepted count reaches 2
- or when attempts reach `B_max`

Questions that fail to reach 2 accepted traces should still contribute any accepted traces they do have.

### 5. Prop2Diff mode

Prop2Diff mode should allocate accepted traces proportionally to estimated difficulty.

Implementation concept:

1. Estimate question difficulty from teacher success rate under a small probe budget.
2. Convert difficulty estimates into target accepted counts and/or attempt budgets.
3. Run the same bounded queue-based rejection-tuning loop with those per-question targets.

Important fairness rule:

- keep the total generation budget comparable to Uniform
- report the total generated samples, accepted samples, and per-question coverage

## Step-by-step implementation plan

### Step 1. Add a `dart/` config

Create a YAML config that specifies:

- teacher model path
- train parquet path
- validation parquet paths
- mode:
  - `uniform`
  - `prop2diff`
- target accepted traces
- max attempts
- generation batch size
- max response length
- prompt truncation settings
- output root
- dataset-build settings
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

### Step 4. Add verifier scoring

Create an offline scorer that:

- applies `reward_minerva`
- records:
  - extracted answer
  - verifier score
  - verifier success

Suggested file:

- `dart/score.py`

### Step 5. Add bounded queue-based sampler

Implement the queue logic for accepted-trace collection.

Uniform mode:

- initialize all questions with:
  - `accepted = 0`
  - `attempts = 0`
- keep generating for active questions
- update accepted / attempts after each batch
- remove a question once:
  - `accepted >= 2`, or
  - `attempts >= B_max`

Prop2Diff mode:

- use the same loop
- but with per-question targets and/or budgets derived from difficulty

Suggested file:

- `dart/collect.py`

This should be the core of the DART pipeline.

### Step 6. Add difficulty estimation for Prop2Diff

Implement a difficulty-estimation pass.

Possible approach:

- run a small probe generation budget per question
- compute estimated teacher success probability
- derive difficulty as:
  - lower success probability = higher difficulty

Then map difficulty to:

- target accepted count `k_i`, or
- max attempts `B_i`, or
- both

Suggested file:

- `dart/difficulty.py`

### Step 7. Add accepted-trace dataset builder

Build the accepted traces into one fixed SFT parquet that will be reused for all target student models.

Suggested file:

- `dart/build_sft_dataset.py`

Expected contents:

- original prompt
- accepted reasoning trace
- final answer
- metadata:
  - uid
  - source question
  - mode
  - attempt count
  - teacher model

### Step 8. Add SFT training wrapper

Reuse the existing Minerva SFT path for student training on the fixed DART dataset.

Suggested file:

- `dart/train_sft.sh`

Requirements:

- standard full fine-tuning
- same validation target as Minerva RL checkpoint selection
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
3. Run teacher candidate generation with bounded rejection tuning.
4. Score all candidates.
5. Collect accepted traces according to:
   - Uniform, or
   - Prop2Diff
6. Build one fixed DART SFT parquet.
7. Save dataset-level summaries and coverage statistics.

Then the student-side driver should:

1. Load the fixed DART dataset.
2. Train each target student model on that same dataset.
3. Evaluate each trained model.
4. Save per-model summaries.

## Metrics to log

At minimum log:

- total teacher samples generated
- accepted traces total
- accepted traces per question
- fraction of questions with:
  - 0 accepted traces
  - 1 accepted trace
  - 2 accepted traces
  - more than 2 if Prop2Diff allows it
- average attempts per question
- attempts distribution
- Minerva dev score
- Athena bench score
- RL global validation score

## Recommended first baseline order

Implement in this order:

1. Uniform DART-CTI
2. Fixed DART dataset build and dataset-level reporting
3. Student training for the four Minerva baseline models
4. Full evaluation and logging
5. Prop2Diff DART-CTI

Reason:

- Uniform is simpler and cleaner
- it provides a direct first comparison against STaR and plain SFT
- Prop2Diff can then be added as the stronger DART variant

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
3. Uniform cap:
   - use `B_max = 50` or `B_max = 100`?
4. Prop2Diff allocation:
   - scale accepted-trace targets, attempt budgets, or both?
5. Total target dataset size:
   - keep approximately `64k` accepted traces for the first Uniform run?

## Non-goals for the first version

Do not add these in the first implementation:

- answer-conditioned rescue prompting
- DPO or PPO updates
- mixed-method hybrid baselines
- retrieval augmentation

First make the core DART-CTI rejection-tuning baseline correct and auditable.

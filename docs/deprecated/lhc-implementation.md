# LHC Implementation (Current)

This document describes the current Label-Hint Curriculum (LHC) implementation
used in RLVR for Minerva, including multi-label hint support and the trainer
integration that adapts hint difficulty over time.

## Overview

The LHC implementation injects candidate ID hints into prompts during training.
Hints are adaptive per task (keyed by `data_source`) and are adjusted to keep
rollout accuracy near a target value. Candidate pools are provided as
`candidate_pool_top100` in the dataset.

Key components:
- Dataset wrapper: `AdaptiveOptionRLHFDataset`
- Controller: `OptionCurriculumController`
- Trainer hook: `RayPPOTrainer.fit` updates the controller after rewards

## Controller: `OptionCurriculumController`

Per task key, the controller tracks:
- `k`: the configured *option count* target (used to compute distractors)
- `p_drop`: probability of omitting hints
- `ema_acc`: EMA of per-task rollout accuracy

Update behavior:
- Warmup can be configured (`warmup_steps`).
- `k` increases or decreases based on `ema_acc` vs `target_acc` with tolerance.
- When `k` is at `k_max` and the task is still easy, `p_drop` increases.

Logged metrics (prefix `option_curriculum/`):
- `K/<task>` (current `k`)
- `p_drop/<task>`
- `ema_acc/<task>`

## Dataset: `AdaptiveOptionRLHFDataset`

Hints are added in `__getitem__` by appending a candidate list to the last user
message. The list is built from `candidate_pool_top100` and includes all gold
IDs for the example.

Multi-label support:
- Ground truth is parsed into a **list** of gold IDs (e.g., mitigations, tactics,
  CWE lists).
- The candidate list header uses **EXACTLY N** where `N = len(gold_ids)`.
- Total options are computed as:
  - `total_options = len(gold_ids) + max(0, k - 1)`
  - Example: `k=5`, `len(gold_ids)=2` → 6 options.

If the prompt exceeds `max_prompt_length`, the number of **distractors** is
reduced until the prompt fits, or hints are skipped.

Hint metadata stored in `extra_info`:
- `hint_used` (bool)
- `hint_K` (total options injected)
- `hint_pool_size` (pool size after normalization)
- `hint_seed`

## Prompt format

The injected list uses:

```
Candidate ID(s) (choose EXACTLY N):
1) <ID>
2) <ID>
...
```

The tail note differs for single vs multi-label:
- Single: “The correct ID is one of the candidates above.”
- Multi: “The correct IDs are among the candidates above.”

## Trainer integration

In `RayPPOTrainer.fit`, after reward computation:
- `is_correct` is derived per rollout either from `reward_extra_info["is_correct"]`
  or from `reward_tensor >= score_threshold`.
- Rollouts are grouped by `uid` (one prompt group per `rollout.n`).
- The dataset is updated via `update_option_curriculum_from_rollout` using
  per-uid accuracy and `data_source`.

## Split metadata examples

`minerva.split` generates `candidate_examples` in `metadata.json` using the same
rules as the adaptive dataset:
- “choose EXACTLY N” for multi-label tasks.
- Total options = `k + (N - 1)` with all gold IDs included first if missing.

Regenerating `dataset/minerva_lhc_split/metadata.json` does **not** change the
Parquet dataset; it only updates the split outputs and metadata examples.

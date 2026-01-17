# ACRD-SFT (Answer-Conditioned Reasoning + Distillation, SFT Only)

This document describes the SFT-only ACRD training path as currently
implemented in the Minerva RLVR trainer. It focuses on the per-batch ACR
(Answer-Conditioned Reasoning) generation + SFT distillation loop and
records the exact hyperparameters used by
`rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.1.sh`.

## Algorithm (SFT only)

Let each training sample be a pair (x, y*) where x is the Minerva prompt and
 y* is the ground-truth answer. Training iterates over RLVR batches and adds a
per-batch ACR + distill phase.

### Step 1 - RLVR rollouts (baseline)
1. Sample a batch of prompts from the Minerva train parquet.
2. Generate n RLVR rollouts per prompt using the actor.
3. Score each rollout with the rule-based verifier (`reward_minerva`).

### Step 2 - ACR prompt construction
For each prompt in the batch:
1. Append an ACR block to the last user message using
   `minerva/acr_prompt.py`:
   - `GROUND_TRUTH_LABELS` (one per line)
   - Optional `LABEL_REFERENCE` (from `dataset/label_details/`)
   - Instructions to produce reasoning + final answer
   - Task-specific reasoning instruction (from `data.acr.task_reasoning_hints`,
     fallback to `data.acr.entity_reasoning_hints`)
2. If the ACR prompt is too long, progressively truncate or omit details; if it
   still exceeds the max length, skip ACR for that sample.

### Step 3 - Hard-example gating (SFT only)
ACR rollouts are generated only for "hard" prompts.
With `hard_reward_mode = no_perfect`, a prompt is hard iff none of its
RLVR rollouts achieve reward 1.0.

### Step 4 - ACR rollouts + ACR reward
1. For each hard prompt, generate n_acr ACR rollouts using the actor.
2. Score each ACR rollout with `reward_acr`:
   - `acr_base_score = reward_minerva(...)` in [0, 1]
   - `score = r_correct * acr_base_score - leak_penalty`
   - `acr_leak_hit`, `acr_banned_phrase_hit`, `acr_verbatim_hit`, `acr_short_hit`, and `acr_overlap_hit` are logged
     (`acr_id_leak_hit` is logged only when ID-in-reasoning checks are enabled)

### Step 5 - SFT candidate selection
For each UID, select one ACR rollout to distill:
1. Eligibility filter (ACR batch):
   - `acr_base_score >= reward_threshold`
   - `acr_leak_hit == false` (includes short reasoning <100 chars and low overlap Jaccard <0.05)
   - Optional degenerate filter based on repeated 3- / 4-grams
   Parse success is implicit in `acr_base_score` (parse failures score 0); `acr_extracted` is logged but not used.
2. Selection mode (current run): `random` among eligible rollouts.
   (Entropy tie-break settings are present but not used under random mode.)
3. Store the selected rollout as an SFT record with:
   - `prompt_nohint` = original Minerva prompt (no ACR hints)
   - `response_ids` = selected ACR rollout tokens

### Step 6 - Periodic SFT distillation
Every `distill.interval` steps (rolling/flush), or when the global buffer reaches
`distill.min_buffer` (buffer mode):
1. Build an SFT minibatch from the distill buffer.
2. Run a single-epoch SFT update on the actor using `prompt_nohint` -> response.
3. Apply a distill LR scale (`lr_scale`), i.e., the SFT update uses
   `actor_lr * lr_scale`.
4. Buffer handling:
   - `rolling`: keep the buffer as a rolling queue capped by `distill.max_buffer`.
   - `flush`: clear the buffer after each SFT run.
   - `buffer`: run only when the buffer reaches `distill.min_buffer`, then clear.

## Parameters used (Llama-8B, LR scale = 0.1)

The following values are the effective defaults when running
`rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.1.sh`.
The `lr_scale=0.1` is a tunable hyperparameter (explicitly set by the script).

### Model + data
- `actor_rollout_ref.model.path`: `meta-llama/Llama-3.1-8B-Instruct`
- Train data: `rlvr/mydata/minerva_base/minerva_base_train.parquet`
- Val data: Minerva dev + Athena CTI parquets
  (`athena_cti_ate`, `athena_cti_ckt`, `athena_cti_rcm`, `athena_cti_rms`, `athena_cti_vsp`)

### RLVR rollout
- `algorithm.adv_estimator`: `grpo`
- `actor_rollout_ref.rollout.name`: `vllm`
- `actor_rollout_ref.rollout.n`: `8` rollouts per prompt
- `actor_rollout_ref.rollout.gpu_memory_utilization`: `0.95`

### Prompt/response lengths
- `data.max_prompt_length` (RLVR): `2048`
- `data.acr.max_prompt_length` (ACR): `4096`
- `data.max_response_length`: `1024`
- `data.filter_overlong_prompts`: `true`
- `data.truncation`: `error`

### ACR prompt + gating
- `data.acr.per_batch`: `true`
- `data.acr.label_details_dir`: `dataset/label_details`
- `data.acr.max_details_chars`: `8096`
- `data.acr.rollout_n`: `4`
- `data.acr.update_actor`: `false`
- `data.acr.enforce_no_id_in_reasoning`: `false`
- `data.acr.hard_reward_mode`: `no_perfect`
- `data.acr.hard_reward_threshold`: `0.05` (unused under `no_perfect`)
- `data.acr.skip_cvss`: `true` (skip ACR generation + distillation for CVSS v3.1/v4 tasks)
- `data.acr.rl_weight`: `0.3` (only relevant if `update_actor=true`)

### ACR reward
- `data.acr.reward_manager`: `naive`
- `data.acr.reward_kwargs.r_correct`: `0.1`
- `data.acr.reward_kwargs.leak_penalty`: `0.5`
- `data.acr.reward_kwargs.multilabel_match`: `exact`
- `data.acr.reward_kwargs.max_id_mentions`: `0`
- `data.acr.reward_kwargs.verbatim_min_details_chars`: `100`
- `data.acr.reward_kwargs.verbatim_ngram_size`: `10`
- `data.acr.reward_kwargs.verbatim_min_matches`: `2`
- `data.acr.reward_kwargs.min_reasoning_chars`: `100`
- `data.acr.reward_kwargs.min_overlap_jaccard`: `0.05` (Jaccard over combined task description + label reference)
- Judge rubric: disabled (`judge_enabled=false`)

### SFT distillation (ACRD-SFT)
- `data.acr.distill.enabled`: `true`
- `data.acr.distill.method`: `sft`
- `data.acr.distill.interval`: `10` steps
- `data.acr.distill.reward_threshold`: `0.99`
- `data.acr.distill.selection_mode`: `random`
- `data.acr.distill.batch_size`: `256` (ignored in `buffer` mode)
- `data.acr.distill.max_buffer`: `1024`
- `data.acr.distill.min_buffer`: `256` (used only when `buffer_mode=buffer`)
- `data.acr.distill.buffer_mode`: `rolling`
- `data.acr.distill.lr_scale`: `0.1` (hyperparameter)
- `data.acr.distill.entropy_sampling`: `true` (not used under random mode)
- `data.acr.distill.entropy_beta`: `1.0`
- `data.acr.distill.degenerate_filter`: `true`
- `data.acr.distill.degenerate_min_tokens`: `30`
- `data.acr.distill.degenerate_rep_3_max`: `0.70`
- `data.acr.distill.degenerate_rep_4_max`: `0.75`

### Optimization + runtime
- `actor_rollout_ref.actor.optim.lr`: `1e-6`
- `actor_rollout_ref.actor.use_kl_loss`: `false`
- `actor_rollout_ref.actor.entropy_coeff`: `0.0`
- `actor_rollout_ref.model.enable_gradient_checkpointing`: `true`
- `actor_rollout_ref.actor.fsdp_config.param_offload`: `true`
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload`: `true`
- `actor_rollout_ref.ref.fsdp_config.param_offload`: `true`

### Trainer
- `data.train_batch_size`: `64`
- `trainer.total_training_steps`: `500`
- `trainer.save_freq`: `10`
- `trainer.test_freq`: `10`
- `trainer.n_gpus_per_node`: `4`
- `trainer.nnodes`: `1`
- `trainer.save_best_only`: `true`
- `trainer.save_best_metric`: `val-core/global-val/reward/mean`
- `trainer.save_best_mode`: `max`
- `trainer.save_best_dir`: `best`

## Notes
- SFT distillation uses the original Minerva prompt (no hints) as input and
  the selected ACR rollout as the target completion.
- ACR generation is generation-only; no PPO update is applied to ACR rollouts
  (`data.acr.update_actor=false`).
- The `lr_scale=0.1` scales the actor base learning rate during SFT updates.

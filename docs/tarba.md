# TARBA (Tool-Augmented Retrieval Budget Annealing) in Minerva RLVR

This document describes the current TARBA implementation and configuration used in this repo.

## Summary
- Multi-turn RL training with an optional CTI retrieval tool.
- Per-task controller anneals retrieval availability over training.
- Retrieval reward shaping uses a linear decay with a floor.
- Training entrypoint: `rlvr/cti-scripts/train_minerva_tarba_llama3b.sh`.
- Qwen3 4B entrypoint: `rlvr/cti-scripts/train_minerva_tarba_qwen4b.sh`.
- Llama 8B entrypoint: `rlvr/cti-scripts/train_minerva_tarba_llama8b.sh`.
- Qwen3 8B entrypoint: `rlvr/cti-scripts/train_minerva_tarba_qwen8b.sh`.

## High-level flow
1) TARBA wraps each CTI sample with optional tool instructions that let the model request retrieval once using a short JSON query (hidden tool schema mode).
2) The rollout executes the retrieval tool when requested, injects the results into the prompt, and continues multi-turn generation.
3) Rewards combine the task answer score with a retrieval shaping term based on the rank of gold labels in the returned docs; a small bonus is added for a valid tool call.
4) A per-task controller adjusts the retrieval budget and the probability of disabling retrieval as training progresses.

## Data and artifacts
- Label docs: `dataset/retrieval/label_docs/`
- Retrieval index: `dataset/retrieval/index/`
- Training data: `rlvr/mydata/minerva_base/minerva_base_train.parquet`
- Validation data:
  - `rlvr/mydata/minerva_base/minerva_base_dev.parquet`
  - `rlvr/mydata/athena/athena_cti_ate.parquet`
  - `rlvr/mydata/athena/athena_cti_ckt.parquet`
  - `rlvr/mydata/athena/athena_cti_rcm.parquet`
  - `rlvr/mydata/athena/athena_cti_rms.parquet`

## Retrieval tool
- Tool: `cti_retrieve` implemented by `rlvr/verl/tools/cti_retrieval_tool.py`.
- Tool config: `rlvr/cti-scripts/tool_config/cti_retrieval_tool.yaml`.
  - `topk_cap=8`, `max_query_chars=128`, `max_snippet_chars=1536`.
- Tool output includes machine-parsable IDs:
  - `DOC_IDS: ["attack_technique_id:T1059.003", ...]`

## Tool invocation (hidden schema mode)
- Training uses `TARBA_HIDE_TOOL_SCHEMA=1`.
- The model is instructed to output a JSON object with `query` and `topk` only.
- The rollout parses that JSON and injects the tool call.
- The rollout captures tool metadata (`doc_ids`, call counts) and forwards it to the reward function.

## TARBA dataset wrapper
`rlvr/verl/utils/dataset/minerva_tarba_retrieval_dataset.py`

Per sample:
- If retrieval is allowed, append tool instructions to the last user message.
- If retrieval is disabled, no tool instructions are appended.
- Add to `extra_info`:
  - `tarba_allow_retrieval`, `tarba_budget_B`, `tarba_label_type`, `tarba_gold_doc_ids`, `tarba_seed`
  - `tools_kwargs` with `{budget_B, label_type}` for the tool.
- Retrieval label scope defaults to **global**:
  - `data.tarba.train_label_type=all`
  - `data.tarba.eval_label_type=all`
  - Override with `task`/`per_type` (or a concrete label type) to force task-specific retrieval.

Prompt filtering:
- `data.max_prompt_length=1536` filters prompts **before** tool responses are added.
- Tool responses are appended during rollout; runtime length is capped by `max_model_len`.

## Retrieval budget controller (current config)
Per task key (`data_source`), TARBA maintains:
- `B`: max retrieval budget
- `p_noret`: probability retrieval is disabled
- EMA accuracy for retrieval-on/off paths

Current training config (from `train_minerva_tarba_llama3b.sh`):
- `p_noret_init=0.10`
- `p_noret_max=0.90`
- `p_step=0.05`
- `default_B_max=5`
- `ema_beta=0.90`
- `target_acc_noret=0.60`
- `tol=0.05`
- `min_steps_before_anneal=50`

Sampling rule:
- `allow_retrieval = rng.random() >= p_noret`
- `budget_B = B` if allowed, else 0

Update rule:
- Uses group accuracy from the trainer (per-uid rollouts), aggregated per task per step.
- Updates EMA for retrieval-on/off accuracy.
- Computes `acc_mix = (1 - p_noret) * ema_ret + p_noret * ema_noret`.
- If `acc_mix > target_acc_noret + tol` (and after warmup), increases `p_noret` by `p_step` up to `p_noret_max=0.9`.
- `B` remains fixed at its initial value (no budget annealing in the current policy).

## Self-distillation (optional)
TARBA can run periodic SFT on successful samples, mirroring the SLHC distillation flow.
It is disabled by default and configured under `data.tarba.distill`.

Behavior:
- Collects the **best** rollout per prompt (reward ≥ `reward_threshold`) into a buffer.
- Tie-breaks equal rewards by **highest entropy** (mean NLL).
- Works for both retrieval-enabled and retrieval-disabled samples.
- Runs an SFT pass every `interval` steps using the **original prompt without retrieval** and the
  selected response, for exactly one epoch. The buffer is cleared after each SFT pass.

Config (defaults):
- `enabled=false`
- `interval=10`
- `reward_threshold=0.5`
- `max_buffer=4096`
- `batch_size=None` (falls back to PPO mini-batch size)
- `max_seq_len=None` (falls back to `max_prompt_length + max_response_length`)
- `entropy_tiebreak=mean_nll`
- `drop_last=true`
- `dedup_by_uid=true`

Metrics:
- Selection stats: `distill/uid_group_*`, `distill/selected_reward_mean`, `distill/entropy_proxy_*`
- Buffer stats: `distill/buffer_size_local`, `distill/buffer_size_global`
- SFT stats: `distill/sft_*` and `distill_sft/actor/sft_loss`

## Reward computation (current config)
`rlvr/verl/utils/reward_score/reward_tarba.py`

Components:
- `r_ans`: task reward from `reward_minerva.reward_minerva`.
- `r_ret`: retrieval shaping reward using **linear decay** over retrieval rank, plus a tool-call bonus.

Linear decay:
- Per-label rank score for a gold label at rank `r` (1-indexed):
  - `s(r) = max(floor, 1 - slope * (r - 1))`
  - `slope = (1 - floor) / (B - 1)`
  - If the gold label is missing from retrieval, `s(r) = 0`.
- `floor=0.2`, `B` is the per-sample budget (`budget_B`).
- Multi-label aggregation:
  - `r_ret_base = (1/|G|) * sum_{g in G} s(rank_g)` where missing labels contribute 0.
- Tool bonus:
  - If exactly one valid tool call is made and retrieval is allowed, `r_ret = r_ret_base + tool_call_bonus`.
  - Otherwise, `r_ret = r_ret_base`.

Final score:
- `score = r_ans + lambda_ret * r_ret`

Current reward kwargs (train script):
- `lambda_ret=0.2`
- `floor=0.2`
- `tool_call_bonus=0.0`

## Validation summary metrics
During validation, we log aggregate reward means:
- `val-core/minerva-dev/reward/mean` for the Minerva dev split.
- `val-core/athena-bench/reward/mean` for the AthenaBench dev sets.
- `val-core/global-val/reward/mean` as the average of the two.

Optional retrieval-on evaluation for AthenaBench subsets can be added via
`trainer.extra_val_runs` in TARBA scripts. These runs log under a separate
`val-<prefix>-core/...` namespace and skip the global averages by default.

## Not used in the current configuration (but present in code)
- Tool penalties:
  - `penalty_tool_call`
  - `penalty_illegal_tool`
- Alternative aggregation policies for multi-label retrieval:
  - `min`, `all_or_nothing`
- Visible tool schema mode (`TARBA_HIDE_TOOL_SCHEMA=0`)
- Ret-on/ret-off eval scripts (available but not part of the base training run)
- Fallback parsing of `<tool_call>` or `DOC_IDS:` from model text when tool metadata is missing

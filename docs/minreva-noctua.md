# Minerva Noctua: EMA-ACR + SFT Distillation (Per-Batch, Flush Buffer)

This document specifies the exact algorithm and hyperparameters used by:
- `rlvr/cti-scripts/train_minerva_noctua_llama3b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.05_textcnn_mlh_t0.7_p0.9_defer_ema.sh`
- `rlvr/cti-scripts/train_minerva_noctua_qwen4b.sh`

These wrappers call `rlvr/cti-scripts/train_minerva_noctua.sh` and differ only in
the base model path; all other settings are identical.

## Algorithm (per training step)

1. **RLVR baseline (GRPO):** sample a batch of prompts, generate 8 rollouts per
   prompt with the actor, score with `reward_minerva`, and update the actor with GRPO.
2. **Hard-example gating:** compute max RLVR reward per UID; only UIDs with
   max reward < 1.0 are eligible for ACR. CVSS tasks are skipped for ACR.
3. **ACR prompt construction:** for eligible UIDs, append an ACR block to the
   last user message (ground-truth labels + label details). If too long, trim
   details, then omit details, else skip the sample. Prompts are buffered
   (deferred generation).
4. **Deferred ACR generation with EMA teacher:** every 10 steps, generate
   4 ACR rollouts per buffered prompt using an EMA teacher (alpha 0.995),
   sampling with temperature 0.7 and top-p 0.9. ACR rollouts are
   generation-only (no PPO update).
5. **ACR scoring and filtering:** score each ACR rollout with `reward_acr`
   (base verifier score + leakage/overlap/length checks). Apply heuristic
   filters and an ML leak classifier (TextCNN, response-only).
6. **Per-UID selection + SFT distillation:** keep ACR rollouts with base
   score >= 1.0 that pass the filters, then select the rollout with the
   highest ML score (random tie-break). Store the original prompt (no hints)
   and selected response as an SFT record. At the same interval, sample up to
   256 records, run one SFT update with LR scaled by 0.05, and flush the buffer.
7. **EMA update:** after each actor update, update the EMA teacher parameters
   (alpha 0.995) for the next ACR generation interval.

## Pseudocode

```text
for step in 1..T:
  batch = sample_train_batch()
  rlvr_rollouts = actor.generate(batch, n=8)
  rlvr_reward = reward_minerva(rlvr_rollouts)
  actor = GRPO_update(actor, rlvr_rollouts, rlvr_reward)

  hard_uids = {uid | max_reward(uid) < 1.0 and not CVSS(uid)}
  acr_prompts = build_acr_prompts(batch[hard_uids])
  acr_prompt_buffer.append(acr_prompts)

  if step % 10 == 0:
    acr_rollouts = ema_teacher.generate(acr_prompt_buffer, n=4, temp=0.7, top_p=0.9)
    acr_scores = reward_acr(acr_rollouts)
    eligible = base_score >= 1.0 and pass_heuristics and textcnn_score >= 0.5
    chosen = argmax_ml_score_per_uid(eligible)  # random tie-break
    distill_buffer.add(prompt_nohint, chosen_response)
    SFT_update(actor, sample(distill_buffer, 256), lr_scale=0.05)
    distill_buffer.clear()
    acr_prompt_buffer.clear()
    ema_teacher = EMA_update(actor, alpha=0.995)
```

## Parameter values (Noctua EMA-SFT configuration)

### Data and model

| Parameter | Value |
| --- | --- |
| Base model | Llama-3.2-3B-Instruct OR Llama-3.1-8B-Instruct OR Qwen3-4B-Base |
| Train data | `rlvr/mydata/minerva_base/minerva_base_train.parquet` |
| Val data | `rlvr/mydata/minerva_base/minerva_base_dev.parquet` + Athena CTI parquets (`athena_cti_ate`, `athena_cti_ckt`, `athena_cti_rcm`, `athena_cti_rms`, `athena_cti_taa`, `athena_cti_vsp`) + `seceval_mini` |
| Label details | `dataset/label_details` |
| Return raw chat | `true` (required for ACR prompt edits) |

### RLVR baseline (GRPO)

| Parameter | Value |
| --- | --- |
| Advantage estimator | `grpo` |
| RLVR rollouts per prompt | `8` |
| Rollout backend | `vllm` |
| Max prompt length | `2048` |
| Max response length | `1024` |
| Filter overlong prompts | `true` |
| Truncation | `error` |
| Actor LR | `1e-6` |
| PPO minibatch size | `128` |

### ACR prompt and gating

| Parameter | Value |
| --- | --- |
| ACR per-batch | `true` |
| ACR max prompt length | `4096` |
| ACR max details chars | `8096` |
| Hard gating mode | `max` (UID hard if max RLVR reward < 1.0) |
| Hard reward threshold | `1.0` |
| Skip CVSS for ACR | `true` |
| Enforce no ID in reasoning | `false` |

### ACR generation (EMA teacher)

| Parameter | Value |
| --- | --- |
| ACR rollouts per prompt | `4` |
| Sampling temperature | `0.7` |
| Sampling top-p | `0.9` |
| Deferred ACR generation | `true` |
| EMA teacher enabled | `true` |
| EMA alpha | `0.995` |
| ACR PPO update | `false` (generation-only) |

### ACR reward and filtering

| Parameter | Value |
| --- | --- |
| Reward function | `reward_acr` (base verifier = `reward_minerva`) |
| r_correct | `0.1` |
| leak_penalty | `0.5` |
| multilabel_match | `exact` |
| min_reasoning_chars | `100` |
| min_overlap_jaccard | `0.05` |
| max_id_mentions | `0` (ID-overuse check disabled) |
| ACR reward manager | `naive` |
| Filter mode | `ml+heuristic` |
| ML filter model | `xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25` |
| ML filter threshold | `0.5` |
| ML filter text mode | `response` |
| ML filter max length | `1024` |
| Degenerate filter | `true` |
| Degenerate min tokens | `30` |
| Degenerate rep-3 max | `0.70` |
| Degenerate rep-4 max | `0.75` |

### Distillation (SFT, flush buffer)

| Parameter | Value |
| --- | --- |
| Distill method | `sft` |
| Distill interval | `10` steps |
| Reward threshold | `1.0` (uses ACR base score) |
| Selection mode | `ml_score` (forced by ML filter) |
| Distill batch size | `256` |
| LR scale | `0.05` |
| Buffer mode | `flush` |
| Max buffer | `0` |
| Min buffer | `0` |
| Entropy sampling | `true` (not used under ml_score) |
| Entropy beta | `1.0` |

### Training/runtime

| Parameter | Value |
| --- | --- |
| Train batch size | `128` |
| Val batch size | `3000` |
| Total steps | `500` |
| Save/test frequency | `10` |
| Save best metric | `val-core/global-val/reward/mean` (mode `max`) |
| GPUs per node | `4` |
| Rollout GPU utilization | `0.95` |

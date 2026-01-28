# Experimental Settings (GRPO + Noctua/ACRD)

This document summarizes the experimental setup encoded in the training scripts under
`rlvr/cti-scripts/`. Defaults come from `train_minerva_grpo.sh` (GRPO baseline) and
`train_minerva_noctua.sh` (ACRD per-batch). Model wrappers in the same folder override
specific env vars for the study runs.

## GRPO baseline (RLVR)

Entry point: `rlvr/cti-scripts/train_minerva_grpo.sh` (wrappers call this script).

The GRPO baseline runs verifiable RL on Minerva Base using `reward_minerva`, with
VLLM rollouts and FSDP offload. It uses the same training/validation files across
models and logs to console + W&B.

Training/validation data (defaults):

```text
train: rlvr/mydata/minerva_base/minerva_base_train.parquet
val:   rlvr/mydata/minerva_base/minerva_base_dev.parquet
       rlvr/mydata/athena/athena_cti_ate.parquet
       rlvr/mydata/athena/athena_cti_ckt.parquet
       rlvr/mydata/athena/athena_cti_rcm.parquet
       rlvr/mydata/athena/athena_cti_rms.parquet
       rlvr/mydata/athena/athena_cti_taa.parquet
       rlvr/mydata/athena/athena_cti_vsp.parquet
       rlvr/mydata/seceval/seceval_mini.parquet
```

Model wrappers (default model paths):

```text
train_minerva_grpo_llama3b.sh  -> meta-llama/Llama-3.2-3B-Instruct
train_minerva_base_llama8b.sh  -> meta-llama/Llama-3.1-8B-Instruct
train_minerva_grpo_qwen4b.sh   -> Qwen/Qwen3-4B-Base
train_minerva_grpo_qwen8b.sh   -> Qwen/Qwen3-8B-Base
train_minerva_base_llama8b_rollout12.sh -> meta-llama/Llama-3.1-8B-Instruct (rollout n=12)
```

## Noctua (ACRD per-batch: RLVR -> ACR -> distill)

Entry point: `rlvr/cti-scripts/train_minerva_noctua.sh` (wrappers call this script).

Noctua runs the same GRPO RLVR loop as the baseline, then constructs answer-conditioned
prompts per batch, generates ACR rollouts, and distills via SFT or DPO. It relies on
label details from `dataset/label_details` and requires raw chat prompts for ACR.

Key mechanics (defaults from the script + wrapper presets):

- RLVR still uses GRPO with `reward_minerva` and vLLM rollouts (`rollout.n=8`).
- ACR prompt length can exceed RLVR prompt length (4096 vs 2048 by default).
- ACR rollouts default to `n=4`; wrappers set temperature/top_p to 0.7/0.9.
- ACR reward uses verifiable scoring with leak checks and short/overlap guards.
- Distillation defaults to SFT; wrappers typically use `buffer_mode=flush` with
  batch size 256 and `lr_scale=0.05` (unless using ablation scripts).
- Filters: default ML+heuristic with a TextCNN response-only classifier; ablations
  disable or downgrade the filter set.

Model wrappers (default model paths and presets):

```text
train_minerva_noctua_llama3b.sh -> meta-llama/Llama-3.2-3B-Instruct
train_minerva_noctua_qwen4b.sh  -> Qwen/Qwen3-4B-Base
train_minerva_noctua_qwen8b.sh  -> Qwen/Qwen3-8B-Base
train_minerva_noctua_gptoss.sh  -> openai/gpt-oss-20b
```

Ablations (Llama 8B):

```text
train_minerva_noctua_llama8b_abl_emaoff.sh      (EMA teacher disabled)
train_minerva_noctua_llama8b_abl_filteroff.sh   (filters disabled)
train_minerva_noctua_llama8b_abl_mlfilteroff.sh (heuristic-only filter)
train_minerva_noctua_llama8b_lr0.01_*.sh        (distill lr_scale=0.01)
train_minerva_noctua_llama8b_lr0.02_*.sh        (distill lr_scale=0.02)
train_minerva_noctua_llama8b_lr0.05_*.sh        (distill lr_scale=0.05)
train_minerva_noctua_llama8b_lr0.1_*.sh         (distill lr_scale=0.1)
```

---

## Hyperparameters (defaults from scripts)

### GRPO defaults (`train_minerva_grpo.sh`)

```yaml
# Model and experiment
GRPO_MODEL_PATH: meta-llama/Llama-3.2-3B-Instruct
GRPO_EXPERIMENT_NAME: minerva_grpo_<model-slug>
GRPO_OUTPUT_ROOT: rlvr/checkpoints/minerva/<experiment>

# Data
GRPO_TRAIN_PATH: rlvr/mydata/minerva_base/minerva_base_train.parquet
GRPO_VAL_PATH_1: rlvr/mydata/minerva_base/minerva_base_dev.parquet
GRPO_VAL_PATH_2: rlvr/mydata/athena/athena_cti_ate.parquet
GRPO_VAL_PATH_3: rlvr/mydata/athena/athena_cti_ckt.parquet
GRPO_VAL_PATH_4: rlvr/mydata/athena/athena_cti_rcm.parquet
GRPO_VAL_PATH_5: rlvr/mydata/athena/athena_cti_rms.parquet
GRPO_VAL_PATH_6: rlvr/mydata/athena/athena_cti_taa.parquet
GRPO_VAL_PATH_7: rlvr/mydata/athena/athena_cti_vsp.parquet
GRPO_VAL_PATH_8: rlvr/mydata/seceval/seceval_mini.parquet

# Batching and lengths
GRPO_TRAIN_BATCH_SIZE: 128
GRPO_VAL_BATCH_SIZE: 3000
GRPO_MAX_PROMPT_LEN: 2048
GRPO_MAX_RESPONSE_LEN: 1024
GRPO_ROLLOUT_N: 8
GRPO_ROLLOUT_GPU_UTIL: 0.95

# Training schedule
GRPO_TOTAL_STEPS: 500
GRPO_N_GPUS_PER_NODE: 4
GRPO_VAL_BEFORE_TRAIN: false
GRPO_SAVE_FREQ: 10
GRPO_TEST_FREQ: 10
GRPO_SAVE_BEST_ONLY: true
GRPO_SAVE_BEST_METRIC: val-core/global-val/reward/mean
GRPO_SAVE_BEST_MODE: max
GRPO_SAVE_BEST_DIR: best

# Optimization
GRPO_ACTOR_LR: 1e-6
GRPO_ENTROPY_COEFF: 0.0
GRPO_TENSOR_PARALLEL_SIZE: 1

# Fixed config values
algorithm.adv_estimator: grpo
custom_reward_function: reward_minerva
custom_reward_function.reward_kwargs.return_dict: true
actor_rollout_ref.rollout.name: vllm
actor_rollout_ref.rollout.n: ${GRPO_ROLLOUT_N}
actor_rollout_ref.actor.use_kl_loss: false
actor_rollout_ref.actor.kl_loss_coef: 0.0
actor_rollout_ref.actor.kl_loss_type: low_var_kl
actor_rollout_ref.model.enable_gradient_checkpointing: true
actor_rollout_ref.actor.fsdp_config.param_offload: true
actor_rollout_ref.actor.fsdp_config.optimizer_offload: true
actor_rollout_ref.ref.fsdp_config.param_offload: true
trainer.logger: [console, wandb]
trainer.project_name: minerva
trainer.critic_warmup: 0
algorithm.use_kl_in_reward: false
```

### Noctua defaults (`train_minerva_noctua.sh`)

```yaml
# Model and experiment
ACRD_MODEL_PATH: meta-llama/Llama-3.2-3B-Instruct
ACRD_EXPERIMENT_NAME: minerva_noctua_<model-slug>
ACRD_OUTPUT_ROOT: rlvr/checkpoints/minerva/<experiment>
ACRD_LABEL_DETAILS_DIR: dataset/label_details

# Data (same as GRPO)
ACRD_TRAIN_PATH: rlvr/mydata/minerva_base/minerva_base_train.parquet
ACRD_VAL_PATH_1: rlvr/mydata/minerva_base/minerva_base_dev.parquet
ACRD_VAL_PATH_2: rlvr/mydata/athena/athena_cti_ate.parquet
ACRD_VAL_PATH_3: rlvr/mydata/athena/athena_cti_ckt.parquet
ACRD_VAL_PATH_4: rlvr/mydata/athena/athena_cti_rcm.parquet
ACRD_VAL_PATH_5: rlvr/mydata/athena/athena_cti_rms.parquet
ACRD_VAL_PATH_6: rlvr/mydata/athena/athena_cti_taa.parquet
ACRD_VAL_PATH_7: rlvr/mydata/athena/athena_cti_vsp.parquet
ACRD_VAL_PATH_8: rlvr/mydata/seceval/seceval_mini.parquet

# RLVR prompt/response lengths
ACRD_RLVR_MAX_PROMPT_LEN: 2048
ACRD_ACR_MAX_PROMPT_LEN: 4096
ACRD_MAX_RESPONSE_LEN: 1024
ACRD_ACR_MAX_DETAILS_CHARS: 8096

# ACR generation
ACRD_ACR_ROLLOUT_N: 4
ACRD_ACR_ROLLOUT_TEMPERATURE: ""   # unset -> model defaults
ACRD_ACR_ROLLOUT_TOP_P: ""         # unset -> model defaults
ACRD_ACR_ROLLOUT_TOP_K: ""         # unset -> model defaults
ACRD_ACR_DEFER_GENERATION: false
ACRD_ACR_EMA_TEACHER_ENABLED: false
ACRD_ACR_EMA_TEACHER_ALPHA: 0.995
ACRD_ACR_RL_WEIGHT: 0.3
ACRD_ACR_HARD_REWARD_MODE: max
ACRD_ACR_HARD_REWARD_THRESHOLD: 1.0
ACRD_ACR_SKIP_CVSS: true

# Distillation (SFT/DPO)
ACRD_ACR_DISTILL_METHOD: sft
ACRD_ACR_DISTILL_INTERVAL: 10
ACRD_ACR_DISTILL_THRESHOLD: 1.0
ACRD_ACR_DISTILL_SELECTION_MODE: random
ACRD_ACR_DISTILL_BUFFER_MODE: rolling
ACRD_ACR_DISTILL_BATCH_SIZE: 256   # auto-selected unless overridden
ACRD_ACR_DISTILL_MAX_BUFFER: 1024  # auto-selected unless overridden
ACRD_ACR_DISTILL_MIN_BUFFER: 0     # auto-selected unless overridden
ACRD_ACR_DISTILL_DEGENERATE_FILTER: true
ACRD_ACR_DISTILL_DEGENERATE_MIN_TOKENS: 30
ACRD_ACR_DISTILL_DEGENERATE_REP3_MAX: 0.70
ACRD_ACR_DISTILL_DEGENERATE_REP4_MAX: 0.75
ACRD_ACR_DISTILL_LR_SCALE: 1.0
ACRD_ACR_DISTILL_ENTROPY_BETA: 1.0
ACRD_ACR_DISTILL_ENTROPY_SAMPLING: true
ACRD_ACR_DISTILL_DISABLE_FILTERS: false
ACRD_DPO_BETA: 0.1

# ACR reward settings
ACRD_ACR_REWARD_MANAGER: naive
ACRD_ACR_MAX_ID_MENTIONS: 0
ACRD_ACR_MIN_REASONING_CHARS: 100
ACRD_ACR_MIN_OVERLAP_JACCARD: 0.05
ACRD_ACR_EXCLUDE_CVSS_TRAIN: false

# Judge (disabled by default)
ACRD_JUDGE_ENABLED: false
ACRD_JUDGE_MODEL: openai/gpt-oss-20b
ACRD_JUDGE_BACKEND: hf
ACRD_JUDGE_MAX_NEW_TOKENS: 32
ACRD_JUDGE_TEMPERATURE: 0.0
ACRD_JUDGE_TOP_P: 1.0
ACRD_JUDGE_DEVICE: auto
ACRD_JUDGE_DEVICE_MAP: auto
ACRD_JUDGE_DTYPE: auto
ACRD_JUDGE_TRUST_REMOTE_CODE: true

# Training schedule and rollout
ACRD_TRAIN_BATCH_SIZE: 128
ACRD_VAL_BATCH_SIZE: 3000
ACRD_JUDGE_BATCH_SIZE: ${ACRD_TRAIN_BATCH_SIZE}
ACRD_N_GPUS_PER_NODE: 4
ACRD_ROLLOUT_GPU_UTIL: 0.95
ACRD_TOTAL_STEPS: 500
ACRD_SAVE_FREQ: 10
ACRD_TEST_FREQ: 10
ACRD_SAVE_BEST_ONLY: true
ACRD_SAVE_BEST_METRIC: val-core/global-val/reward/mean
ACRD_SAVE_BEST_MODE: max
ACRD_SAVE_BEST_DIR: best

# Distill filter (TextCNN classifier)
ACRD_ACR_DISTILL_FILTER_MODE: ml
ACRD_ACR_DISTILL_FILTER_MODEL: xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25
ACRD_ACR_DISTILL_FILTER_MODEL_TYPE: textcnn
ACRD_ACR_DISTILL_FILTER_THRESHOLD: 0.5
ACRD_ACR_DISTILL_FILTER_BATCH_SIZE: ${ACRD_TRAIN_BATCH_SIZE}
ACRD_ACR_DISTILL_FILTER_MAX_LENGTH: 1024
ACRD_ACR_DISTILL_FILTER_TEXT_MODE: response
ACRD_ACR_DISTILL_FILTER_DEVICE: auto
ACRD_ACR_DISTILL_FILTER_DTYPE: auto
ACRD_ACR_DISTILL_FILTER_TOKENIZER: ""
ACRD_ACR_DISTILL_FILTER_TRUST_REMOTE_CODE: false

# Fixed config values
algorithm.adv_estimator: grpo
custom_reward_function: reward_minerva
custom_reward_function.reward_kwargs.return_dict: true
actor_rollout_ref.rollout.name: vllm
actor_rollout_ref.rollout.n: 8
actor_rollout_ref.actor.optim.lr: 1e-6
actor_rollout_ref.actor.use_kl_loss: false
actor_rollout_ref.actor.kl_loss_coef: 0.0
actor_rollout_ref.actor.kl_loss_type: low_var_kl
actor_rollout_ref.model.enable_gradient_checkpointing: true
actor_rollout_ref.actor.fsdp_config.param_offload: true
actor_rollout_ref.actor.fsdp_config.optimizer_offload: true
actor_rollout_ref.ref.fsdp_config.param_offload: true
trainer.logger: [console, wandb]
trainer.project_name: minerva
trainer.critic_warmup: 0
algorithm.use_kl_in_reward: false

# ACR config forced in the launcher
+data.acr.per_batch: true
+data.acr.update_actor: false
+data.acr.enforce_no_id_in_reasoning: false
+data.acr.reward_kwargs.r_correct: 0.1
+data.acr.reward_kwargs.leak_penalty: 0.5
+data.acr.reward_kwargs.multilabel_match: exact
+data.acr.reward_kwargs.enforce_no_id_in_reasoning: false
```

### Noctua wrapper presets (used in study)

```text
train_minerva_noctua_llama3b.sh / qwen4b.sh / qwen8b.sh / gptoss.sh
  ACRD_ACR_DISTILL_LR_SCALE=0.05
  ACRD_ACR_DISTILL_BUFFER_MODE=flush
  ACRD_ACR_DISTILL_BATCH_SIZE=256
  ACRD_ACR_DISTILL_FILTER_MODE=ml+heuristic
  ACRD_ACR_DISTILL_FILTER_MODEL=textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25
  ACRD_ACR_DISTILL_FILTER_THRESHOLD=0.5
  ACRD_ACR_DEFER_GENERATION=true
  ACRD_ACR_EMA_TEACHER_ENABLED=true
  ACRD_ACR_EMA_TEACHER_ALPHA=0.995
  ACRD_ACR_ROLLOUT_TEMPERATURE=0.7
  ACRD_ACR_ROLLOUT_TOP_P=0.9
```

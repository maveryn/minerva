#!/usr/bin/env bash
set -e

# Training script for Minerva Base GRPO (no TARBA) with Qwen3 8B.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true

train_path="$DATA_DIR/minerva_base/minerva_base_train.parquet"

val_paths=(
  "$DATA_DIR/minerva_base/minerva_base_dev.parquet"
  "$DATA_DIR/athena/athena_cti_ate.parquet"
  "$DATA_DIR/athena/athena_cti_ckt.parquet"
  "$DATA_DIR/athena/athena_cti_rcm.parquet"
  "$DATA_DIR/athena/athena_cti_rms.parquet"
)

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"

train_files="['$train_path']"
val_files="['${val_paths[0]}','${val_paths[1]}','${val_paths[2]}','${val_paths[3]}','${val_paths[4]}']"
save_best_only="${BASE_SAVE_BEST_ONLY:-true}"
save_best_metric="${BASE_SAVE_BEST_METRIC:-val-core/global-val/reward/mean}"
save_best_mode="${BASE_SAVE_BEST_MODE:-max}"
save_best_dir="${BASE_SAVE_BEST_DIR:-best}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.dataloader_num_workers=0 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_minerva \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name='minerva_base_grpo_qwen8b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    +trainer.save_best_only="$save_best_only" \
    +trainer.save_best_metric="$save_best_metric" \
    +trainer.save_best_mode="$save_best_mode" \
    +trainer.save_best_dir="$save_best_dir" \
    trainer.total_training_steps=500 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

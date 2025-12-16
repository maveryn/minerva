#!/usr/bin/env bash
set -e

#2x h200

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

activity_train_path="$DATA_DIR/activity_train.parquet"
activity_test_path="$DATA_DIR/activity_test.parquet"
lis_train_path="$DATA_DIR/lis_train.parquet"
lis_test_path="$DATA_DIR/lis_test.parquet"
reward_fn_path=$ROOT_DIR/verl/utils/reward_score/myreward.py

# train_files="['$activity_train_path', '$lis_train_path']"
# test_files="['$activity_test_path', '$lis_test_path']"

train_files="['$lis_train_path']"
test_files="['$lis_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_ids_prefix \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='rlvr' \
    trainer.experiment_name='lis_id_qwen2_7b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=20 \
    trainer.max_actor_ckpt_to_keep=1\
    trainer.max_critic_ckpt_to_keep=1
    "$@"

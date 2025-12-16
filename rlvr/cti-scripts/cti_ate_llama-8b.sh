#!/usr/bin/env bash
set -e

# CTI training on the ATE split (Llama-3.1-8B-Instruct, GRPO)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata/cti_out"

cti_ate_train_path="$DATA_DIR/cti_ate_train.parquet"
cti_ate_test_path="$DATA_DIR/cti_ate_test.parquet"
cti_mcq_test_path="$DATA_DIR/cti_mcq_test.parquet"
cti_rcm_test_path="$DATA_DIR/cti_rcm_test.parquet"
cti_rms_test_path="$DATA_DIR/cti_rms_test.parquet"

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/myreward_boxed.py"

train_files="['$cti_ate_train_path']"
test_files="['$cti_ate_test_path', '$cti_mcq_test_path', '$cti_rcm_test_path', '$cti_rms_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_answer_only \
    data.train_batch_size=256 \
    data.max_prompt_length=768 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
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
    trainer.experiment_name='cti_ate_llama_8b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

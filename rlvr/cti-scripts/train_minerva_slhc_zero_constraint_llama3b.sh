#!/usr/bin/env bash
set -e

# Training script for Minerva SLHC (zero_constraint) with Llama 3B.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true

train_path="$DATA_DIR/minerva_lhc/minerva_lhc_train.parquet"

val_paths=(
  "$DATA_DIR/minerva_lhc/minerva_lhc_dev.parquet"
  "$DATA_DIR/athena/athena_cti_ate.parquet"
  "$DATA_DIR/athena/athena_cti_rcm.parquet"
  "$DATA_DIR/athena/athena_cti_rms.parquet"
)

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"
custom_dataset_path="$ROOT_DIR/verl/utils/dataset/minerva_stochastic_slhc_dataset.py"

train_files="['$train_path']"
val_files="['${val_paths[0]}','${val_paths[1]}','${val_paths[2]}','${val_paths[3]}']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.custom_cls.path="$custom_dataset_path" \
    data.custom_cls.name=StochasticSLHCRLHFDataset \
    data.dataloader_num_workers=0 \
    +data.stochastic_slhc.enabled=true \
    +data.stochastic_slhc.control_mode=zero_constraint \
    +data.stochastic_slhc.success_threshold=0.1 \
    +data.stochastic_slhc.candidate_pool_key=candidate_pool_top100 \
    +data.stochastic_slhc.buffer=10 \
    +data.stochastic_slhc.K_max_total=30 \
    +data.stochastic_slhc.d_min=1 \
    +data.stochastic_slhc.sigma_init=2.0 \
    +data.stochastic_slhc.p_nohint_start=0.05 \
    +data.stochastic_slhc.p_nohint_end=0.50 \
    +data.stochastic_slhc.total_steps=300 \
    +data.stochastic_slhc.ema_beta=0.90 \
    +data.stochastic_slhc.lr_mu=0.50 \
    +data.stochastic_slhc.target_acc=0.50 \
    +data.stochastic_slhc.zero_target=0.15 \
    +data.stochastic_slhc.all_target=0.15 \
    +data.stochastic_slhc.drift_delta=0.25 \
    +data.stochastic_slhc.alpha_zero=1.0 \
    +data.stochastic_slhc.alpha_all=0.7 \
    +data.stochastic_slhc.temp_zero=0.05 \
    +data.stochastic_slhc.temp_all=0.05 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_minerva \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
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
    trainer.experiment_name='minerva_slhc_zero_constraint_llama3b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps=300 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

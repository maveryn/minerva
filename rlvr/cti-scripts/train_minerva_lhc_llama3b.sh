#!/usr/bin/env bash
set -e

# Training script for Minerva LHC CTI datasets with adaptive label-hint curriculum (Llama 3B).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

train_path="$DATA_DIR/minerva_lhc/minerva_lhc_train.parquet"

val_paths=(
  "$DATA_DIR/minerva_lhc/minerva_lhc_dev.parquet"
  "$DATA_DIR/athena/athena_cti_ate.parquet"
  "$DATA_DIR/athena/athena_cti_rcm.parquet"
  "$DATA_DIR/athena/athena_cti_rms.parquet"
)

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_minerva.py"
custom_dataset_path="$ROOT_DIR/verl/utils/dataset/minerva_adaptive_dataset.py"

train_files="['$train_path']"
val_files="['${val_paths[0]}','${val_paths[1]}','${val_paths[2]}','${val_paths[3]}']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.custom_cls.path="$custom_dataset_path" \
    data.custom_cls.name=AdaptiveOptionRLHFDataset \
    data.dataloader_num_workers=0 \
    +data.adaptive_options.enabled=true \
    +data.adaptive_options.candidate_pool_key=candidate_pool_top100 \
    +data.adaptive_options.target_acc=0.5 \
    +data.adaptive_options.tol=0.05 \
    +data.adaptive_options.warmup_steps=0 \
    +data.adaptive_options.ema_beta=0.9 \
    +data.adaptive_options.k_min=2 \
    +data.adaptive_options.k_max=30 \
    +data.adaptive_options.k_step=2 \
    +data.adaptive_options.buffer=10 \
    +data.adaptive_options.p_drop_init=0.0 \
    +data.adaptive_options.p_drop_step=0.05 \
    +data.adaptive_options.p_drop_max=1.0 \
    +data.adaptive_options.score_threshold=0.5 \
    +data.adaptive_options.option_desc_max_chars=200 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_minerva \
    data.train_batch_size=64 \
    data.max_prompt_length=1536 \
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
    trainer.experiment_name='minerva_lhc_llama3b' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=500 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

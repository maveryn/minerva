#!/usr/bin/env bash
set -e

# Training script for Minerva TARBA (LHC) with Llama 3B.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

# Avoid wandb service teardown failures in some environments.
export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export SGLANG_FORCE_CPU_WEIGHT_SYNC=1

train_path="$DATA_DIR/minerva_lhc/minerva_lhc_train.parquet"

val_paths=(
  "$DATA_DIR/minerva_lhc/minerva_lhc_dev.parquet"
  "$DATA_DIR/athena/athena_cti_ate.parquet"
  "$DATA_DIR/athena/athena_cti_rcm.parquet"
  "$DATA_DIR/athena/athena_cti_rms.parquet"
)

reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_tarba.py"
custom_dataset_path="$ROOT_DIR/verl/utils/dataset/minerva_tarba_retrieval_dataset.py"
tool_config_path="$ROOT_DIR/cti-scripts/tool_config/cti_retrieval_tool.yaml"
sglang_engine_kwargs="{max_total_tokens: 16384, max_running_requests: 16, disable_cuda_graph: false, disable_radix_cache: false, disable_overlap_schedule: false, attention_backend: torch_native, prefill_attention_backend: torch_native}"

train_files="['$train_path']"
val_files="['${val_paths[0]}','${val_paths[1]}','${val_paths[2]}','${val_paths[3]}']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.custom_cls.path="$custom_dataset_path" \
    data.custom_cls.name=TarbaRLHFDataset \
    data.dataloader_num_workers=0 \
    data.return_raw_chat=True \
    +data.tarba.enabled=true \
    +data.tarba.p_noret_init=0.10 \
    +data.tarba.default_B_max=8 \
    +data.tarba.B_min=0 \
    +data.tarba.B_step=1 \
    +data.tarba.ema_beta=0.90 \
    +data.tarba.target_acc_noret=0.70 \
    +data.tarba.tol=0.05 \
    +data.tarba.min_steps_before_anneal=50 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_tarba \
    +custom_reward_function.reward_kwargs.lambda_ret=0.2 \
    +custom_reward_function.reward_kwargs.alpha=0.6931 \
    +custom_reward_function.reward_kwargs.ans_threshold=0.5 \
    +custom_reward_function.reward_kwargs.penalty_tool_call=0.02 \
    +custom_reward_function.reward_kwargs.penalty_illegal_tool=0.5 \
    data.train_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.engine_kwargs.sglang="$sglang_engine_kwargs" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name='minerva_tarba_lhc_llama3b' \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_training_steps=2 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    "$@"

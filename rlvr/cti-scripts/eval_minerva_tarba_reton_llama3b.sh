#!/usr/bin/env bash
set -e

# Validation for TARBA with retrieval enabled (ret-on).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/mydata"

export WANDB_START_METHOD=thread
export WANDB_DISABLE_SERVICE=true
export SGLANG_FORCE_CPU_WEIGHT_SYNC=1

variant="${TARBA_VARIANT:-lhc}"
if [[ "$variant" == "base" ]]; then
  train_path="$DATA_DIR/minerva_base/minerva_base_train.parquet"
  val_paths=(
    "$DATA_DIR/minerva_base/minerva_base_dev.parquet"
    "$DATA_DIR/athena/athena_cti_ate.parquet"
    "$DATA_DIR/athena/athena_cti_rcm.parquet"
    "$DATA_DIR/athena/athena_cti_rms.parquet"
  )
else
  train_path="$DATA_DIR/minerva_lhc/minerva_lhc_train.parquet"
  val_paths=(
    "$DATA_DIR/minerva_lhc/minerva_lhc_dev.parquet"
    "$DATA_DIR/athena/athena_cti_ate.parquet"
    "$DATA_DIR/athena/athena_cti_rcm.parquet"
    "$DATA_DIR/athena/athena_cti_rms.parquet"
  )
fi

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
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    +data.tarba.enabled=true \
    +data.tarba.eval_mode=ret_on \
    +data.tarba.eval_budget_B=5 \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=reward_tarba \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
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
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='minerva' \
    trainer.experiment_name='minerva_tarba_reton_llama3b' \
    "$@"

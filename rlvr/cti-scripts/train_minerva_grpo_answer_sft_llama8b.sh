#!/usr/bin/env bash
set -euo pipefail

# Matched auxiliary answer-only SFT ablation for Llama 8B.
# Mirrors the selected Llama-8B Noctua distillation schedule:
# lr_scale=0.05, flush buffer, batch size 256, interval 10, 4 GPUs.

export ANSWER_SFT_MODEL_PATH="${ANSWER_SFT_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export ANSWER_SFT_DISTILL_LR_SCALE="${ANSWER_SFT_DISTILL_LR_SCALE:-0.05}"
export ANSWER_SFT_DISTILL_BUFFER_MODE="${ANSWER_SFT_DISTILL_BUFFER_MODE:-flush}"
export ANSWER_SFT_DISTILL_BATCH_SIZE="${ANSWER_SFT_DISTILL_BATCH_SIZE:-256}"
export ANSWER_SFT_DISTILL_MAX_BUFFER="${ANSWER_SFT_DISTILL_MAX_BUFFER:-0}"
export ANSWER_SFT_DISTILL_MIN_BUFFER="${ANSWER_SFT_DISTILL_MIN_BUFFER:-0}"
export ANSWER_SFT_DISTILL_INTERVAL="${ANSWER_SFT_DISTILL_INTERVAL:-10}"
export ANSWER_SFT_HARD_ONLY="${ANSWER_SFT_HARD_ONLY:-true}"
export ANSWER_SFT_HARD_REWARD_MODE="${ANSWER_SFT_HARD_REWARD_MODE:-max}"
export ANSWER_SFT_HARD_REWARD_THRESHOLD="${ANSWER_SFT_HARD_REWARD_THRESHOLD:-1.0}"
export ANSWER_SFT_SKIP_CVSS="${ANSWER_SFT_SKIP_CVSS:-true}"
export ANSWER_SFT_ROLLOUT_GPU_UTIL="${ANSWER_SFT_ROLLOUT_GPU_UTIL:-0.95}"
export ANSWER_SFT_N_GPUS_PER_NODE="${ANSWER_SFT_N_GPUS_PER_NODE:-4}"

MODEL_NAME="$(basename "$ANSWER_SFT_MODEL_PATH")"
MODEL_SLUG="$(printf "%s" "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [ -z "$MODEL_SLUG" ]; then
  MODEL_SLUG="model"
fi
export ANSWER_SFT_EXPERIMENT_NAME="${ANSWER_SFT_EXPERIMENT_NAME:-minerva_grpo_answer_sft_${MODEL_SLUG}_lr0.05_flush_bs256}"

exec "$(dirname "$0")/train_minerva_grpo_answer_sft.sh" "$@"

# Minerva SFT

This document describes the SFT datasets and training scripts used for Minerva.

## Datasets

### Judge-selected SFT (reasoned responses)

Source:
- `minerva-judge/data/responses_gpt_oss_120b.jsonl`
- `rlvr/mydata/minerva_base/minerva_base_train.parquet`

Output:
- `rlvr/mydata/minerva_base_sft/minerva_base_sft_train.parquet`

Selection logic per `source_index`:
1. If plain response is correct (`reward == 1.0`), use `response_final`.
2. Else if hinted response is correct, use `response_final`.
3. Else use the ground-truth answer boxed as `\boxed{...}`.

Builder:
```
python minerva-sft/scripts/build_sft_from_judge.py \
  --responses minerva-judge/data/responses_gpt_oss_120b.jsonl \
  --base-parquet rlvr/mydata/minerva_base/minerva_base_train.parquet \
  --output rlvr/mydata/minerva_base_sft/minerva_base_sft_train.parquet
```

### Answer-only SFT (fast loss-based validation)

Source (train):
- `rlvr/mydata/minerva_base/minerva_base_train.parquet`

Source (val, same splits as GRPO):
- `rlvr/mydata/minerva_base/minerva_base_dev.parquet`
- `rlvr/mydata/athena/athena_cti_ate.parquet`
- `rlvr/mydata/athena/athena_cti_ckt.parquet`
- `rlvr/mydata/athena/athena_cti_rcm.parquet`
- `rlvr/mydata/athena/athena_cti_rms.parquet`
- `rlvr/mydata/athena/athena_cti_taa.parquet`
- `rlvr/mydata/athena/athena_cti_vsp.parquet`
- `rlvr/mydata/seceval/seceval_mini.parquet`

Outputs:
- `rlvr/mydata/minerva_base_sft/minerva_base_sft_answer_train.parquet`
- `rlvr/mydata/minerva_base_sft/minerva_base_sft_answer_val.parquet`

Builder:
```
python minerva-sft/scripts/build_answer_only_sft.py \
  --inputs rlvr/mydata/minerva_base/minerva_base_train.parquet \
  --output rlvr/mydata/minerva_base_sft/minerva_base_sft_answer_train.parquet

python minerva-sft/scripts/build_answer_only_sft.py \
  --inputs \
    rlvr/mydata/minerva_base/minerva_base_dev.parquet \
    rlvr/mydata/athena/athena_cti_ate.parquet \
    rlvr/mydata/athena/athena_cti_ckt.parquet \
    rlvr/mydata/athena/athena_cti_rcm.parquet \
    rlvr/mydata/athena/athena_cti_rms.parquet \
    rlvr/mydata/athena/athena_cti_taa.parquet \
    rlvr/mydata/athena/athena_cti_vsp.parquet \
    rlvr/mydata/seceval/seceval_mini.parquet \
  --output rlvr/mydata/minerva_base_sft/minerva_base_sft_answer_val.parquet
```

All answer-only responses are boxed as `\boxed{...}`.

## Training scripts

### Judge-selected SFT with reward eval

Scripts:
- `rlvr/cti-scripts/train_minerva_sft.sh`
- `rlvr/cti-scripts/train_minerva_sft_llama8b.sh` (Llama 8B + 4 GPUs)

Behavior:
- Uses `minerva_base_sft_train.parquet` for training.
- Skips val loss and runs GRPO-style reward eval on the RL val parquets.
- Saves best checkpoint by `val/reward_mean`.

Run:
```
bash rlvr/cti-scripts/train_minerva_sft_llama8b.sh
```

### Answer-only SFT with loss-based eval

Scripts:
- `rlvr/cti-scripts/train_minerva_sft_answer_only.sh`
- `rlvr/cti-scripts/train_minerva_sft_answer_only_llama8b.sh` (Llama 8B + 4 GPUs)

Behavior:
- Uses `minerva_base_sft_answer_train.parquet` for training.
- Uses `minerva_base_sft_answer_val.parquet` for validation.
- Saves best checkpoint by `val/loss` (min).

Run:
```
bash rlvr/cti-scripts/train_minerva_sft_answer_only_llama8b.sh
```

## Default training parameters (current scripts)

Common defaults:
- Global batch: 128
- Micro-batch per GPU: 8
- Max length: 4048
- Truncation: left (preserve answer)
- Total epochs: 2
- Validation frequency: every 10 steps

Reward-eval SFT defaults (train_minerva_sft.sh):
- Reward eval batch size: 64
- Reward eval max prompt length: 2048
- Reward eval max response length: 1024
- Temperature: 0.0, Top-p: 1.0
- Save best by `val/reward_mean`

Answer-only SFT defaults (train_minerva_sft_answer_only.sh):
- Save best by `val/loss` (min)

Override any setting with env vars or Hydra args, e.g.:
```
SFT_MICRO_BATCH_SIZE_PER_GPU=4 \
SFT_TEST_FREQ=5 \
bash rlvr/cti-scripts/train_minerva_sft_answer_only_llama8b.sh
```

## Notes

- `rlvr/mydata/minerva_base_sft/**` is tracked in Git LFS.
- Answer-only SFT is faster to validate because it uses loss, not generation.

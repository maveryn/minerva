# STaR-CTI Experiment Details (Paper Draft Notes)

This document captures the implementation and run details for the STaR-CTI experiments
in `star/` as of 2026-04-02. It is intended to be lifted into the paper’s methods and
reproducibility sections.

## 1) Scope and objective

We implement an offline STaR-style self-improving supervised fine-tuning loop for
Minerva-CTI tasks.

- Data source: Minerva base train split and Minerva/ Athena validation splits.
- Baseline comparator: MinervaRL-style checkpoint-selection metric and reward checker.
- Training objective: SFT only (no PPO/GRPO in this workflow).
- Goal: quantify whether iterative self-generated rationalized traces improve performance
  versus the base model without RL updates.

## 2) Core STaR algorithm implemented here

The loop is implemented in `star/run_star_cti.py` (`run_round` and `main`):

1. Use base model `M_r` and a fixed train split.
2. For each training example, generate:
   - `original` mode trace conditioned only on the original prompt.
   - `rationalization` mode trace conditioned on the gold answer appended to the final user turn.
3. Score both traces with `reward_minerva` (`verifier_score`, `verifier_success`).
4. Select the final training trace:
   - keep `original` if verifier-correct;
   - else keep `rationalization` if verifier-correct;
   - else keep nothing.
5. Build `star_sft_train.parquet` from selected traces (deduplicated).
6. Fine-tune a fresh STaR checkpoint for one round.
7. Evaluate checkpoint on validation split using:
   - `minerva_base_dev`
   - `athena_cti_ate`, `athena_cti_ckt`, `athena_cti_rcm`, `athena_cti_rms`,
     `athena_cti_taa`, `athena_cti_vsp`
8. Track round summary + checkpoint path.
9. Use best checkpoint by validation metric and continue next round from it.

In this repository, `selected_count == round_sft_rows` unless some selected examples are
dropped by downstream SFT formatting, which did not occur in these runs.

## 3) Key implementation files

- `star/run_star_cti.py`: round orchestration, selection, training launcher, eval launch.
- `star/generate.py`: generation for `original` and `rationalization`.
- `star/score.py`: adds `extracted_answer`, `verifier_score`, `verifier_success`, `reward_info`.
- `star/prompting.py`: rationalization prompt injection block.
- `star/build_sft_dataset.py`: converts selected traces to SFT parquet.
- `star/select.py`: selection policy implementation.
- `star/train_round.sh`: calls `rlvr/cti-scripts/train_minerva_sft.sh` via `torchrun`.
- `rlvr/cti-scripts/train_minerva_sft.sh`: FSDP SFT trainer wrapper.
- `star/eval.py`: combined validation generation + scoring + per-dataset/per-source breakdown.
- `star/scripts/train_star_llama8b.sh`, `star/scripts/train_star_qwen8b.sh`: launchers.
- `star/configs/star_cti_llama8b.yaml`, `star/configs/star_cti_qwen8b.yaml`: full run settings.

## 4) Data and validation protocol

- Train parquet: `rlvr/mydata/minerva_base/minerva_base_train.parquet` (`32,000` rows)
- Val suite:
  - `rlvr/mydata/minerva_base/minerva_base_dev.parquet` (`1,200` rows)
  - `rlvr/mydata/athena/athena_cti_ate.parquet` (`100`)
  - `rlvr/mydata/athena/athena_cti_ckt.parquet` (`300`)
  - `rlvr/mydata/athena/athena_cti_rcm.parquet` (`200`)
  - `rlvr/mydata/athena/athena_cti_rms.parquet` (`100`)
  - `rlvr/mydata/athena/athena_cti_taa.parquet` (`50`)
  - `rlvr/mydata/athena/athena_cti_vsp.parquet` (`200`)
- Total validation rows per round: `2,150`.
- Verifier/success threshold: `1.0`.
- Reward source: `rlvr/utils/reward_score/reward_minerva.py`.

## 5) Hyperparameters used (final configs)

Both completed 8B runs currently share these values, with model/path differences.

### Generation settings (training traces)

- backend: `vllm`
- batch_size: `2048`
- max_new_tokens: `2048`
- temperature: `0.7`
- top_p: `0.95`
- gpu_memory_utilization: `0.95`
- trust_remote_code: `false`
- max_prompt_length: `0` (in code, this disables explicit token truncation)

### Evaluation settings

- backend: `vllm`
- batch_size: `128`
- max_new_tokens: `2048`
- max_prompt_length: `0`
- gpu_memory_utilization: `0.95`

### SFT training settings

- backend: `verl`
- total_epochs: `1`
- train_batch_size: `128`
- micro_batch_size_per_gpu: `8`
- max_length: `3072`
- response_max_tokens: `2048`
- model_dtype: `bfloat16`
- model_strategy: `fsdp`
- enable_gradient_checkpointing: `true`
- reward_eval_batch_size: `8`
- reward_eval_max_prompt_length: `2048`
- reward_eval_max_response_length: `2048`
- n_gpus: `1`
- save_freq: `250`
- test_freq: `250`
- save_best_only: `true`

### Common run options

- wandb enabled (`project=minerva`, `job_type=star`)
- max planned rounds: `5`
- completed rounds in both runs: `3` (`round_00`, `round_01`, `round_02`)

## 6) Launch commands used

- Llama-3.1-8B-Instruct:
  - `bash star/scripts/train_star_llama8b.sh --from-scratch`
- Qwen3-8B-Base:
  - `bash star/scripts/train_star_qwen8b.sh --from-scratch`

## 7) Exact run configs

### Llama-3.1-8B-Instruct
- Config: `star/configs/star_cti_llama8b.yaml`
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Artifact root: `star-artifacts/star_cti_llama31_8b`

### Qwen3-8B-Base
- Config: `star/configs/star_cti_qwen8b.yaml`
- Base model: `Qwen/Qwen3-8B-Base`
- Artifact root: `star-artifacts/star_cti_qwen3_8b_base`

## 8) Round-wise results from saved summaries

### Combined metrics (percent)

Formula:
- combined checkpoint score = `0.5 * rl_minerva_dev_mean_score + 0.5 * rl_athena_bench_mean_score`.

#### Llama-3.1-8B-Instruct STaR

| Round | Original succ. | Rationalization succ. | Selected | Minerva-dev | Athena mean | Global mean | Combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_00 | 1,738 / 32,000 (5.4%) | 27,495 / 32,000 (85.9%) | 27,666 / 32,000 (86.5%) | 43.0 | 41.5 | 47.7 | 42.3 |
| round_01 | 8,301 / 32,000 (25.9%) | 30,712 / 32,000 (96.0%) | 30,866 / 32,000 (96.5%) | 43.4 | 39.8 | 47.4 | 41.6 |
| round_02 | 8,686 / 32,000 (27.1%) | 30,595 / 32,000 (95.6%) | 30,774 / 32,000 (96.2%) | 43.5 | 35.7 | 45.5 | 39.6 |

#### Qwen3-8B-Base STaR

| Round | Original succ. | Rationalization succ. | Selected | Minerva-dev | Athena mean | Global mean | Combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_00 | 3,911 / 32,000 (12.2%) | 26,237 / 32,000 (82.0%) | 26,825 / 32,000 (83.8%) | 34.7 | 30.8 | 38.6 | 32.7 |
| round_01 | 5,079 / 32,000 (15.9%) | 26,995 / 32,000 (84.4%) | 27,682 / 32,000 (86.5%) | 36.7 | 28.7 | 38.8 | 32.7 |
| round_02 | 4,443 / 32,000 (13.9%) | 20,843 / 32,000 (65.1%) | 22,285 / 32,000 (69.6%) | 19.0 | 28.2 | 28.8 | 23.6 |

### Per-dataset validation means (%)

#### Llama-3.1-8B-Instruct STaR

| Round | CKT | ATE | RCM | RMS | TAA | VSP | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_00 | 70.0 | 26.0 | 50.5 | 9.4 | 16.0 | 77.1 | 43.0 |
| round_01 | 69.0 | 25.0 | 46.0 | 8.8 | 10.0 | 80.2 | 43.4 |
| round_02 | 68.0 | 12.0 | 37.5 | 9.7 | 12.0 | 75.3 | 43.5 |

#### Qwen3-8B-Base STaR

| Round | CKT | ATE | RCM | RMS | TAA | VSP | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| round_00 | 69.7 | 17.0 | 48.0 | 2.0 | 4.0 | 44.0 | 34.7 |
| round_01 | 68.7 | 18.0 | 46.5 | 0.0 | 2.0 | 37.1 | 36.7 |
| round_02 | 73.0 | 17.0 | 36.0 | 2.8 | 0.0 | 40.4 | 19.0 |

### Best checkpoints

- Llama: `round_00` is best by combined score (`42.3`), checkpoint path:
  `star-artifacts/star_cti_llama31_8b/round_00/checkpoint/hf_model`.
- Qwen: `round_00` is best by combined score (`32.7`, ties with round_01), checkpoint path:
  `star-artifacts/star_cti_qwen3_8b_base/round_00/checkpoint/hf_model`.

## 9) Artifact structure and what to keep

For sharing and paper-level reproducibility, prefer storing:
- `round_i/summary.json` (generator/training metadata per round),
- `round_i/eval/summary.json` (val metrics + per-task/per-source breakdown),
- plus git-checked `star/` configuration and scripts.

Artifacts intentionally excluded from lightweight reporting:
- checkpoints under `checkpoint/hf_model`,
- generated `.jsonl` traces,
- selected/scored sample files,
- `star_sft_train.parquet`.

## 10) Notes for write-up

- The implementation is faithful to STaR-style iterative SFT refresh, not policy-gradient RL.
- `round_00` is the strongest checkpoint for both tested 8B models on the combined metric.
- Later rounds show stronger trace selection coverage but mixed or reduced validation generalization,
  so checkpoint selection is essential.

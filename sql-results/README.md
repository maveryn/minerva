# MinervaRL on SQL-R1

This directory documents the SQL-R1 experiment we ran with the Minerva RL/Noctua stack.

In this document, `MinervaRL` means:

- SQL-R1 dataset and reward function
- trained inside the Minerva/VeRL stack
- with Minerva's Noctua-style ACR + distillation loop enabled

The main model/result files in this directory are:

- [sql_6_benchmark_results.md](/home/jovyan/work/minerva/sql-results/sql_6_benchmark_results.md)
- [sql_r1_3b_three_runs_comparison.md](/home/jovyan/work/minerva/sql-results/sql_r1_3b_three_runs_comparison.md)

## Summary

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- Training data: `SQL-R1/example_data/train.parquet`
- Validation data: `SQL-R1/example_data/test.parquet`
- Reward: SQL-R1 SynSQL reward ported into Minerva
- Training stack: Minerva + VeRL GRPO + Noctua ACRD
- Best checkpoint used for the main `MinervaRL` benchmark report: step `400`
- Checkpoint root:
  - `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps`
- Best checkpoint step marker:
  - `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best/checkpoint_step.txt`

## Code Paths

Training:

- [train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh)
- [train_sql_r1_noctua.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/train_sql_r1_noctua.sh)

SQL-R1 integration:

- [sql_r1_dataset.py](/home/jovyan/work/minerva/rlvr/verl/utils/dataset/sql_r1_dataset.py)
- [reward_sql_r1.py](/home/jovyan/work/minerva/rlvr/verl/utils/reward_score/reward_sql_r1.py)
- [reward_sql_r1_acr.py](/home/jovyan/work/minerva/rlvr/verl/utils/reward_score/reward_sql_r1_acr.py)
- [sql_r1_acr_prompt.py](/home/jovyan/work/minerva/minerva/sql_r1_acr_prompt.py)
- [ray_trainer.py](/home/jovyan/work/minerva/rlvr/verl/trainer/ppo/ray_trainer.py)

Evaluation:

- [eval_sql_r1_table1_single.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_table1_single.sh)
- [eval_sql_r1_spider_robustness.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_spider_robustness.sh)
- [run_sql_r1_table1_parallel.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/run_sql_r1_table1_parallel.sh)

LiveSQLBench package:

- [sql-benchmark/livesqlbench/README.md](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/README.md)
- [run_qwen25coder3b_base.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/run_qwen25coder3b_base.sh)
- [run_sqlr1_3b_released.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/run_sqlr1_3b_released.sh)
- [run_grpo_qwen25coder3b.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/run_grpo_qwen25coder3b.sh)
- [run_minervarl_qwen25coder3b.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/run_minervarl_qwen25coder3b.sh)

## Data

Expected SQL-R1 checkout:

```bash
/home/jovyan/work/SQL-R1
```

Training/validation files:

- train: `/home/jovyan/work/SQL-R1/example_data/train.parquet`
- val: `/home/jovyan/work/SQL-R1/example_data/test.parquet`

Database root used by the reward:

- `/home/jovyan/work/SQL-R1/data/NL2SQL/SynSQL-2.5M/databases`

## Training Setup

The 8-GPU launcher defaults used for the main `MinervaRL` run were:

- GPUs: `8`
- train batch size: `128`
- val batch size: `128`
- max prompt length: `4096`
- ACR max prompt length: `4608`
- max response length: `2048`
- GRPO rollout count: `8`
- GRPO rollout temperature: `1.1`
- actor learning rate: `3e-7`
- KL coefficient: `0.001`
- rollout backend: `vllm`
- rollout GPU utilization: `0.7`
- free cache engine: `true`
- PPO micro batch size per GPU: `4`
- logprob micro batch size per GPU: `16`

### Noctua / ACR settings

- ACR rollout count: `4`
- ACR RL weight: `0.3`
- deferred generation: `true`
- EMA teacher: `true`
- EMA alpha: `0.995`
- hardness gate metric: `exec_match`
- hardness gate mode: `max`
- hardness threshold: `1.0`

### Distillation settings

- enabled: `true`
- interval: `10` steps
- reward threshold: `1.0`
- selection mode: `random`
- buffer mode: `flush`
- distill batch size: `256`
- max buffer: `0` (unbounded)
- distill LR scale: `0.05`
- filter mode: `heuristic`

### Reward / ACR reward settings

Main reward:

- return dict: `true`
- timeout: `10s`

ACR reward:

- timeout: `10s`
- `r_correct = 0.1`
- `leak_penalty = 0.5`
- `min_reasoning_chars = 100`
- `query_copy_min_common_run = 12`
- `query_copy_min_ratio = 0.8`

## What We Actually Ran

### Initial 100-step run

```bash
cd /home/jovyan/work/minerva

bash rlvr/cti-scripts/train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh
```

That launcher hard-sets:

- `trainer.total_training_steps=100`
- `trainer.val_before_train=False`
- `save_freq=20`
- `test_freq=20`
- `save_best_only=true`

### Resume past 100 steps

The same experiment was then resumed with a larger total-step target. The clean way to continue the run is:

```bash
cd /home/jovyan/work/minerva

SQLR1_EXPERIMENT_NAME=sql_r1_noctua_qwen25coder3b_8gpu_100steps \
SQLR1_TOTAL_EPOCHS=16 \
bash rlvr/cti-scripts/train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh \
  trainer.total_training_steps=500
```

Notes:

- `trainer.total_training_steps` is the final target, not an increment.
- We needed to raise `SQLR1_TOTAL_EPOCHS` above `10` so the resumed run would not stop early due to epoch budget.
- The best checkpoint selected by validation reward for this experiment ended up at step `400`.

## Best Checkpoint Used For Reported MinervaRL Results

Checkpoint:

- `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best`

Best step:

- `400`

Merged HF export used for evaluation:

- `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf`

If the merged HF model is missing, create it with:

```bash
cd /home/jovyan/work/minerva/rlvr

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best/actor \
  --target_dir /home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf
```

## Evaluation Setup

We used SQL-R1-style self-consistency evaluation:

- number of sampled candidates: `8`
- sampling temperature: `0.8`
- final prediction selected by execution-based `major_voting`

This matches the SQL-R1 paper/repo inference configuration.

### Table 1 benchmarks

Benchmarks:

- Spider Dev
- Spider Test
- BIRD Dev

Single-dataset command pattern:

```bash
cd /home/jovyan/work/minerva

SQLR1_MERGED_MODEL_DIR=/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf \
SQLR1_EVAL_RESULTS_ROOT=/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table1/spider-dev \
SQLR1_EVAL_DATASET=spider-dev \
SQLR1_EVAL_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0 \
bash rlvr/cti-scripts/eval_sql_r1_table1_single.sh
```

Repeat with:

- `SQLR1_EVAL_DATASET=spider-test`
- `SQLR1_EVAL_DATASET=bird-dev`

The script writes a per-run `summary.tsv`.

### Table 2 robustness benchmarks

Benchmarks:

- Spider-DK
- Spider-Syn
- Spider-Realistic

Command:

```bash
cd /home/jovyan/work/minerva

SQLR1_MERGED_MODEL_DIR=/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf \
SQLR1_EVAL_RESULTS_ROOT=/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table2 \
SQLR1_EVAL_NUM_GPUS=1 \
CUDA_VISIBLE_DEVICES=0 \
SQLR1_MERGE_IF_MISSING=false \
SQLR1_EVAL_DATASETS='spider-dk spider-syn spider-realistic' \
bash rlvr/cti-scripts/eval_sql_r1_spider_robustness.sh
```

Notes:

- The robustness script sanitizes malformed gold SQL before evaluation.
- It then runs the same `n=8`, `temperature=0.8`, major-voting pipeline.

## Reported MinervaRL Numbers

These are the step-400 best-checkpoint numbers captured in [sql_6_benchmark_results.md](/home/jovyan/work/minerva/sql-results/sql_6_benchmark_results.md):

| Model | Spider Dev | Spider Test | BIRD Dev | Spider-DK | Spider-Syn | Spider-Realistic | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MinervaRL (best step 400) | 78.24 | 79.27 | 52.54 | 69.16 | 70.70 | 69.29 | 69.87 |

Raw committed benchmark artifacts for this model are in:

- `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table1`
- `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table2`

## Reproducibility Caveat

Evaluation is stochastic because of self-consistency sampling:

- `n = 8`
- `temperature = 0.8`

So exact per-dataset numbers can move slightly across reruns even for the same checkpoint. We observed this clearly on the released SQL-R1-3B model. The 6-task average was much more stable than every individual benchmark score.

## Minimal Reproduction Recipe

If you want the shortest path to re-run the `MinervaRL` experiment end to end:

1. Clone SQL-R1 to `/home/jovyan/work/SQL-R1`
2. Train from base Qwen:

```bash
cd /home/jovyan/work/minerva

SQLR1_TOTAL_EPOCHS=16 \
bash rlvr/cti-scripts/train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh \
  trainer.total_training_steps=500
```

3. Use the best checkpoint selected by validation.
4. Merge it to HF format if needed.
5. Run:
   - [eval_sql_r1_table1_single.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_table1_single.sh) on Spider Dev/Test and BIRD Dev
   - [eval_sql_r1_spider_robustness.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_spider_robustness.sh) on Spider-DK/Syn/Realistic
6. Compare the resulting `summary.tsv` files against:
   - [sql_6_benchmark_results.md](/home/jovyan/work/minerva/sql-results/sql_6_benchmark_results.md)

# SQL-E1 Training Setup

## Task and Training Data

We train on the public SQL-R1 SynSQL split:

- training parquet: `/home/jovyan/work/SQL-R1/example_data/train.parquet`
- validation parquet: `/home/jovyan/work/SQL-R1/example_data/test.parquet`
- train size: `4,000`
- validation size: `1,000`

The task is single-turn text-to-SQL generation. Each example maps a natural-language question and database context to one final SQL answer. The reward is the SQL-R1 executable SynSQL reward ported into the Minerva/VeRL trainer.

Database execution during training uses:

- database root: `/home/jovyan/work/SQL-R1/data/NL2SQL/SynSQL-2.5M/databases`
- reward timeout: `10s`

## Base Model

All trained models in this experiment start from:

- `Qwen/Qwen2.5-Coder-3B-Instruct`

We report four evaluation rows overall:

1. base `Qwen2.5-Coder-3B-Instruct`
2. released `SQL-R1-3B`
3. our `GRPO` model
4. our `MinervaRL` model

## Common Training Settings

These settings were shared by both of our trained variants unless noted otherwise:

| Setting | Value |
| --- | --- |
| max prompt length | `4096` |
| max response length | `2048` |
| rollout count | `8` |
| rollout temperature | `1.1` |
| actor learning rate | `3e-7` |
| KL coefficient | `0.001` |
| reward backend | SQL-R1 SynSQL executable reward |
| rollout backend | `vllm` |
| reward timeout | `10s` |

## GRPO Run

The vanilla GRPO run was launched from:

- [train_sql_r1_grpo_qwen25coder3b_4gpu_400steps.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/train_sql_r1_grpo_qwen25coder3b_4gpu_400steps.sh)

Key settings:

| Setting | Value |
| --- | --- |
| GPUs | `4` |
| train batch size | `128` |
| val batch size | `128` |
| total training steps | `400` |
| total epoch budget | `16` |
| PPO micro batch size / GPU | `4` |
| logprob micro batch size / GPU | `16` |
| rollout GPU utilization | `0.7` |
| save frequency | every `20` steps |
| validation frequency | every `20` steps |
| checkpoint selection | `best` + `last` |

Best checkpoint used for reporting:

- checkpoint root: `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_grpo_qwen25coder3b_4gpu_400steps`
- best checkpoint step: `380`
- merged HF model: `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_grpo_qwen25coder3b_4gpu_400steps/best_merged_hf`

## MinervaRL Run

`MinervaRL` is our paper-facing name for the Minerva Noctua-style SQL run:

- SQL-R1 dataset and reward
- Minerva/VeRL GRPO backbone
- Minerva ACR prompt generation and distillation

The main launcher was:

- [train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/train_sql_r1_noctua_qwen25coder3b_8gpu_100steps.sh)

We first ran a fixed `100`-step job, then resumed the same experiment to continue training. The best validation checkpoint selected for reporting is step `400`.

Core run settings:

| Setting | Value |
| --- | --- |
| GPUs | `8` |
| train batch size | `128` |
| val batch size | `128` |
| initial scripted budget | `100` steps |
| resumed total-step target | `500` |
| resumed epoch budget | `16` |
| PPO micro batch size / GPU | `4` |
| logprob micro batch size / GPU | `16` |
| rollout GPU utilization | `0.7` |
| save frequency | every `20` steps |
| validation frequency | every `20` steps |
| checkpoint selection | `best` + `last` |

### MinervaRL-specific settings

| Setting | Value |
| --- | --- |
| ACR max prompt length | `4608` |
| ACR rollout count | `4` |
| ACR RL weight | `0.3` |
| deferred generation | `true` |
| EMA teacher | `true` |
| EMA alpha | `0.995` |
| hard-example gate metric | `exec_match` |
| hard-example gate mode | `max` |
| hard-example threshold | `1.0` |
| distillation interval | every `10` steps |
| distillation threshold | `1.0` |
| distillation selection | `random` |
| distillation buffer mode | `flush` |
| distillation batch size | `256` |
| distillation LR scale | `0.05` |
| filter mode | `heuristic` |

Best checkpoint used for reporting:

- checkpoint root: `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps`
- best checkpoint step: `400`
- merged HF model: `/home/jovyan/work/minerva/rlvr/checkpoints/sql-r1/sql_r1_noctua_qwen25coder3b_8gpu_100steps/best_merged_hf`

## MinervaRL vs GRPO

The paper-relevant difference between the two trained variants is:

- `GRPO`: SQL-R1 reward with vanilla GRPO only
- `MinervaRL`: the same SQL-R1 reward and base training setup, plus Minerva Noctua-style answer-conditioned reasoning generation and periodic SFT distillation on hard examples

For SQL, hard examples are defined by execution correctness:

- a prompt is treated as solved if at least one rollout has `exec_match = 1.0`
- otherwise it remains eligible for ACR/MinervaRL processing

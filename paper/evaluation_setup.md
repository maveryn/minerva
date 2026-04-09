# SQL-E1 Evaluation Setup

## Evaluated Models

We evaluate four models:

1. `Qwen2.5-Coder-3B-Instruct` base
2. released `SQL-R1-3B`
3. our `GRPO` best checkpoint (step `380`)
4. our `MinervaRL` best checkpoint (step `400`)

For the six Spider/BIRD-style SQL benchmarks, the released `SQL-R1-3B` row is reported as the best score per dataset across three repeated evaluation runs because the SQL-R1 self-consistency evaluation is stochastic.

## Six SQL Benchmarks

These six benchmarks are all single-turn text-to-SQL tasks evaluated with execution-style accuracy.

| Dataset | Size | Notes |
| --- | ---: | --- |
| Spider Dev | `1,034` | official Spider dev split |
| Spider Test | `2,147` | official Spider test split |
| BIRD Dev | `1,534` | official BIRD dev split |
| Spider-DK | `535` | Spider robustness split |
| Spider-Syn | `1,034` | Spider robustness split |
| Spider-Realistic | `508` | Spider robustness split |

Evaluation settings for all six:

- sampled candidates: `8`
- temperature: `0.8`
- final SQL chosen by execution-based `major_voting`

This matches the SQL-R1 paper/repo inference style.

Implementation paths:

- [eval_sql_r1_table1_single.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_table1_single.sh)
- [eval_sql_r1_spider_robustness.sh](/home/jovyan/work/minerva/rlvr/cti-scripts/eval_sql_r1_spider_robustness.sh)

Metric:

- `EX` / execution accuracy, reported as a percentage

## LiveSQLBench Benchmarks

We also evaluate on two LiveSQLBench open-dev benchmarks using the official local evaluator.

### LiveSQLBench Base-Full-v1

| Property | Value |
| --- | --- |
| dataset file | `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/artifacts/livesqlbench_base_full_v1_full.jsonl` |
| size | `600` |
| Query tasks | `410` |
| Management tasks | `190` |
| backend | PostgreSQL |

### LiveSQLBench Base-Lite

| Property | Value |
| --- | --- |
| dataset file | `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/artifacts/livesqlbench_base_lite_full.jsonl` |
| size | `270` |
| Query tasks | `180` |
| Management tasks | `90` |
| backend | PostgreSQL |

Base-Lite setup files added in this repo:

- [setup_assets_base_lite.py](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/setup_assets_base_lite.py)
- [start_local_postgres_base_lite.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/start_local_postgres_base_lite.sh)

### LiveSQLBench generation settings

For both Base-Full-v1 and Base-Lite:

- backend: `vllm`
- sampled candidates: `8`
- temperature: `0.8`
- top-p: `0.95`
- max new tokens: `768`
- max model length: `32768`
- tensor parallel size: `1`
- final SQL chosen by normalized SQL majority vote over the `8` sampled candidates

Implementation path:

- [run_livesqlbench_model.sh](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/run_livesqlbench_model.sh)
- [generate_sql_predictions.py](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/generate_sql_predictions.py)

### LiveSQLBench metric

LiveSQLBench does **not** use Spider/BIRD-style execution accuracy.

The official evaluator reports:

- `Overall Accuracy`
- `Number of Execution Errors`
- `Number of Timeouts`
- `Number of Assertion Errors`
- `Total Errors`

`Overall Accuracy` is an instance-level task-pass metric under the official evaluator. A case fails if it encounters an execution error, timeout, or assertion failure under the benchmark's test-case checks.

Implementation path:

- [evaluation.py](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/official/evaluation/src/evaluation.py)
- [utils.py](/home/jovyan/work/minerva/sql-benchmark/livesqlbench/official/evaluation/src/utils.py)

## Artifact Paths

### Six SQL benchmarks

- base model summaries: `/home/jovyan/work/minerva/sql-benchmark/summary_table1.tsv` and `/home/jovyan/work/minerva/sql-benchmark/qwen25coder3b_instruct_base_table2/summary.tsv`
- released SQL-R1-3B repeated-run comparison: `/home/jovyan/work/minerva/sql-results/sql_r1_3b_three_runs_comparison.md`
- GRPO summaries:
  - `/home/jovyan/work/minerva/sql-benchmark/grpo_qwen25coder3b_best_step380_table1/summary.tsv`
  - `/home/jovyan/work/minerva/sql-benchmark/grpo_qwen25coder3b_best_step380_table2/summary.tsv`
- MinervaRL summaries:
  - `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table1/summary.tsv`
  - `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table2/summary.tsv`

### LiveSQLBench Base-Full-v1

- base: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_base_32k/predictions/report.txt`
- released SQL-R1-3B: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_sqlr1_3b_released_32k/predictions/report.txt`
- GRPO: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_grpo_32k/predictions/report.txt`
- MinervaRL: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_minervarl_32k/predictions/report.txt`

### LiveSQLBench Base-Lite

- base: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_base_lite_qwen25coder3b_base_32k/predictions/report.txt`
- released SQL-R1-3B: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_base_lite_sqlr1_3b_released_32k/predictions/report.txt`
- GRPO: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_base_lite_qwen25coder3b_grpo_32k/predictions/report.txt`
- MinervaRL: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_base_lite_qwen25coder3b_minervarl_32k/predictions/report.txt`

# SQL Benchmark Results

Execution accuracy (`EX`, %) on the 6 SQL benchmarks we evaluated locally:

- Spider Dev
- Spider Test
- BIRD Dev
- Spider-DK
- Spider-Syn
- Spider-Realistic

All evals used the SQL-R1-style self-consistency setup:

- `n = 8`
- `temperature = 0.8`
- final SQL chosen by execution-based major voting

LiveSQLBench is shown separately for reference. It uses the official LiveSQLBench task-pass metric (`Overall Accuracy`), not Spider/BIRD-style execution accuracy, so it is not included in the 6-benchmark average.

## Results

| Model | Spider Dev | Spider Test | BIRD Dev | Spider-DK | Spider-Syn | Spider-Realistic | LiveSQLBench | Avg (6 SQL) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-3B-Instruct (base) | 77.37 | 77.60 | 50.33 | 65.61 | 67.21 | 69.69 | 3.67 | 67.97 |
| SQL-R1-3B (released, best of 3 eval runs) | 77.08 | 77.41 | 53.26 | 70.47 | 67.50 | 66.54 | 6.33 | 68.71 |
| GRPO Qwen2.5-Coder-3B (best step 380) | 77.47 | 78.53 | 54.24 | 68.97 | 67.89 | 68.70 | 6.33 | 69.30 |
| Noctua Qwen2.5-Coder-3B (best step 400) | 78.24 | 79.27 | 52.54 | 69.16 | 70.70 | 69.29 | 6.67 | 69.87 |

## Dataset Sizes

| Dataset | Size |
| --- | ---: |
| Spider Dev | 1,034 |
| Spider Test | 2,147 |
| BIRD Dev | 1,534 |
| Spider-DK | 535 |
| Spider-Syn | 1,034 |
| Spider-Realistic | 508 |

## Source Artifacts

- Base model Table 1: `/home/jovyan/work/minerva/sql-benchmark/summary_table1.tsv`
- Base model Table 2: `/home/jovyan/work/minerva/sql-benchmark/qwen25coder3b_instruct_base_table2/summary.tsv`
- Released SQL-R1-3B best-of-3 comparison: `/home/jovyan/work/minerva/sql-results/sql_r1_3b_three_runs_comparison.md`
- GRPO step-380 Table 1: `/home/jovyan/work/minerva/sql-benchmark/grpo_qwen25coder3b_best_step380_table1/summary.tsv`
- GRPO step-380 Table 2: `/home/jovyan/work/minerva/sql-benchmark/grpo_qwen25coder3b_best_step380_table2/summary.tsv`
- Noctua step-400 Table 1: `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table1/summary.tsv`
- Noctua step-400 Table 2: `/home/jovyan/work/minerva/sql-benchmark/noctua_qwen25coder3b_best_step400_table2/summary.tsv`
- LiveSQLBench base: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_base_32k/predictions/report.txt`
- LiveSQLBench released SQL-R1-3B: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_sqlr1_3b_released_32k/predictions/report.txt`
- LiveSQLBench GRPO step-380: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_grpo_32k/predictions/report.txt`
- LiveSQLBench MinervaRL step-400: `/home/jovyan/work/minerva/sql-benchmark/livesqlbench/runs/livesqlbench_qwen25coder3b_minervarl_32k/predictions/report.txt`

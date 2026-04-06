# SQL 6-Benchmark Results

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

## Results

| Model | Spider Dev | Spider Test | BIRD Dev | Spider-DK | Spider-Syn | Spider-Realistic | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-3B-Instruct (base) | 77.37 | 77.60 | 50.33 | 65.61 | 67.21 | 69.69 | 67.97 |
| SQL-R1-3B (released, best of 3 eval runs) | 77.08 | 77.41 | 53.26 | 70.47 | 67.50 | 66.54 | 68.71 |
| GRPO Qwen2.5-Coder-3B (best step 380) | 77.47 | 78.53 | 54.24 | 68.97 | 67.89 | 68.70 | 69.30 |
| Noctua Qwen2.5-Coder-3B (best step 400) | 78.24 | 79.27 | 52.54 | 69.16 | 70.70 | 69.29 | 69.87 |

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

# SQL-R1-3B Released Checkpoint: Three Eval Runs

Fresh local reruns of the released `MPX0222forHF/SQL-R1-3B` checkpoint on the same 6 SQL benchmarks.

All runs used the same SQL-R1-style evaluation setup:

- `n = 8`
- `temperature = 0.8`
- execution-based major voting

## Results

| Run | Spider Dev | Spider Test | BIRD Dev | Spider-DK | Spider-Syn | Spider-Realistic | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 76.11 | 77.41 | 52.93 | 70.47 | 66.83 | 65.55 | 68.22 |
| Run 2 | 76.21 | 77.32 | 53.26 | 69.16 | 67.50 | 65.94 | 68.23 |
| Run 3 | 77.08 | 77.08 | 52.93 | 68.22 | 67.50 | 66.54 | 68.23 |
| Best of Runs 1-3 | 77.08 | 77.41 | 53.26 | 70.47 | 67.50 | 66.54 | 68.71 |

## Notes

- The 6-task mean is very stable across runs: `68.22`, `68.23`, `68.23`.
- Per-dataset variance is larger, which is expected because this evaluation is stochastic.
- The biggest movement across these three runs is on `Spider-DK`.

## Source Artifacts

- Run 1 Table 1: `/home/jovyan/work/minerva/sql-benchmark/summary_table1.tsv`
- Run 1 Table 2: `/home/jovyan/work/minerva/sql-benchmark/sql_r1_3b_released_table2/summary.tsv`
- Run 2 Table 1: `/home/jovyan/work/minerva/sql-benchmark/sql_r1_3b_released_rerun_table1/summary.tsv`
- Run 2 Table 2: `/home/jovyan/work/minerva/sql-benchmark/sql_r1_3b_released_rerun_table2/summary.tsv`
- Run 3 Table 1: `/home/jovyan/work/minerva/sql-benchmark/sql_r1_3b_released_rerun3_table1/summary.tsv`
- Run 3 Table 2: `/home/jovyan/work/minerva/sql-benchmark/sql_r1_3b_released_rerun3_table2/summary.tsv`

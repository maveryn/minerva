# SQL-E1 Results

## Main Six-Benchmark SQL Results

Execution accuracy (`EX`, %) on Spider/BIRD-style SQL benchmarks:

| Model | Spider Dev | Spider Test | BIRD Dev | Spider-DK | Spider-Syn | Spider-Realistic | Avg (6 SQL) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-3B-Instruct (base) | 77.37 | 77.60 | 50.33 | 65.61 | 67.21 | 69.69 | 67.97 |
| SQL-R1-3B (released, best of 3 eval runs) | 77.08 | 77.41 | 53.26 | 70.47 | 67.50 | 66.54 | 68.71 |
| GRPO Qwen2.5-Coder-3B (best step 380) | 77.47 | 78.53 | 54.24 | 68.97 | 67.89 | 68.70 | 69.30 |
| MinervaRL Qwen2.5-Coder-3B (best step 400) | 78.24 | 79.27 | 52.54 | 69.16 | 70.70 | 69.29 | 69.87 |

Main comparison points:

- `GRPO` improves over the base model on the 6-benchmark average by `+1.33` points.
- `MinervaRL` improves over the base model by `+1.90` points.
- `MinervaRL` improves over the released `SQL-R1-3B` best-of-3 row by `+1.16` points.
- `MinervaRL` improves over our `GRPO` run by `+0.57` points.

## LiveSQLBench Base-Full-v1

Metric: official `Overall Accuracy` (task-pass metric, not Spider/BIRD `EX`).

| Model | Overall Accuracy | Execution Errors | Timeouts | Assertion Errors | Total Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-3B-Instruct (base) | 3.67 | 475 | 0 | 103 | 578 |
| SQL-R1-3B (released) | 6.33 | 388 | 0 | 174 | 562 |
| GRPO Qwen2.5-Coder-3B (best step 380) | 6.33 | 385 | 0 | 177 | 562 |
| MinervaRL Qwen2.5-Coder-3B (best step 400) | 6.67 | 381 | 0 | 179 | 560 |

## LiveSQLBench Base-Lite

Metric: official `Overall Accuracy` (task-pass metric, not Spider/BIRD `EX`).

| Model | Overall Accuracy | Execution Errors | Timeouts | Assertion Errors | Total Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-3B-Instruct (base) | 5.93 | 217 | 0 | 37 | 254 |
| SQL-R1-3B (released) | 7.04 | 141 | 0 | 110 | 251 |
| GRPO Qwen2.5-Coder-3B (best step 380) | 7.41 | 131 | 0 | 119 | 250 |
| MinervaRL Qwen2.5-Coder-3B (best step 400) | 7.41 | 135 | 0 | 115 | 250 |

## Notes

- The released `SQL-R1-3B` six-benchmark row uses the best score per dataset across three repeated self-consistency evaluations.
- The LiveSQLBench tables use single runs per model.
- LiveSQLBench numbers should not be averaged together with Spider/BIRD `EX` because the metric is different.

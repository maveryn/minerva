# LLMBench Eval12 Rebuttal Statistics

Generated with `analyze_llmbench_eval12.py` using `2000` bootstrap resamples and seed `8809`.

All values are percentages. Delta is `MinervaRL - GRPO` in percentage points.
Intervals are nonparametric paired bootstrap percentile intervals over evaluation examples or grouped examples for aggregate metrics.

Source root: `/home/jovyan/work/llmbench/runs`.

## 12-Column Average

| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Llama-8B | 56.0 | 60.2 | 4.2 | [3.5, 5.1] |
| Llama-3B | 42.1 | 48.3 | 6.2 | [5.3, 7.2] |
| Qwen-8B | 46.9 | 53.4 | 6.4 | [5.6, 7.7] |
| Qwen-4B | 47.6 | 48.0 | 0.4 | [-0.8, 1.3] |

## Task-Family Averages

### Choice/reasoning

| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Llama-8B | 73.4 | 74.3 | 0.9 | [-0.1, 1.8] |
| Llama-3B | 69.1 | 67.9 | -1.2 | [-2.1, -0.2] |
| Qwen-8B | 70.5 | 77.8 | 7.4 | [6.4, 8.3] |
| Qwen-4B | 74.6 | 71.4 | -3.2 | [-4.3, -2.3] |

### Taxonomy mapping

| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Llama-8B | 40.4 | 49.9 | 9.6 | [7.8, 11.4] |
| Llama-3B | 18.7 | 32.1 | 13.4 | [11.9, 14.8] |
| Qwen-8B | 27.6 | 34.7 | 7.2 | [5.7, 8.6] |
| Qwen-4B | 26.3 | 29.5 | 3.2 | [2.0, 4.4] |

### Structured extraction/scoring

| Backbone | GRPO | MinervaRL | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Llama-8B | 58.1 | 60.1 | 2.0 | [0.9, 3.2] |
| Llama-3B | 44.6 | 49.5 | 4.9 | [3.1, 6.9] |
| Qwen-8B | 48.3 | 53.6 | 5.3 | [3.7, 7.9] |
| Qwen-4B | 48.5 | 48.8 | 0.4 | [-2.3, 2.2] |

## Per-Task Paired Deltas

| Backbone | Task | GRPO | MinervaRL | Delta | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-8B | CKT | 71.9 | 73.9 | 2.0 | [0.5, 3.4] |
| Llama-8B | CyberMetric | 85.4 | 84.2 | -1.1 | [-2.5, 0.2] |
| Llama-8B | SOCEval | 63.0 | 64.7 | 1.7 | [-0.3, 3.7] |
| Llama-8B | RCM | 66.3 | 68.8 | 2.4 | [1.0, 3.8] |
| Llama-8B | VSP | 82.6 | 87.6 | 5.0 | [4.4, 5.7] |
| Llama-8B | ATE | 32.0 | 48.4 | 16.4 | [12.6, 20.4] |
| Llama-8B | RMS | 30.9 | 42.1 | 11.2 | [8.0, 14.6] |
| Llama-8B | ElasticRule | 32.2 | 40.5 | 8.3 | [4.2, 12.7] |
| Llama-8B | APTNER | 32.7 | 34.1 | 1.4 | [0.5, 2.5] |
| Llama-8B | LANCE | 86.2 | 84.6 | -1.5 | [-5.5, 2.6] |
| Llama-8B | AnnoCTR | 49.6 | 50.3 | 0.7 | [-1.6, 2.9] |
| Llama-8B | AZERG | 39.5 | 43.7 | 4.1 | [2.4, 7.8] |
| Llama-3B | CKT | 71.2 | 71.0 | -0.2 | [-1.6, 1.1] |
| Llama-3B | CyberMetric | 77.8 | 78.1 | 0.3 | [-1.0, 1.7] |
| Llama-3B | SOCEval | 58.3 | 54.6 | -3.7 | [-5.8, -1.3] |
| Llama-3B | RCM | 48.9 | 57.2 | 8.3 | [6.8, 9.9] |
| Llama-3B | VSP | 56.5 | 77.1 | 20.6 | [18.9, 22.4] |
| Llama-3B | ATE | 5.2 | 21.8 | 16.6 | [13.4, 20.0] |
| Llama-3B | RMS | 15.4 | 29.3 | 13.9 | [11.0, 16.8] |
| Llama-3B | ElasticRule | 5.3 | 20.1 | 14.8 | [11.3, 18.3] |
| Llama-3B | APTNER | 27.5 | 16.7 | -10.8 | [-12.5, -9.3] |
| Llama-3B | LANCE | 70.6 | 82.2 | 11.7 | [3.6, 19.7] |
| Llama-3B | AnnoCTR | 38.5 | 37.9 | -0.7 | [-2.5, 1.2] |
| Llama-3B | AZERG | 29.7 | 33.7 | 4.0 | [1.7, 6.9] |
| Qwen-8B | CKT | 69.8 | 77.8 | 8.0 | [6.4, 9.7] |
| Qwen-8B | CyberMetric | 72.7 | 88.2 | 15.5 | [13.5, 17.6] |
| Qwen-8B | SOCEval | 69.0 | 67.5 | -1.4 | [-2.8, -0.1] |
| Qwen-8B | RCM | 60.5 | 64.8 | 4.3 | [2.8, 5.8] |
| Qwen-8B | VSP | 68.8 | 79.4 | 10.6 | [9.7, 11.6] |
| Qwen-8B | ATE | 23.6 | 32.0 | 8.4 | [5.6, 11.4] |
| Qwen-8B | RMS | 8.2 | 20.1 | 12.0 | [8.8, 14.7] |
| Qwen-8B | ElasticRule | 18.1 | 22.0 | 3.9 | [-0.2, 7.9] |
| Qwen-8B | APTNER | 37.7 | 37.4 | -0.3 | [-1.5, 0.9] |
| Qwen-8B | LANCE | 66.1 | 74.9 | 8.8 | [1.0, 19.6] |
| Qwen-8B | AnnoCTR | 37.8 | 43.0 | 5.2 | [3.1, 8.2] |
| Qwen-8B | AZERG | 31.1 | 33.1 | 2.0 | [0.2, 5.6] |
| Qwen-4B | CKT | 74.2 | 70.0 | -4.2 | [-5.7, -2.6] |
| Qwen-4B | CyberMetric | 85.8 | 80.0 | -5.7 | [-7.5, -4.0] |
| Qwen-4B | SOCEval | 63.9 | 64.0 | 0.2 | [-1.8, 2.1] |
| Qwen-4B | RCM | 60.8 | 59.9 | -0.9 | [-2.1, 0.2] |
| Qwen-4B | VSP | 88.1 | 80.0 | -8.1 | [-8.9, -7.2] |
| Qwen-4B | ATE | 20.2 | 25.4 | 5.2 | [2.8, 7.8] |
| Qwen-4B | RMS | 3.8 | 5.1 | 1.3 | [-0.6, 3.1] |
| Qwen-4B | ElasticRule | 20.4 | 27.5 | 7.2 | [3.5, 11.1] |
| Qwen-4B | APTNER | 32.4 | 28.0 | -4.4 | [-5.9, -2.9] |
| Qwen-4B | LANCE | 58.2 | 56.7 | -1.5 | [-9.8, 9.3] |
| Qwen-4B | AnnoCTR | 39.6 | 50.4 | 10.7 | [4.1, 12.8] |
| Qwen-4B | AZERG | 24.0 | 29.1 | 5.1 | [0.3, 8.0] |

## Notes

- The script verifies every computed point estimate against `evalall-summary.json` before writing the report.
- `Avg.` is recomputed over the 12 paper columns and intentionally does not use `overall_avg`, which includes additional tasks.
- PRISM/LANCE CIs resample report/type groups within each IoC subtype, then average subtype F1 values.
- AZERG and AnnoCTR CIs resample examples within each subtask, using macro-F1 for T2 and exact accuracy for T1/T3/T4.

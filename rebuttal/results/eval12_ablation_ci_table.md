# Eval12 Llama-8B Ablation Tables

Generated with `build_eval12_ablation_ci_table.py` using `2000` paired bootstrap resamples and seed `8809`.

All scores are actual Eval12 task scores in percent. The CI sign summary compares `Full MinervaRL - control/ablation` over the 12 displayed tasks.

## Task Scores

| Model | CKT | CyberMetric | SOCEval | RCM | VSP | ATE | RMS | ElasticRule | APTNER | LANCE | AnnoCTR | AZERG | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO / no ACR | 71.9 | 85.4 | 63.0 | 66.3 | 82.6 | 32.0 | 30.9 | 32.2 | 32.7 | 86.2 | 49.6 | 39.5 | 56.0 |
| GRPO-12 | 71.2 | 85.0 | 64.6 | 66.8 | 76.3 | 31.4 | 17.0 | 31.7 | 36.3 | 82.6 | 48.8 | 35.4 | 53.9 |
| Answer-only SFT | 76.3 | 85.9 | 65.2 | 64.7 | 64.8 | 36.6 | 17.2 | 34.7 | 40.7 | 56.5 | 45.3 | 43.2 | 52.6 |
| In-loop answer-only SFT | 75.6 | 84.5 | 66.8 | 65.8 | 79.2 | 38.4 | 28.6 | 33.3 | 38.7 | 78.1 | 53.9 | 50.5 | 57.8 |
| No EMA teacher | 72.5 | 83.7 | 62.4 | 65.3 | 85.9 | 42.2 | 43.0 | 39.8 | 33.1 | 75.1 | 43.7 | 40.2 | 57.2 |
| No TextCNN filter | 71.2 | 85.3 | 63.6 | 64.0 | 78.6 | 45.0 | 36.5 | 41.4 | 36.0 | 82.5 | 47.5 | 41.1 | 57.7 |
| No filtering | 73.1 | 84.3 | 63.3 | 67.2 | 83.9 | 40.0 | 37.2 | 38.2 | 34.2 | 81.2 | 45.9 | 44.6 | 57.8 |
| Full MinervaRL | 73.9 | 84.2 | 64.7 | 68.8 | 87.6 | 48.4 | 42.1 | 40.5 | 34.1 | 84.6 | 50.3 | 43.7 | 60.2 |

## Paired Bootstrap CI Sign Summary

| Control / ablation | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO / no ACR | 10/12 | 2/12 | 8/12 | 0/12 | 4/12 |
| GRPO-12 | 10/12 | 2/12 | 7/12 | 1/12 | 4/12 |
| Answer-only SFT | 8/12 | 4/12 | 7/12 | 3/12 | 2/12 |
| In-loop answer-only SFT | 6/12 | 6/12 | 6/12 | 4/12 | 2/12 |
| No EMA teacher | 11/12 | 1/12 | 8/12 | 0/12 | 4/12 |
| No TextCNN filter | 9/12 | 3/12 | 7/12 | 1/12 | 4/12 |
| No filtering | 9/12 | 3/12 | 7/12 | 0/12 | 5/12 |

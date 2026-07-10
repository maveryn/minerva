# Eval12 Athena Non-Mini Subset Check

Generated from saved LLMBench row-level outputs by removing AthenaBench-Mini validation instances from the corresponding Athena-derived Eval12 columns.

All deltas are `MinervaRL - GRPO` in percentage points. Intervals are paired nonparametric bootstrap 95% CIs over the remaining evaluation examples, using 2000 bootstrap resamples and seed components keyed by backbone/task.

These intervals quantify evaluation-sample uncertainty, not training-seed variance.

## Selection-Subset Overlap

AthenaBench-Mini is not instance-disjoint from these final Athena-derived Eval12 columns; it is a subset of the full files by prompt hash or CVE ID.

| Task | Full Eval12 examples | AthenaBench-Mini selected | Remaining examples |
| --- | ---: | ---: | ---: |
| CKT | 3000 | 300 | 2700 |
| RCM | 2000 | 200 | 1800 |
| VSP | 2000 | 200 | 1800 |
| ATE | 500 | 100 | 400 |
| RMS | 500 | 100 | 400 |

## Per-Task Non-Mini Deltas

| Task | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |
| --- | ---: | ---: | ---: | ---: |
| CKT | +2.0 [0.5, 3.7] | -0.3 [-1.7, 1.1] | +8.1 [6.5, 9.7] | -4.6 [-6.3, -3.0] |
| RCM | +2.4 [0.9, 3.9] | +8.5 [6.9, 10.1] | +4.3 [2.8, 5.8] | -1.2 [-2.4, 0.0] |
| VSP | +5.2 [4.5, 6.0] | +20.7 [18.9, 22.6] | +10.8 [9.8, 11.7] | -8.0 [-8.9, -7.1] |
| ATE | +17.2 [12.8, 22.0] | +17.0 [13.2, 20.8] | +8.5 [5.2, 12.0] | +6.2 [3.7, 9.2] |
| RMS | +11.7 [8.0, 15.4] | +13.7 [10.3, 17.0] | +11.5 [8.1, 14.5] | +0.9 [-1.2, 2.9] |

## Five-Task Macro-Average

This average is only over the five Athena-derived columns above after Mini instances are removed.

| Backbone | Delta | 95% CI |
| --- | ---: | ---: |
| Llama-8B | +7.7 | [6.4, 9.0] |
| Llama-3B | +11.9 | [10.8, 13.1] |
| Qwen-8B | +8.6 | [7.6, 9.7] |
| Qwen-4B | -1.3 | [-2.1, -0.5] |

## Reporting Note

Use this only for the checkpoint-selection-overlap response. It shows that the Llama-8B, Llama-3B, and Qwen-8B Athena-derived results are not explained only by the selected Mini subset, while Qwen-4B should be described cautiously because GRPO is stronger on this Athena-only non-mini subset.

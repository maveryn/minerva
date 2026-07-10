# Review 1 Response Draft

## Statistical uncertainty and heterogeneous averaging

We agree that the original submission relied too heavily on a heterogeneous 12-task average. We will revise the paper so that the main evidence is per-task and paired, with the average retained only as a compact comparability number where needed.

For the revised analysis, we computed paired nonparametric bootstrap confidence intervals from the saved row-level Eval12 outputs. For each task and model pair, we resample the evaluation examples with replacement at the original task size, use the same sampled examples for both models, recompute the task metric and the paired delta, and repeat this for 2000 bootstrap replicates. The Eval12 task denominators and metric definitions are reported in Appendix Table 9. We report percentile 95% CIs for `MinervaRL - baseline` in percentage points. For grouped metrics, the resampling follows the original metric aggregation: LANCE resamples report/type groups within each IoC subtype, and AnnoCTR/AZERG resample examples within each subtask before macro-averaging. These intervals quantify evaluation-sample uncertainty and paired model-comparison uncertainty; they do not estimate training-seed variation.

### Table R1.1: Per-task paired CIs versus GRPO

All entries are `MinervaRL - GRPO` in percentage points with paired bootstrap 95% CIs.

| Task | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |
| --- | ---: | ---: | ---: | ---: |
| CKT | +2.0 [0.6, 3.5] | -0.2 [-1.6, 1.1] | +8.0 [6.5, 9.5] | -4.2 [-5.8, -2.6] |
| CyberMetric | -1.1 [-2.5, 0.3] | +0.3 [-1.1, 1.5] | +15.5 [13.5, 17.5] | -5.7 [-7.4, -4.0] |
| SOCEval | +1.7 [-0.4, 3.8] | -3.7 [-5.8, -1.4] | -1.4 [-2.7, -0.1] | +0.2 [-1.6, 2.1] |
| RCM | +2.4 [1.1, 3.8] | +8.3 [6.8, 9.8] | +4.3 [2.9, 5.8] | -0.9 [-2.1, 0.2] |
| VSP | +5.0 [4.3, 5.7] | +20.6 [18.9, 22.4] | +10.6 [9.7, 11.7] | -8.1 [-9.0, -7.3] |
| ATE | +16.4 [12.6, 20.4] | +16.6 [13.2, 20.2] | +8.4 [5.8, 11.4] | +5.2 [2.8, 7.6] |
| RMS | +11.2 [7.8, 14.4] | +13.9 [11.1, 16.9] | +12.0 [9.1, 14.8] | +1.3 [-0.7, 3.2] |
| ElasticRule | +8.3 [3.9, 12.7] | +14.8 [11.6, 18.3] | +3.9 [-0.2, 7.9] | +7.2 [3.7, 10.9] |
| APTNER | +1.4 [0.5, 2.4] | -10.8 [-12.3, -9.3] | -0.3 [-1.5, 1.0] | -4.4 [-5.9, -2.9] |
| LANCE | -1.5 [-5.7, 2.2] | +11.7 [2.7, 19.3] | +8.8 [0.8, 19.9] | -1.5 [-10.5, 9.3] |
| AnnoCTR | +0.7 [-1.6, 2.9] | -0.7 [-2.4, 1.1] | +5.2 [3.2, 8.1] | +10.7 [4.4, 12.9] |
| AZERG | +4.1 [2.3, 7.7] | +4.0 [1.5, 7.0] | +2.0 [0.1, 5.6] | +5.1 [0.3, 7.8] |

This table gives a more direct view than the heterogeneous average. Against GRPO, MinervaRL has positive point deltas on 34/48 backbone-task comparisons. The paired 95% CI is strictly positive on 28/48 comparisons, strictly negative on 7/48, and overlaps zero on 13/48. We will therefore state the improvements in task-level terms and qualify cases where the paired interval overlaps zero or is negative, especially for Qwen-4B and for specific extraction/MCQ tasks.

### Table R1.2: Per-task paired CI sign summary versus all baselines

Point wins/losses count the raw task-level delta sign over 48 backbone-task comparisons. `CI positive` means the entire paired 95% CI is above zero, `CI negative` means the entire interval is below zero, and `CI overlaps 0` means the bootstrap interval does not support a separated positive or negative comparison at this level.

| Baseline | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 42/48 | 6/48 | 35/48 | 2/48 | 11/48 |
| STaR | 38/48 | 10/48 | 35/48 | 6/48 | 7/48 |
| DART | 38/48 | 10/48 | 34/48 | 3/48 | 11/48 |
| GRPO | 34/48 | 14/48 | 28/48 | 7/48 | 13/48 |
| LUFFY | 32/48 | 16/48 | 20/48 | 6/48 | 22/48 |

This summary addresses the aggregation concern without making the heterogeneous average the primary evidence. The results show that MinervaRL is not uniformly better on every task, but its gains are not driven by only a few aggregate columns: it has more positive than negative point deltas against every baseline, and many of those task-level deltas remain positive under paired bootstrap uncertainty. We will revise the claims to reflect this pattern rather than implying uniform dominance.

## Rollout-aware held-out best-of-k

We also added held-out rollout-aware evaluation under matched sampling budgets for GRPO and MinervaRL. Each task cell below is `GRPO/MinervaRL` and reports the actual task score in percent for that backbone and best-of-k budget. For exact-match, MCQ, and taxonomy tasks this is verifier success/best-of-k accuracy; for graded, extraction, and multi-label tasks, we select the best scored sample per example and recompute the original Eval12 metric.

| Backbone | k | CKT | CyberMetric | SOCEval | RCM | VSP | ATE | RMS | ElasticRule | APTNER | LANCE | AnnoCTR | AZERG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-8B | 1 | 71.3/71.4 | 85.7/85.2 | 63.2/65.4 | 66.5/67.2 | 82.4/87.4 | 30.8/46.8 | 31.3/38.9 | 31.7/40.3 | 32.6/32.5 | 87.0/81.6 | 52.9/47.6 | 39.3/42.5 |
| Llama-8B | 2 | 75.4/80.5 | 89.0/89.7 | 65.8/69.3 | 67.2/70.3 | 83.9/89.9 | 32.2/51.6 | 33.8/47.1 | 32.4/45.1 | 36.0/36.2 | 88.0/85.3 | 55.7/52.9 | 47.8/45.8 |
| Llama-8B | 4 | 77.9/86.0 | 91.4/92.5 | 69.7/76.7 | 68.4/72.9 | 85.1/91.3 | 35.0/55.4 | 35.0/52.9 | 34.0/50.5 | 38.6/40.0 | 89.2/87.8 | 58.8/57.4 | 49.7/50.8 |
| Llama-8B | 8 | 80.0/89.4 | 92.9/94.8 | 71.6/80.1 | 69.2/74.8 | 86.0/92.3 | 35.4/59.0 | 36.0/58.4 | 34.3/52.3 | 41.0/42.8 | 90.1/88.8 | 61.8/64.5 | 51.4/53.9 |
| Llama-3B | 1 | 68.6/68.2 | 77.7/79.0 | 60.5/55.4 | 47.9/55.5 | 54.6/74.6 | 5.4/22.2 | 17.1/28.2 | 5.6/20.6 | 26.8/15.1 | 66.2/76.5 | 36.9/42.5 | 25.0/43.0 |
| Llama-3B | 2 | 73.1/76.2 | 80.5/82.5 | 63.0/60.0 | 48.9/58.0 | 70.0/81.9 | 5.6/25.8 | 22.3/33.4 | 6.5/21.1 | 30.7/15.1 | 77.6/83.3 | 43.5/47.6 | 47.9/52.3 |
| Llama-3B | 4 | 81.6/85.7 | 83.9/86.2 | 67.2/66.6 | 49.5/60.1 | 79.8/86.8 | 5.8/29.0 | 24.5/37.2 | 7.2/23.1 | 34.0/17.7 | 81.8/88.1 | 46.7/52.1 | 51.1/58.9 |
| Llama-3B | 8 | 82.9/89.5 | 85.7/88.8 | 68.8/69.8 | 50.3/62.8 | 85.1/89.7 | 5.8/31.2 | 26.3/41.6 | 7.9/25.7 | 37.0/21.3 | 87.1/89.7 | 51.6/58.9 | 55.2/63.0 |
| Qwen-8B | 1 | 69.0/76.0 | 84.5/87.4 | 68.4/66.1 | 60.0/64.0 | 69.5/78.8 | 22.6/31.2 | 7.9/18.4 | 24.1/24.3 | 36.2/33.1 | 66.4/69.2 | 45.1/45.3 | 38.4/32.3 |
| Qwen-8B | 2 | 82.3/83.1 | 92.1/92.5 | 74.7/74.0 | 61.5/67.0 | 73.8/82.6 | 24.6/33.2 | 8.3/25.2 | 26.6/30.8 | 45.1/43.6 | 74.7/79.3 | 53.5/53.9 | 49.2/47.9 |
| Qwen-8B | 4 | 88.2/87.8 | 94.9/94.5 | 80.9/81.1 | 62.3/69.5 | 81.6/86.0 | 25.2/35.2 | 8.6/31.0 | 28.5/34.0 | 51.7/49.6 | 83.2/85.2 | 56.8/59.1 | 53.4/55.9 |
| Qwen-8B | 8 | 92.8/91.1 | 96.8/96.0 | 83.7/84.8 | 62.8/71.5 | 91.6/88.4 | 25.8/37.4 | 9.0/35.7 | 29.4/37.7 | 55.3/53.9 | 92.3/91.8 | 59.5/60.9 | 58.1/61.8 |
| Qwen-4B | 1 | 62.7/72.2 | 54.1/82.8 | 62.1/61.8 | 60.2/59.5 | 88.1/80.0 | 20.6/24.4 | 12.2/7.9 | 22.0/26.6 | 21.8/32.2 | 45.4/57.4 | 44.8/43.4 | 38.5/33.9 |
| Qwen-4B | 2 | 80.4/81.4 | 90.6/90.3 | 69.2/69.5 | 61.5/62.5 | 89.6/82.1 | 22.0/26.6 | 14.7/9.4 | 25.0/29.2 | 34.4/39.3 | 64.2/69.0 | 54.3/58.9 | 48.6/47.5 |
| Qwen-4B | 4 | 86.3/86.9 | 95.0/94.2 | 75.2/75.8 | 62.2/64.5 | 90.1/84.3 | 22.4/29.4 | 15.1/12.0 | 25.7/30.8 | 41.3/46.0 | 76.7/78.1 | 59.2/63.4 | 53.5/59.4 |
| Qwen-4B | 8 | 89.9/91.9 | 96.3/96.5 | 80.0/80.5 | 62.8/65.4 | 90.8/89.9 | 22.4/32.6 | 15.9/16.5 | 26.4/32.9 | 46.0/51.3 | 89.7/87.8 | 65.6/73.5 | 58.4/62.9 |

We also computed paired bootstrap CIs for these rollout-aware scores. Counts below are over 48 backbone-task comparisons at each `k`; `CI positive` means the paired 95% CI for `MinervaRL - GRPO` is entirely above zero.

| k | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32/48 | 16/48 | 27/48 | 10/48 | 11/48 |
| 2 | 36/48 | 12/48 | 26/48 | 6/48 | 16/48 |
| 4 | 38/48 | 10/48 | 31/48 | 4/48 | 13/48 |
| 8 | 39/48 | 9/48 | 31/48 | 6/48 | 11/48 |

## Component-wise Llama-8B ablations

We also added final Eval12 Llama-8B component controls. `GRPO / no ACR` removes the answer-conditioned rationale path, `GRPO-12` controls for extra rollout compute, and `In-loop answer-only SFT` matches the in-loop distillation schedule while replacing ACR rationales with direct answer-only SFT targets.

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

For each control, we computed paired bootstrap CIs over the same 12 task outputs for `Full MinervaRL - control/ablation`.

| Control / ablation | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO / no ACR | 10/12 | 2/12 | 8/12 | 0/12 | 4/12 |
| GRPO-12 | 10/12 | 2/12 | 7/12 | 1/12 | 4/12 |
| Answer-only SFT | 8/12 | 4/12 | 7/12 | 3/12 | 2/12 |
| In-loop answer-only SFT | 6/12 | 6/12 | 6/12 | 4/12 | 2/12 |
| No EMA teacher | 11/12 | 1/12 | 8/12 | 0/12 | 4/12 |
| No TextCNN filter | 9/12 | 3/12 | 7/12 | 1/12 | 4/12 |
| No filtering | 9/12 | 3/12 | 7/12 | 0/12 | 5/12 |

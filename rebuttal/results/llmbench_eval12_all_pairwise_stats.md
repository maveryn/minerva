# LLMBench Eval12 All-Pair Comparisons

Generated with `analyze_llmbench_eval12_all_pairs.py` using `2000` bootstrap resamples and seed `8809`.

All values are percentages. Delta is `right model - left model` over the 12 Eval12 paper columns.
Task wins/losses count point-estimate per-task deltas across the 12 columns.

## 12-Column Averages

| Backbone | Model | Avg. |
| --- | --- | ---: |
| Llama-8B | Base | 48.6 |
| Llama-8B | STaR | 49.7 |
| Llama-8B | DART | 53.7 |
| Llama-8B | GRPO | 56.0 |
| Llama-8B | LUFFY | 57.0 |
| Llama-8B | MinervaRL | 60.2 |
| Llama-3B | Base | 33.1 |
| Llama-3B | STaR | 37.3 |
| Llama-3B | DART | 44.0 |
| Llama-3B | GRPO | 42.1 |
| Llama-3B | LUFFY | 45.4 |
| Llama-3B | MinervaRL | 48.3 |
| Qwen-8B | Base | 37.6 |
| Qwen-8B | STaR | 36.8 |
| Qwen-8B | DART | 42.9 |
| Qwen-8B | GRPO | 46.9 |
| Qwen-8B | LUFFY | 53.2 |
| Qwen-8B | MinervaRL | 53.4 |
| Qwen-4B | Base | 27.6 |
| Qwen-4B | STaR | 45.4 |
| Qwen-4B | DART | 39.4 |
| Qwen-4B | GRPO | 47.6 |
| Qwen-4B | LUFFY | 44.0 |
| Qwen-4B | MinervaRL | 48.0 |

## MinervaRL Against Each Baseline

| Backbone | Comparison | Delta | Paired 95% CI | Task >0 | Task <0 | Task =0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama-8B | MinervaRL - Base | 11.6 | [10.7, 12.5] | 10 | 2 | 0 |
| Llama-8B | MinervaRL - STaR | 10.5 | [9.8, 11.3] | 12 | 0 | 0 |
| Llama-8B | MinervaRL - DART | 6.6 | [5.7, 7.5] | 12 | 0 | 0 |
| Llama-8B | MinervaRL - GRPO | 4.2 | [3.5, 5.1] | 10 | 2 | 0 |
| Llama-8B | MinervaRL - LUFFY | 3.3 | [2.5, 4.0] | 11 | 1 | 0 |
| Llama-3B | MinervaRL - Base | 15.2 | [14.5, 16.1] | 9 | 3 | 0 |
| Llama-3B | MinervaRL - STaR | 11.0 | [10.3, 11.8] | 11 | 1 | 0 |
| Llama-3B | MinervaRL - DART | 4.3 | [3.5, 5.0] | 8 | 4 | 0 |
| Llama-3B | MinervaRL - GRPO | 6.2 | [5.3, 7.2] | 8 | 4 | 0 |
| Llama-3B | MinervaRL - LUFFY | 2.9 | [2.0, 3.9] | 8 | 4 | 0 |
| Qwen-8B | MinervaRL - Base | 15.7 | [14.6, 17.1] | 12 | 0 | 0 |
| Qwen-8B | MinervaRL - STaR | 16.5 | [15.4, 17.6] | 11 | 1 | 0 |
| Qwen-8B | MinervaRL - DART | 10.5 | [9.5, 11.8] | 8 | 4 | 0 |
| Qwen-8B | MinervaRL - GRPO | 6.4 | [5.6, 7.7] | 10 | 2 | 0 |
| Qwen-8B | MinervaRL - LUFFY | 0.1 | [-0.7, 0.8] | 6 | 6 | 0 |
| Qwen-4B | MinervaRL - Base | 20.4 | [19.0, 21.2] | 11 | 1 | 0 |
| Qwen-4B | MinervaRL - STaR | 2.6 | [1.5, 3.5] | 4 | 8 | 0 |
| Qwen-4B | MinervaRL - DART | 8.6 | [7.3, 9.6] | 10 | 2 | 0 |
| Qwen-4B | MinervaRL - GRPO | 0.4 | [-0.8, 1.3] | 6 | 6 | 0 |
| Qwen-4B | MinervaRL - LUFFY | 4.0 | [3.1, 4.9] | 7 | 5 | 0 |

## MinervaRL Against Strongest Non-MinervaRL Baseline

| Backbone | Strongest baseline | Baseline Avg. | MinervaRL Avg. | Delta | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-8B | LUFFY | 57.0 | 60.2 | 3.3 | [2.5, 4.0] |
| Llama-3B | LUFFY | 45.4 | 48.3 | 2.9 | [2.0, 3.9] |
| Qwen-8B | LUFFY | 53.2 | 53.4 | 0.1 | [-0.7, 0.8] |
| Qwen-4B | GRPO | 47.6 | 48.0 | 0.4 | [-0.8, 1.3] |

## All Unordered Model Pairs

| Backbone | Comparison | Delta | Task >0 | Task <0 | Task =0 |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-8B | STaR - Base | 1.1 | 6 | 6 | 0 |
| Llama-8B | DART - Base | 5.0 | 6 | 6 | 0 |
| Llama-8B | GRPO - Base | 7.4 | 8 | 4 | 0 |
| Llama-8B | LUFFY - Base | 8.3 | 10 | 2 | 0 |
| Llama-8B | MinervaRL - Base | 11.6 | 10 | 2 | 0 |
| Llama-8B | DART - STaR | 3.9 | 9 | 3 | 0 |
| Llama-8B | GRPO - STaR | 6.3 | 11 | 1 | 0 |
| Llama-8B | LUFFY - STaR | 7.3 | 12 | 0 | 0 |
| Llama-8B | MinervaRL - STaR | 10.5 | 12 | 0 | 0 |
| Llama-8B | GRPO - DART | 2.4 | 10 | 2 | 0 |
| Llama-8B | LUFFY - DART | 3.3 | 11 | 1 | 0 |
| Llama-8B | MinervaRL - DART | 6.6 | 12 | 0 | 0 |
| Llama-8B | LUFFY - GRPO | 0.9 | 8 | 4 | 0 |
| Llama-8B | MinervaRL - GRPO | 4.2 | 10 | 2 | 0 |
| Llama-8B | MinervaRL - LUFFY | 3.3 | 11 | 1 | 0 |
| Llama-3B | STaR - Base | 4.2 | 5 | 7 | 0 |
| Llama-3B | DART - Base | 11.0 | 8 | 4 | 0 |
| Llama-3B | GRPO - Base | 9.0 | 9 | 3 | 0 |
| Llama-3B | LUFFY - Base | 12.3 | 7 | 5 | 0 |
| Llama-3B | MinervaRL - Base | 15.2 | 9 | 3 | 0 |
| Llama-3B | DART - STaR | 6.8 | 10 | 2 | 0 |
| Llama-3B | GRPO - STaR | 4.8 | 11 | 1 | 0 |
| Llama-3B | LUFFY - STaR | 8.1 | 10 | 2 | 0 |
| Llama-3B | MinervaRL - STaR | 11.0 | 11 | 1 | 0 |
| Llama-3B | GRPO - DART | -2.0 | 3 | 9 | 0 |
| Llama-3B | LUFFY - DART | 1.4 | 8 | 4 | 0 |
| Llama-3B | MinervaRL - DART | 4.3 | 8 | 4 | 0 |
| Llama-3B | LUFFY - GRPO | 3.3 | 7 | 5 | 0 |
| Llama-3B | MinervaRL - GRPO | 6.2 | 8 | 4 | 0 |
| Llama-3B | MinervaRL - LUFFY | 2.9 | 8 | 4 | 0 |
| Qwen-8B | STaR - Base | -0.8 | 6 | 6 | 0 |
| Qwen-8B | DART - Base | 5.2 | 9 | 3 | 0 |
| Qwen-8B | GRPO - Base | 9.3 | 10 | 2 | 0 |
| Qwen-8B | LUFFY - Base | 15.6 | 12 | 0 | 0 |
| Qwen-8B | MinervaRL - Base | 15.7 | 12 | 0 | 0 |
| Qwen-8B | DART - STaR | 6.0 | 10 | 2 | 0 |
| Qwen-8B | GRPO - STaR | 10.1 | 10 | 2 | 0 |
| Qwen-8B | LUFFY - STaR | 16.4 | 10 | 2 | 0 |
| Qwen-8B | MinervaRL - STaR | 16.5 | 11 | 1 | 0 |
| Qwen-8B | GRPO - DART | 4.1 | 7 | 5 | 0 |
| Qwen-8B | LUFFY - DART | 10.4 | 7 | 5 | 0 |
| Qwen-8B | MinervaRL - DART | 10.5 | 8 | 4 | 0 |
| Qwen-8B | LUFFY - GRPO | 6.3 | 9 | 3 | 0 |
| Qwen-8B | MinervaRL - GRPO | 6.4 | 10 | 2 | 0 |
| Qwen-8B | MinervaRL - LUFFY | 0.1 | 6 | 6 | 0 |
| Qwen-4B | STaR - Base | 17.8 | 11 | 1 | 0 |
| Qwen-4B | DART - Base | 11.8 | 8 | 4 | 0 |
| Qwen-4B | GRPO - Base | 20.0 | 11 | 1 | 0 |
| Qwen-4B | LUFFY - Base | 16.3 | 11 | 1 | 0 |
| Qwen-4B | MinervaRL - Base | 20.4 | 11 | 1 | 0 |
| Qwen-4B | DART - STaR | -6.0 | 3 | 9 | 0 |
| Qwen-4B | GRPO - STaR | 2.2 | 5 | 7 | 0 |
| Qwen-4B | LUFFY - STaR | -1.4 | 4 | 8 | 0 |
| Qwen-4B | MinervaRL - STaR | 2.6 | 4 | 8 | 0 |
| Qwen-4B | GRPO - DART | 8.2 | 8 | 4 | 0 |
| Qwen-4B | LUFFY - DART | 4.6 | 10 | 2 | 0 |
| Qwen-4B | MinervaRL - DART | 8.6 | 10 | 2 | 0 |
| Qwen-4B | LUFFY - GRPO | -3.6 | 5 | 7 | 0 |
| Qwen-4B | MinervaRL - GRPO | 0.4 | 6 | 6 | 0 |
| Qwen-4B | MinervaRL - LUFFY | 4.0 | 7 | 5 | 0 |

## Notes

- Point estimates are recomputed from row-level `*-scored.jsonl` files and validated against saved Eval12/EvalAll summaries where present.
- The main rebuttal should prioritize MinervaRL-vs-GRPO because MinervaRL is introduced as a GRPO extension.
- MinervaRL-vs-Base is useful for showing post-training improvement; MinervaRL-vs-strongest-baseline is useful for aggregate leaderboard claims.
- Full all-pair tables are mainly a supplementary audit, not ideal for the main rebuttal text.
- By default, all unordered model pairs use point estimates only; pass `--all-pair-cis` to bootstrap every pair.

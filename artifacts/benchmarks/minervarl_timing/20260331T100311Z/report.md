# MinervaRL Timing Benchmark

Generated: 2026-03-31T10:21:55.399841+00:00
Steps per run: 10

## Totals

| Run | Training wall s (startup excluded) | Process wall s | Startup/non-step overhead s |
| --- | ---: | ---: | ---: |
| GRPO | 337.9024 | 469.8172 | 131.9149 |
| Noctua | 481.4931 | 654.5329 | 173.0398 |

## Noctua Components

| Component | Sum s |
| --- | ---: |
| `timing_s/acr_build` | 2.8907 |
| `timing_s/acr_distill_sft` | 9.3066 |
| `timing_s/acr_gen` | 36.3803 |
| `timing_s/acr_reward` | 5.8833 |
| `timing_s/gen` | 110.2817 |
| `timing_s/reward` | 2.3079 |
| `timing_s/update_actor` | 191.8049 |

## Overhead

- Noctua minus GRPO: 143.5907 s
- Noctua over GRPO: 42.4947 %

## Files

- Raw GRPO log: `artifacts/benchmarks/minervarl_timing/20260331T100311Z/grpo.log`
- Raw Noctua log: `artifacts/benchmarks/minervarl_timing/20260331T100311Z/noctua.log`
- GRPO summary: `artifacts/benchmarks/minervarl_timing/20260331T100311Z/grpo.summary.json`
- Noctua summary: `artifacts/benchmarks/minervarl_timing/20260331T100311Z/noctua.summary.json`
- Comparison: `artifacts/benchmarks/minervarl_timing/20260331T100311Z/comparison.json`

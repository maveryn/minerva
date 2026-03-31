# STaR-CTI Results

This document summarizes the completed STaR runs saved under `star-artifacts/`
as of 2026-03-26.

## Evaluation protocol

- `Combined avg.` below is the RL-matched checkpoint-selection metric:
  `0.5 * rl_minerva_dev_mean_score + 0.5 * rl_athena_bench_mean_score`
- `All-val mean` is the mean reward across the full 2,150-row validation suite.
- Validation suite composition:
  - `minerva_base_dev`: 1,200 rows
  - Athena CTI: 950 rows total
    - `CKT`: 300
    - `ATE`: 100
    - `RCM`: 200
    - `RMS`: 100
    - `TAA`: 50
    - `VSP`: 200
- Every STaR round uses the same 32,000-example train split.
- `selected_count == round_sft_rows` because one trace is kept at most once per
  train example.
- Both configs were set up for 5 rounds, but only `round_00` to `round_02`
  completed. `round_03/` directories exist for both runs, but they do not
  contain finished `summary.json` and `eval/summary.json` files.

All scores below are shown as percentages for readability, but they come from
the saved reward means in the artifact JSON.

## Best checkpoints

| Model | Best round | Minerva dev | Athena mean | Combined avg. | All-val mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct STaR | `round_00` | 43.0 | 41.5 | 42.3 | 47.7 |
| Qwen3-8B-Base STaR | `round_00` | 34.7 | 30.8 | 32.7 | 38.6 |

Best checkpoint paths:
- Llama-3.1-8B-Instruct STaR:
  `star-artifacts/star_cti_llama31_8b/round_00/checkpoint/hf_model`
- Qwen3-8B-Base STaR:
  `star-artifacts/star_cti_qwen3_8b_base/round_00/checkpoint/hf_model`

## Llama-3.1-8B-Instruct STaR

Run details:
- Launcher: `bash star/scripts/train_star_llama8b.sh --from-scratch`
- Artifact root: `star-artifacts/star_cti_llama31_8b`
- Base model: `meta-llama/Llama-3.1-8B-Instruct`
- Best round by `Combined avg.`: `round_00`

### Round-wise validation and trace selection

| Round | Original success | Rationalization success | Selected traces | Minerva dev | Athena mean | Combined avg. | All-val mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_00` | 1,738 / 32,000 (5.4%) | 27,495 / 32,000 (85.9%) | 27,666 / 32,000 (86.5%) | 43.0 | 41.5 | 42.3 | 47.7 |
| `round_01` | 8,301 / 32,000 (25.9%) | 30,712 / 32,000 (96.0%) | 30,866 / 32,000 (96.5%) | 43.4 | 39.8 | 41.6 | 47.4 |
| `round_02` | 8,686 / 32,000 (27.1%) | 30,595 / 32,000 (95.6%) | 30,774 / 32,000 (96.2%) | 43.5 | 35.7 | 39.6 | 45.5 |

### Athena CTI task breakdown by round

| Round | CKT | ATE | RCM | RMS | TAA | VSP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_00` | 70.0 | 26.0 | 50.5 | 9.4 | 16.0 | 77.1 |
| `round_01` | 69.0 | 25.0 | 46.0 | 8.8 | 10.0 | 80.2 |
| `round_02` | 68.0 | 12.0 | 37.5 | 9.7 | 12.0 | 75.3 |

Observations:
- `round_00` is the best checkpoint by the RL-matched validation metric.
- Minerva dev stayed roughly flat to slightly higher after `round_00`, but the
  Athena mean degraded every round.
- The biggest late-round drops came from `ATE`, `RCM`, and `TAA`.
- Trace selection became very dense after `round_00`, but that did not translate
  into better combined validation.

## Qwen3-8B-Base STaR

Run details:
- Launcher: `bash star/scripts/train_star_qwen8b.sh --from-scratch`
- Artifact root: `star-artifacts/star_cti_qwen3_8b_base`
- Base model: `Qwen/Qwen3-8B-Base`
- Best round by `Combined avg.`: `round_00`

### Round-wise validation and trace selection

| Round | Original success | Rationalization success | Selected traces | Minerva dev | Athena mean | Combined avg. | All-val mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_00` | 3,911 / 32,000 (12.2%) | 26,237 / 32,000 (82.0%) | 26,825 / 32,000 (83.8%) | 34.7 | 30.8 | 32.7 | 38.6 |
| `round_01` | 5,079 / 32,000 (15.9%) | 26,995 / 32,000 (84.4%) | 27,682 / 32,000 (86.5%) | 36.7 | 28.7 | 32.7 | 38.8 |
| `round_02` | 4,443 / 32,000 (13.9%) | 20,843 / 32,000 (65.1%) | 22,285 / 32,000 (69.6%) | 19.0 | 28.2 | 23.6 | 28.8 |

### Athena CTI task breakdown by round

| Round | CKT | ATE | RCM | RMS | TAA | VSP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `round_00` | 69.7 | 17.0 | 48.0 | 2.0 | 4.0 | 44.0 |
| `round_01` | 68.7 | 18.0 | 46.5 | 0.0 | 2.0 | 37.1 |
| `round_02` | 73.0 | 17.0 | 36.0 | 2.8 | 0.0 | 40.4 |

Observations:
- `round_00` remains the best checkpoint. `round_01` was effectively flat on
  `Combined avg.` and slightly better on Minerva dev, but worse on Athena.
- `round_02` collapsed on both data generation quality and validation:
  rationalization success dropped from 84.4 to 65.1, selected traces dropped
  from 86.5 to 69.6, and Minerva dev fell from 36.7 to 19.0.
- `CKT` stayed relatively stable across rounds, but `RCM`, `TAA`, and `VSP`
  remained weak enough to keep the Athena average low.

## Cross-run summary

| Model | Best round | Completed rounds | Best checkpoint | Notes |
| --- | ---: | ---: | --- | --- |
| Llama-3.1-8B-Instruct STaR | `round_00` | 3 | `star-artifacts/star_cti_llama31_8b/round_00/checkpoint/hf_model` | Stronger of the two STaR runs; later rounds regressed mainly on Athena CTI. |
| Qwen3-8B-Base STaR | `round_00` | 3 | `star-artifacts/star_cti_qwen3_8b_base/round_00/checkpoint/hf_model` | Weaker overall and unstable by `round_02`, with a large Minerva-dev collapse. |

## Source artifacts

Primary files used for this summary:
- `star-artifacts/star_cti_llama31_8b/round_00/summary.json`
- `star-artifacts/star_cti_llama31_8b/round_01/summary.json`
- `star-artifacts/star_cti_llama31_8b/round_02/summary.json`
- `star-artifacts/star_cti_qwen3_8b_base/round_00/summary.json`
- `star-artifacts/star_cti_qwen3_8b_base/round_01/summary.json`
- `star-artifacts/star_cti_qwen3_8b_base/round_02/summary.json`

# Matched Answer-Only SFT Ablation

This ablation trains GRPO with the same online auxiliary SFT optimization
schedule used by the Llama-8B Noctua run, but replaces ACR trace generation
with direct answer-only SFT targets.

Run:

```bash
bash rlvr/cti-scripts/train_minerva_grpo_answer_sft_llama8b.sh
```

The Llama-8B wrapper calls:

- `rlvr/cti-scripts/train_minerva_grpo_answer_sft.sh`

## What It Matches

The Llama-8B preset mirrors the selected Noctua distillation schedule:

| Setting | Value |
| --- | --- |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| GRPO rollouts per prompt | `8` |
| Train batch size | `128` |
| Total steps | `500` |
| Auxiliary SFT interval | `10` steps |
| Auxiliary SFT batch size | `256` |
| Auxiliary SFT LR scale | `0.05` |
| Buffer mode | `flush` |
| GPUs per node | `4` |
| Hard-sample gate | UID max RLVR reward `< 1.0` |
| CVSS auxiliary SFT | skipped by default |

This matches the online SFT optimization cadence, batch size, LR scale, and
training length. It does not match ACR generation compute, because this ablation
intentionally removes the ACR rollout phase.

## What It Replaces

Noctua:

```text
GRPO rollout -> hard UID gate -> ACR prompt -> ACR rollouts -> filtering -> SFT on selected ACR trace
```

Matched answer-only SFT:

```text
GRPO rollout -> hard UID gate -> SFT on original prompt -> boxed ground-truth answer
```

The SFT target is formatted as:

```text
\boxed{answer}
```

No ACR block is appended, and no ACR-specific generation, EMA teacher, TextCNN
filter, or heuristic ACR filter is used.

## Useful Overrides

```bash
ANSWER_SFT_EXPERIMENT_NAME=my_run \
ANSWER_SFT_TOTAL_STEPS=500 \
ANSWER_SFT_N_GPUS_PER_NODE=4 \
bash rlvr/cti-scripts/train_minerva_grpo_answer_sft_llama8b.sh
```

Common knobs:

| Environment variable | Default |
| --- | --- |
| `ANSWER_SFT_MODEL_PATH` | `meta-llama/Llama-3.1-8B-Instruct` |
| `ANSWER_SFT_DISTILL_INTERVAL` | `10` |
| `ANSWER_SFT_DISTILL_BATCH_SIZE` | `256` |
| `ANSWER_SFT_DISTILL_LR_SCALE` | `0.05` |
| `ANSWER_SFT_DISTILL_BUFFER_MODE` | `flush` |
| `ANSWER_SFT_HARD_REWARD_MODE` | `max` |
| `ANSWER_SFT_HARD_REWARD_THRESHOLD` | `1.0` |
| `ANSWER_SFT_SKIP_CVSS` | `true` |
| `ANSWER_SFT_TOTAL_STEPS` | `500` |
| `ANSWER_SFT_OUTPUT_ROOT` | `rlvr/checkpoints/minerva/$ANSWER_SFT_EXPERIMENT_NAME` |

## Expected Metrics

The trainer logs answer-SFT specific metrics such as:

- `answer_sft/hard_uid_total`
- `answer_sft/hard_uid_count`
- `answer_sft/selected_count`
- `answer_sft/selected_frac`
- `answer_sft/skip_not_hard`
- `distill/ran`
- `distill/sft_sampled_count`
- `distill_sft/*`

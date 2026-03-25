# STaR-CTI Experiment Plan

## Purpose

Define a fair baseline protocol for comparing STaR-CTI against MinervaRL / Noctua.

## Primary comparison

Compare:

- Plain SFT
- STaR-CTI
- Noctua / MinervaRL

The question is whether iterative rationale bootstrapping with verifier filtering
and supervised updates is sufficient, without RL.

## Required controls

- Same base model
- Same Minerva train split
- Same validation suite
- Same answer extraction + verifier
- Same max prompt and response lengths where possible

## STaR protocol

Run one faithful STaR loop only.

For each round on the full train set:

1. Generate one original reasoning trace per example.
2. Generate one rationalization trace per example conditioned on the gold answer.
3. Keep the original trace if it is verifier-correct.
4. Otherwise keep the rationalization trace if the original failed and the
   rationalization is verifier-correct.
5. Otherwise keep nothing for that example.
6. Build or update the STaR SFT dataset.
7. Train or continue training.
8. Save a checkpoint.
9. Evaluate on the combined validation set.

Repeat iteratively across rounds.

For reporting:

- report the best checkpoint based on validation
- optionally report the round where the best checkpoint occurred

## Metrics

Use the same metrics already reported for MinervaRL:

- Minerva dev reward mean
- per-task reward means
- Athena CTI reward means
- aggregate validation reward

Also log STaR-specific diagnostics:

- original success rate
- rationalization success rate
- overall selected-trace rate
- percentage of examples contributing SFT traces
- average trace length
- per-task success rates

## Artifact layout

Suggested output tree:

- `star-artifacts/<exp>/round_0/original_samples.jsonl`
- `star-artifacts/<exp>/round_0/original_scored.jsonl`
- `star-artifacts/<exp>/round_0/rationalization_samples.jsonl`
- `star-artifacts/<exp>/round_0/rationalization_scored.jsonl`
- `star-artifacts/<exp>/round_0/star_sft_train.parquet`
- `star-artifacts/<exp>/round_0/summary.json`
- `star-artifacts/<exp>/round_1/...`

Each `summary.json` should contain:

- counts of attempted and successful traces
- task-level breakdowns
- train parquet row count
- selected checkpoint path
- eval metrics

## Reporting language

Use the following framing:

"STaR-CTI is an offline rationale-bootstrapping baseline. It uses the same Minerva
train split and verifier as MinervaRL, but replaces RL updates with iterative SFT
on verifier-correct original and gold-conditioned rationalization traces."

## Success criterion for implementation

The baseline is implementation-complete when:

1. one command can run one full STaR round
2. it emits scored original and rationalization traces
3. it writes an SFT parquet
4. it trains a checkpoint
5. it evaluates that checkpoint on the standard CTI validation suite
6. it records round-wise validation scores so the best checkpoint can be selected

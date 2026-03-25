# STaR-CTI Baseline

This folder is for implementing a STaR-style supervised baseline for Minerva-CTI.

Goal: provide a fair baseline against MinervaRL / Noctua by adapting the original
STaR loop to CTI tasks while reusing the same train split, verifier, and
checkpoint-selection validation metric already used in this repository.

## Baseline definition

For each Minerva training example `(x, y*)`:

1. Generate one reasoning trace from the original prompt `x`.
2. Generate one rationalization trace conditioned on the correct answer `y*`.
3. Score both traces with the existing Minerva verifier.
4. Keep the original trace if it is verifier-correct.
5. Otherwise, keep the rationalization trace if the original failed and the
   rationalization is verifier-correct.
6. Otherwise, keep nothing for that example in that round.
7. Build the current round's STaR SFT dataset from the kept traces.
8. Train a fresh copy of the base pretrained model on that round's dataset.
9. Save the final round checkpoint and evaluate it on the RL-matched validation
   suite.
10. Repeat for multiple rounds.

The resulting round checkpoint is then used as the generator for the next round.
This matches the original STaR repo behavior more closely than cumulative SFT or
continued finetuning.

## Fairness requirements

- Use the same train split as MinervaRL:
  `rlvr/mydata/minerva_base/minerva_base_train.parquet`
- Use the same reward parser/verifier:
  `rlvr/verl/utils/reward_score/reward_minerva.py`
- Use the same checkpoint-selection validation target used by current Minerva RL:
  `minerva_base_dev` plus Athena CTI (`ate`, `ckt`, `rcm`, `rms`, `taa`, `vsp`)
- Select the best STaR checkpoint by the RL-style metric:
  `0.5 * minerva_dev_mean + 0.5 * athena_bench_mean`
- Use one faithful STaR loop, run iteratively round by round.

## Recommended default comparison

For the main comparison:

- Original-trace samples per example: `1`
- Rationalization-trace samples per example: `1`
- Train only with SFT updates
- No PPO / GRPO update

This gives a simple statement in the paper: "STaR-CTI uses the same verifier and
the same train split as MinervaRL, but replaces RL with iterative rationale
bootstrapping and supervised fine-tuning."

## Current implementation

- Main entrypoint:
  - `python -m star.run_star_cti --config <yaml>`
- Round generation uses `vllm` in subprocesses for:
  - original traces
  - rationalization traces
  - validation generation
- Each STaR round saves one final checkpoint at:
  - `star-artifacts/<exp>/round_XX/checkpoint/hf_model`
- Current launchers:
  - `bash star/scripts/train_star_llama3b.sh --from-scratch`
  - `bash star/scripts/train_star_llama8b.sh --from-scratch`
- Default output root:
  - `star-artifacts/`

## Planned contents

- `star/IMPLEMENTATION_PLAN.md`
  Concrete implementation plan and file map
- `star/EXPERIMENT_PLAN.md`
  Main comparison protocol, metrics, and reporting rules
- future:
  - generation scripts
  - rationalization-prompt builder
  - trace filtering and packing utilities
  - SFT training wrappers

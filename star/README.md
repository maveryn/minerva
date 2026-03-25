# STaR-CTI Baseline

This folder is for implementing a STaR-style supervised baseline for Minerva-CTI.

Goal: provide a fair baseline against MinervaRL / Noctua by adapting the original
STaR loop to CTI tasks while reusing the same train split, verifier, and validation
suite already used in this repository.

## Baseline definition

For each Minerva training example `(x, y*)`:

1. Generate one reasoning trace from the original prompt `x`.
2. Generate one rationalization trace conditioned on the correct answer `y*`.
3. Score both traces with the existing Minerva verifier.
4. Keep the original trace if it is verifier-correct.
5. Otherwise, keep the rationalization trace if the original failed and the
   rationalization is verifier-correct.
6. Otherwise, keep nothing for that example in that round.
7. Build or update the STaR SFT dataset from the kept traces.
8. Train or continue training with SFT.
9. Save a checkpoint and evaluate on the combined validation set.
10. Repeat for multiple rounds.

This is the closest CTI analogue of the original STaR procedure.

## Fairness requirements

- Use the same train split as MinervaRL:
  `rlvr/mydata/minerva_base/minerva_base_train.parquet`
- Use the same reward parser/verifier:
  `rlvr/verl/utils/reward_score/reward_minerva.py`
- Use the same validation suite used by current Minerva scripts:
  Minerva dev + Athena CTI + SecEval where applicable
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

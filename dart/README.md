# DART-CTI Baseline

This folder is for planning a DART-style rejection-tuning baseline for Minerva-CTI.

Goal: implement a faithful CTI adaptation of DART-Math using:

- a stronger teacher model for trace generation
- verifier-based rejection filtering
- Uniform data construction with `k_u = 2`
- a `B_max = 100` safety cap for plain rejection sampling
- two fixed dataset variants:
  - `v1`: plain original-prompt rejection tuning only
  - `v2`: `v1` plus answer-guided fill for missing traces
- one fixed accepted-trace dataset built once by the teacher
- standard SFT on that fixed dataset for multiple student models
- checkpoint selection by validation loss on synthetic DART validation traces

This folder currently contains planning only. No implementation code has been added yet.

Contents:

- `dart/IMPLEMENTATION_PLAN.md`
  Step-by-step implementation plan for the DART-CTI baseline

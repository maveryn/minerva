# DART-CTI Baseline

This folder is for planning a DART-style rejection-tuning baseline for Minerva-CTI.

Goal: implement a faithful CTI adaptation of DART-Math using:

- a stronger teacher model for trace generation
- verifier-based rejection filtering
- bounded generation budgets
- Uniform and Prop2Diff data construction modes
- one fixed accepted-trace dataset built once by the teacher
- standard SFT on that fixed dataset for multiple student models

This folder currently contains planning only. No implementation code has been added yet.

Contents:

- `dart/IMPLEMENTATION_PLAN.md`
  Step-by-step implementation plan for the DART-CTI baseline

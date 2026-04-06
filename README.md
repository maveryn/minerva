# Minerva

Minerva is a cyber threat intelligence training workspace built around three layers:

1. `minerva/`: dataset construction from public CTI sources
2. `rlvr/`: RLVR training with Minerva rewards, including GRPO and Minerva-Noctua
3. baseline and analysis code such as `dart/`, `star/`, `minerva-sft/`, `minerva-judge/`, and `paper/`

The repo is no longer just a dataset builder. `main` contains the dataset pipeline, RLVR training stack, STaR and DART baselines, judge tooling, and tracked paper-source artifacts. Additional experiments live on dedicated branches such as `luffy` and `sql-r1`.

## Repo Map

- `minerva/`
  - Core dataset builder, task definitions, prompt templates, and reward helpers
  - Main entrypoints: `minerva/pipeline.py`, `minerva/split.py`
- `rlvr/`
  - VeRL-based RLVR training stack
  - Main CTI launchers: `rlvr/cti-scripts/train_minerva_grpo.sh`, `rlvr/cti-scripts/train_minerva_noctua.sh`
- `dart/`
  - DART-CTI data construction and fixed-dataset SFT scripts
- `dart-artifacts/`
  - Tracked DART summaries plus selected train/validation Parquets in Git LFS
- `star/`
  - STaR-CTI iterative supervised baseline
- `minerva-sft/`
  - Single-trace SFT builders over the Minerva train split
- `minerva-judge/`
  - LLM judge, classifier, and human annotation tooling
- `paper/`
  - Figure scripts and small tracked source CSVs for paper plots
- `rlvr/recipe/r1/`
  - upstream DeepSeek-R1 reproduction recipe kept inside the VeRL subtree

## Dataset Builder

The dataset pipeline builds RL-friendly CTI tasks from NVD CVEs, MITRE ATT&CK procedures, CAPEC, Sigma, mappings-explorer, and related sources.

Main code:

- `minerva/pipeline.py`
- `minerva/split.py`
- `minerva/tasks/`
- `minerva/data_sources/`
- `minerva/reward.py`

Primary outputs:

- base JSONL dataset: `dataset/minerva_base/`
- LHC JSONL dataset: `dataset/minerva_lhc/`
- base split: `dataset/minerva_base_split/`
- LHC split: `dataset/minerva_lhc_split/`
- label details for ACRD: `dataset/label_details/`

Typical flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional for faster NVD access
export NVD_API_KEY=...

# Build base dataset
python -m minerva.pipeline \
  --config minerva/config.yaml \
  --variant base \
  --output-root dataset/minerva_base

# Optional LHC build
python -m minerva.pipeline \
  --config minerva/config.yaml \
  --variant lhc \
  --output-root dataset/minerva_base \
  --lhc-output-root dataset/minerva_lhc

# Build train/dev splits
python -m minerva.split \
  --input-dir dataset/minerva_base \
  --output-dir dataset/minerva_base_split \
  --file-prefix minerva-base

python -m minerva.split \
  --input-dir dataset/minerva_lhc \
  --output-dir dataset/minerva_lhc_split \
  --file-prefix minerva-lhc \
  --reuse-split dataset/minerva_base_split

# Build ACRD label details
python scripts/build_label_details.py
```

Convert split JSONLs to VeRL-ready Parquet:

```bash
python rlvr/mydata/data_prepare/cti.py \
  --data_dir dataset/minerva_base_split \
  --out_dir rlvr/mydata/minerva_base
```

See `minerva/README.md` and `docs/task-descriptions.md` for task-level details.

## RLVR Training On `main`

### GRPO

Core entrypoint:

- `rlvr/cti-scripts/train_minerva_grpo.sh`

Current model wrappers on `main`:

- `rlvr/cti-scripts/train_minerva_grpo_llama3b.sh`
- `rlvr/cti-scripts/train_minerva_base_llama8b.sh`
- `rlvr/cti-scripts/train_minerva_grpo_qwen4b.sh`
- `rlvr/cti-scripts/train_minerva_grpo_qwen8b.sh`

Example:

```bash
bash rlvr/cti-scripts/train_minerva_grpo_llama3b.sh
```

This uses:

- train: `rlvr/mydata/minerva_base/minerva_base_train.parquet`
- validation: `minerva_base_dev.parquet` + Athena CTI Parquets + `seceval_mini.parquet`
- reward: `rlvr/verl/utils/reward_score/reward_minerva.py`

### Minerva-Noctua

Core entrypoint:

- `rlvr/cti-scripts/train_minerva_noctua.sh`

Current model/preset wrappers on `main`:

- `rlvr/cti-scripts/train_minerva_noctua_llama3b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_qwen4b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_qwen8b.sh`
- Llama-8B presets and ablations:
  - `rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.05_textcnn_mlh_t0.7_p0.9_defer_ema.sh`
  - `rlvr/cti-scripts/train_minerva_noctua_llama8b_abl_filteroff.sh`
  - `rlvr/cti-scripts/train_minerva_noctua_llama8b_abl_mlfilteroff.sh`
  - `rlvr/cti-scripts/train_minerva_noctua_llama8b_abl_emaoff.sh`

Example:

```bash
bash rlvr/cti-scripts/train_minerva_noctua_llama3b.sh
```

Noctua adds per-batch answer-conditioned reasoning and distillation on top of GRPO. It uses:

- label details from `dataset/label_details/`
- ACR reward logic in `rlvr/verl/utils/reward_score/reward_acr.py`
- the same Minerva train/validation Parquets used by GRPO

### DART-Initialized GRPO

Wrappers for RLVR initialized from uploaded DART checkpoints:

- `rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_llama3b.sh`
- `rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_llama8b.sh`
- `rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_qwen4b.sh`
- `rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_qwen8b.sh`

## Fixed-Data Baselines And Supporting Tooling

### DART-CTI

Code:

- `dart/run_dart_cti.py`
- `dart/run_dart_v2.py`
- `dart/run_dart_validation.py`
- `dart/build_sft_dataset.py`
- `dart/collect.py`
- `dart/fill.py`
- `dart/fill_direct_answers.py`
- `dart/score.py`

Student-model SFT launchers:

- `dart/train_dart_sft_llama3b.sh`
- `dart/train_dart_sft_llama8b.sh`
- `dart/train_dart_sft_qwen4b.sh`
- `dart/train_dart_sft_qwen8b.sh`

Tracked artifact docs and summaries:

- `dart-artifacts/README.md`
- final 64k DART SFT train Parquet:
  - `dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet`
- tracked synthetic validation Parquets:
  - `dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet`
  - `dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet`
  - `dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v2_from_athena_bench_v3_from_v2.parquet`

### STaR-CTI

Code:

- `python -m star.run_star_cti --config <yaml>`
- `star/train_round.sh`

Launchers:

- `star/scripts/train_star_llama3b.sh`
- `star/scripts/train_star_llama8b.sh`
- `star/scripts/train_star_qwen4b.sh`
- `star/scripts/train_star_qwen8b.sh`

Outputs land under `star-artifacts/`.

### Minerva-SFT Trace Builders

Code:

- `minerva-sft/scripts/build_sft_trace.py`
- `minerva-sft/scripts/build_answer_only_sft.py`
- `minerva-sft/scripts/build_sft_from_judge.py`

These build one-trace-per-sample supervised datasets over the Minerva train split.

### Judge And Human Annotation Tooling

Code:

- pointwise/pairwise judge tooling under `minerva-judge/`
- human annotation app under `minerva-judge/human-judge/`
- pairwise judge runner:
  - `minerva-judge/judge-pairwise-quality-eval/run_pairwise_judge.py`

## Paper Assets And Small Tracked Sources

`paper/` mostly keeps scripts and small source tables, while large generated figures and bundles stay ignored.

Tracked examples:

- `paper/valacc/*.csv`
- `paper/expansion/*.csv`
- `paper/reward_sparsity/target_level_stats_v1_*.csv`
- plotting scripts under `paper/`

## Branch-Specific Extensions

### `luffy` branch

The `luffy` branch contains the Minerva adaptation of LUFFY off-policy RLVR. That branch includes:

- LUFFY source bundle and setup notes
- Minerva train/validation copies prepared for LUFFY
- 32k one-trace-per-prompt off-policy dataset derived from DART
- LUFFY launchers for Llama/Qwen models

Switch to it with:

```bash
git switch luffy
```

or from a fresh clone:

```bash
git switch --track origin/luffy
```

### `sql-r1` branch

`main` already contains the generic R1 recipe under `rlvr/recipe/r1/`. The `sql-r1` branch contains the SQL-specialized extension and results, including:

- SQL-specific reward code and prompts
- SQL-R1 GRPO and Noctua launchers
- `sql-benchmark/` and `sql-results/`
- additional evaluation scripts such as:
  - `rlvr/cti-scripts/train_sql_r1_grpo.sh`
  - `rlvr/cti-scripts/train_sql_r1_noctua.sh`
  - `rlvr/cti-scripts/run_sql_r1_table1_parallel.sh`

Switch to it with:

```bash
git switch --track origin/sql-r1
```

## Additional References

- `AGENTS.md`
- `minerva/README.md`
- `rlvr/README.md`
- `dart-artifacts/README.md`
- `docs/task-descriptions.md`

# AGENTS.md

## Project summary
- `main` is a combined workspace for:
  - Minerva CTI dataset construction in `minerva/`
  - VeRL-based RLVR training in `rlvr/`
  - fixed-data baselines in `dart/`, `star/`, and `minerva-sft/`
  - judge / annotation tooling in `minerva-judge/`
  - paper analysis scripts in `paper/`
- The repo is not dataset-only anymore. When users ask about training or baselines, inspect the baseline folders directly instead of assuming everything lives under `minerva/`.

## Key entry points on `main`

### Dataset builder
- `minerva/pipeline.py`: builds task JSONLs from CTI sources using `minerva/config.yaml`
- `minerva/split.py`: builds train/dev splits from generated task JSONLs
- `minerva/tasks/`: per-task builders
- `minerva/data_sources/`: source fetchers/loaders
- `minerva/reward.py`: dataset-side reward helpers

### RLVR training
- `rlvr/cti-scripts/train_minerva_grpo.sh`: core Minerva GRPO launcher
- `rlvr/cti-scripts/train_minerva_noctua.sh`: core Minerva-Noctua launcher
- `rlvr/verl/utils/reward_score/reward_minerva.py`: main RL reward
- `rlvr/verl/utils/reward_score/reward_acr.py`: ACR/Noctua reward path

### Baselines
- `dart/run_dart_cti.py`, `dart/run_dart_v2.py`, `dart/run_dart_validation.py`
- `star/run_star_cti.py`
- `minerva-sft/scripts/build_sft_trace.py`
- `minerva-judge/judge-pairwise-quality-eval/run_pairwise_judge.py`

## Important data locations

### Dataset builder outputs
- base dataset: `dataset/minerva_base/`
- LHC dataset: `dataset/minerva_lhc/`
- base split: `dataset/minerva_base_split/`
- LHC split: `dataset/minerva_lhc_split/`
- ACRD label details: `dataset/label_details/`

### VeRL-ready Parquets
- base RL train/dev: `rlvr/mydata/minerva_base/`
- Athena validation Parquets: `rlvr/mydata/athena/`
- SecEval mini validation: `rlvr/mydata/seceval/seceval_mini.parquet`

### DART tracked artifacts
- docs: `dart-artifacts/README.md`
- final 64k SFT train Parquet:
  - `dart-artifacts/dart_cti_llama8b/datasets/train/train_v2_from_llama8b_direct_from_v3_cap200_seed.parquet`
- tracked validation Parquets:
  - `dart-artifacts/dart_cti_llama8b/datasets/minerva_dev/minerva_dev_v2_from_minerva_dev_v3_from_v2.parquet`
  - `dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v1.parquet`
  - `dart-artifacts/dart_cti_llama8b/datasets/athena_bench/athena_bench_v2_from_athena_bench_v3_from_v2.parquet`
- small tracked summaries live alongside them; raw `attempts_*.jsonl` and most generated artifacts stay ignored

## Common commands

### Build datasets
- `python -m minerva.pipeline --config minerva/config.yaml --variant base --output-root dataset/minerva_base`
- `python -m minerva.pipeline --config minerva/config.yaml --variant lhc --output-root dataset/minerva_base --lhc-output-root dataset/minerva_lhc`
- `python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base`
- `python -m minerva.split --input-dir dataset/minerva_lhc --output-dir dataset/minerva_lhc_split --file-prefix minerva-lhc --reuse-split dataset/minerva_base_split`
- `python scripts/build_label_details.py`

### Convert to VeRL Parquet
- `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_base_split --out_dir rlvr/mydata/minerva_base`
- `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_lhc_split --out_dir rlvr/mydata/minerva_lhc`

### RLVR launchers
- GRPO:
  - `bash rlvr/cti-scripts/train_minerva_grpo_llama3b.sh`
  - `bash rlvr/cti-scripts/train_minerva_base_llama8b.sh`
  - `bash rlvr/cti-scripts/train_minerva_grpo_qwen4b.sh`
  - `bash rlvr/cti-scripts/train_minerva_grpo_qwen8b.sh`
- Minerva-Noctua:
  - `bash rlvr/cti-scripts/train_minerva_noctua_llama3b.sh`
  - `bash rlvr/cti-scripts/train_minerva_noctua_qwen4b.sh`
  - `bash rlvr/cti-scripts/train_minerva_noctua_qwen8b.sh`
  - Llama-8B preset on `main`:
    - `bash rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.05_textcnn_mlh_t0.7_p0.9_defer_ema.sh`
- DART-initialized GRPO:
  - `bash rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_llama3b.sh`
  - `bash rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_llama8b.sh`
  - `bash rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_qwen4b.sh`
  - `bash rlvr/cti-scripts/dart-init-grpo/train_minerva_dart_qwen8b.sh`

### Baseline launchers
- STaR:
  - `bash star/scripts/train_star_llama3b.sh`
  - `bash star/scripts/train_star_llama8b.sh`
  - `bash star/scripts/train_star_qwen4b.sh`
  - `bash star/scripts/train_star_qwen8b.sh`
- DART SFT:
  - `bash dart/train_dart_sft_llama3b.sh`
  - `bash dart/train_dart_sft_llama8b.sh`
  - `bash dart/train_dart_sft_qwen4b.sh`
  - `bash dart/train_dart_sft_qwen8b.sh`

## Branch-specific areas

### `luffy`
- LUFFY integration is maintained on the `luffy` branch, not `main`
- If a user asks about LUFFY code, scripts, datasets, or setup, check `origin/luffy` or switch to that branch
- Do not assume the top-level `luffy/` directory on `main` is a complete runnable bundle

### `sql-r1`
- SQL-R1 work is maintained on the `sql-r1` branch
- `main` still contains the generic upstream R1 recipe at `rlvr/recipe/r1/`
- Relevant files there include:
  - `rlvr/cti-scripts/train_sql_r1_grpo.sh`
  - `rlvr/cti-scripts/train_sql_r1_noctua.sh`
  - `sql-benchmark/`
  - `sql-results/`
  - `minerva/sql_r1_acr_prompt.py`
- If a user asks about SQL-R1 and the current branch is `main`, inspect `origin/sql-r1` or switch branches

## Repo hygiene and gotchas
- Treat `dataset/` caches and most of `dart-artifacts/`, `star-artifacts/`, `rlvr/checkpoints/`, and `wandb/` as generated artifacts
- Do not hand-edit dataset outputs unless explicitly asked
- Large tracked artifacts use Git LFS; raw run logs and large intermediate JSONLs remain ignored
- `paper/` keeps scripts and selected small source tables; generated PDFs/PNGs are mostly ignored
- `rlvr/` is heavy; avoid running broad test suites unless the user asks
- External fetches from NVD/MITRE/CAPEC can be slow or rate-limited; prefer cached data when present

## Useful references
- `README.md`
- `minerva/README.md`
- `dart-artifacts/README.md`
- `docs/task-descriptions.md`

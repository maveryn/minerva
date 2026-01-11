# AGENTS.md

## Project summary
- Minerva builds RL-friendly cyber threat intelligence datasets (CVE, CAPEC, ATT&CK, Sigma, mappings-explorer).
- The repo also contains related tooling (`athena_*`) and a large RL training stack (`rlvr/`); most dataset work lives under `minerva/`.

## Key entry points
- `minerva/pipeline.py` orchestrates dataset generation from sources using `minerva/config.yaml`.
- `minerva/split.py` builds train/dev splits from per-task JSONL outputs.
- `minerva/tasks/` implements per-task builders; `minerva/data_sources/` fetches/caches external data.
- Reward logic lives in `minerva/reward.py` and `rlvr/verl/utils/reward_score/reward_minerva.py`.

## Data & outputs
- Input caches live under `dataset/` (e.g., `dataset/nvd/`, `dataset/mitre/`, `dataset/capec/`, `dataset/sigma/`, `dataset/mappings-explorer/`).
- Dataset outputs:
  - Base: `dataset/minerva_base/*.jsonl` plus `dataset/minerva_base/metadata.json`.
  - LHC: `dataset/minerva_lhc/*.jsonl` plus `dataset/minerva_lhc/metadata.json` (train rows include `candidate_pool_top100`).
- Split outputs:
  - Base: `dataset/minerva_base_split/minerva-base-{train,dev}.jsonl` + `metadata.json`.
  - LHC: `dataset/minerva_lhc_split/minerva-lhc-{train,dev}.jsonl` + `metadata.json` (dev reused from base).
- `--detailed-prompts` writes to suffixed output roots (e.g., `dataset/minerva_base_detailed`).

## RLVR CTI dataset prep (VeRL format)
- Preferred flow: write splits with a `minerva-*` prefix so `cti.py` can read them directly.
  - Base split: `python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base`
  - LHC split (reuse dev): `python -m minerva.split --input-dir dataset/minerva_lhc --output-dir dataset/minerva_lhc_split --file-prefix minerva-lhc --reuse-split dataset/minerva_base_split`
- Convert to VeRL-ready Parquet + Excel:
  - Base: `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_base_split --out_dir rlvr/mydata/minerva_base`
  - LHC: `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_lhc_split --out_dir rlvr/mydata/minerva_lhc`
  - Outputs `rlvr/mydata/<name>/*.parquet` and `.xlsx`, caps `minerva-*-dev.jsonl` at 2000 rows, and injects CTI system prompts.
- To mix in AthenaBench JSONLs (e.g., `athena-cti-ate.jsonl`, `athena-cti-rcm.jsonl`, `athena-cti-rms.jsonl`, `athena-cti-taa.jsonl`), drop them in the same `--data_dir` before running `cti.py`.
- Athena Parquet outputs live under `rlvr/mydata/athena/`.
- Optional: `python rlvr/mydata/data_prepare/cti_rms_detailed.py` builds `athena-cti-rms-detailed.jsonl` with MITRE mitigation catalogs; rerun the converter to emit `athena_cti_rms_detailed.parquet`.

## Common commands
- Setup:
  - `python -m venv .venv`
  - `.venv\Scripts\python -m pip install -r requirements.txt`
- Build datasets:
  - `.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml --variant base --output-root dataset/minerva_base`
  - `.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml --variant lhc --output-root dataset/minerva_base --lhc-output-root dataset/minerva_lhc`
- Build splits:
  - `.venv\Scripts\python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base`
  - `.venv\Scripts\python -m minerva.split --input-dir dataset/minerva_lhc --output-dir dataset/minerva_lhc_split --file-prefix minerva-lhc --reuse-split dataset/minerva_base_split`
- Tests (lightweight):
  - `.venv\Scripts\python -m pytest minerva/test/test_reward_minerva.py`

## RLVR GRPO training (old-cti-scripts)
- Primary CTI runs use `rlvr/old-cti-scripts/train_minerva_llama.sh` or `rlvr/old-cti-scripts/train_minerva_qwen.sh`.
  - Both call `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and `custom_reward_function.path=rlvr/verl/utils/reward_score/reward_minerva.py`.
  - Training data: `rlvr/mydata/minerva_base/minerva_base_train.parquet` (or `rlvr/mydata/minerva_lhc/minerva_lhc_train.parquet`); validation data: matching `*_dev.parquet` plus Athena CTI parquets from `rlvr/mydata/athena/`.
  - Defaults target 8 GPUs, vLLM rollouts, and minimal KL (see the script flags).
- Variants: `rlvr/old-cti-scripts/v0.1.sh` and `rlvr/old-cti-scripts/v0.2.sh` are config snapshots with different models/prompt lengths.
- Single-task runs: `rlvr/old-cti-scripts/cti_ate_llama-8b.sh` and `rlvr/old-cti-scripts/cti_rcm_llama-8b.sh` train on `rlvr/mydata/cti_out/*.parquet` using `myreward_boxed.py` and the `reward_answer_only` scorer.

## TARBA tool-augmented retrieval (RLVR)
- Code flow: `minerva/retrieval/` builds canonical label docs + BM25 indexes; `minerva/retrieval/server.py` serves `/retrieve`; `rlvr/verl/tools/cti_retrieval_tool.py` calls the server; `rlvr/verl/utils/dataset/minerva_tarba_retrieval_dataset.py` injects tool instructions + per-task budgets; `rlvr/verl/utils/reward_score/reward_tarba.py` adds retrieval shaping; controller updates in `rlvr/verl/trainer/ppo/ray_trainer.py`.
- Build label docs + index (generated artifacts):
  - `python -m minerva.retrieval.build_label_docs --config configs/retrieval/label_docs.yaml --out_dir dataset/retrieval/label_docs`
  - `python -m minerva.retrieval.build_index --label_docs_dir dataset/retrieval/label_docs --out_dir dataset/retrieval/index`
- Run retrieval server:
  - `python -m minerva.retrieval.server --index_dir dataset/retrieval/index --host 0.0.0.0 --port 8000`
- Train TARBA (same train/val datasets as base/LHC/SLHC):
  - `rlvr/cti-scripts/train_minerva_tarba_llama3b.sh`
  - `rlvr/cti-scripts/train_minerva_tarba_llama8b.sh`
  - `rlvr/cti-scripts/train_minerva_tarba_qwen3b.sh`
  - `rlvr/cti-scripts/train_minerva_tarba_qwen8b.sh`
- Evaluate validation with retrieval on/off:
  - `rlvr/old-cti-scripts/eval_minerva_tarba_reton_llama3b.sh`
  - `rlvr/old-cti-scripts/eval_minerva_tarba_retoff_llama3b.sh`

## Config and environment
- `minerva/config.yaml` defines data paths, NVD date windows, and per-task output settings.
- Environment variables:
  - `NVD_API_KEY` to avoid strict NVD rate limits.
  - `NVD_DELAY_S` to override the request delay between NVD pages.

## Repo hygiene and gotchas
- Treat `dataset/` outputs and caches as generated artifacts; avoid hand-editing unless asked.
- External fetches (NVD/MITRE/CAPEC) can be slow and rate-limited; prefer cached files when available.
- `rlvr/` is heavy and has extensive tests; only run those if explicitly requested.
- vLLM is currently avoided because SGLang pins `outlines_core`/`xgrammar` to versions that conflict with vLLM; use `actor_rollout_ref.rollout.name=sglang` in CTI scripts.

## Reference docs
- `README.md` and `minerva/README.md` explain the dataset tasks and prompts.
- `docs/task-descriptions.md` summarizes per-task inputs, outputs, and reward functions.

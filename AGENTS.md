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
- Dataset outputs: `dataset/minerva/*.jsonl` plus `dataset/minerva/metadata.json`.
- Split outputs: `dataset/minerva_split/train.jsonl`, `dataset/minerva_split/dev.jsonl`, `dataset/minerva_split/metadata.json`.
- `--detailed-prompts` writes to suffixed folders (e.g., `dataset/minerva_detailed`).

## RLVR CTI dataset prep (VeRL format)
- Source CTI JSONL inputs live in `rlvr/mydata/cti-in/` (not committed); the converter scans `*.jsonl` there.
- Minerva splits must be named `minerva-train.jsonl` and `minerva-dev.jsonl` so `rlvr/mydata/data_prepare/cti.py` routes `data_source` from each row's `reward_fn` field.
  - Typical flow: generate `dataset/minerva_split/{train,dev}.jsonl`, then copy/symlink into `rlvr/mydata/cti-in/` with the `minerva-*.jsonl` names.
  - Example (copy):
    - `cp dataset/minerva_split/train.jsonl rlvr/mydata/cti-in/minerva-train.jsonl`
    - `cp dataset/minerva_split/dev.jsonl rlvr/mydata/cti-in/minerva-dev.jsonl`
  - Example (symlink):
    - `ln -s ../../dataset/minerva_split/train.jsonl rlvr/mydata/cti-in/minerva-train.jsonl`
    - `ln -s ../../dataset/minerva_split/dev.jsonl rlvr/mydata/cti-in/minerva-dev.jsonl`
- AthenaBench JSONLs (e.g., `athena-cti-ate.jsonl`, `athena-cti-rcm.jsonl`, `athena-cti-rms.jsonl`, `athena-cti-taa.jsonl`) can be dropped into the same folder; their filename becomes the `data_source`.
- Run the converter to build VeRL-ready Parquet + Excel outputs:
  - `python rlvr/mydata/data_prepare/cti.py --data_dir rlvr/mydata/cti-in --out_dir rlvr/mydata/cti`
  - Produces `rlvr/mydata/cti/*.parquet` and `.xlsx`, caps `minerva-dev.jsonl` at 2000 rows, and injects CTI system prompts.
- Optional: `python rlvr/mydata/data_prepare/cti_rms_detailed.py` builds `athena-cti-rms-detailed.jsonl` with MITRE mitigation catalogs; rerun the converter to emit `athena_cti_rms_detailed.parquet`.

## Common commands
- Setup:
  - `python -m venv .venv`
  - `.venv\Scripts\python -m pip install -r requirements.txt`
- Build datasets:
  - `.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml`
- Build splits:
  - `.venv\Scripts\python -m minerva.split --input-dir dataset/minerva`
- Tests (lightweight):
  - `.venv\Scripts\python -m pytest minerva/test/test_reward_minerva.py`

## RLVR GRPO training (cti-scripts)
- Primary CTI runs use `rlvr/cti-scripts/train_minerva_llama.sh` or `rlvr/cti-scripts/train_minerva_qwen.sh`.
  - Both call `python -m verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` and `custom_reward_function.path=rlvr/verl/utils/reward_score/reward_minerva.py`.
  - Training data: `rlvr/mydata/cti/minerva_train.parquet`; validation data: `minerva_dev.parquet` plus Athena CTI parquets.
  - Defaults target 8 GPUs, vLLM rollouts, and minimal KL (see the script flags).
- Variants: `rlvr/cti-scripts/v0.1.sh` and `rlvr/cti-scripts/v0.2.sh` are config snapshots with different models/prompt lengths.
- Single-task runs: `rlvr/cti-scripts/cti_ate_llama-8b.sh` and `rlvr/cti-scripts/cti_rcm_llama-8b.sh` train on `rlvr/mydata/cti_out/*.parquet` using `myreward_boxed.py` and the `reward_answer_only` scorer.

## Config and environment
- `minerva/config.yaml` defines data paths, NVD date windows, and per-task output settings.
- Environment variables:
  - `NVD_API_KEY` to avoid strict NVD rate limits.
  - `NVD_DELAY_S` to override the request delay between NVD pages.

## Repo hygiene and gotchas
- Treat `dataset/` outputs and caches as generated artifacts; avoid hand-editing unless asked.
- External fetches (NVD/MITRE/CAPEC) can be slow and rate-limited; prefer cached files when available.
- `rlvr/` is heavy and has extensive tests; only run those if explicitly requested.

## Reference docs
- `README.md` and `minerva/README.md` explain the dataset tasks and prompts.
- `task-descriptions.md` summarizes per-task inputs, outputs, and reward functions.

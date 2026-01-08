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

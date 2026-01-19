# Minerva RLVR Dataset Builder

Minerva builds reinforcement-learning friendly cyber threat intelligence (CTI) datasets with embedded reward functions. Sources include NVD CVEs, MITRE ATT&CK procedures, Sigma rules, and mappings-explorer.

## What it produces
- CVE → ATT&CK (exploitation/primary/secondary impact) from mappings-explorer
- CVE → CWE, CVE → CVSS v3.1
- Sigma → ATT&CK technique / tactics
- Detection rules (ART/Sentinel/Splunk) → ATT&CK technique
- ATT&CK procedure scenarios → technique / tactics / mitigations / detections
- CAPEC examples → CAPEC / CWE
- Threat actor (procedures) → name attribution
- Train/dev splits (32k/1.2k) with minimal fields (`task`, `prompt`, `ground_truth`, `reward_fn`)

See `docs/task-descriptions.md` for inputs/outputs/rewards, prompts, and per-task split statistics.

## Layout
- `minerva/pipeline.py` – orchestrates full dataset build
- `minerva/tasks/` – task builders (CVE, Sigma, scenarios, mappings-explorer)
- `minerva/data_sources/` – loaders for NVD, MITRE ATT&CK, mappings-explorer
- `minerva/reward.py` – reward functions (binary, technique/tactic F1, CVSS parsers, etc.)
- Outputs: `dataset/minerva_base/*.jsonl` + `dataset/minerva_base/metadata.json` (LHC: `dataset/minerva_lhc/*`)
- Splits: `dataset/minerva_base_split/minerva-base-{train,dev}.jsonl` (LHC: `dataset/minerva_lhc_split/minerva-lhc-{train,dev}.jsonl`)
- Optional validation-only Parquet (IFEval): `rlvr/mydata/ifeval/ifeval_dev.parquet`

## Quickstart
1) Create/activate venv and install deps:
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
2) (Optional) Set NVD API key for faster quota:
```
$env:NVD_API_KEY="..."
```
3) Build datasets:
```
.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml --variant base --output-root dataset/minerva_base
```
4) Build train/dev splits (uses existing JSONL files):
```
.venv\Scripts\python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base
```

Artifacts land under `dataset/minerva_base/`; splits under `dataset/minerva_base_split/`.

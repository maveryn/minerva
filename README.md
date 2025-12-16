# Minerva RLVR Dataset Builder

Minerva builds reinforcement-learning friendly cyber threat intelligence (CTI) datasets with embedded reward functions. Sources include NVD CVEs, MITRE ATT&CK (procedures, intrusion sets), CAPEC, Sigma rules, and mappings-explorer.

## What it produces
- CVE → ATT&CK (exploitation/primary/secondary impact) from mappings-explorer
- CVE → CWE, CVE → CVSS v3.1, CVE → CVSS v4.0
- Sigma → ATT&CK technique / tactics
- ATT&CK procedure scenarios → technique / tactics / mitigations / detections
- CAPEC examples → CAPEC ID / CWE IDs / ATT&CK technique
- Threat actor MCQ from ATT&CK procedures
- Train/dev splits (32k/8k) with minimal fields (`task`, `prompt`, `ground_truth`, `reward_fn`)

See `task-descriptions.md` for inputs/outputs/rewards, prompts, and per-task split statistics.

## Layout
- `minerva/pipeline.py` – orchestrates full dataset build
- `minerva/tasks/` – task builders (CVE, Sigma, scenarios, CAPEC, mappings-explorer, threat actors)
- `minerva/data_sources/` – loaders for NVD, MITRE ATT&CK, CAPEC, mappings-explorer
- `minerva/reward.py` – reward functions (binary, technique/tactic F1, CVSS parsers, etc.)
- Outputs: `dataset/minerva/*.jsonl` + `dataset/minerva/metadata.json`
- Splits: `dataset/minerva_split/train.jsonl`, `dataset/minerva_split/dev.jsonl`, `dataset/minerva_split/metadata.json`

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
.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml
```
4) Build train/dev splits (uses existing JSONL files):
```
.venv\Scripts\python -m minerva.split
```

Artifacts land under `dataset/minerva/`; splits under `dataset/minerva_split/`.

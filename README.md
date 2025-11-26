# Minerva RLVR Dataset Builder

Minerva constructs reinforcement-learning friendly, verifiable CTI tasks from
authoritative sources (NVD, MITRE ATT&CK, CAPEC, CWE). It emits JSONL datasets
with embedded reward metadata for:

- CVE → CWE (binary ID reward)
- CVE → CVSS v3.1 (exact or distance-based reward)
- ATT&CK scenario → Technique (binary ID reward)
- Scenario → Tactics (multi-label F1)
- Scenario → Mitigations (multi-label F1)
- Scenario → Detection Strategies (multi-label F1)
- CVE → CWE → CAPEC → ATT&CK Technique chain (multi-label F1, with hop metadata)
- CWE code snippet → CWE (currently logged as skipped; placeholder file written)

## Layout
- `minerva/config.yaml` – knobs for date ranges, caches, and output paths.
- `minerva/pipeline.py` – orchestration CLI; downloads sources and builds all tasks.
- `minerva/data_sources/` – loaders for NVD, MITRE ATT&CK, and CAPEC STIX bundles.
- `minerva/tasks.py` – task builders with reward specs.
- `minerva/reward.py` – reference verifiers (binary, F1, CVSS distance).

## Quickstart
1) Install deps (inside your venv):
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

2) Export your NVD API key if you have one (faster quota):
```
$env:NVD_API_KEY="..."
```

3) Build all datasets (uses defaults in `minerva/config.yaml`):
```
.venv\Scripts\python -m minerva.pipeline
```

Artifacts land under `dataset/minerva/*.jsonl` with a summary at
`dataset/minerva/metadata.json`.

### Notes
- Paraphrasing is intentionally skipped per current scope; inputs come directly
  from source descriptions.
- If a task cannot be built (e.g., missing CWE code examples), the pipeline logs
  the skip and writes an empty JSONL so downstream steps remain reproducible.

# Minerva Dataset Pipeline

This package builds the RL-friendly CTI datasets from public sources. The
pipeline fetches NVD CVE records, CAPEC patterns, and MITRE ATT&CK scenarios
and emits JSONL tasks with reward metadata.

## What the pipeline produces
- CVE to CWE mapping tasks
- CVE to CVSS v3.1 scoring tasks
- ATT&CK procedure scenarios to techniques, tactics, mitigations, and detection strategies
- CVE to CAPEC to ATT&CK technique chains
- CWE code snippet placeholder file (currently empty but kept for reproducibility)

## Run it
1) Activate your virtualenv and install deps:
```
.venv\Scripts\python -m pip install -r requirements.txt
```

2) (Optional) Set an NVD API key for better rate limits:
```
$env:NVD_API_KEY="..."
```

3) Build all tasks using the default config:
```
.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml
```

Outputs go to the paths defined in `minerva/config.yaml` (defaults under
`dataset/minerva/` with metadata at `dataset/minerva/metadata.json`). Logs are
written to `dataset/logs`.

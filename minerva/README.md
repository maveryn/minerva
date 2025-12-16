# Minerva Dataset Pipeline

Builds RL-friendly CTI datasets from public sources (NVD CVEs, CAPEC, MITRE ATT&CK procedures, Sigma rules, mappings-explorer).

## What gets built
- CVE → CWE, CVE → CVSS v3.1, CVE → CVSS v4.0
- CVE → ATT&CK exploitation/impact (mappings-explorer)
- Sigma → ATT&CK technique/tactics
- Procedure scenarios → ATT&CK technique/tactics/mitigations/detections
- CAPEC examples → CAPEC ID / CWE IDs / ATT&CK technique
- Threat actor MCQ from procedure descriptions

Outputs are JSONL files under `dataset/minerva/` with `dataset/minerva/metadata.json` for counts/examples. Train/dev splits (32k/8k) live under `dataset/minerva_split/` with minimal fields (`task`, `prompt`, `ground_truth`, `reward_fn`).

## Run it
```
.venv\Scripts\python -m pip install -r requirements.txt
$env:NVD_API_KEY="..."   # optional, for NVD rate limits
.venv\Scripts\python -m minerva.pipeline --config minerva/config.yaml
```

## Prompt references

### Mapping-explorer (CVE → ATT&CK)
- **Exploitation Technique / Primary Impact / Secondary Impact**
```
Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

<task-specific line>

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Vulnerability description:
{CVE_DESCRIPTION}
```
`<task-specific line>` is one of:
- Exploitation Technique - the method (technique) used to exploit the vulnerability.
- Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability.
- Secondary Impact - what the adversary can do by gaining the benefit of the primary impact. (Includes primary impact ID in the prompt.)

### CVE (NVD)
- **CVE → CWE**: `Given the Common Vulnerabilities and Exposures (CVE) description below, list EXACTLY {COUNT} Common Weakness Enumeration (CWE) ID{S_SUFFIX} in CWE-<number> format.`
- **CVE → CVSS v3.1**: `Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v3.1 base vector string (format: CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X).`
- **CVE → CVSS v4.0**: `Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v4.0 base vector string (format: CVSS:4.0/AV:X/AC:X/AT:X/PR:X/UI:X/VC:X/VI:X/VA:X/SC:X/SI:X/SA:X).`

### Sigma → ATT&CK
- **Technique**: single ATT&CK technique ID (with requirements block; prefers sub-techniques).
- **Tactics**: list all applicable ATT&CK tactic IDs (TA000x) (with requirements block).

### Procedure scenarios → ATT&CK
- **Technique**: single ATT&CK technique ID (requirements block; prefers sub-techniques).
- **Tactics/Mitigations**: list EXACTLY the count provided per row (requirements block).
- **Detection Strategy**: single ATT&CK detection strategy ID (DET####) with requirements block.

### CAPEC examples
- **Example → CAPEC ID**: single CAPEC-<number>.
- **Example → CWE IDs**: EXACTLY the provided count of CWE-<number> IDs (up to 3).
- **Example → ATT&CK technique**: single ATT&CK technique ID (requirements block; prefers sub-techniques) when only one technique mapping exists.

### Threat actor MCQ
```
Given the observed adversary procedures below, choose the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Select the correct option (A-E) and return the option letter.

Options:
{OPTIONS_TEXT}
```
Multiple variants are generated per actor alias by resampling procedures and shuffling options.

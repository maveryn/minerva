# Minerva Dataset Pipeline

Builds RL-friendly CTI datasets from public sources (NVD CVEs, MITRE ATT&CK procedures, Sigma rules, mappings-explorer).

## What gets built
- CVE → CWE, CVE → CVSS v3.1
- CVE → ATT&CK exploitation/impact (mappings-explorer)
- Sigma → ATT&CK technique/tactics
- Procedure scenarios → ATT&CK technique/tactics/mitigations/detections
- Train/dev splits (32k/1k) with minimal fields (`task`, `prompt`, `ground_truth`, `reward_fn`)

Outputs are JSONL files under `dataset/minerva/` with `dataset/minerva/metadata.json` for counts/examples. Train/dev splits (32k/1k) live under `dataset/minerva_split/` with minimal fields (`task`, `prompt`, `ground_truth`, `reward_fn`).

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

### Sigma → ATT&CK
- **Technique**: single ATT&CK technique ID (with requirements block; prefers sub-techniques).
- **Tactics**: list all applicable ATT&CK tactic IDs (TA000x) (with requirements block).

### Procedure scenarios → ATT&CK
- **Technique**: single ATT&CK technique ID (requirements block; prefers sub-techniques).
- **Tactics/Mitigations**: list EXACTLY the count provided per row (requirements block).
- **Detection Strategy**: single ATT&CK detection strategy ID (DET####) with requirements block.

### CAPEC examples
Note: CAPEC example tasks are currently excluded from the base Minerva splits.
- **Example → CAPEC ID**: single CAPEC-<number>.
- **Example → CWE IDs**: EXACTLY the provided count of CWE-<number> IDs (up to 3).
- **Example → ATT&CK technique**: single ATT&CK technique ID (requirements block; prefers sub-techniques) when only one technique mapping exists.

### Threat actor (procedures)
Note: Threat actor tasks are currently excluded from the base Minerva splits.
```
Given the observed adversary procedures below, identify the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Return only the threat actor name.
```
Multiple variants are generated per actor by sampling 60-90% of techniques (min 3) and assigning a question count based on technique-count bins chosen to balance total questions per bin (~100 each across 8 bins). Each bin assigns 3-10 questions per actor (bin 0 -> 3, bin 7 -> 10); the last bin is open-ended.
Rewarding uses `threat_actor_lookup.json`, which merges MITRE intrusion-set aliases with supplemental aliases from `rlvr/verl/utils/reward_score/aliases.csv`.

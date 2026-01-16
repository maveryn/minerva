# Task Definitions and Prompts

Structured reference for how each dataset is built, the inputs/outputs, reward used, and the exact prompt. Train/dev counts come from `dataset/minerva_split/`.

---

## Mapping-explorer: CVE to ATT&CK
Uses `dataset/mappings-explorer/kev-02.13.2025_attack-15.1-enterprise.yaml`; CVE description from `comments` (CVE IDs replaced with "This vulnerability"/"this vulnerability"); label from `attack_object_id`.

- Input: CVE description
- Output: MITRE ATT&CK Enterprise technique ID
- Reward: `reward_technique_id` (1 if full ID match; 0.5 if technique matches but sub-tech differs; 0 otherwise)

**Exploitation Technique**
```
Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

Exploitation Technique - the method (technique) used to exploit the vulnerability.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Vulnerability description:
{CVE_DESCRIPTION}
```

**Primary Impact**
```
Given the vulnerability description below, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

Primary Impact - the initial benefit (impact) gained through exploitation of the vulnerability.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Vulnerability description:
{CVE_DESCRIPTION}
```

**Secondary Impact**
```
Given the vulnerability description and the primary impact already obtained, output the single most appropriate MITRE ATT&CK Enterprise technique ID for:

Secondary Impact - what the adversary can do by gaining the benefit of the primary impact.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Vulnerability description:
{CVE_DESCRIPTION}

Primary impact:
{PRIMARY_IMPACT_TECHNIQUE_ID}
```

---

## CVE to CWE / CVSS (NVD)
Built from NVD CVEs.

**CVE -> CWE**
- Input: CVE description
- Output: CWE IDs
- Reward: `reward_cwe_ids` (F1 over sets)
```
Given the Common Vulnerabilities and Exposures (CVE) description below, list EXACTLY {COUNT} Common Weakness Enumeration (CWE) ID{S_SUFFIX} in CWE-<number> format.

CVE description:
{CVE_DESCRIPTION}
```

**CVE -> CVSS v3.1 base vector**
- Input: CVE description
- Output: CVSS v3.1 base vector string
- Reward: `reward_cvss_v31` (1 - |score diff| / 10 using CVSS3; invalid vector -> 0)
```
Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v3.1 base vector string (format: CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X).

CVE description:
{CVE_DESCRIPTION}
```

---

## Sigma to ATT&CK
Built from Sigma rules in `dataset/sigma/rules` and `dataset/sigma/rules-threat-hunting`; excerpt includes title, logsource, detection.

**Technique**
- Input: Sigma rule excerpt
- Output: ATT&CK technique ID
- Reward: `reward_technique_id` (1 full, 0.5 parent technique match, else 0)
```
Given the Sigma rule excerpt below (log source + detection logic), provide the single most appropriate MITRE ATT&CK Enterprise technique ID that best represents the adversary behavior this rule is intended to detect.

Technique - how an adversary achieves a tactical objective by performing an action.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable than the parent technique; otherwise return the parent technique ID (e.g., T1059).

Sigma rule excerpt:
{SIGMA_RULE_EXCERPT}
```

**Tactics**
- Input: Sigma rule excerpt
- Output: ATT&CK tactic IDs (TA000x), multiple allowed
- Reward: `reward_tactic_ids` (F1 over sets)
```
Given the Sigma rule excerpt below (log source + detection logic), enumerate all MITRE ATT&CK Enterprise tactic IDs (TA000x) that are valid for the adversary behavior this rule is intended to detect.

Tactic - why an adversary performs an action (the adversary's tactical objective).

Requirements:
- Use MITRE ATT&CK Enterprise tactic IDs (TA000x) only.
- List ALL applicable tactic IDs (multiple may apply).

Sigma rule excerpt:
{SIGMA_RULE_EXCERPT}
```

---

## Scenario to ATT&CK (MITRE procedures)
Built from MITRE ATT&CK procedure scenarios.

**Technique**
- Input: Adversary procedure description
- Output: ATT&CK technique ID
- Reward: `reward_technique_id` (1 full, 0.5 parent technique match, else 0)
```
Given the adversary procedure description below, provide the single most appropriate MITRE ATT&CK Enterprise technique ID that best represents the behavior.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Adversary procedure:
{SCENARIO_TEXT}
```

**Tactics**
- Input: Adversary procedure description
- Output: ATT&CK tactic IDs (TA000x), exact count provided per row
- Reward: `reward_tactic_ids` (F1 over sets)
```
Given the adversary procedure description below, list EXACTLY {COUNT} MITRE ATT&CK Enterprise tactic ID{S_SUFFIX} (TA000x) that apply to this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise tactic IDs (TA000x) only.
- List EXACTLY {COUNT} tactic ID{S_SUFFIX} that are most appropriate.

Adversary procedure:
{SCENARIO_TEXT}
```

**Mitigations**
- Input: Adversary procedure description
- Output: ATT&CK mitigation IDs (M####), exact count provided per row
- Reward: `reward_mitigation_ids` (F1 over sets)
```
Given the adversary procedure description below, list EXACTLY {COUNT} MITRE ATT&CK Enterprise mitigation ID{S_SUFFIX} (M####) that best mitigate this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise mitigation IDs only.
- List EXACTLY {COUNT} mitigation ID{S_SUFFIX} that are most appropriate.

Adversary procedure:
{SCENARIO_TEXT}
```

---

## CAPEC example tasks
Built from CAPEC example instances (parent fallback for ATT&CK mappings). CWE limited to <=3 per example; ATT&CK technique only when a single mapping exists.

**Example -> CAPEC ID**
- Input: Example attack description
- Output: CAPEC-<number>
- Reward: `binary_id` (1 if ID matches, else 0)
```
Given the example attack description below, provide the single best matching CAPEC ID in CAPEC-<number> format.

Example description:
{EXAMPLE_TEXT}
```

**Example -> CWE IDs**
- Input: Example attack description
- Output: CWE IDs (count provided per row, up to 3)
- Reward: `reward_cwe_ids` (F1 over sets)
```
Given the example attack description below, list EXACTLY {COUNT} Common Weakness Enumeration (CWE) ID{S_SUFFIX} in CWE-<number> format that apply.

Example description:
{EXAMPLE_TEXT}
```

**Example -> ATT&CK Technique ID**
- Input: Example attack description
- Output: ATT&CK technique ID (single)
- Reward: `reward_technique_id` (1 full, 0.5 parent technique match, else 0)
```
Given the example attack description below, provide the single most appropriate MITRE ATT&CK Enterprise technique ID that best represents the adversary behavior.

Technique - how an adversary achieves a tactical objective by performing an action.

Requirements:
- Use MITRE ATT&CK Enterprise technique IDs only.
- Return exactly ONE technique ID.
- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059).

Example description:
{EXAMPLE_TEXT}
```

---

## Threat actor (procedures)
Built from MITRE ATT&CK intrusion-set procedure descriptions. For each actor, we sample 60-90% of its techniques (min 3) and generate a number of questions based on technique-count bins chosen to balance total questions per bin (~100 each across 8 bins). Each bin assigns 3..10 questions per actor (bin 0 -> 3, bin 7 -> 10); the last bin is open-ended.

- Input: List of observed procedures
- Output: Threat actor name (free-form)
- Reward: `reward_threat_actor_name` (1 if prediction matches the actor name or an alias; else 0)
- Lookup: `threat_actor_lookup.json` is emitted alongside the dataset and copied into `rlvr/mydata/minerva_base` for reward scoring (canonical names + aliases from MITRE ATT&CK + supplemental aliases from `rlvr/verl/utils/reward_score/aliases.csv`).
```
Given the observed adversary procedures below, identify the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Return only the threat actor name.
```

---

# Dataset Statistics (samples and split counts)
All counts derive from `dataset/minerva_split/metadata.json`.

file	Input	Output	Reward fn	#sample	#Train	#Val
cve_to_attack_exploitation	CVE description	ATT&CK Technique ID	reward_technique_id	265	245	20
cve_to_attack_primary_impact	CVE description	ATT&CK Technique ID	reward_technique_id	230	210	20
cve_to_attack_secondary_impact	CVE description	ATT&CK Technique ID	reward_technique_id	74	54	20
sigma_to_attack_technique	Sigma detection + logsource	ATT&CK Technique ID	reward_technique_id	2000	1950	50
sigma_to_attack_tactics	Sigma detection + logsource	ATT&CK Tactic IDs	reward_tactic_ids	1000	950	50
scenario_to_technique	ATT&CK procedure scenario	ATT&CK Technique ID	reward_technique_id	8000	7800	200
scenario_to_tactics	ATT&CK procedure scenario	ATT&CK Tactic IDs	reward_tactic_ids	1844	1794	50
scenario_to_mitigations	ATT&CK procedure scenario	ATT&CK Mitigation IDs	reward_mitigation_ids	8000	7800	200
cve_to_cwe	CVE description	CWE IDs	reward_cwe_ids	8000	7800	200
cve_to_cvss_v31	CVE description	CVSS v3.1 vector string	reward_cvss_v31	2000	1920	80
capec_example_to_capec	CAPEC example text	CAPEC ID	binary_id	380	360	20
capec_example_to_cwe	CAPEC example text	CWE IDs	reward_cwe_ids	196	176	20
capec_example_to_attack	CAPEC example text	ATT&CK Technique ID	reward_technique_id	144	124	20
threat_actor	Procedures list	Threat actor name	reward_threat_actor_name	867	817	50

---

# Reward Function Notes
- `reward_technique_id`: 1.0 if technique+subtechnique match; 0.5 if parent technique matches but subtech differs/is missing; else 0.
- `reward_tactic_ids`: F1 over predicted vs. ground-truth tactic ID sets.
- `reward_cwe_ids`: F1 over predicted vs. ground-truth CWE ID sets.
- `reward_cvss_v31`: compute CVSS v3.1 score distance (1 - |truth - pred| / 10), invalid vector -> 0. AthenaBench `athena-cti-vsp` uses delta=7.7.
- `reward_mitigation_ids`: F1 over predicted vs. ground-truth mitigation ID sets.
- `binary_id`: 1.0 if IDs match; else 0.
- `reward_threat_actor_name`: 1.0 if prediction matches the actor name or an alias; else 0.

See `docs/task-descriptions.tex` for a LaTeX version of this document.

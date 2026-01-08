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
- Reward: `reward_cvss_v31` (per-metric F1 over base metrics)
```
Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v3.1 base vector string (format: CVSS:3.1/AV:X/AC:X/PR:X/UI:X/S:X/C:X/I:X/A:X).

CVE description:
{CVE_DESCRIPTION}
```

**CVE -> CVSS v4.0 base vector**
- Input: CVE description
- Output: CVSS v4.0 base vector string
- Reward: `reward_cvss_v40` (per-metric F1 over base metrics)
```
Given the Common Vulnerabilities and Exposures (CVE) description below, provide the CVSS v4.0 base vector string (format: CVSS:4.0/AV:X/AC:X/AT:X/PR:X/UI:X/VC:X/VI:X/VA:X/SC:X/SI:X/SA:X).

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

**Detection Strategy**
- Input: Adversary procedure description
- Output: ATT&CK detection strategy ID (DET####)
- Reward: `reward_detection_id` (1 if ID matches, else 0)
```
Given the adversary procedure description below, provide the single most appropriate MITRE ATT&CK Enterprise detection strategy ID (DET####) that best detects this behavior.

Requirements:
- Use MITRE ATT&CK Enterprise detection strategy IDs only.
- Return exactly ONE detection strategy ID.

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

## Threat actor MCQ (procedures)
Built from MITRE ATT&CK intrusion-set procedure descriptions. For each actor alias, five variants are generated by sampling 60-100% of procedures and shuffling options; distractors are actors that do not cover the sampled techniques.

- Input: List of observed procedures
- Output: Option letter (A-E)
- Reward: `binary_id` (1 if correct option letter, else 0)
```
Given the observed adversary procedures below, choose the most likely threat actor.

Observed procedures:
{PROCEDURE_LIST}

Select the correct option (A-E) and return the option letter.

Options:
{OPTIONS_TEXT}
```

---

# Dataset Statistics (samples and split counts)
All counts derive from `dataset/minerva_split/metadata.json`.

file	Input	Output	Reward fn	#sample	#Train	#Val
cve_to_attack_exploitation	CVE description	ATT&CK Technique ID	reward_technique_id	265	212	53
cve_to_attack_primary_impact	CVE description	ATT&CK Technique ID	reward_technique_id	230	184	46
cve_to_attack_secondary_impact	CVE description	ATT&CK Technique ID	reward_technique_id	74	59	15
sigma_to_attack_technique	Sigma detection + logsource	ATT&CK Technique ID	reward_technique_id	1500	1200	300
sigma_to_attack_tactics	Sigma detection + logsource	ATT&CK Tactic IDs	reward_tactic_ids	1500	1200	300
scenario_to_technique	ATT&CK procedure scenario	ATT&CK Technique ID	reward_technique_id	8000	6400	1600
scenario_to_tactics	ATT&CK procedure scenario	ATT&CK Tactic IDs	reward_tactic_ids	3000	2400	600
scenario_to_detections	ATT&CK procedure scenario	ATT&CK Detection ID	reward_detection_id	3000	2400	600
scenario_to_mitigations	ATT&CK procedure scenario	ATT&CK Mitigation IDs	reward_mitigation_ids	3000	2400	600
cve_to_cwe	CVE description	CWE IDs	reward_cwe_ids	10000	8000	2000
cve_to_cvss_v31	CVE description	CVSS v3.1 vector string	reward_cvss_v31	4081	3265	816
cve_to_cvss_v40	CVE description	CVSS v4.0 vector string	reward_cvss_v40	1000	800	200
capec_example_to_capec	CAPEC example text	CAPEC ID	binary_id	380	304	76
capec_example_to_cwe	CAPEC example text	CWE IDs	reward_cwe_ids	196	157	39
capec_example_to_attack	CAPEC example text	ATT&CK Technique ID	reward_technique_id	144	115	29
threat_actor_mcq	Procedures list	Threat actor option (A-E)	binary_id	3630	2904	726

---

# Reward Function Notes
- `reward_technique_id`: 1.0 if technique+subtechnique match; 0.5 if parent technique matches but subtech differs/is missing; else 0.
- `reward_tactic_ids`: F1 over predicted vs. ground-truth tactic ID sets.
- `reward_cwe_ids`: F1 over predicted vs. ground-truth CWE ID sets.
- `reward_cvss_v31` / `reward_cvss_v40`: parse base metrics from vectors (v3.1: AV, AC, PR, UI, S, C, I, A; v4.0: AV, AC, AT, PR, UI, VC, VI, VA, SC, SI, SA) and compute per-metric F1.
- `reward_mitigation_ids`: F1 over predicted vs. ground-truth mitigation ID sets.
- `reward_detection_id`: 1.0 if detection ID matches; else 0.
- `binary_id`: 1.0 if IDs match; else 0.

See `docs/task-descriptions.tex` for a LaTeX version of this document.

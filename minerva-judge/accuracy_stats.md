# Minerva Judge Accuracy Stats (Overall)

Source file: `minerva-judge/data/responses_llama3_2_3b.jsonl`

## Prompt variants

| Variant | Correct | Incorrect | Total | Accuracy |
| --- | --- | --- | --- | --- |
| hinted | 23436 | 8564 | 32000 | 73.24% |
| plain | 770 | 31230 | 32000 | 2.41% |

## Task-wise accuracy (plain)

| Task | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| cve_to_cwe | 623 | 6696 | 9.30% |
| cve_to_attack_exploitation | 10 | 245 | 4.08% |
| capec_example_to_cwe | 4 | 176 | 2.27% |
| scenario_to_attack_tactics | 35 | 1950 | 1.79% |
| sentinel_to_attack_technique | 14 | 950 | 1.47% |
| sigma_to_attack_tactics | 14 | 950 | 1.47% |
| cve_to_attack_primary_impact | 2 | 210 | 0.95% |
| art_to_attack_technique | 7 | 950 | 0.74% |
| scenario_to_attack_technique | 45 | 7780 | 0.58% |
| sigma_to_attack_technique | 5 | 950 | 0.53% |
| threat_actor | 4 | 779 | 0.51% |
| cve_to_cvss_v31 | 6 | 1900 | 0.32% |
| scenario_to_attack_mitigations | 1 | 7780 | 0.01% |
| capec_example_to_capec | 0 | 340 | 0.00% |
| splunk_to_attack_technique | 0 | 280 | 0.00% |
| cve_to_attack_secondary_impact | 0 | 64 | 0.00% |

## Task-wise accuracy (hinted)

| Task | Correct | Total | Accuracy |
| --- | --- | --- | --- |
| threat_actor | 735 | 779 | 94.35% |
| scenario_to_attack_tactics | 1801 | 1950 | 92.36% |
| capec_example_to_cwe | 151 | 176 | 85.80% |
| cve_to_cwe | 5487 | 6696 | 81.94% |
| cve_to_attack_exploitation | 190 | 245 | 77.55% |
| sentinel_to_attack_technique | 722 | 950 | 76.00% |
| art_to_attack_technique | 712 | 950 | 74.95% |
| scenario_to_attack_technique | 5815 | 7780 | 74.74% |
| splunk_to_attack_technique | 209 | 280 | 74.64% |
| sigma_to_attack_technique | 685 | 950 | 72.11% |
| cve_to_attack_primary_impact | 142 | 210 | 67.62% |
| sigma_to_attack_tactics | 637 | 950 | 67.05% |
| capec_example_to_capec | 223 | 340 | 65.59% |
| cve_to_cvss_v31 | 1187 | 1900 | 62.47% |
| scenario_to_attack_mitigations | 4705 | 7780 | 60.48% |
| cve_to_attack_secondary_impact | 35 | 64 | 54.69% |

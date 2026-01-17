# Minerva Dataset Preparation

This document describes the full dataset preparation pipeline for Minerva, suitable for inclusion in a research methods section. It covers data sources, task construction, resampling, train/validation splitting, LHC augmentation, retrieval artifacts, and conversion to RL-ready formats.

## Overview
Minerva builds RL-friendly cyber threat intelligence datasets from public CTI sources. The pipeline produces:
- Base datasets (`dataset/minerva_base/*.jsonl` + metadata)
- LHC datasets with retrieval candidate pools (`dataset/minerva_lhc/*.jsonl` + metadata)
- Train/dev splits in minimal schema (`dataset/minerva_base_split/` and `dataset/minerva_lhc_split/`)
- RLVR-ready Parquet and Excel outputs (`rlvr/mydata/minerva_base/`, `rlvr/mydata/minerva_lhc/`)

All paths and task parameters are configured through `minerva/config.yaml` and cached source files under `dataset/`.

## Data sources and caching
Inputs are fetched once and cached locally:
- NVD CVE records: `dataset/nvd/`
- MITRE ATT&CK bundle: `dataset/mitre/enterprise-attack.json`
- CAPEC bundle: `dataset/capec/stix-capec.json`
- Sigma rules: `dataset/sigma/`
- Mappings-explorer CVE->ATT&CK mappings: `dataset/mappings-explorer/`

Network fetches are rate-limited; use `NVD_API_KEY` to avoid strict NVD throttling.

## Task construction
The pipeline generates per-task JSONL files with standardized `task`, `input`, `ground_truth`, and `reward_fn` fields. Tasks include:
- CVE -> ATT&CK exploitation / primary impact / secondary impact
- Sigma -> ATT&CK technique / tactics
- Scenario -> technique / tactics / mitigations
- CVE -> CWE
- CVE -> CVSS v3.1

Generation uses deterministic seeds where applicable; prompt text is embedded in each row.

Filtering notes:
- CVE -> CWE drops any CVE whose CWE list includes `NVD-CWE-noinfo` (noinfo-only and mixed).
- CVE -> CVSS v3.1/v4.0 also skips CVEs with `NVD-CWE-noinfo` to keep label-less rows out of CVE-derived tasks.

## Resampling policy (fixed per-task totals)
After task generation, each per-task JSONL is resampled to a fixed target size using a seeded reservoir sampler (seed 1337). This yields a total of 33,000 examples across tasks.

| Task | Train samples | Validation samples | Total samples |
| --- | --- | --- |
| cve_to_attack_exploitation | 245 | 20 | 265 |
| cve_to_attack_primary_impact | 210 | 20 | 230 |
| cve_to_attack_secondary_impact | 54 | 20 | 74 |
| sigma_to_attack_technique | 1450 | 50 | 1500 |
| sigma_to_attack_tactics | 881 | 50 | 931 |
| scenario_to_technique | 7780 | 220 | 8000 |
| scenario_to_tactics | 1950 | 50 | 2000 |
| scenario_to_mitigations | 7780 | 220 | 8000 |
| cve_to_cwe | 9750 | 250 | 10000 |
| cve_to_cvss_v31 | 1900 | 100 | 2000 |
| **Total** | **32000** | **1000** | **33000** |

## Train/validation split policy
Splits are produced by sampling the per-task totals above, shuffling with a fixed RNG seed (1337), and assigning a fixed number of validation rows per task (table above). The remainder of each task is assigned to training. This yields exactly 32,000 training rows and 1,000 validation rows. No cross-task balancing is applied after the per-task split.

Split outputs:
- Base: `dataset/minerva_base_split/minerva-base-train.jsonl`, `minerva-base-dev.jsonl`
- LHC: `dataset/minerva_lhc_split/minerva-lhc-train.jsonl`, `minerva-lhc-dev.jsonl`

## LHC augmentation (candidate pools)
The LHC dataset augments each base example with a `candidate_pool_top100` list using task-specific retrieval strategies. Candidate pools are computed with a combination of BM25 and dense retrieval using the retrieval selection spec at `minerva/analysis/retrieval_selection/retrieval_selection.json`.

## Retrieval artifacts
For TARBA and retrieval-enabled training/evaluation, canonical label documents and BM25 indexes are built into:
- `dataset/retrieval/label_docs/`
- `dataset/retrieval/index/`

These artifacts are generated from the cached MITRE/CAPEC/CWE sources.

## RLVR conversion
Split JSONLs are converted to VeRL-ready Parquet and Excel files using `rlvr/mydata/data_prepare/cti.py`. The conversion injects CTI system prompts and preserves the per-row `reward_fn` as the data source for Minerva datasets.

Outputs:
- `rlvr/mydata/minerva_base/*.parquet` and `.xlsx`
- `rlvr/mydata/minerva_lhc/*.parquet` and `.xlsx`

## ACR leakage guardrails
Answer-conditioned reasoning (ACR) runs apply explicit meta-leak filters across Minerva tasks
(CWE, CVSS, ATT&CK technique/tactic/mitigation). Responses are flagged if they refer to
label/reference/details text or hint at provided options, candidate pools, or given IDs
(e.g., “label reference,” “according to the label,” “given mitigation list”). The check
uses explicit phrase and regex matching (fuzzy similarity is disabled) to focus on
leakage about provided materials rather than content-level reasoning.

## Auxiliary validation: IFEval
Instruction-following validation uses the IFEval prompts from
`google-research/instruction_following_eval/data/input_data.jsonl`.
Convert it into VeRL-ready Parquet via:

```bash
python rlvr/mydata/data_prepare/ifeval.py \
  --input_data /path/to/instruction_following_eval/data/input_data.jsonl \
  --out_dir rlvr/mydata/ifeval
```

The evaluation logic lives under `minerva/instruction_following_eval/` and
requires `absl-py`, `langdetect`, `nltk`, and `immutabledict` (plus the NLTK
`punkt` tokenizer data).

## Reproducible command sequence
The following sequence rebuilds all artifacts end-to-end:

```bash
# Base + LHC datasets (includes resampling)
python -m minerva.pipeline --config minerva/config.yaml --variant base --output-root dataset/minerva_base
python -m minerva.pipeline --config minerva/config.yaml --variant lhc --output-root dataset/minerva_base --lhc-output-root dataset/minerva_lhc

# Splits (fixed per-task validation counts)
python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base
python -m minerva.split --input-dir dataset/minerva_lhc --output-dir dataset/minerva_lhc_split --file-prefix minerva-lhc --reuse-split dataset/minerva_base_split
python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_split

# Retrieval artifacts
python -m minerva.retrieval.build_label_docs --config configs/retrieval/label_docs.yaml --out_dir dataset/retrieval/label_docs
python -m minerva.retrieval.build_index --label_docs_dir dataset/retrieval/label_docs --out_dir dataset/retrieval/index

# RLVR conversion
python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_base_split --out_dir rlvr/mydata/minerva_base
python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_lhc_split --out_dir rlvr/mydata/minerva_lhc
```

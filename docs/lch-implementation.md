# LHC (Label Hint Curriculum) Implementation

This document describes the LHC data flow and the custom dataloader that injects label hints at train time to stabilize a ~50% solve rate.

## Dataset creation (base + LHC)
1) Build base datasets:
   - `python -m minerva.pipeline --config minerva/config.yaml --variant base --output-root dataset/minerva_base`
2) Build LHC datasets by augmenting base rows with candidate pools:
   - `python -m minerva.pipeline --config minerva/config.yaml --variant lhc --output-root dataset/minerva_base --lhc-output-root dataset/minerva_lhc`

### Candidate pools for LHC
- `minerva/analysis/select_retrieval_method.py` evaluates BM25, dense, and hybrid retrieval (alpha ∈ {0.1..0.9}) on ≤1000 sampled rows per label family and records the best MRR in `minerva/analysis/retrieval_selection/retrieval_selection.json`.
- `minerva/analysis/retrieval_candidates.py` builds corpora from MITRE/CAPEC/CWE sources and uses the selected method per label family to retrieve top-100 candidates.
- The LHC pipeline adds `candidate_pool_top100` as a list of dicts in each row:
  - `{"id": "...", "name": "...", "description": "..."}` ordered by retrieval score (best-first).
  - Gold IDs are ensured to appear in the pool; if missing, they are appended and non-gold entries are trimmed.
- CVSS tasks do not use retrieval pools and are copied unchanged from base.

## Train/dev split (shared dev set)
- Base split:
  - `python -m minerva.split --input-dir dataset/minerva_base --output-dir dataset/minerva_base_split --file-prefix minerva-base`
- LHC split (reuses base dev):
  - `python -m minerva.split --input-dir dataset/minerva_lhc --output-dir dataset/minerva_lhc_split --file-prefix minerva-lhc --reuse-split dataset/minerva_base_split`
- Reuse is keyed on `(task, reward_fn, prompt)` so the dev set matches exactly across base and LHC.

## Parquet conversion (VeRL format)
- Base:
  - `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_base_split --out_dir rlvr/mydata/minerva_base`
- LHC:
  - `python rlvr/mydata/data_prepare/cti.py --data_dir dataset/minerva_lhc_split --out_dir rlvr/mydata/minerva_lhc`
- `candidate_pool_top100` is preserved at the top-level row for the LHC dataset.

## Custom dataloader (LHC)
LHC uses a custom dataset class that extends `RLHFDataset`:
- `rlvr/verl/utils/dataset/minerva_adaptive_dataset.py`
- Class: `AdaptiveOptionRLHFDataset`

### How it injects hints
For each example (with a candidate pool):
1) Select a task key (`reward_fn`/`data_source`) and read the current curriculum state `(K, p_drop)`.
2) Sample K-1 distractors from the top-N retrieved candidates and add the gold ID.
3) Append a “Candidate IDs” block to the user prompt:
   - `ID | name` by default; set `include_descriptions=true` to add `description` (truncated to `option_desc_max_chars`).
4) If the prompt exceeds `max_prompt_length`, reduce K until it fits.

### Curriculum controller
- `OptionCurriculumController` tracks EMA accuracy per task and adapts K and `p_drop` to target ~50% solve rate.
  - If accuracy is high, increase K (harder) or raise `p_drop`.
  - If accuracy is low, decrease K or reset to `k_min`.
- Update hook: `rlvr/verl/trainer/ppo/ray_trainer.py` calls
  `update_option_curriculum_from_rollout(...)` after reward computation.

### Config snippet (Hydra CLI)
```
data.custom_cls.path=rlvr/verl/utils/dataset/minerva_adaptive_dataset.py
data.custom_cls.name=AdaptiveOptionRLHFDataset
data.adaptive_options.enabled=true
data.adaptive_options.candidate_pool_key=candidate_pool_top100
data.adaptive_options.target_acc=0.5
data.adaptive_options.k_min=2
data.adaptive_options.k_max=30
data.adaptive_options.k_step=2
data.adaptive_options.p_drop_step=0.05
data.adaptive_options.score_threshold=0.5
data.adaptive_options.include_descriptions=false
data.adaptive_options.option_desc_max_chars=400
```

## Training scripts (Llama 3B)
- Base: `rlvr/cti-scripts/train_minerva_base_llama3b.sh`
- LHC: `rlvr/cti-scripts/train_minerva_lhc_llama3b.sh`

The LHC script enables the adaptive dataset class and curriculum options, while the base script uses the default `RLHFDataset`.

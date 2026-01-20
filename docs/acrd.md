# ACRD: Answer-Conditioned Reasoning + Distillation (Per-Batch)

This spec replaces LHC-style candidate hints and TARBA retrieval with a per-batch loop:
1) regular RLVR on the original prompt,
2) answer-conditioned reasoning trace generation (SFT: hard examples only; DPO: all UIDs with RLVR reuse on max reward),
3) periodic distillation (SFT or DPO; default SFT) from accepted ACR traces.

ACRD uses the same RLVR trainer, with a per-batch ACR + distill phase inserted into
`rlvr/verl/trainer/ppo/ray_trainer.py`.

---

## High-level algorithm

Let training samples be (x, y*) with `data_source` identifying the task.

### Step 1: RLVR (baseline)
Run your existing RLVR training loop on the standard prompts (no ground truth revealed).

### Step 2: ACR trace generation (answer-conditioned)
Per-batch ACR is built inside `rlvr/verl/trainer/ppo/ray_trainer.py` from the same
training batch used in Step 1. It appends an ACR block to the last user message of
`raw_prompt` (so `data.return_raw_chat=true` is required).

The ACR block is produced by `minerva/acr_prompt.build_acr_block(...)` and includes:
- GROUND_TRUTH_LABELS (one per line)
- LABEL_REFERENCE (from `LabelDetailsStore`, joined with `\n\n---\n\n` for multi-label)
- Instructions to produce reasoning + final answer in the original task format
- Task-specific reasoning instruction from `data.acr.task_reasoning_hints`
  (fallback to `data.acr.entity_reasoning_hints` by label type)

Prompt-length handling (`try_build_acr_messages`):
- First try full details (up to `data.acr.max_details_chars`).
- If still too long, omit details entirely.
- If still too long, skip ACR for that sample.
If `entity_type` is not resolved (e.g., CVSS tasks), the canonical details section is omitted.
For MCQ tasks with option letters, the ACR builder maps the gold option letter to the
option text and uses that text (or its extracted ID) to fetch label details.

ACR rollouts use the same actor as RLVR but a separate repeat count:
`data.acr.rollout_n` (default 4). No PPO update is run by default; the ACR score is
used only for filtering/ranking traces before distillation.
Hard-example gating (SFT only): per-batch ACR runs according to `data.acr.hard_reward_mode`:
- `no_perfect` (default): run ACR only when the UID has no rollout with reward >= 1.0 (max reward < 1.0).
- `mean_reward`: run ACR only when mean RLVR reward across rollouts is below
  `data.acr.hard_reward_threshold` (default 0.05).
This keeps distillation focused on unsolved cases and avoids over-teaching already-correct samples.
For DPO, we skip the hard-threshold filter; if a UID's **max** RLVR reward is 1.0, we do **not** generate
ACR responses and instead reuse the top `data.acr.rollout_n` RLVR rollouts (sorted by RLVR reward, random tie-break).
Otherwise we generate `data.acr.rollout_n` ACR responses with answer hints as usual.

Reward (`reward_acr`) uses the RLVR parsing logic (`reward_minerva`) and returns:
- `acr_base_score` = `reward_minerva` score (0..1), before scaling
- `score` = `r_correct * acr_base_score - leak_penalty` (no ID leak penalty)
- `acr_leak_hit` from banned phrase/regex detection (explicit match by default; fuzzy optional),
  plus short reasoning, low-overlap reasoning, verbatim overlap, and optional ID-overuse checks
- `acr_banned_phrase_hit` for the phrase/regex match component only
- `acr_short_hit` and `acr_overlap_hit` for minimum-reasoning and overlap guards (logged only)
- `acr_id_leak_hit` if the reasoning section contains any gold IDs (logged only)
Optional judge rubric (when enabled) adds:
- `acr_rubric_score` (float, 0..1) used for DPO ranking (chosen/rejected selection); computed as the
  weighted sum of Q1-Q4 (1-4 scale, weights 0.30/0.30/0.20/0.20) normalized via
  `(S_1to4 - 1.0) / 3.0` from `rlvr/verl/utils/reward_score/prompts/acr_rubric_prompt.txt`. The
  prompt expects JSON keys `Q1`, `Q2`, `Q3`, `Q4` and does not mention weights.

### Step 3: Distillation (SFT or DPO)
Per-batch distillation is handled by the PPO trainer and uses the same batch.
Set `data.acr.distill.method = sft | dpo` (default: `sft`).

See `docs/acrd-dpo.md` for the current ACRD-DPO selection logic and rubric.

For `method=sft`, input is the original RLVR prompt (`acr_orig_prompt`), while the target is the ACR trace.
The ACR prompt (with answer hints) is kept only for debugging and selection analysis.
Selection is per-UID (one record per original prompt):

- Eligibility: `acr_base_score >= data.acr.distill.reward_threshold` (default 1.0 in Minerva cti-scripts),
  and `acr_leak_hit == false`.
- Leakage includes banned-phrase detection (fuzzy match optional), short reasoning (default: <100 chars),
  low overlap with the combined task description + label reference (Jaccard <0.05; task description
  extracted from the original prompt block; overlap computed on the reasoning portion only),
  verbatim overlap (default: two 10-word spans from `LABEL_REFERENCE`, with details >= 100 chars),
  plus optional ID overuse when `max_id_mentions > 0` (default 0 in `cti-scripts`; reward default 3).
- Degenerate filter (default on): drop eligible responses with high n-gram repetition using
  `rep_n = 1 - distinct_n` over token IDs (defaults: `rep_3 >= 0.85` or `rep_4 >= 0.90`,
  applied when `len(response_tokens) >= 40`).
- Default selection (`selection_mode=random`): sample uniformly from eligible rollouts.
- Optional selection (`selection_mode=top_reward`): pick the response with the highest weighted ACR reward
  (`score` includes leak penalty), then tie-break by `entropy_tiebreak` (mean NLL or response length).
  If `entropy_sampling=true`, sample among top-rewarded candidates with probability proportional to
  `exp(beta * mean_nll)` (stochastic entropy tie-break). Mean NLL is computed over assistant tokens only (SFT only).

Accepted SFT records (`method=sft`) are stored as:
- `prompt_nohint`: `extra_info["acr_orig_prompt"]` (original RLVR prompt; no answer hints)
- `response_ids`: selected response token IDs
- `reward`: weighted ACR score
- `entropy_proxy`: mean NLL of the selected response (assistant tokens only)

Per distill run, we sample up to `data.acr.distill.batch_size` records from the buffer
uniformly at random (no replacement) and run a single SFT update. If fewer records exist,
we use all available records. The buffer is a rolling queue capped at
`data.acr.distill.max_buffer`; once the cap is exceeded, the oldest records are dropped.

Buffer behavior is controlled by `data.acr.distill.buffer_mode`:
- `rolling` (default): run on `interval`, keep buffer between runs.
- `flush`: run on `interval`, sample up to `batch_size` records for SFT, then clear the buffer
  (no max-buffer trimming in flush mode).
- `buffer`: ignore `interval`, run only when the global buffer reaches `data.acr.distill.min_buffer`
  (default 256); use all buffered records, then clear.
`buffer_mode` applies to SFT; DPO remains interval-based.

For `method=dpo`, we consider every UID in the batch (no hard-threshold filtering) and build a
preference pair from a pool of size `data.acr.rollout_n`:
- If max RLVR reward == 1.0, use the top-N RLVR rollouts (sorted by RLVR reward, random tie-break).
- Otherwise, use the N ACR rollouts generated with answer hints.

For DPO selection, the reward is the base verifier reward (`reward_minerva` for RLVR rollouts,
`acr_base_score` for ACR rollouts). Leak penalties and `reward_acr.score` are not used.

Chosen: among responses with reward >= `data.acr.distill.reward_threshold`, pick the highest rubric
score (if present).
Rejected: among responses with reward < `data.acr.distill.reward_threshold`, pick the highest rubric
(tie-break by higher reward, then deterministic).
If no reward<threshold response (all reward>=threshold), pick the lowest rubric among reward>=threshold
candidates as rejected (distinct from chosen). Skip if you cannot form a pair.

DPO stores `(prompt_nohint, chosen_ids, rejected_ids)` (plus metadata), not `response_ids`,
and does not use the entropy/mean-NLL tie-break logic (SFT only).

Distillation runs every `data.acr.distill.interval` steps (default 10) when
`buffer_mode` is `rolling` or `flush`. For `buffer` mode, SFT triggers only when
the global buffer size reaches `data.acr.distill.min_buffer`, then clears the buffer.

---

## Implementation map

### 0) Task specs (data_source -> entity type)
Implemented in `minerva/cti_task_specs.py`:

```python
TASK_SPECS[data_source] = {
  "entity_type": "attack_technique_id" | "cwe_id" | "threat_actor_name" | None,
  "is_multilabel": bool,
  "label_id_regex": str | None,  # optional for ID leak checks
}
```

Rules:
- Prefer `extra_info["source_file"]` (JSONL stem) or `extra_info["task"]` to resolve the task key.
- If a task key maps to `minerva.analysis.retrieval_candidates.TASK_SPECS`, use that.
- Fall back to `minerva.retrieval.task_specs.get_task_spec(data_source, ground_truth)`.
- If nothing resolves, infer `is_multilabel` from ground_truth type (list/tuple -> multi).

This avoids ambiguity in `reward_capec_id` tasks by using the JSONL file stem or task name.

---

### 1) Canonical label-details store (for Step 2 prompts)
We need a fast lookup from `(entity_type, label_id)` to a short, canonical details string.

#### 1.1 Build details JSONL
`scripts/build_label_details.py` generates:
- `dataset/label_details/<entity_type>.jsonl`
- `dataset/label_details/_manifest.json`
- `dataset/label_details/metadata.json` (counts + examples per label type)

Each JSONL record:

```json
{
  "key": "attack_technique_id:T1059.003",
  "entity_type": "attack_technique_id",
  "canonical_id": "T1059.003",
  "name": "Windows Command Shell",
  "aliases": ["cmd.exe", "Command Prompt"],
  "definition": "...",
  "metadata": {"tactics": ["Execution"], "platforms": ["Windows"]},
  "details_text": "ID: T1059.003\nName: Windows Command Shell\nAdversaries may...\n"
}
```

Rules:
- No dataset examples.
- Store full `details_text` (no build-time cap). Cap at runtime in the ACR dataset if needed.
- Source MITRE/CAPEC/CWE from cached files in `dataset/`.
- Details are intentionally compact; relationship lists are omitted from label details text.
- Preserve newlines in `details_text`; they are passed through into the ACR prompt.

Details formatting (current):
- `attack_tactic_id`: `ID:` + `Name:` + tactic description only.
- `attack_technique_id`: `ID:` + `Name:`; optional `Sub-techniques (N)` table (`ID\tName`); then the technique description
  (no mitigations/detections lists).
- `mitigation_id`: `ID:` + `Name:` + full MITRE description in `Definition:` (no techniques list).
- `cwe_id`: `<CWE-####>: <Name>` header, `Description` section, optional `Extended Description`,
  optional `Background Details` (each section separated by a blank line).
Notes:
- ATT&CK tactics/techniques use the full MITRE description (no paragraph truncation).
- Mitigations use the full MITRE description (no paragraph truncation).

#### 1.2 Runtime loader
`minerva/label_details_store.py` loads all JSONL files into a dict mapping `key -> details_text`.
It also builds an alias index (canonical ID, name, and aliases) so values like
"Pawn Storm" resolve to `APT28`.

```python
def get_details(entity_type: str, label_ids: list[str]) -> str:
    # concatenate details blocks with "\n\n---\n\n" between labels
```

---

### 2) ACR dataset (answer-conditioned RL)
Standalone ACR runs can use `rlvr/verl/utils/dataset/minerva_acr_dataset.py` (`ACRRLHFDataset`).
Per-batch ACRD does not use this class; it builds the ACR batch directly in the trainer.

Key points:
- Inherit from `RLHFDataset`.
- Set `data.dataloader_num_workers=0` in configs.
- Store the original prompt for distillation in `extra_info["acr_orig_prompt"]` (deep copy).
- Store ACR metadata:
  - `extra_info["acr_entity_type"]`
  - `extra_info["acr_gold_labels"]`
  - `extra_info["acr_gold_keys"]` (e.g., `entity_type:ID`)
  - `extra_info["acr_mode"] = "acr"`

#### 2.1 ACR prompt template
The ACR block is appended to the last user message:

```text
You are generating a reasoning trace for training.

GROUND_TRUTH_LABELS:
- <ID1>
- <ID2>

LABEL_REFERENCE:
<details_text from store>

Instructions:
- Write a short reasoning that would justify selecting the correct label(s) from the input.
- <TASK_SPECIFIC_REASONING_HINT>
- Do NOT say or imply that the answer was provided (no phrases like "given the answer", "based on the provided label", "ground truth", etc.).
- End with the final answer in the same format required by the original task.
```

Prompt-length control:
- If the prompt exceeds `data.acr.max_prompt_length`, truncate `details_text` first.
- If still too long, keep the GROUND_TRUTH block but omit details.
- If still too long, skip ACR for that sample.

---

### 3) ACR reward (verifiable, no LLM judge)
Implemented in `rlvr/verl/utils/reward_score/reward_acr.py` (`reward_acr`).

#### 3.1 Parsing and extraction
- Use the same extraction logic as `reward_minerva` (RLVR) for the task's answer format.
- Treat the final answer as the last non-empty line (reasoning comes before it).

#### 3.2 Correctness
- Uses the `reward_minerva` score as `acr_base_score`.
- `acr_is_correct` is `acr_base_score == 1.0` (full match).

#### 3.3 Leakage phrase penalties
- Use a small banned-phrase list (case-insensitive substring + regexes).
- Fuzzy check via `difflib.SequenceMatcher` is optional (disabled by default).

#### 3.4 Minimum reasoning length
- If the reasoning section is shorter than `min_reasoning_chars` (default 100), mark `acr_short_hit`
  and treat as leakage.

#### 3.5 Minimum overlap with task context
- Compute Jaccard overlap between reasoning and the combined task description + `LABEL_REFERENCE`
  (tokenized with IDs stripped). If overlap < `min_overlap_jaccard` (default 0.05), mark `acr_overlap_hit`
  and treat as leakage.

#### 3.6 Verbatim label-detail overlap
- If `LABEL_REFERENCE` is at least `verbatim_min_details_chars` (default 100 chars) and the reasoning contains
  at least `verbatim_min_matches` exact `verbatim_ngram_size`-word spans (defaults: 2 spans, size 10),
  mark `acr_verbatim_hit` and treat as leakage.

#### 3.7 "ID in reasoning" detection
- The reasoning section (everything before the last non-empty line) is checked for gold IDs.
- This is logged as `acr_id_leak_hit` but is not penalized in the score.

#### 3.8 Reward scalar
Example (current defaults):

```text
r = 0
if extracted and acr_base_score > 0:
  r += r_correct * acr_base_score
if leak_hit:
  r -= leak_penalty       # e.g. 0.5
```

Return a dict with:
- `score` (float)
- `acr_base_score`
- `acr_is_correct`
- `acr_extracted`
- `acr_leak_hit`
- `acr_banned_phrase_hit`
- `acr_short_hit`
- `acr_overlap_hit`
- `acr_id_leak_hit`
- `acr_verbatim_hit`
- `acr_id_overuse_hit` / `acr_id_overuse_max` (only when `max_id_mentions > 0`)
The score is used for filtering/ranking traces, not for PPO updates by default.

---

### 4) Accepted-trace collection for distillation
Per-batch ACRD uses the trainer's in-memory distill buffer. Offline collection is optional.

#### 4.1 Buffer
Per-batch ACRD uses the trainer's in-memory distill buffer (configured under `data.acr.distill.*`).
The JSONL buffer is only needed for the optional offline SFT pipeline.

For `method=sft`, each per-batch record includes:
- `uid` (original prompt ID)
- `task_key` (data_source)
- `prompt_nohint` (historical name; contains the original RLVR prompt)
- `response_ids` (selected rollout response)
- `reward` (weighted ACR score)
- `entropy_proxy` (mean NLL of the selected response)

For `method=dpo`, each per-batch record includes:
- `uid` (original prompt ID)
- `task_key` (data_source)
- `prompt_nohint` (historical name; contains the original RLVR prompt)
- `chosen_ids`, `rejected_ids`
- `chosen_meta`, `rejected_meta` (ACR scores)

#### 4.2 Acceptance criteria (per-batch)
- Eligibility (SFT): `acr_base_score >= reward_threshold` (default 1.0 in Minerva cti-scripts),
  and `acr_leak_hit == false`.
- Degenerate filter (SFT): drop eligible responses with high n-gram repetition
  using `rep_n = 1 - distinct_n` over token IDs (defaults: `rep_3 >= 0.85` or `rep_4 >= 0.90`,
  applied when `len(response_tokens) >= 40`).
- SFT selection: default random among eligible (`selection_mode=random`). Set `selection_mode=top_reward`
  to pick max weighted `score` with optional entropy/length tie-break (mean NLL over assistant tokens).
- DPO pool: `data.acr.rollout_n` responses per UID (RLVR rollouts if max reward == 1.0,
  otherwise ACR rollouts with answer hints).
- DPO selection: chosen = highest rubric among reward>=threshold responses; rejected = highest
  rubric among reward<threshold responses (tie-break by higher reward). If all reward>=threshold,
  rejected falls back to the lowest rubric among reward>=threshold responses (distinct from chosen).
  DPO ignores leak penalties.

#### 4.3 Collection mode
Option A (preferred): per-batch collection via `data.acr.distill.enabled=true` and
`data.return_raw_chat=true` (SFT/DPO).

Option B (offline, SFT only): collect traces from a checkpoint.

```bash
python scripts/collect_acr_traces.py \
  --ckpt <hf_model_path> \
  --dataset <train_parquet> \
  --out outputs/acr_sft_buffer.jsonl
```

The offline pipeline is optional and mainly useful for debugging or analysis.

---

### 5) Optional SFT dataset build (offline pipeline, SFT only)
Applies when `data.acr.distill.method=sft`.
Use `scripts/build_sft_from_acr.py` to transform the JSONL buffer into an SFT dataset.

Output format (recommended for MultiTurnSFTDataset):
- `messages`: `orig_prompt + [{"role":"assistant","content": target_completion}]`
Uses `orig_prompt` only (ACR prompt is not used for SFT).

---

### 6) Per-batch training script (cti-scripts)
Use the single entrypoint `rlvr/cti-scripts/train_minerva_noctua.sh` to run RLVR -> ACR -> distill (SFT or DPO)
in a single PPO job (same batch per iteration). Model-specific wrappers are provided:

- `rlvr/cti-scripts/train_minerva_noctua_llama3b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_llama8b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_qwen4b.sh`
- `rlvr/cti-scripts/train_minerva_noctua_qwen8b.sh`

Example:

```bash
rlvr/cti-scripts/train_minerva_noctua.sh
```

Key env overrides:
- `ACRD_ACR_RL_WEIGHT` (scales the ACR PPO update; only used if `data.acr.update_actor=true`)
- `ACRD_ACR_ROLLOUT_N` (number of ACR samples per prompt; default 4)
- `ACRD_ACR_DISTILL_METHOD` (`sft` or `dpo`; default `sft`)
- `ACRD_ACR_DISTILL_INTERVAL` (distill interval in steps; default 10)
- `ACRD_ACR_DISTILL_LR_SCALE` (distill LR scale vs RLVR; default 1.0)
- `ACRD_ACR_DISTILL_THRESHOLD` (threshold applied to `acr_base_score` before r_correct scaling; default 1.0 in Minerva cti-scripts)
- `ACRD_ACR_DISTILL_SELECTION_MODE` (SFT only; `random` or `top_reward`; default `random`)
- `ACRD_ACR_DISTILL_BATCH_SIZE` (SFT only; sample size per distill run; default 256 in `cti-scripts`; ignored in `buffer` mode)
- `ACRD_ACR_DISTILL_MAX_BUFFER` (SFT only; rolling buffer cap; default 1024 in `cti-scripts`)
- `ACRD_ACR_DISTILL_MIN_BUFFER` (SFT only; `buffer` mode threshold; default 256 in `cti-scripts`)
- `ACRD_ACR_DISTILL_BUFFER_MODE` (SFT only; `rolling`, `flush`, or `buffer`; default `rolling`)
- `ACRD_ACR_DISTILL_DEGENERATE_FILTER` (SFT only; enable repetition filter; default `true`)
- `ACRD_ACR_DISTILL_DEGENERATE_MIN_TOKENS` (SFT only; min tokens before repetition filter; default 30)
- `ACRD_ACR_DISTILL_DEGENERATE_REP3_MAX` (SFT only; reject if `rep_3` >= this; default 0.70)
- `ACRD_ACR_DISTILL_DEGENERATE_REP4_MAX` (SFT only; reject if `rep_4` >= this; default 0.75)
- `ACRD_ACR_DISTILL_ENTROPY_BETA` (SFT only; softmax beta for entropy sampling when `selection_mode=top_reward`; default 1.0)
- `ACRD_ACR_DISTILL_ENTROPY_SAMPLING` (SFT only; enable stochastic entropy tie-break when `selection_mode=top_reward`; default true)
- `ACRD_DPO_BETA` (DPO only; default 0.1)
- `ACRD_ACR_REWARD_MANAGER` (`naive` or `batch`; use `batch` to enable judge batching)
- `ACRD_ACR_MAX_ID_MENTIONS` (leakage check; reject if gold ID appears more than this; default 0)
- `ACRD_ACR_MIN_REASONING_CHARS` (leakage check; reject if reasoning < this; default 100)
- `ACRD_ACR_MIN_OVERLAP_JACCARD` (leakage check; reject if overlap < this; default 0.05)
- `ACRD_ACR_VERBATIM_MIN_DETAILS_CHARS` (verbatim guard; default 100)
- `ACRD_ACR_VERBATIM_NGRAM_SIZE` (verbatim guard; default 10)
- `ACRD_ACR_VERBATIM_MIN_MATCHES` (verbatim guard; default 2)
- `ACRD_JUDGE_ENABLED` (true/false; only used with `ACRD_ACR_REWARD_MANAGER=batch`)
- `ACRD_JUDGE_MODEL` (default `openai/gpt-oss-20b`)
- `ACRD_JUDGE_BATCH_SIZE`, `ACRD_JUDGE_MAX_NEW_TOKENS`, `ACRD_JUDGE_TEMPERATURE`, `ACRD_JUDGE_TOP_P`
- `ACRD_JUDGE_DEVICE`, `ACRD_JUDGE_DEVICE_MAP`, `ACRD_JUDGE_DTYPE`, `ACRD_JUDGE_TRUST_REMOTE_CODE`
- `ACRD_ACR_HARD_REWARD_MODE` (`no_perfect` or `mean_reward`; default `no_perfect`)
- `ACRD_ACR_HARD_REWARD_THRESHOLD` (used when `hard_reward_mode=mean_reward`; default 1.0 in `train_minerva_noctua.sh`)
- `ACRD_ACR_SKIP_CVSS` (true/false; when true, CVSS v3.1/v4 tasks skip ACR generation + distillation; default true in `train_minerva_noctua.sh`)
- `ACRD_DEBUG_SAMPLES` (prints original prompt -> ACR prompt -> rollouts -> selected response; default 2)
- `ACRD_DETAILS_DEBUG_SAMPLES` (prints details-missing/omitted locations; default 2)
- `ACRD_MAX_DETAILS_CHARS`, `ACRD_ACR_MAX_PROMPT_LEN`
- `ACRD_TOTAL_STEPS`, `ACRD_TRAIN_BATCH_SIZE`
- `ACRD_EXPERIMENT_NAME` (defaults to `minerva_acrd_grpo_<model-slug>`)

---

### 7) Optional offline pipeline
The earlier offline ACR->SFT pipeline (collect traces -> build SFT parquet, SFT only) is still available
via `scripts/collect_acr_traces.py` and `scripts/build_sft_from_acr.py`, but the per-batch
trainer flow is the preferred path for ACRD.

---

### 8) Configs
Use `configs/acr_rl.yaml` for standalone ACR-only runs, or rely on the per-batch overrides
from `rlvr/cti-scripts/train_minerva_noctua.sh`.

Standalone ACR example:

```yaml
data:
  custom_cls:
    path: rlvr/verl/utils/dataset/minerva_acr_dataset.py
    name: ACRRLHFDataset
  dataloader_num_workers: 0

actor_rollout_ref:
  rollout:
    n: 1

custom_reward_function:
  path: rlvr/verl/utils/reward_score/reward_acr.py
  name: reward_acr
  reward_kwargs:
    r_correct: 0.2
    leak_penalty: 0.5
    multilabel_match: exact
    enforce_no_id_in_reasoning: true
```

Defaults:
- `banned_phrases` is defined in `rlvr/verl/utils/reward_score/reward_acr.py` (answer/label variants).
- `use_fuzzy_leak_check` is disabled by default (explicit phrase/regex leak checks only).
- `max_id_mentions` defaults to 3 in `reward_acr` but is 0 (disabled) in `cti-scripts` unless overridden.
- Verbatim guard defaults in `cti-scripts`: `verbatim_min_details_chars=100`, `verbatim_ngram_size=10`,
  `verbatim_min_matches=2`.
- SFT distill selection defaults to `selection_mode=random`. If you set `selection_mode=top_reward`,
  the default tie-break is `entropy_tiebreak=mean_nll` with `entropy_sampling=true` and `entropy_beta=1.0` (SFT only).
- SFT degenerate filter defaults to on for ACRD in `cti-scripts`:
  - `degenerate_min_tokens=30`
  - `rep_3 >= 0.70` or `rep_4 >= 0.75`
  - repeated-window check (`degenerate_window_size=24`, `degenerate_window_jaccard=0.9`)
  - near-duplicate sentence check (`degenerate_sentence_sim=0.75`, `degenerate_sentence_window=6`,
    `degenerate_sentence_min_words=6`)
- The default `cti-scripts` run samples up to 256 SFT records per distill step
  (`ACRD_ACR_DISTILL_BATCH_SIZE=256`).
- Judge rubric is disabled by default; enable via `ACRD_ACR_REWARD_MANAGER=batch` and `ACRD_JUDGE_ENABLED=true`.

Per-batch ACRD needs:
- `data.return_raw_chat=true`
- `data.acr.per_batch=true` plus `data.acr.*` overrides (weight, rollout_n, distill interval)
Set `data.acr.update_actor=true` only if you explicitly want PPO updates on ACR (default false).

---

### 9) Validation / evaluation
Use `scripts/eval_acr_outputs.py` to compute parse rate, leak-hit rate,
ID leak rate, and mean score on a JSONL file with outputs + ground_truth.
Supports `--multilabel-match` and `--enforce-no-id`.

---

## Implementation notes / gotchas
- Keep ACR reward small. It is a format/compliance shaper, not a second task objective.
- Keep the final answer on the last line, using the same format as the original task.
- Keep label details short. Do not blow up prompt length.
- Periodic distillation (`interval=10`) stabilizes training; for SFT this replays from the buffer.
- SFT eligibility uses `acr_base_score` (pre-scaling); SFT selection defaults to random among eligible
  rollouts. Set `selection_mode=top_reward` to use weighted reward with the optional mean-NLL/length tie-break.
- When `data.acr.distill.method=dpo`, ensure a reference policy is instantiated even if KL is disabled;
  DPO always needs logp_ref.
- Hard-example gating (SFT only) uses `data.acr.hard_reward_mode` (default `no_perfect`: max reward < 1.0);
  `mean_reward` uses `data.acr.hard_reward_threshold`. DPO skips the hard-threshold filter and uses max
  reward == 1.0 to decide whether to reuse RLVR rollouts or generate ACR rollouts.
- `ACRD_DEBUG_SAMPLES` prints original prompt -> ACR prompt -> rollouts -> selected response for quick inspection.
- `acr/details_missing`, `acr/details_omitted`, `acr/details_truncated` metrics track label-detail coverage.
- `acr/details_not_applicable` counts samples where no label details are expected (e.g., CVSS).
- `acr/hard_uid_total`, `acr/hard_uid_count`, `acr/hard_uid_used` report hard-example gating coverage.
- `acr/ppo_update_skipped=1` indicates ACR is running in generation-only mode (no PPO update).

## Practical recommendations
- Treat Step 2 as a trace generator + filter, not a direct accuracy booster.
- Tighten the output contract (final answer line only).
- Run distillation frequently enough to internalize traces, but not so frequently that RLVR stops exploring.

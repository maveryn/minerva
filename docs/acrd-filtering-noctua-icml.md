# ACRD Filtering (Noctua Llama-3B Run)

This document describes the **filtering step** (heuristics + ML) used during
ACRD self-distillation when running:

- `rlvr/cti-scripts/train_minerva_noctua_llama3b.sh`

It focuses strictly on knobs that are **active in filtering**, ignoring other
training settings. Implementation lives in:

- `rlvr/verl/trainer/ppo/ray_trainer.py` (distill candidate filtering)
- `rlvr/verl/utils/reward_score/reward_acr.py` (leak/overlap checks)
- `rlvr/cti-scripts/train_minerva_noctua.sh` (default env wiring)

---

## 1) Engineering Documentation (Noctua Llama-3B Defaults)

### 1.1 Active filter knobs for this run

**Enable/disable:**
- `ACRD_ACR_DISTILL_DISABLE_FILTERS=false` -> filters enabled
- `ACRD_ACR_DISTILL_FILTER_MODE=ml+heuristic`

**Base eligibility:**
- `ACRD_ACR_DISTILL_THRESHOLD=1.0`
  - Applied to `acr_base_score` from `reward_acr` (verifier score, pre-penalty).

**Heuristic leakage + degeneracy:**
- `ACRD_ACR_DISTILL_DEGENERATE_FILTER=true`
- `ACRD_ACR_DISTILL_DEGENERATE_MIN_TOKENS=30`
- `ACRD_ACR_DISTILL_DEGENERATE_REP3_MAX=0.70`
- `ACRD_ACR_DISTILL_DEGENERATE_REP4_MAX=0.75`
- `ACRD_ACR_MIN_REASONING_CHARS=100`
- `ACRD_ACR_MIN_OVERLAP_JACCARD=0.05`
- `ACRD_ACR_MAX_ID_MENTIONS=0` (ID-overuse disabled)
- `ACRD_ACR_ENFORCE_NO_ID_IN_REASONING=false` (ID-in-reasoning disabled)

**ML filter:**
- `ACRD_ACR_DISTILL_FILTER_MODEL=xashru/textcnn-response-only-lr6e-4-k345-f384-e300-t1024-d0p25`
- `ACRD_ACR_DISTILL_FILTER_MODEL_TYPE=textcnn`
- `ACRD_ACR_DISTILL_FILTER_THRESHOLD=0.5`
- `ACRD_ACR_DISTILL_FILTER_TEXT_MODE=response`
- `ACRD_ACR_DISTILL_FILTER_MAX_LENGTH=1024`
- `ACRD_ACR_DISTILL_FILTER_BATCH_SIZE=ACRD_TRAIN_BATCH_SIZE=128`
- `ACRD_ACR_DISTILL_FILTER_DEVICE=auto`
- `ACRD_ACR_DISTILL_FILTER_DTYPE=auto`
- `ACRD_ACR_DISTILL_FILTER_TRUST_REMOTE_CODE=false`

### 1.2 What "filtering" means in code

Filtering is the process of selecting **one ACR rollout per UID** to be saved
as an SFT distillation record. For ACRD SFT (the Noctua run), the filter is
applied inside `_collect_distill_candidates(...)` in `ray_trainer.py`.

The final SFT record uses:
- `prompt_nohint`: original prompt without ACR hints
- `response_ids`: tokens of the chosen ACR rollout

The filtering **does not use** the ACR reward's scaled score (`reward_acr.score`).
It uses:
- `acr_base_score` (base verifier score)
- `acr_leak_hit` (heuristic leak signal)
- degeneracy checks
- ML classifier score

### 1.3 Filtering pipeline (exact behavior)

For each UID, ACR rollouts are filtered and then one is selected:

1. **Base eligibility (correctness gate)**  
   Keep a rollout only if:
   - `acr_base_score >= 1.0`

2. **Heuristic leak gate**  
   Rollout must have `acr_leak_hit == false`.  
   In this run, `acr_leak_hit` is set true if **any** of:
   - banned phrase/regex matched in the response (explicit "label/reference"
     leakage cues in `reward_acr.py`)
   - reasoning length < 100 characters
   - Jaccard overlap between reasoning and (task description + label reference)
     is below 0.05

   ID-in-reasoning and ID-overuse checks are disabled for this run, so they do
   not affect `acr_leak_hit`.

3. **Degenerate response gate**  
   Enabled because filter mode includes heuristics. A response is rejected if:
   - `rep_3 >= 0.70` or `rep_4 >= 0.75` (repetition ratio over token n-grams), or
   - repeated token windows are found (exact repeat or high 3-gram Jaccard
     across windows), or
   - near-duplicate sentences detected (SequenceMatcher >= 0.75).

   Degenerate checks activate only when response length >= 30 tokens.

4. **ML filter (TextCNN)**  
   Remaining candidates are scored by a TextCNN classifier:
   - input tokens are extracted from the **response only** using
     `[A-Za-z0-9_]+` regex
   - sequence truncated to max 1024 tokens
   - model output uses class-1 ("good") probability
   - keep if `p_good >= 0.5`

   If the ML model fails to load or returns invalid scores, the pipeline
   automatically falls back to heuristic-only filtering (unless filters are
   explicitly disabled).

5. **Per-UID selection (after filtering)**  
   Because ML filtering is enabled, selection mode is forced to `ml_score`:
   - choose the rollout with the highest ML score
   - random tie-break among maxima

### 1.4 Leak features computed by `reward_acr`

Leak checks are implemented in `rlvr/verl/utils/reward_score/reward_acr.py`:

- **Banned phrase/regex leak**: explicit cues like "label reference", "given
  answer", etc. (lists in `DEFAULT_BANNED_PHRASES` and `DEFAULT_BANNED_REGEXES`).
- **Short reasoning**: reasoning portion (all lines except the final answer line)
  has fewer than 100 characters.
- **Low overlap**: Jaccard overlap of reasoning tokens vs.
  `(task_description + label_reference)` is below 0.05.

ID-in-reasoning and ID-overuse checks exist in the code but are disabled in this
run (by configuration).

### 1.5 Filtering metrics (logged)

The trainer logs filtering stats, useful for debugging and ablations:

- `distill/heuristic_input`, `distill/heuristic_passed`
- `distill/degenerate_filtered`
- `distill/ml_filter_input`, `distill/ml_filter_passed`
- `distill/ml_filter_threshold`

---

## 2) ICML-Style Description (Heuristic + ML Filtering)

This section is designed to be dropped into a paper appendix or methods section.

### Problem setup

For each training prompt UID \(i\), ACRD generates \(K\) answer-conditioned
rollouts \(\{r_{i,k}\}_{k=1}^K\). Each rollout is scored by a verifier
(\(\texttt{reward\_acr}\)) which returns:

- base verifier score \(s_{i,k} \in [0,1]\) (accuracy signal)
- leakage indicator \(L_{i,k} \in \{0,1\}\)

We apply a two-stage filter: a deterministic heuristic gate followed by an
optional ML gate. The Noctua configuration enables both.

### Heuristic filter

**Base correctness gate.** A rollout is eligible only if:

\[
s_{i,k} \ge \tau_s,\quad \tau_s = 1.0.
\]

**Leakage gate.** A rollout is rejected if \(L_{i,k}=1\). In this run,
\(L_{i,k}\) is set by three checks:

1. **Banned phrase/regex leak:** detect explicit references to the label,
   label details, or provided answers using a curated phrase list and regexes.
2. **Short reasoning:** reasoning text length \(< L_{\min}\) with
   \(L_{\min}=100\) characters. The reasoning is defined as all lines except
   the last non-empty line (which holds the final answer).
3. **Low overlap:** Jaccard overlap between reasoning tokens and the combined
   text of (task description + label reference) is below \(J_{\min}=0.05\).
   Tokens are alphanumeric word tokens (lowercased), length >= 3, with any
   ID-like tokens (e.g., T####, CWE-##) removed.

ID-in-reasoning and ID-overuse checks are disabled in the Noctua run and
therefore do not affect \(L_{i,k}\).

### Degenerate-response filter

We reject repetitive or degenerate responses using token-level heuristics.
Let \(\text{rep}_n\) be the repetition ratio over token IDs:

\[
\text{rep}_n = 1 - \frac{\lvert \text{distinct } n\text{-grams}\rvert}
{\lvert \text{all } n\text{-grams}\rvert}.
\]

A rollout is rejected if **any** of the following holds (only applied when
response length \(\ge 30\) tokens):

- \(\text{rep}_3 \ge 0.70\) or \(\text{rep}_4 \ge 0.75\)
- repeated token windows: window size 24, stride 12, reject if an exact
  window repeats or if two non-overlapping windows have 3-gram Jaccard
  similarity \(\ge 0.9\)
- near-duplicate sentences: within a 6-sentence window, reject if sentence
  similarity (SequenceMatcher) \(\ge 0.75\)

### ML filter

We apply a lightweight TextCNN classifier \(f_\theta\) as a second-stage filter.
Inputs are **response-only** tokens, extracted with regex `[A-Za-z0-9_]+` and
truncated to a max length of 1024 tokens. The classifier outputs a "good"
probability \(m_{i,k} \in [0,1]\), and we keep rollouts that satisfy:

\[
m_{i,k} \ge \tau_m,\quad \tau_m = 0.5.
\]

### Final eligibility and selection

The final eligible set for UID \(i\) is:

\[
\mathcal{E}_i = \{k \mid s_{i,k}\ge \tau_s,\; L_{i,k}=0,\; D_{i,k}=0,\;
m_{i,k}\ge \tau_m\}.
\]

If \(\mathcal{E}_i\) is empty, we do not distill a sample for that UID. Otherwise
we select one rollout:

\[
k^\* = \arg\max_{k\in \mathcal{E}_i} m_{i,k},
\]

breaking ties uniformly at random. The distillation target is the selected ACR
response, while the input prompt is the original RLVR prompt without ACR hints.

### Optional fallback (implementation guard)

If the ML model fails to load or returns invalid scores, the pipeline falls
back to heuristic-only filtering (unless filters are explicitly disabled). This
is a safety mechanism rather than an intended training mode.

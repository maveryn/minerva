# ACRD Step 3 DPO (Design + Implementation Guide)

This document describes how to add **DPO** as an alternative to **SFT** for
ACRD Step 3 (distillation) in the current Minerva codebase. Step 1 is unchanged;
Step 2 is method-aware for DPO (reuse RLVR rollouts when max reward is 1.0).

## ACRD-DPO Summary (Current Implementation)

- Step 1 (RLVR): generate rollouts and compute base verifier reward (`reward_minerva`).
- Step 2 (candidate pool, all UIDs):
  - If a UID's max RLVR reward is 1.0, **do not** run ACR; reuse the top `data.acr.rollout_n`
    RLVR rollouts (sorted by reward, deterministic random tie-break).
  - Otherwise, generate `data.acr.rollout_n` ACR rollouts with answer hints.
- DPO selection uses base rewards only (`reward_minerva` / `acr_base_score`), plus optional
  judge rubric `acr_rubric_score` (0..1). Leak penalties and `reward_acr.score` are not used.
- Pair per UID:
  - Chosen: highest rubric among reward>=threshold responses (deterministic tie-break).
  - Rejected: highest rubric among reward<threshold responses (tie-break by higher reward, then deterministic).
  - If no reward<threshold response (all reward>=threshold), fall back to reward>=threshold candidates and pick the
    worst rubric as rejected (still distinct from chosen). Skip if you cannot form a pair.
- Distill buffer stores `(prompt_nohint, chosen_ids, rejected_ids)` and DPO updates run
  every `data.acr.distill.interval` using a frozen reference policy; logprobs sum over
  assistant tokens only.

## Overview

ACRD currently performs periodic distillation (SFT by default) from accepted ACR traces.
We want Step 3 to support:

- `method: sft` (default, current behavior)
- `method: dpo` (new)

DPO builds preference pairs from a pool of size `data.acr.rollout_n`:
top-N RLVR rollouts when max reward is 1.0, otherwise ACR rollouts with hints.
It runs a DPO update every `data.acr.distill.interval` steps, then clears the buffer.

ACR rollouts use `data.acr.rollout_n` (default 4). A value of 3 is also supported,
but the default remains 4.

## Scope

Goals:

- Keep Step 1 unchanged (RLVR rollouts + verifier reward).
- Make Step 2 method-aware for DPO (reuse RLVR rollouts when max reward == 1.0).
- Add `distill.method = sft | dpo` switch.
- Reuse the same frozen reference policy as GRPO (no separate DPO ref path).

Non-goals (for now):

- No custom ref-model overrides (keep a single ref policy).

## Integration Points

ACRD Step 3 lives in:

- `rlvr/verl/trainer/ppo/ray_trainer.py`
  - `_compute_uid_max_rewards(...)` and `_filter_acr_batch_by_uid_max_reward(...)` for DPO gating.
  - `_collect_distill_candidates_dpo(...)` to build chosen/rejected pairs.
  - `_maybe_run_distill_dpo(...)` for the DPO update.

The distill buffer is still periodic and is cleared after each update.

## Configuration

### Hydra config

Add new fields under `data.acr.distill`:

```yaml
data:
  acr:
    distill:
      enabled: true  # required for SFT/DPO
      method: sft  # sft | dpo
      interval: 10

      # existing SFT fields (also used by DPO as the chosen/rejected threshold)
      reward_threshold: 0.99
      lr_scale: 1.0

      # DPO config
      dpo:
        beta: 0.1
```

Note: DPO is still gated by `data.acr.distill.enabled=true`; method selection only applies when distill is enabled.

### Script env overrides (train_minerva_noctua.sh)

Add:

- `ACRD_ACR_DISTILL_METHOD` (`sft` or `dpo`)
- `ACRD_DPO_BETA`

Map them to Hydra overrides like the existing SFT knobs.

## Distill Buffer Schema

SFT records (current):

- `uid`, `task_key`
- `prompt_nohint` (original RLVR prompt; no hints)
- `response_ids`
- `reward`, `entropy_proxy` (mean NLL)

DPO records (new):

- `uid`, `task_key`
- `prompt_nohint`
- `chosen_ids`, `rejected_ids`
- `chosen_meta` / `rejected_meta`:
  - `acr_score`, `acr_base_score`
  - `rubric_score` (from `acr_rubric_score`, 0..1 when judge is enabled)

If a judge is enabled, the ACR reward path should populate `acr_rubric_score`
(0..1). The current implementation uses the rubric prompt in
`rlvr/verl/utils/reward_score/prompts/acr_rubric_prompt.txt` and computes a
weighted sum of Q1-Q4 (1-4 scale, weights 0.30/0.30/0.20/0.20) normalized via
`(S_1to4 - 1.0) / 3.0`.

Recommended helper:

```python
ACRDistillBuffer.add_sft(...)
ACRDistillBuffer.add_dpo_pair(...)
ACRDistillBuffer.flush()  # returns records + clears
```

## Pair Construction (DPO)

### Candidate pool (per UID)

We consider every UID in the batch (no success-rate filtering).

- If max RLVR reward == 1.0, reuse the top `data.acr.rollout_n` RLVR rollouts
  (sorted by RLVR reward, random tie-break). No extra ACR generation.
- Otherwise, generate `data.acr.rollout_n` ACR rollouts with answer hints.

The "reward" used for DPO selection is the base verifier reward:
- RLVR rollouts: `reward_minerva` score (0..1).
- ACR rollouts: `acr_base_score` (0..1).
Leak penalties and `reward_acr.score` are not used for DPO selection.

### Selection (one pair per UID)

- Chosen: among responses with reward >= `data.acr.distill.reward_threshold`, pick the highest rubric score (if present).
  Tie-break is deterministic random (seeded by UID + index).
- Rejected: among responses with reward < `data.acr.distill.reward_threshold`, pick the highest rubric (tie-break by higher
  reward, then deterministic random).
- If no reward<threshold response (all reward>=threshold), pick the lowest rubric among reward>=threshold candidates
  as rejected (distinct from chosen). Skip if you cannot form a pair.

Notes:

- DPO uses `data.acr.distill.reward_threshold` for the chosen/rejected split and does not use
  entropy/mean-NLL tie-breaks.
- DPO always forms a single pair per UID in the current implementation.

## DPO Update (Step 3)

For each pair `(x, y+, y-)`:

```text
logp_pi(y+|x), logp_pi(y-|x)   # actor
logp_ref(y+|x), logp_ref(y-|x) # reference

pi_lograt  = logp_pi(y+)  - logp_pi(y-)
ref_lograt = logp_ref(y+) - logp_ref(y-)
margin = pi_lograt - ref_lograt

loss = -log(sigmoid(beta * margin))
```

### Logprob details

Use **assistant tokens only**:

- build `input_ids = prompt + response`
- compute logprobs for **response tokens**
- compute sequence log probability by summing token log-probs over assistant tokens in the completion only

This must match the assistant-token definition used in ACRD selection.

## Reference Model (DPO)

DPO uses the same **frozen reference policy** as GRPO: the pre-RL checkpoint
used to limit drift (i.e., `actor_rollout_ref.ref`, which defaults to the same
path as the actor).

Notes:

- DPO always needs a frozen reference, even if KL loss is disabled.
- The PPO runner ensures a ref worker is created when `distill.method=dpo`.
- No custom ref model path or tokenizer checks are used in the current design.

## Metrics to Log (DPO)

Add:

- `acr_distill_dpo/dpo_loss`
- `acr_distill_dpo/dpo_margin_mean`
- `acr_distill_dpo/dpo_pair_acc` (mean of `margin > 0`)
- `acr_distill_dpo/dpo_beta`

Counts:

- `acr_distill/dpo_pairs_added`
- `acr_distill/dpo_pairs_skipped`

Keep existing distill logs (buffer size, selected counts, etc.).

## Script Usage (ACRD)

Example:

```bash
ACRD_ACR_DISTILL_METHOD=dpo \
ACRD_DPO_BETA=0.1 \
bash rlvr/cti-scripts/train_minerva_noctua_llama8b.sh
```

Switch back to SFT by setting:

```bash
ACRD_ACR_DISTILL_METHOD=sft
```

## Judge Rubric (Current)

The judge is already implemented. Enable the batch reward manager and set
`judge_enabled=true` so `reward_acr_batch` emits `acr_rubric_score` (0..1).
This does not change DPO mechanics, only the ranking signal.

The prompt is stored in `rlvr/verl/utils/reward_score/prompts/acr_rubric_prompt.txt`
and is reproduced here for clarity:

```
You are an evaluation judge scoring the QUALITY of a CTI reasoning trace.
Final answer correctness and output-format compliance are verified separately by rule-based checks.
IGNORE correctness and formatting; score only the reasoning quality.

You will be given:
- QUESTION: the original prompt
- RESPONSE: the model's full response (reasoning + final answer line)

Return ONLY a valid JSON object with integer scores (no rationale, no extra keys, no markdown).
IMPORTANT: Output must be a single JSON object and nothing else. Do not add any commentary
or rationale or surrounding text. The response must start with "{" and end with "}".

========================
QUESTION:
{QUESTION}

RESPONSE:
{RESPONSE}
========================

Use a 1-4 Likert scale for each axis:
1 = Poor, 2 = Fair, 3 = Good, 4 = Excellent

Q1 - No hint/label leakage
Score how well the RESPONSE avoids stating or implying that the answer/labels/options were provided.
Includes phrases like: "given the answer", "ground truth", "based on provided label/options",
"since we know the correct label", etc., and indirect meta-language implying label access.
1: explicit leakage / strong implication of provided answer
2: indirect but clear implication / notable leakage language
3: minor borderline phrasing but generally clean
4: completely clean (no leakage or implication)

Q2 - Clarity + conciseness + non-redundancy
Score readability and efficiency: clear structure, no repetition, no rambling, no degenerate loops.
1: confusing/gibberish OR highly repetitive/verbose
2: understandable but wordy/redundant
3: mostly clear and reasonably concise; minor redundancy
4: very clear, crisp, minimal redundancy

Q3 - Groundedness to the input
Score how well the reasoning is anchored in specific evidence from the QUESTION
and avoids hallucinated specifics not supported by the QUESTION.
1: generic/unanchored OR introduces major unsupported specifics
2: somewhat grounded but vague OR some unsupported specifics
3: grounded with multiple concrete ties to the QUESTION; minimal unsupported detail
4: strongly grounded; evidence-based; no unsupported specifics

Q4 - Evidence->answer alignment
Score whether the reasoning logically supports the final answer stated in the RESPONSE,
without contradictions or non sequiturs. (Do NOT judge whether the answer is correct.)
1: reasoning contradicts the final answer or is unrelated
2: weak/partial support; key steps missing or shaky
3: generally supports the final answer with a coherent link
4: strong, consistent support with clear linkage from evidence to the stated answer

OUTPUT (STRICT): return ONLY a JSON object with keys Q1, Q2, Q3, Q4 and integer values 1-4.

Example valid output:
{
  "Q1": 4,
  "Q2": 3,
  "Q3": 3,
  "Q4": 4
}
```

The rubric score is computed in code (not by the LLM); the weights are not provided to the judge:

```
S_1to4 = 0.30*Q1 + 0.30*Q2 + 0.20*Q3 + 0.20*Q4
S_0to1 = (S_1to4 - 1.0) / 3.0
```

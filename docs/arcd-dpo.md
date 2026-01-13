# ACRD Step 3 DPO (Design + Implementation Guide)

This document describes how to add **DPO** as an alternative to **SFT** for
ACRD Step 3 (distillation) in the current Minerva codebase. Steps 1–2 remain
unchanged; only the distill phase becomes pluggable.

## Overview

ACRD currently performs periodic distillation (SFT by default) from accepted ACR traces.
We want Step 3 to support:

- `method: sft` (default, current behavior)
- `method: dpo` (new)

DPO should build preference pairs from the same ACR rollouts and run a DPO update
every `data.acr.distill.interval` steps, then clear the buffer.

ACR rollouts use `data.acr.rollout_n` (default 4). A value of 3 is also supported,
but the default remains 4.

## Scope

Goals:

- Keep Steps 1–2 unchanged (RLVR + ACR trace generation).
- Add `distill.method = sft | dpo` switch.
- Reuse the same frozen reference policy as GRPO (no separate DPO ref path).
- Keep distill eligibility gates the same as today.

Non-goals (for now):

- No LLM judge (can be added later).
- No changes to ACR reward or gating logic.
- No custom ref-model overrides (keep a single ref policy).

## Integration Points

ACRD Step 3 lives in:

- `rlvr/verl/trainer/ppo/ray_trainer.py`
  - `_collect_distill_candidates(...)`
  - `_maybe_run_distill_sft(...)`

Refactor to:

```python
if distill_method == "sft":
    _run_acr_sft_distill(...)
elif distill_method == "dpo":
    _run_acr_dpo_distill(...)
```

Buffer handling stays periodic and is cleared after each update.

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

      # existing SFT fields
      reward_threshold: 1.0
      lr_scale: 0.1

      # DPO config
      dpo:
        beta: 0.1
        require_rejected_parses: true
```

Note: DPO is still gated by `data.acr.distill.enabled=true`; method selection only applies when distill is enabled.

### Script env overrides (train_minerva_acrd.sh)

Add:

- `ACRD_ACR_DISTILL_METHOD` (`sft` or `dpo`)
- `ACRD_DPO_BETA`
- `ACRD_DPO_REQUIRE_REJECTED_PARSES`

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
  - `rubric_score` (0 until a judge is added; future hook is `acr_rubric_score`)

Recommended helper:

```python
ACRDistillBuffer.add_sft(...)
ACRDistillBuffer.add_dpo_pair(...)
ACRDistillBuffer.flush()  # returns records + clears
```

## Pair Construction (DPO)

### Eligibility (unchanged)

Chosen candidates must satisfy:

- `acr_base_score >= reward_threshold`
- `acr_extracted == true`
- `acr_leak_hit == false`

Do not alter this gate unless explicitly requested.

### Ranking signal (lexicographic)

Define for each rollout `i`:

```
is_clean_correct_i := (acr_extracted && acr_base_score >= τ && !acr_leak_hit)
```

where `τ = data.acr.distill.reward_threshold` (default 1.0).

We use a lexicographic rank key:

1) `is_clean_correct` (True > False)  
2) `rubric_i` (higher better; currently 0 for all rollouts until a judge is added)  
3) `acr_base_score` (optional, for partial-credit tasks)  
4) shorter response length (tie-break)  
5) deterministic random (seeded by `uid` + rollout index)  

If you prefer a single scalar for writeups:

```
combined_i = 100 * I[is_clean_correct_i] + rubric_i
```

Then `max(combined_i) >= 100` iff at least one clean+correct rollout exists.

### Pair construction (one pair per UID)

Step A — pick CHOSEN:

- If no rollout is clean+correct, **skip the UID** (no DPO pair).
- Otherwise, `chosen = argmax` among clean+correct by the rank key above.

Step B — pick REJECTED:

- Candidate pool is non-clean rollouts (not clean+correct). If `require_rejected_parses=true`,
  restrict to `acr_extracted==true`; if false, allow any non-clean rollout (including non-extracted).
- If there is a valid rejected candidate, choose the best such rollout by the same rank key (hard negative).
- Otherwise, pick the **worst** clean+correct rollout (style-only negative).
- If no valid rejected exists distinct from chosen, skip the UID.

Notes:

- Entropy/mean-NLL tie-breaks are **not used** in the DPO path.
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
ACRD_DPO_REQUIRE_REJECTED_PARSES=true \
bash rlvr/cti-scripts/train_minerva_acrd_llama8b.sh
```

Switch back to SFT by setting:

```bash
ACRD_ACR_DISTILL_METHOD=sft
```

## Optional Judge Rubric (Future)

If you add a judge later, use a task-agnostic rubric to rank rollouts.
This does not change the DPO mechanics, only the ranking signal.

### Hard-reject checks (verdict = REJECT)

Reject if any:

- leakage / hint mention
- no clear final answer line
- gibberish / degeneration
- final answer mismatch with ground truth (simple normalization)

### Scored criteria (0–2 each; total 10)

- instruction compliance
- groundedness to prompt
- faithfulness (no hallucinated details)
- conciseness
- reasoning -> answer alignment

### Judge output

Return strict JSON:

```json
{
  "verdict": "ACCEPT|REJECT",
  "overall": 0-10,
  "subscores": {
    "instruction": 0-2,
    "groundedness": 0-2,
    "faithfulness": 0-2,
    "conciseness": 0-2,
    "alignment": 0-2
  },
  "flags": ["leak", "no_final_answer", "gibberish", "mismatch"],
  "rationale": "≤2 sentences"
}
```

If all candidates are rejected, skip the UID (or resample).

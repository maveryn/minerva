# ACRD-DPO Process Summary (Defaults from Docs)

This is the end-to-end per-batch loop when you run ACRD with Step 3 = DPO,
using the defaults documented in `docs/acrd.md` and `docs/arcd-dpo.md`.

## Key knobs (DPO-relevant defaults)

- `data.acr.rollout_n = 4` (3 supported, but default 4)
- `data.acr.hard_reward_threshold = 0.5` (run ACR only if mean RLVR reward < threshold)
- `data.acr.distill.enabled = true` (required for distillation, SFT/DPO)
- `data.acr.distill.method = dpo` (default is `sft`, so you must set to `dpo`)
- `data.acr.distill.interval = 10`
- `data.acr.distill.reward_threshold = 1.0` (clean+correct only)
- `data.acr.distill.lr_scale = 0.1` (distill step uses a smaller LR than RLVR)
- `data.acr.distill.dpo.beta = 0.1`
- `data.acr.distill.dpo.require_rejected_parses = true`

## Step 1: RLVR baseline update (unchanged)

For each training batch, run the existing RLVR training loop on the original
prompts (no ground truth revealed). This is the normal GRPO/PPO-style RL with
verifiable rewards. Nothing in DPO changes this step.

## Step 2: Per-batch ACR trace generation on hard examples (unchanged)

Still within the same training batch, ACRD does an answer-conditioned "teacher
trace" generation pass, but only for hard prompts.

### 2.1 Hard-example gating (default = 0.5)

For each prompt UID in the batch:

- Compute mean RLVR reward across the RLVR rollouts.
- Only if that mean is below `data.acr.hard_reward_threshold` (default 0.5) do
  you generate ACR traces.

This concentrates Step 2/3 compute on prompts the model is not reliably solving.

### 2.2 Build the answer-conditioned ACR prompt

For those hard prompts, you build an ACR block and append it to the last user
message of `raw_prompt` (so `data.return_raw_chat=true` is required).

The ACR block includes:

- `GROUND_TRUTH_LABELS` (the gold label(s))
- Optional `CANONICAL_LABEL_DETAILS` from `LabelDetailsStore` (joined with `---`
  for multi-label)
- Instructions: "write reasoning + final answer in original task format" and
  "do not say/imply answer was provided"
- Optional "Do not include exact label(s) in reasoning" when
  `data.acr.enforce_no_id_in_reasoning=true`

Prompt-length control (`try_build_acr_messages`):

- Truncate details first (up to `data.acr.max_details_chars`).
- If still too long, omit details.
- If still too long, skip ACR for that sample.

### 2.3 Generate ACR rollouts (default `n=4`)

Using the same actor as RLVR, you sample `data.acr.rollout_n` ACR responses per
hard UID (default 4). These are answer-conditioned traces and do not do PPO
update by default (ACR reward is used for filtering/ranking only).

### 2.4 Score ACR rollouts with verifiers + leak checks

Each ACR rollout is scored via `reward_acr`, which reuses RLVR parsing logic
(`reward_minerva`) and returns:

- `acr_base_score` in `[0, 1]` (verifier score)
- `acr_extracted` (parse success)
- `acr_leak_hit` (banned-phrase + fuzzy check for "given the answer",
  "ground truth", etc.)
- `acr_id_leak_hit` (gold ID appears in reasoning; logged only, not penalized)
- Weighted score = `r_correct * acr_base_score - leak_penalty` (used for
  ranking/filtering)

This produces the pool of candidate traces that Step 3 will distill from.

## Step 3: Distillation via DPO

When `data.acr.distill.enabled=true` and `data.acr.distill.method=dpo`, you do
preference-based distillation rather than imitation.

### 3.1 Buffer entries are DPO pairs (not single targets)

Instead of storing one accepted trace per UID, DPO stores (per UID):

- `prompt_nohint = extra_info["acr_orig_prompt"]` (original RLVR prompt; no hints)
- `chosen_ids`, `rejected_ids`
- Metadata (ACR scores and a rubric placeholder `rubric_score=0` until a judge
  exists)

### 3.2 Per-UID pair construction (one pair per UID)

For each UID, you build exactly one preference pair.

Eligibility for "chosen" (unchanged gate):

- `acr_base_score >= reward_threshold` (default 1.0)
- `acr_extracted == true`
- `acr_leak_hit == false`

If no rollout meets this, you skip the UID (no DPO pair).

Ranking rule (lexicographic):

Define:

- `is_clean_correct_i := (acr_extracted && acr_base_score >= tau && !acr_leak_hit)`
  with `tau=1.0` by default.

Rank key (highest wins):

1. `is_clean_correct` (True > False)
2. `rubric_i` (currently 0 everywhere; hook for later judge)
3. `acr_base_score` (optional use for partial-credit tasks)
4. Shorter response length
5. Deterministic random (seeded)

Choose CHOSEN:

- Choose the best rollout among `is_clean_correct=True` by the rank key.

Choose REJECTED (default `require_rejected_parses=true`):

- Candidate pool = rollouts that are not clean+correct.
- If `require_rejected_parses=true`, restrict that pool to rollouts with
  `acr_extracted==true`.
- If there is at least one valid rejected candidate, pick the best rejected
  candidate by the same rank key (hard negative).
- Else, fallback: pick the worst clean+correct rollout (style-only negative).
- If you still cannot get a rejected distinct from chosen, skip the UID.

No entropy/mean-NLL tie-break is used in DPO.

### 3.3 DPO update runs every 10 steps (default)

Every `data.acr.distill.interval=10` steps, you:

- Flush the DPO pair buffer.
- Run a DPO optimization step over these (prompt, chosen, rejected) pairs.
- Clear the buffer.

### 3.4 DPO loss (exactly as in the guide)

For each pair `(x, y+, y-)`:

- Compute sequence logprobs under the current actor:
  `logp_pi(y+|x)`, `logp_pi(y-|x)`.
- Compute sequence logprobs under the frozen reference:
  `logp_ref(y+|x)`, `logp_ref(y-|x)`.

Then:

```text
pi_lograt  = logp_pi(y+) - logp_pi(y-)
ref_lograt = logp_ref(y+) - logp_ref(y-)
margin     = pi_lograt - ref_lograt
loss       = -log(sigmoid(beta * margin))
```

Logprob definition (important):

- Logprobs are computed over assistant tokens only.
- Sequence log probability is the sum of token log-probs over completion tokens
  only.

### 3.5 Reference policy behavior

DPO uses the same frozen reference policy concept as GRPO: the pre-RL checkpoint
used to limit drift (i.e., `actor_rollout_ref.ref`, defaulting to the actor path).

Two key implementation requirements are explicitly called out:

- DPO always needs `logp_ref`, even if KL loss is disabled.
- The runner should ensure a ref worker exists when `distill.method=dpo`.

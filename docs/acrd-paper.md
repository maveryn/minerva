# ACRD-DPO Process Summary (Defaults from Docs)

This is the end-to-end per-batch loop when you run ACRD with Step 3 = DPO,
using the defaults documented in `docs/acrd.md` and `docs/acrd-dpo.md`.

## Key knobs (DPO-relevant defaults)

- `data.acr.rollout_n = 4` (3 supported, but default 4)
- `data.acr.hard_reward_mode = no_perfect` (run ACR only if no rollout reaches reward 1.0)
- `data.acr.hard_reward_threshold = 0.05` (used when `hard_reward_mode=mean_reward`)
- `data.acr.distill.enabled = true` (required for distillation, SFT/DPO)
- `data.acr.distill.method = dpo` (default is `sft`, so you must set to `dpo`)
- `data.acr.distill.interval = 10`
- `data.acr.distill.reward_threshold = 0.99` (chosen/rejected split; default matches SFT/DPO)
- `data.acr.distill.lr_scale = 1.0` (distill step uses the same LR as RLVR)
- `data.acr.distill.dpo.beta = 0.1`

## Step 1: RLVR baseline update (unchanged)

For each training batch, run the existing RLVR training loop on the original
prompts (no ground truth revealed). This is the normal GRPO/PPO-style RL with
verifiable rewards. Nothing in DPO changes this step.

## Step 2: Per-batch ACR trace generation (method-aware)

Still within the same training batch, ACRD does an answer-conditioned "teacher
trace" generation pass, but only for hard prompts.

### 2.1 Hard-example gating (SFT only)

For each prompt UID in the batch:

- `hard_reward_mode=no_perfect` (default): generate ACR traces only if max RLVR reward < 1.0.
- `hard_reward_mode=mean_reward`: generate ACR traces only if mean RLVR reward across
  rollouts is below `data.acr.hard_reward_threshold` (default 0.05).

This concentrates Step 2/3 compute on prompts the model is not reliably solving.
For DPO, we skip hard-example gating and reuse RLVR rollouts when max reward is 1.0.

### 2.2 Build the answer-conditioned ACR prompt

For those hard prompts, you build an ACR block and append it to the last user
message of `raw_prompt` (so `data.return_raw_chat=true` is required).

The ACR block includes:

- `GROUND_TRUTH_LABELS` (the gold label(s))
- Optional `LABEL_REFERENCE` from `LabelDetailsStore` (joined with `---`
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

Using the same actor as RLVR, you sample `data.acr.rollout_n` responses per UID.
DPO does not filter by success rate or hard threshold.

- If max RLVR reward == 1.0, **do not** generate ACR responses. Reuse the top-N
  RLVR rollouts (sorted by RLVR reward, random tie-break).
- Otherwise, generate `data.acr.rollout_n` ACR responses with answer hints.

ACR rollouts are answer-conditioned traces and do not do PPO update by default
(ACR reward is used for filtering/ranking only).

### 2.4 Score candidate rollouts with verifiers + leak checks

Each candidate rollout (ACR or reused RLVR) is scored via `reward_acr`
(or `reward_acr_batch` when judge batching is enabled), which reuses RLVR
parsing logic (`reward_minerva`) and returns:

- `acr_base_score` in `[0, 1]` (verifier score)
- `acr_extracted` (parse success)
- `acr_leak_hit` (banned-phrase + fuzzy check for "given the answer",
  "ground truth", etc.)
- `acr_id_leak_hit` (gold ID appears in reasoning; logged only, not penalized)
- Weighted score = `r_correct * acr_base_score - leak_penalty` (used for ACR
  filtering; DPO selection uses base reward with rubric tie-breaks)
- Optional `acr_rubric_score` when a judge is enabled (used for DPO ranking);
  weighted sum of Q1-Q4 from the rubric JSON prompt, normalized via
  `(S_1to4 - 1.0) / 3.0`.

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

Candidate pool:

- If max RLVR reward == 1.0, use the top `data.acr.rollout_n` RLVR rollouts.
- Otherwise, use the `data.acr.rollout_n` ACR rollouts.

Choose CHOSEN:

- Among responses with reward >= `data.acr.distill.reward_threshold`, pick the highest rubric score
  (if present). If no reward>=threshold response exists, skip the UID.

Choose REJECTED:

- Among responses with reward < `data.acr.distill.reward_threshold`, pick the highest rubric (tie-break by higher reward).
- If no reward<threshold response (all reward>=threshold), pick the lowest rubric among reward>=threshold candidates
  (distinct from chosen). Skip if you cannot form a pair.

No entropy/mean-NLL tie-break is used in DPO. The reward here is the base verifier
reward (RLVR: `reward_minerva`; ACR: `acr_base_score`).

### 3.3 DPO update runs every 10 steps (default)

Every `data.acr.distill.interval=10` steps, you:

- Flush the DPO pair buffer.
- Run a DPO optimization step over these (prompt, chosen, rejected) pairs.
- Clear the buffer.

## SFT note (default method)

When `data.acr.distill.method = sft`, each distill step samples up to
`data.acr.distill.batch_size` records uniformly at random from the buffer
(no replacement) and runs a single SFT update. If fewer records exist, all
available records are used.

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

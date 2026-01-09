# IMPLEMENTATION TASK: Stochastic Label-Hint Curriculum (SLHC) + No-Hint Ramp + 3 Variants

You are modifying https://github.com/volcengine/verl (our fork).
We need a stochastic label-hint curriculum that:
- injects candidate ID hints into prompts during RLVR/GRPO training,
- supports multi-label tasks (choose EXACTLY N) using the same prompt format as the existing LHC,
- includes no-hint prompts during training (to match evaluation where no hints are provided),
- uses success threshold reward>=0.1 to define success per rollout,
- provides THREE controller variants selectable by ONE config arg:
    (A) acc_target
    (B) zero_constraint
    (C) two_tail_smooth

IMPORTANT PROMPT RULE:
- Do NOT add any instruction like "Return ONLY the ID(s)..." in the injected hint block.
- Keep the injected hint block format identical to current LHC (Candidate ID(s) (choose EXACTLY N): 1) ...).
- The base prompt already includes whatever reasoning + answer format we want; do not override it.

We will implement this by:
1) Creating a new dataset wrapper and controller (based on lhc-updated).
2) Patching RayPPOTrainer.fit to compute group metrics and update the controller.
3) Adding YAML config knobs, but ensuring the ONLY ablation switch is `control_mode`.

Reference docs (use them as implementation guidance):
- lhc-updated.md: architecture + stochastic sampling around mu (discrete Gaussian). 
- lhc-implementation.md: exact prompt injection text, multi-label EXACTLY N logic, and prompt-length trimming strategy.

---

## 0) Definitions and requirements

### Group metrics (GRPO)
GRPO uses G=rollout.n rollouts per prompt uid (G is usually 8).
We define per-rollout success using reward threshold:
- success_i = 1 if reward_scalar_i >= 0.1 else 0

Per uid group:
- success_count = sum(success_i over i=1..G)
- acc_g  = success_count / G
- zero_g = 1 if success_count == 0 else 0     # all rollouts < 0.1
- all_g  = 1 if success_count == G else 0     # all rollouts >= 0.1

These metrics are used to update per-task controllers keyed by data_source.

### Hint size / sentinel state
Each example has gold_ids (list) of size c = len(gold_ids).
We show total candidates K_total = c + d, where d is number of distractors.

We cap total candidates shown to K_max_total (e.g., 30).
d_max_sample = min(
    K_max_total - c,
    pool_size - c
)
If d_max_sample < d_min -> cannot inject hints (fallback to no-hint).

SENTINEL: treat "next harder step after max hints" as no-hint:
- Expand d domain to include sentinel state: d = d_max_sample + 1 => NO HINT
- This gives K_max_total+1 concept without adding a separate knob.

### No-hint ramp schedule (simple, global, no tuning)
We add a global forced no-hint probability schedule (evaluation alignment):
- linearly ramp from 0.05 to 0.50 over total training steps S
- we will set S = total training steps we plan to run (passed via config or trainer)

p_forced_nohint(step) = p_start + (p_end - p_start) * clamp(step / S, 0, 1)
Defaults: p_start=0.05, p_end=0.50.
This is shared across all ablations to keep comparisons fair.

Crucial: controller updates should NOT be driven by forced-nohint groups (since they are not chosen by the controller).
We only update the controller using "controller-active" groups where forced-nohint did NOT occur.
Within controller-active groups, d sampling may still land on sentinel => no-hint; those DO affect controller.

We will track:
- extra_info["slhc_ctrl_active"] = True/False
- extra_info["slhc_hint_used"] = True/False
- extra_info["slhc_nohint_reason"] in {"forced_schedule", "sentinel", "unavailable"} when slhc_hint_used==False

---

## 1) Files to create / modify

### (1) Create file: `minerva_stochastic_slhc_dataset.py`
Implement:
- `class StochasticSLHCController`
- `class StochasticSLHCRLHFDataset(RLHFDataset)`

### (2) Patch file: `verl/trainer/ppo/ray_trainer.py`
Modify `RayPPOTrainer.fit`:
- after reward computation, compute reward_scalar and success_i (reward>=0.1)
- group by uid, compute acc_g, zero_g, all_g per uid
- call dataset update hook if present:
    self.train_dataset.update_slhc_controller_from_groups(group_summaries, global_step=self.global_steps, total_steps=TOTAL_STEPS)

---

## 2) Controller design: `StochasticSLHCController`

### Controller state (per task_key = data_source)
For each task t:
- mu_d[t]: float, init = 1.0          # mean distractor count
- sigma_d[t]: float, init = sigma_init (fixed globally, no per-task tuning)
- ema_acc[t], ema_zero[t], ema_all[t]: float, init 0

Global:
- global_step: int (updated from trainer)
- total_steps: int (passed from trainer or config)
- rng: random.Random seeded from cfg.seed (plus task hash if you want)

Config fields (cfg.data.stochastic_slhc):
- enabled: bool
- control_mode: "acc_target" | "zero_constraint" | "two_tail_smooth"  # SINGLE ablation arg
- success_threshold: float default 0.1

Hint construction:
- candidate_pool_key: "candidate_pool_top100"
- buffer: int default 10
- K_max_total: int default 30
- d_min: int default 1
- sigma_init: float default 2.0

Forced nohint schedule:
- p_nohint_start: float default 0.05
- p_nohint_end: float default 0.50
- (nohint_ramp_steps) we will use TOTAL training steps passed from trainer; no separate tuning

EMA and update parameters (keep fixed across ablations):
- ema_beta: float default 0.90
- lr_mu: float default 0.50      # used in acc_target + zero_constraint
- For two_tail_smooth (fixed defaults; do not tune for ablation):
    drift_delta: float default 0.25
    alpha_zero: float default 1.0
    alpha_all: float default 0.7
    temp_zero: float default 0.05
    temp_all: float default 0.05

### Helper: forced nohint probability
Implement:
get_forced_nohint_p(global_step, total_steps):
  if total_steps<=0: return p_nohint_end
  x = clamp(global_step / total_steps, 0, 1)
  return p_start + (p_end - p_start) * x

### Sampling API used by dataset
Implement:
sample_decision(task_key, c, pool_size, uid) -> dict with:
- ctrl_active: bool
- hint_used: bool
- nohint_reason: Optional[str]
- d: Optional[int]         # distractor count if hinted (or sentinel d if you prefer logging)
- seed: int                # used to shuffle options deterministically
- K_total: Optional[int]

Logic:
1) compute p_forced = get_forced_nohint_p(global_step, total_steps)
2) if rng.random() < p_forced:
     return {ctrl_active=False, hint_used=False, nohint_reason="forced_schedule"}
3) ctrl_active=True
4) compute d_max_sample = min(K_max_total - c, pool_size - c)
   if d_max_sample < d_min:
     return {ctrl_active=True, hint_used=False, nohint_reason="unavailable"}   # cannot inject
5) define domain D = [d_min .. d_max_sample+1]
   where (d_max_sample+1) is sentinel => no-hint
6) sample d from truncated discrete Gaussian centered at mu_d[task]:
     w(d) = exp(-(d - mu)^2 / (2*sigma^2))
   normalize over D and sample
7) if d == d_max_sample+1:
     return {ctrl_active=True, hint_used=False, nohint_reason="sentinel"}
   else:
     return {ctrl_active=True, hint_used=True, d=d, K_total=c+d}

Important: clip mu_d[t] in controller update to [d_min, K_max_total]
(because for any sample, d_max_sample+1 <= K_max_total when c>=1).

### Controller update API used by trainer
Implement:
update_from_groups(group_summaries, global_step, total_steps) -> metrics dict

Input `group_summaries`: list of dicts, one per uid group:
{
  "task_key": str,
  "ctrl_active": bool,   # from extra_info
  "acc_g": float,
  "zero_g": float,
  "all_g": float
}

Update steps:
1) store controller.global_step = global_step
2) store controller.total_steps = total_steps
3) filter to ctrl_active groups only for controller learning signals:
   controlled = [g for g in group_summaries if g["ctrl_active"] is True]
4) aggregate per task_key:
   acc_mean[t]  = mean(acc_g)
   zero_mean[t] = mean(zero_g)
   all_mean[t]  = mean(all_g)
5) update EMAs:
   ema_x[t] = ema_beta*ema_x[t] + (1-ema_beta)*x_mean[t]

6) update mu_d[t] depending on control_mode:

(A) control_mode="acc_target":
    mu_d[t] = clip(mu_d[t] + lr_mu*(ema_acc[t] - target_acc), d_min, K_max_total)
    where target_acc is fixed at 0.50

(B) control_mode="zero_constraint":
    mu_d[t] = clip(mu_d[t] + lr_mu*(zero_target - ema_zero[t]), d_min, K_max_total)
    where zero_target is fixed at 0.15
    Intuition: if ema_zero too high => mu decreases => easier; if ema_zero low => mu increases => harder.

(C) control_mode="two_tail_smooth":
    sigmoid(x)=1/(1+exp(-x))
    dz = sigmoid((ema_zero[t] - zero_target)/temp_zero)
    du = sigmoid((ema_all[t]  - all_target )/temp_all)
    mu_d[t] = clip(
        mu_d[t] + drift_delta + alpha_all*du - alpha_zero*dz,
        d_min, K_max_total
    )
    where zero_target=0.15 and all_target=0.15

7) return metrics for logging:
   - slhc/mu_d/<task>
   - slhc/ema_acc/<task>
   - slhc/ema_zero/<task>
   - slhc/ema_all/<task>
   - slhc/p_forced_nohint (global scalar)
   - slhc/control_mode (string or int-coded)

---

## 3) Dataset wrapper: `StochasticSLHCRLHFDataset(RLHFDataset)`

### Requirements
- MUST run with dataloader_num_workers=0 (controller is mutable).
- Reuse prompt injection format and multi-label EXACTLY N logic from current LHC.
- Reuse prompt-length trimming behavior from current LHC:
  if prompt exceeds max_prompt_length, iteratively reduce distractors (or skip hint).
(Refer to existing AdaptiveOptionRLHFDataset behavior.)

### __getitem__(idx) behavior
1) row = super().__getitem__(idx)
2) Ensure uid exists:
   if "uid" not in row:
      row["uid"] = f"{row['data_source']}::{idx}"
2a) If extra_info.split exists and is not "train", skip hint injection (evaluation should be no-hint).
3) Parse gold labels:
   gt = row["reward_model"]["ground_truth"]
   gold_ids = [gt] if isinstance(gt, str) else list(gt)
   c = len(gold_ids)

4) pool = row[candidate_pool_key]  # candidate_pool_top100
   pool_size = len(pool)

5) decision = controller.sample_decision(task_key=row["data_source"], c=c, pool_size=pool_size, uid=row["uid"])

6) Always write these into extra_info:
   extra = row.get("extra_info", {}) or {}
   extra["slhc_ctrl_active"] = decision["ctrl_active"]
   extra["slhc_hint_used"] = decision["hint_used"]
   extra["slhc_nohint_reason"] = decision.get("nohint_reason")
   extra["slhc_hint_seed"] = decision.get("seed")
   extra["slhc_hint_pool_size"] = pool_size
   if decision["hint_used"]:
      extra["slhc_hint_d"] = decision["d"]
      extra["slhc_hint_K_total"] = c + decision["d"]
   row["extra_info"] = extra

7) If hint_used is False: return row unchanged (base prompt)

8) If hint_used is True:
   d = decision["d"]
   K_total = c + d

   Build candidate list exactly like current LHC:
   - Ensure all gold_ids are included in options (even if pool is missing them; insert gold at front if needed).
   - Choose distractors from the top-(K_total + buffer) portion of pool:
       pool_top = pool[:min(pool_size, K_total + buffer)]
       negatives = [x for x in pool_top if x not in set(gold_ids)]
       if len(negatives) < d: shrink d to len(negatives) and recompute K_total
       distractors = sample_without_replacement(negatives, d)
   - options = gold_ids + distractors
   - shuffle options with deterministic RNG using hint_seed

   Inject into the last user message:
   - Append "\n\nCandidate ID(s) (choose EXACTLY {c}):\n"
   - Enumerate "1) {opt}\n2) {opt} ..."
   - Append tail note (same as current LHC prompt format):
       if c==1: "The correct ID is one of the candidates above. You may use them as a reference, but do not mention the list. Reason step by step to arrive at the answer."
       else:    "The correct IDs are among the candidates above. You may use them as a reference, but do not mention the list. Reason step by step to arrive at the answer."

9) Prompt-length trimming:
   - If tokenized prompt length > max_prompt_length:
       iteratively reduce distractors (remove some sampled distractors) until it fits;
       if cannot fit, revert to no-hint (return base prompt).
   Use the same logic pattern as AdaptiveOptionRLHFDataset.

---

## 4) Trainer patch: `RayPPOTrainer.fit`

Patch after reward computation (token_level_scores) and before logging/optimizer step finalization.

### Reward scalar + success_i
We want success_i = 1 if reward_scalar_i >= 0.1.

Compute reward_scalar robustly:
- reward_tensor = batch.batch["token_level_scores"]  (shape could be [B, T] or [B])
- if reward_tensor.ndim == 2:
    reward_scalar = reward_tensor.max(dim=-1).values   # avoids length scaling
  else:
    reward_scalar = reward_tensor

success = (reward_scalar >= cfg.data.stochastic_slhc.success_threshold).float()

### Group by uid
Use:
- uid_arr = batch.non_tensor_batch["uid"]
- task_arr = batch.non_tensor_batch["data_source"]
- extra_arr = batch.non_tensor_batch["extra_info"]

For each uid group:
- idxs = uid_arr == uid
- s = success[idxs]  # length G
- success_count = int(s.sum().item())
- G = len(s)
- acc_g = success_count / G
- zero_g = 1.0 if success_count==0 else 0.0
- all_g  = 1.0 if success_count==G else 0.0
- task_key = task_arr[idxs][0]
- ctrl_active = bool(extra_arr[idxs][0].get("slhc_ctrl_active", False))

Append to group_summaries:
{
  "task_key": task_key,
  "ctrl_active": ctrl_active,
  "acc_g": acc_g,
  "zero_g": zero_g,
  "all_g": all_g,
}

### Call dataset update hook
If hasattr(self.train_dataset, "update_slhc_controller_from_groups"):
   total_steps = cfg.trainer.total_training_steps if exists else cfg.data.stochastic_slhc.total_steps
   metrics_update = self.train_dataset.update_slhc_controller_from_groups(
        group_summaries, global_step=self.global_steps, total_steps=total_steps
   )
   metrics.update(metrics_update)

Keep patch minimal and guard missing fields.

---

## 5) YAML config (single ablation arg)

Add to YAML:

data:
  custom_cls:
    path: /abs/path/minerva_stochastic_slhc_dataset.py
    name: StochasticSLHCRLHFDataset
  dataloader_num_workers: 0

  stochastic_slhc:
    enabled: true
    # SINGLE ARG for ablation:
    control_mode: acc_target   # or zero_constraint, or two_tail_smooth

    success_threshold: 0.1

    candidate_pool_key: candidate_pool_top100
    buffer: 10
    K_max_total: 30
    d_min: 1
    sigma_init: 2.0

    # forced nohint schedule (shared across all ablations)
    p_nohint_start: 0.05
    p_nohint_end: 0.50
    # total_steps can be passed here OR derived from trainer.total_training_steps:
    total_steps: 20000

    # EMA + update params (KEEP FIXED across ablations)
    ema_beta: 0.90
    lr_mu: 0.50

    # targets (KEEP FIXED)
    target_acc: 0.50
    zero_target: 0.15
    all_target: 0.15

    # two-tail smooth params (KEEP FIXED)
    drift_delta: 0.25
    alpha_zero: 1.0
    alpha_all: 0.7
    temp_zero: 0.05
    temp_all: 0.05

Run ablations by changing ONLY:
- data.stochastic_slhc.control_mode

---

## 6) Sanity checks (must do)

1) Ensure dataloader workers == 0. If not, controller state will diverge.
2) Verify prompt injection matches current LHC format exactly:
   - "Candidate ID(s) (choose EXACTLY N):" header
   - enumerated list
   - correct tail note
   - no added output constraints
3) Verify forced no-hint rate ramps from 0.05 to 0.50 over total_steps
   - log slhc/p_forced_nohint each update
4) Verify sentinel no-hint occurs:
   - count fraction of ctrl_active groups where hint_used=False and nohint_reason=="sentinel"
5) Verify metrics computed from reward>=0.1:
   - print a small debug batch with known reward scalars; check acc_g/zero_g/all_g correctness
6) Verify controller updates are based only on ctrl_active groups:
   - forced_schedule groups should not affect mu_d updates.

---

# END TASK

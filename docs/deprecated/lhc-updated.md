Goal
- Implement an Online Stochastic K-options Label-Hint Curriculum (no explicit warmup) for RLVR/GRPO.
- For each task category (keyed by `data_source`), dynamically inject a candidate ID list into the prompt with a stochastically sampled number of distractors d.
- Controller adjusts mean distractor count `mu_d[t]` to keep rollout accuracy across G=8 rollouts near target=0.50.
- Once hinted prompts are “too easy” even at max d, increase `p_nohint[t]` to anneal hints off, so benchmark eval is strong without hints.

Key assumptions
- Dataset is parquet in verl RL format with columns:
  - `prompt`: list of chat messages [{role, content}, ...]
  - `data_source`: string task key (e.g., "scenario_to_mitigations", "cve_to_cwe", etc.)
  - `reward_model`: dict containing `ground_truth` (string for single-label OR list[str] for multi-label)
  - `extra_info`: dict
  - NEW: `candidate_pool_top100`: list[str], gold label(s) included, sorted by similarity (BM25/dense/hybrid precomputed offline)
- GRPO uses rollout.n = 8 and repeats each prompt 8 times.

Deliverables
1) Create `minerva_stochastic_hint_dataset.py` with:
   - class `StochasticHintController`
   - class `StochasticHintRLHFDataset` inheriting from `verl.utils.dataset.rl_dataset.RLHFDataset`

2) Patch `verl/trainer/ppo/ray_trainer.py` (RayPPOTrainer.fit) to:
   - after reward is computed for the repeated batch, compute per-uid group accuracy acc@G,
   - call `self.train_dataset.update_hint_controller_from_rollouts(...)` if present,
   - log controller metrics into `metrics`.

Implementation details

A) Controller behavior (per task = data_source)
State per task_key t:
- mu_d[t]: float, init=1.0  (mean distractor count; implies K_total≈2 for single-label)
- sigma_d[t]: float, init=2.0 (exploration width)
- p_nohint[t]: float, init=0.0 or 0.05 (probability of omitting hints)
- ema_acc_hint[t], ema_acc_nohint[t]: float, init=0
- steps_hint[t], steps_nohint[t]: int counters
Config parameters:
- target_acc = 0.50
- tol = 0.05
- ema_beta = 0.90
- lr_mu = 0.50   # update step size in “d units”, used with EMA so it’s stable
- p_step = 0.05
- d_min = 1
- d_max_by_task: dict[str,int] with default_d_max (e.g., default 32, mitigation maybe 12, cwe maybe 20)
Sampling method:
- For a sample with gold_count=c and pool_size=P:
  d_max = min(d_max_by_task.get(t, default_d_max), P - c)
  If d_max < d_min: disable hints for this sample (hint_used=False).
- Decide hint_used:
  hint_used = (rng.random() >= p_nohint[t])
- If hint_used:
  Sample d in [d_min..d_max] from discrete Gaussian weights around mu_d[t]:
    w(d) = exp(-(d - mu)^2 / (2*sigma^2))
  Normalize weights and sample.
  Return d, K_total=c+d.

Update method per uid (prompt group):
Inputs per uid:
- task_key t
- hint_used (bool)
- d_used (int) if hint_used else None
- acc_g = mean(is_correct over 8 rollouts)
Update:
- If hint_used:
    ema_acc_hint[t] = ema_beta*ema_acc_hint[t] + (1-ema_beta)*acc_g
    mu_d[t] = clip(mu_d[t] + lr_mu*(acc_g - target_acc), d_min, d_max_by_task[t])
    If mu_d[t] >= d_max_by_task[t]-0.5 AND ema_acc_hint[t] > target_acc+tol:
        p_nohint[t] = min(1.0, p_nohint[t] + p_step)
- Else:
    ema_acc_nohint[t] = ema_beta*ema_acc_nohint[t] + (1-ema_beta)*acc_g
    If ema_acc_nohint[t] > target_acc+tol and steps_nohint[t] >= 20:
        p_nohint[t] = 1.0   # always no-hint; task solved w/o hints

Controller metrics:
Implement get_metrics(prefix="hint_ctrl/") returning:
- hint_ctrl/mu_d/<task>
- hint_ctrl/p_nohint/<task>
- hint_ctrl/ema_hint/<task>
- hint_ctrl/ema_nohint/<task>

B) Dataset: `StochasticHintRLHFDataset`
- Read config.data.stochastic_hints.* in __init__.
- Keep `self.hint_controller = StochasticHintController(...)`.
- IMPORTANT: set dataloader_num_workers=0 for correctness because controller has mutable state.
In __getitem__(idx):
1) row = super().__getitem__(idx)
2) task_key = row["data_source"]
3) gt = row["reward_model"]["ground_truth"]
   - if isinstance(gt, str): gold_ids=[gt]
   - else: gold_ids=list(gt)
   c = len(gold_ids)
4) pool = row["candidate_pool_top100"]  # list[str]
5) Ask controller for hint decision:
   hint_used, d, k_total, seed = controller.sample(task_key, c, len(pool), uid=row.get("uid") or idx)
6) If not hint_used:
   - set row["extra_info"]["hint_used"]=False
   - return row
7) Build options:
   - Ensure pool contains all gold ids; if not, insert gold ids at front.
   - Need d distractors:
     pool_top = pool[:min(len(pool), k_total + buffer)]
     distractors = sample without replacement from (pool_top - gold_ids) size=d
   - options = gold_ids + distractors
   - shuffle options with deterministic RNG seeded by `seed`
8) Inject into last user message:
   Append a block like:
   ---
   Candidate IDs (choose EXACTLY {c}):
   1) <ID>
   2) <ID>
   ...
   [Format the option prompts similarly as current LHC implementation]
   ---
9) Write metadata into extra_info:
   extra_info["hint_used"]=True
   extra_info["hint_d"]=d
   extra_info["hint_k_total"]=k_total
   extra_info["hint_seed"]=seed
Return row.

C) Trainer patch: RayPPOTrainer.fit
- Find where reward is computed and assigned to `batch.batch["token_level_scores"]`.
- Immediately after that, compute per-response `is_correct`.
  Preferred:
  - If reward_extra_info contains "is_correct" use it.
  Else:
  - Convert reward_tensor to scalar:
    if reward_tensor.ndim==2: score = reward_tensor.sum(-1)
    else: score = reward_tensor
  - is_correct = (score >= score_threshold)  # from config; default 0.5
- Now compute per-uid acc@G:
  - uid_arr = batch.non_tensor_batch["uid"]
  - task_arr = batch.non_tensor_batch["data_source"]
  - extra_info_arr = batch.non_tensor_batch["extra_info"]
  For each uid:
    - acc_g = mean(is_correct for that uid)
    - hint_used = extra_info["hint_used"]
    - d_used = extra_info.get("hint_d")
    - task_key = task_arr (same for all repeats)
- If hasattr(self.train_dataset, "update_hint_controller_from_rollouts"):
    metrics_update = self.train_dataset.update_hint_controller_from_rollouts(group_summaries, global_step=self.global_steps)
    metrics.update(metrics_update)
    metrics.update(self.train_dataset.hint_controller.get_metrics())
- Keep patch minimal and safe if fields missing.

D) Config
Add to YAML:
data:
  custom_cls:
    path: /abs/path/minerva_stochastic_hint_dataset.py
    name: StochasticHintRLHFDataset
  dataloader_num_workers: 0
  stochastic_hints:
    enabled: true
    candidate_pool_key: candidate_pool_top100
    buffer: 10
    target_acc: 0.50
    tol: 0.05
    ema_beta: 0.90
    lr_mu: 0.50
    sigma_init: 2.0
    d_min: 1
    default_d_max: 32
    d_max_by_task:
      scenario_to_mitigations: 12
      cve_to_cwe: 20
    p_nohint_init: 0.05
    p_step: 0.05
    score_threshold: 0.5

Notes
- Must not break existing reward verifiers; hints only change prompt (consistent with RLVR prompt augmentation).
- Ensure prompt length doesn’t exceed max_prompt_length; if it does, reduce d for that sample.
- Store hint metadata in extra_info because verl keeps extra_info through rollout.
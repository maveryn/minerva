0) Key constraints from verl you must design around

Your dataset “prompt” is chat-format messages (list of {role, content} dicts) in the parquet.

verl will sample rollout.n responses per prompt (n>1 is typical for GRPO). 
Verl

In the Ray trainer, non-tensor fields are aggressively dropped for rollout: _get_gen_batch keeps only data_source, reward_model, extra_info, uid (if present) and pops the rest.
So if you want to do per-task curriculum you should key by data_source (recommended) and/or store any curriculum metadata inside extra_info (since it survives).

Reward computation in verl expects the reward manager to return {"reward_tensor": ..., "reward_extra_info": ...} when called with return_dict=True. Your reward_extra_info is a clean way to pass back booleans like is_correct.

1) Data format you should store (parquet) for K-options

For each training sample row, add:

prompt: list[message] (existing)

data_source: string identifying the “label-space category” (e.g., cti_mitigation_id, cti_attack_technique_id, cti_cwe_id). This is how you’ll track success rate per category. 
Verl

reward_model.ground_truth: gold ID (existing pattern)

extra_info: dict; include at least an index if you want, and optionally task_name, etc.

NEW candidate_pool_top100: list[str] (or list[dict])
Suggested: store IDs only as strings to keep prompts short, e.g.

["T1059", "T1059.003", "T1047", ...]


If you really want names too, store ["T1059 | Command and Scripting Interpreter", ...], but watch max prompt length.

Important:

Make sure the gold label is in this pool (preferably at rank 0).

Your pool should be precomputed offline using BM25+dense similarity as you described.

2) What you will implement

You’ll add a dataset wrapper + a controller:

2.1 AdaptiveOptionController (per data_source)

Maintains for each task category s (keyed by data_source):

K_s: number of options shown (K=0 means no hint)

p_drop_s: probability of hiding options even when K>0 (for “weaning off”)

ema_acc_s: EMA of rollout accuracy (fraction correct across rollouts, i.e., your “~50% of 8 rollouts” metric)

Target:

Keep ema_acc_s ≈ 0.5 (your “~4/8 correct” objective), NOT pass@8.

Warmup:

For first W=50 steps, don’t adapt K; just measure baseline per category.

After warmup, initialize:

if ema_acc_s ≥ 0.5: set K_s=0 (no hints needed)

else set K_s=K_min (e.g., 2)

Update rule (per category, after each training step):

if ema_acc_s > target + tol:

increase K: K_s = min(K_max, K_s + ΔK)

if K_s == K_max and still too easy: increase p_drop_s (eventually options disappear)

if ema_acc_s < target - tol:

decrease K: K_s = max(K_min, K_s - ΔK)

also decrease p_drop_s toward 0 if it was >0

Use hysteresis (tol) to prevent oscillations.

2.2 AdaptiveOptionRLHFDataset (custom dataset class)

On each __getitem__:

Read candidate_pool_top100

Look up current K_s, p_drop_s using row’s data_source

If hint is on, build the option set:

Always include gold label

Take top min(len(pool), K_s + buffer) items, sample K_s-1 distractors from them (excluding gold)

Shuffle options

Inject options text into the prompt (ideally append to the last user message):

Include strict formatting instruction: “Answer with ONLY the ID (e.g., Txxxx / Mxxxx / CWE-xxx).”

Store hint metadata in extra_info:

extra_info["hint_K"], extra_info["hint_used"], extra_info["hint_pool_size"], extra_info["hint_seed"]

Why extra_info? It survives _get_gen_batch and is still present when reward is computed/logged.

3) Where to hook this into verl training (minimal patch)

You need one small trainer patch: after reward is computed on the repeated rollout batch, call train_dataset.option_controller.update_from_batch(...).

In RayPPOTrainer.fit, reward is computed and then assigned:
batch.batch["token_level_scores"] = reward_tensor

Right after that (and after reward_extra_infos_dict is merged), add:

Convert reward_tensor to per-sample scalar score

If 2D token-level: score_i = reward_tensor.sum(-1)

If 1D already: use as is

Determine correctness:

best is: if your reward function returns reward_extra_info["is_correct"], use that (more robust)

else: is_correct = (score_i > 0.5) for binary reward

Get category keys:

data_source = batch.non_tensor_batch["data_source"] (present)

Call dataset update:

self.train_dataset.update_option_curriculum(data_source, is_correct, global_step=self.global_steps, uid=batch.non_tensor_batch["uid"])

Also: log K_s and ema_acc_s into the metrics dict so wandb/swanlab sees the curriculum dynamics.

4) Practical “minor details” that will save you pain
4.1 Dataloader workers

Because your curriculum state changes during training, start with

data.dataloader_num_workers: 0

This avoids stale copies of the dataset in worker processes.

If you later need workers>0, move the controller state into shared memory:

multiprocessing.Manager().dict() (simple, slower)

or shared torch tensors (faster, a bit more code)

4.2 Prompt length limits

You can easily blow data.max_prompt_length by appending many options.
Options:

Keep options to IDs only (recommended)

Cap K_max (e.g., 30)

Add a “length guard”: if tokenized length > max_prompt_length, reduce K for this sample.

4.3 Deterministic randomness for resuming

Because StatefulDataLoader supports checkpoint/resume, you might want deterministic option sampling:

Seed per sample with (global_step, sample_index, base_seed)

Store hint_seed in extra_info for debugging

4.4 Don’t let hints leak into your final eval

For AthenaBench-style eval without hints, your training must include:

p_drop_s annealing to high values (or a final phase with K=0 always)

or mixed batches with hint dropout

Otherwise the model may become “MCQ-dependent.”

5) Config changes (YAML)

You will use verl’s data.custom_cls.path/name to load your dataset class. 
Verl

Example snippet:

data:
  train_files: /path/to/train.parquet
  val_files: /path/to/val.parquet
  prompt_key: prompt
  dataloader_num_workers: 0
  max_prompt_length: 1024
  max_response_length: 128

  custom_cls:
    path: /abs/path/to/minerva_adaptive_dataset.py
    name: AdaptiveOptionRLHFDataset

  adaptive_options:
    enabled: true
    candidate_pool_key: candidate_pool_top100
    target_acc: 0.50
    tol: 0.05
    warmup_steps: 50
    ema_beta: 0.90
    k_min: 2
    k_max: 30
    k_step: 2
    buffer: 10
    p_drop_init: 0.0
    p_drop_step: 0.05
    p_drop_max: 1.0
    score_threshold: 0.5


And make sure your rollout config has n: 8 (GRPO) as you already do. 
Verl

Copy-paste prompt for ChatGPT Codex

Use the block below as-is in Codex. It’s written so Codex produces (1) a new dataset file, (2) a small trainer patch, (3) config glue, with minimal ambiguity.

You are working in a fork of https://github.com/volcengine/verl.
Goal: implement an Adaptive Randomized Label-Hint Curriculum (“K-options”) for RLVR CTI tasks.

Context:
- Training uses RayPPOTrainer / GRPO with rollout.n=8.
- Data is a parquet dataset in verl RL format: columns include
  - "prompt": list of chat messages, each {"role": "...", "content": "..."} (or multimodal list)
  - "data_source": string identifying the task category / label space (e.g. cti_mitigation_id)
  - "reward_model": dict with "ground_truth" (the gold ID string)
  - "extra_info": dict (must survive into rollout; Ray trainer keeps extra_info)
- Add new column: "candidate_pool_top100": list[str] of candidate label IDs (gold included).
- We want to inject a list of K candidate options into the prompt to raise the probability of at least one correct rollout and to maintain ~50% rollout accuracy across the 8 rollouts.
- K is adaptive per data_source: low success → small K (easy), higher success → larger K (harder), and eventually we drop hints (K=0 or hint dropout probability p_drop→1).

Deliverables:
1) Create file minerva_adaptive_dataset.py implementing:
   - class OptionCurriculumController
   - class AdaptiveOptionRLHFDataset inheriting from verl.utils.dataset.rl_dataset.RLHFDataset

2) Patch verl/trainer/ppo/ray_trainer.py (RayPPOTrainer.fit) with a minimal change:
   after reward computation sets batch.batch["token_level_scores"]=reward_tensor and merges reward_extra_infos_dict,
   call self.train_dataset.update_option_curriculum(...) if that method exists.
   Also log curriculum metrics (K per data_source and EMA accuracy) into the existing `metrics` dict.

Implementation requirements:

A) Dataset: AdaptiveOptionRLHFDataset
- Parse config.data.adaptive_options (OmegaConf dict) in __init__. Provide defaults if missing.
- In __getitem__:
  - call super().__getitem__(idx) to obtain row_dict (contains raw_prompt etc in latest verl)
  - determine task_key = row_dict["data_source"]
  - read candidate pool list from row_dict[candidate_pool_key]
  - read gold label from row_dict["reward_model"]["ground_truth"]
  - query controller for (K, p_drop) for this task_key
  - if K==0 or random() < p_drop: do not add options; set extra_info["hint_used"]=False
  - else:
     * ensure gold is included
     * take top_pool = pool[:min(len(pool), K + buffer)]
     * sample (K-1) distractors from top_pool excluding gold
     * options = [gold] + distractors
     * shuffle options deterministically using seed = base_seed ^ (global_step<<16) ^ sample_index
     * format options as:
        "Candidate IDs (choose one):\n1) ID1\n2) ID2\n...\nReturn ONLY the ID."
     * inject into the last user message in raw_prompt (append to content).
     * set extra_info["hint_used"]=True, extra_info["hint_K"]=K, extra_info["hint_seed"]=seed
- Provide method update_option_curriculum_from_rollout(data_source_arr, uid_arr, is_correct_arr, global_step)
  that updates EMA accuracy per task and adjusts K/p_drop after warmup.

B) Controller: OptionCurriculumController
- Maintain per task_key:
  - K (int), p_drop (float), ema_acc (float)
- Config parameters:
  target_acc, tol, warmup_steps, ema_beta, k_min, k_max, k_step, p_drop_step, p_drop_max
- Warmup logic:
  - During steps <= warmup_steps: only accumulate ema_acc, do not change K/p_drop.
  - After warmup, if ema_acc >= target_acc: set K=0; else set K=k_min.
- Update logic after warmup:
  - if ema_acc > target+tol:
      if K < k_max: K += k_step
      else: p_drop = min(p_drop_max, p_drop + p_drop_step)
  - if ema_acc < target-tol:
      if K==0: K = k_min; p_drop = 0
      else: K = max(k_min, K - k_step); p_drop = max(0, p_drop - p_drop_step)
- Provide get_metrics(prefix="option_curriculum/") returning a flat dict like:
  option_curriculum/K/<task_key>, option_curriculum/p_drop/<task_key>, option_curriculum/ema_acc/<task_key>

C) Trainer patch: RayPPOTrainer.fit
- Find the section where reward_tensor is computed and later assigned:
  batch.batch["token_level_scores"] = reward_tensor
- Immediately after that, compute per-sample scalar score:
  if reward_tensor.ndim==2: sample_score = reward_tensor.sum(dim=-1)
  else: sample_score = reward_tensor
- Determine is_correct:
  - Prefer: if "is_correct" exists in batch.non_tensor_batch (from reward_extra_infos_dict), use it.
  - Else: is_correct = (sample_score > score_threshold)
- Get data_source list from batch.non_tensor_batch["data_source"]
- Get uid list from batch.non_tensor_batch["uid"]
- If hasattr(self.train_dataset, "update_option_curriculum_from_rollout"):
    curriculum_metrics = self.train_dataset.update_option_curriculum_from_rollout(
        data_source_arr=data_source, uid_arr=uid, is_correct_arr=is_correct, global_step=self.global_steps
    )
    metrics.update(curriculum_metrics)
  Also metrics.update(self.train_dataset.option_controller.get_metrics(...)) if available.
- Keep the patch minimal and safe; don’t break if keys are missing.

D) Notes:
- Because Ray trainer pops most non_tensor keys during rollout, store hint metadata inside extra_info (which is kept).
- Start with data.dataloader_num_workers=0; add comment in code.

Deliver code with type hints, docstrings, and clear comments.
Make sure imports are correct for current verl.
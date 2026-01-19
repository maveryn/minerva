# CODEX TASK: Add Online Verified Self-Distillation (Inline SFT Micro-Epochs) on top of SLHC (verl)

We already have SLHC implemented per slhc.md:
- dataset wrapper injects stochastic label-hints + stores slhc_* metadata
- RayPPOTrainer.fit computes group metrics and updates SLHC controller
- 3 variants selectable by ONE arg: data.stochastic_slhc.control_mode in {acc_target, zero_constraint, two_tail_smooth}

Now implement an OPTIONAL addon:
"Online Verified Self-Distillation" (OVSD):
- During RL training, collect (no-hint prompt, chosen completion) pairs from rollouts with reward>=0.5
- Every `distill.interval` RL iterations (e.g., 10), run 1 epoch of SFT on those pairs (no-hint prompts only)
- Clear buffer after SFT, then continue RL
- Must be robust to vLLM output ordering (use uid grouping, never rely on batch order)
- Must be distributed-safe (DDP/FSDP): all ranks must do same SFT steps

We do NOT need Codex to re-implement SLHC; only extend it.

========================================================
1) CONFIG CHANGES (YAML)
========================================================
Add a new config block. Keep it minimal; no hyperparameter tuning expected.

data:
  stochastic_slhc:
    ...
    distill:
      enabled: true
      interval: 10              # run SFT every N RL iterations
      reward_threshold: 0.5     # only distill from rollouts with reward>=0.5
      max_buffer: 4096          # cap buffer size (global) per interval
      batch_size: null          # if null, reuse RL microbatch or ppo minibatch
      max_seq_len: null         # if null, reuse model max length / trainer setting
      entropy_tiebreak: "mean_nll"  # proxy entropy: mean(-logp) of sampled tokens
      drop_last: true           # important for distributed consistency
      dedup_by_uid: true        # keep best sample per uid

Only distill.enabled should be toggled for ablation; other values can have safe defaults.

========================================================
2) DATASET CHANGE: store NO-HINT prompt copy in extra_info (Option A)
========================================================
File: minerva_stochastic_slhc_dataset.py
Class: StochasticSLHCRLHFDataset.__getitem__

Requirement:
- Store a deep copy of the base prompt BEFORE injecting hints.
- This prevents later in-place augmentation from mutating the stored no-hint prompt.

In __getitem__ after row=super().__getitem__(idx) and before any hint injection:
- base_prompt = deepcopy(row["prompt"])  # list[{role, content}, ...]
- extra = row.get("extra_info", {}) or {}
- extra["slhc_prompt_nohint"] = base_prompt
- row["extra_info"] = extra

Then continue existing SLHC decision logic and potential hint injection.

Also ensure existing keys remain:
- extra["slhc_ctrl_active"], extra["slhc_hint_used"], extra["slhc_nohint_reason"], etc.

========================================================
3) TRAINER EXTENSION: maintain a distillation buffer and collect samples per uid group
========================================================
File: verl/trainer/ppo/ray_trainer.py
Class: RayPPOTrainer

3.1 Add new members in __init__ (or trainer setup):
- self._distill_cfg = cfg.data.stochastic_slhc.distill
- self._distill_buffer_local = []   # list of dict records (local rank)
- self._distill_last_run_step = -1  # last global step we ran distill SFT

3.2 After reward computation in fit(), we already compute:
- reward_scalar per rollout
- success metrics for SLHC (reward>=0.1 for acc@G/zero@G/all@G)

Now ALSO compute distillation candidates:
- Distillation eligibility per rollout: reward_scalar >= distill.reward_threshold (default 0.5)

IMPORTANT: vLLM order randomization:
- Always group by uid (same as SLHC grouping), then choose one rollout index inside each uid group.

Implementation outline inside RayPPOTrainer.fit after reward_scalar is available:

A) Identify required batch fields (Codex must inspect batch structure):
- uid_arr = batch.non_tensor_batch["uid"]
- task_arr = batch.non_tensor_batch["data_source"]
- extra_arr = batch.non_tensor_batch["extra_info"]
- We also need the generated completion tokens or text:
  - Prefer token IDs: e.g., batch.batch["response_ids"] or similar
  - If only text exists: batch.non_tensor_batch["response"] (fallback)

Codex: search in the code around rollout generation for where responses are stored in Batch.
Add a debug print in a safe dev branch if needed:
- print(batch.batch.keys(), batch.non_tensor_batch.keys())

B) For each uid group:
- idxs = (uid_arr == uid)
- rewards = reward_scalar[idxs]            # shape [G]
- eligible = rewards >= distill.reward_threshold
- if eligible.sum()==0: continue (skip uid)
- candidates = indices where eligible True

Selection rule:
1) max_reward = max(rewards[candidates])
2) top = candidates where rewards == max_reward (or within small eps for float)
3) if len(top)==1: chosen_idx = top[0]
4) else: tie-break by entropy proxy (mean_nll):
   - Need per-token chosen logprobs for each rollout completion.
   - Prefer using logprobs already computed for PPO ratios (search for them in batch).
   - Compute mean_nll = -mean(logp_tokens) over generated tokens.
   - chosen_idx = argmax(mean_nll) among top
If logprobs are not available in batch:
- Run a single forward pass to compute chosen-token logprobs for only the `top` candidates (small set).
- Use teacher-forcing on (prompt_used, response_tokens) OR use a helper that computes logprobs for response tokens given prompt tokens.
- Keep it minimal (only for tie-break).

C) Build the distillation record:
- prompt_nohint = extra_arr[chosen_idx]["slhc_prompt_nohint"]  # list of messages
- completion_tokens (preferred) = response_ids[chosen_idx] OR completion_text
- Save record:
  {
    "uid": uid,
    "task_key": task_arr[chosen_idx],
    "prompt_nohint": prompt_nohint,
    "response_ids": list[int] OR None,
    "response_text": str OR None,
    "reward": float(reward_scalar[chosen_idx]),
    "entropy_proxy": float(mean_nll) if available else None,
  }

Append to self._distill_buffer_local.

D) Cap buffer size locally:
- If len(buffer_local) exceeds max_buffer (or max_buffer/world_size), drop oldest or drop lowest-reward items.
Keep it simple:
- if len(buffer_local) > max_local: buffer_local = buffer_local[-max_local:]

========================================================
4) RUN INLINE SFT EVERY `distill.interval` RL ITERATIONS (1 epoch), then clear buffer
========================================================
Add a method on RayPPOTrainer:

def _maybe_run_distill_sft(self, global_step: int):
  cfg = self._distill_cfg
  if not cfg.enabled: return
  if cfg.interval <= 0: return
  if (global_step + 1) % cfg.interval != 0: return
  if global_step == self._distill_last_run_step: return

  # 1) Gather buffer across ranks (distributed-safe)
  # 2) Build SFT dataset (no-hint prompts)
  # 3) Run 1 epoch of supervised CE training
  # 4) Clear buffer
  # 5) Sync updated weights to rollout workers

Call _maybe_run_distill_sft at the end of each RL iteration in fit(), AFTER the RL (GRPO) optimizer update is done.

4.1 Distributed-safe gather
Use torch.distributed if initialized:
- local_list = self._distill_buffer_local
- global_list = []
- if dist.is_initialized():
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_list)
    global_list = concat(gathered)
  else:
    global_list = local_list

If cfg.dedup_by_uid:
- keep only best per uid:
  - choose higher reward; if tie, higher entropy_proxy

Then cap to cfg.max_buffer (GLOBAL cap):
- sort by reward desc (then entropy desc) and take first cfg.max_buffer

IMPORTANT: For DDP consistency, ensure all ranks use the SAME global_list ordering and length.
- Do sorting and truncation identically on all ranks.
- If dist initialized and you want to be extra safe:
  - do sorting/truncation on rank 0, then broadcast the final list via dist.broadcast_object_list (or all_gather_object already does it but each rank must apply same operations deterministically).

4.2 Convert distill records into tokenized SFT examples
We train on NO-HINT prompts only.

Preferred approach (robust, avoids text/template drift):
- Tokenize prompt_nohint messages to prompt_ids using the SAME tokenizer/chat-template logic as RL prompts.
  - Use tokenizer.apply_chat_template(prompt_nohint, tokenize=True, add_generation_prompt=True)
  - OR reuse existing helper from verl that turns a list[{role,content}] into input_ids.
- Use response_ids from rollout if present (best).
- If response_ids not available, tokenize response_text (without adding a new generation prompt).

Build:
- input_ids = prompt_ids + response_ids
- labels = [-100]*len(prompt_ids) + response_ids
- attention_mask = [1]*len(input_ids)
- Truncate to max_seq_len:
  - If too long, truncate from the LEFT of prompt_ids (keep end of prompt + full answer if possible)
  - Always keep at least some prompt context.
  - Ensure labels align with input_ids after truncation.

Store each as a dict with torch tensors.

4.3 SFT DataLoader
Create a small in-memory Dataset class inside ray_trainer.py OR a new file, e.g.:
- class DistillSFTDataset(torch.utils.data.Dataset)

Implement collate_fn:
- pad input_ids to max length in batch with pad_token_id
- pad attention_mask with 0
- pad labels with -100
Return tensors on correct device later.

Batch size:
- if cfg.batch_size is None:
    use existing RL micro batch size or ppo mini batch size from cfg (Codex to find the most appropriate existing field)
Else use cfg.batch_size.

Set drop_last according to cfg.drop_last (True recommended when dist distributed).

If dist initialized:
- use DistributedSampler(dataset, shuffle=True, drop_last=cfg.drop_last)
- DataLoader(..., sampler=sampler, shuffle=False)

If not distributed:
- DataLoader(..., shuffle=True)

4.4 Run one epoch of SFT
Implement:
def _run_one_epoch_sft(self, dataloader) -> dict metrics

Inside:
- put actor/policy model in train() mode
- for batch in dataloader:
    - move tensors to device
    - forward pass:
        outputs = self.actor_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits
    - compute CE with ignore_index=-100:
        shift logits/labels for causal LM:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]
        loss = cross_entropy(logits.reshape(-1,V), labels.reshape(-1), ignore_index=-100)
    - backward + optimizer step using the SAME strategy/engine used in RL updates:
        - if using deepspeed/accelerate/fsdp wrappers, call the same helper used in RL:
            self.strategy.backward(loss) / self.strategy.optimizer_step(...)
          Codex must locate how RL loss is backprop’d and reuse that exact pattern.
    - zero grads

Return metrics:
- distill_sft/loss_mean
- distill_sft/num_examples
- distill_sft/avg_reward_of_examples

IMPORTANT: keep it simple:
- exactly 1 epoch
- do NOT change SLHC controller state here
- after epoch, clear buffers

4.5 Clear buffers + sync weights to rollout workers
After SFT:
- self._distill_buffer_local = []
- self._distill_last_run_step = global_step

Then ensure rollout workers get updated weights before next rollout:
- Call the same weight-sync method used after RL updates (Codex: search for "sync" / "broadcast" / "update_weight" in RayPPOTrainer).
- If the RL loop already syncs at the start of each iteration, still trigger it here if possible (since weights changed post-RL).

Log metrics into the existing `metrics` dict:
- distill/ran = 1
- distill/buffer_size_global
- distill/sft_loss
etc.

========================================================
5) IMPORTANT EDGE CASES / SAFETY
========================================================
- If global_list is empty: skip SFT.
- If dataset tokenization fails (rare): skip that record.
- If response_ids are empty: skip.
- Ensure prompt_nohint stored is a deep copy; otherwise hint injection will mutate it.
- Ensure DDP consistency: all ranks must perform same number of SFT steps. Prefer drop_last=True.
- Keep distillation prompts NO-HINT: always use slhc_prompt_nohint, never the possibly-hinted prompt.

========================================================
6) OUTPUT
========================================================
- Log fraction of uid groups that produced a distill sample
- Log mean reward of selected samples
- Log how many samples came from hinted vs non-hinted rollouts (slhc_hint_used from extra_info)
- Log entropy_proxy stats if available

END TASK

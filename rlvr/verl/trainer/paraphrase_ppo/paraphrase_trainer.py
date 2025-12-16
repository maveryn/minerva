# verl/trainer/paraphrase_ppo/paraphrase_trainer.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paraphrase PPO Trainer with answer and question paraphrasing variants."""

from collections import deque
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _BasePPO
from verl.trainer.ppo.ray_trainer import (
    compute_advantage as _compute_advantage,
    compute_response_mask as _compute_response_mask,
    apply_kl_penalty as _apply_kl_penalty,
)
from verl.trainer.ppo.reward import compute_reward as _compute_reward, compute_reward
from verl.trainer.ppo.core_algos import agg_loss, AdvantageEstimator
from verl.utils.tracking import Tracking
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
import numpy as np
import uuid
from tqdm import tqdm

from .paraphrase_utils import (
    SampleRecord,
    compute_transport_ratio,
    rescore_under_plain_prompt,
    compute_stable_qid,
    group_by_qid,
    group_by_group_id,
    validate_transport_ratio,
)
from .paraphrase_samplers import (
    rollout_with_tracking,
    score_reward_under_plain,
    collect_answer_paraphrase_batch,
    collect_question_paraphrase_batch,
)
from .paraphrase_logging import (
    ParaphraseMetricsTracker,
    compute_transport_statistics,
    log_paraphrase_examples,
)


class ParaphrasePPOTrainer(_BasePPO):
    """Paraphrase PPO trainer supporting answer and question paraphrasing variants.
    
    Key features:
    - Answer paraphrasing: Generate condensed/verified rewrites of successful solutions
    - Question paraphrasing: Generate answer-preserving rephrases of problems
    - Off-policy to on-policy transport with cross-context ratios
    - Optional CE distillation for discovered successes
    - Balanced group selection for optimal learning signal
    """
    
    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping=None,
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        # Force GRPO mode
        if not hasattr(config, "algorithm"):
            raise ValueError("config.algorithm missing")
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False
        
        # Disable PPO clipping
        if hasattr(config.algorithm, "clip_coef"):
            config.algorithm.clip_coef = float("inf")
        if hasattr(config, "actor_rollout_ref") and hasattr(config.actor_rollout_ref, "actor"):
            try:
                config.actor_rollout_ref.actor.clip_coef = float("inf")
            except Exception:
                pass
        
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        # Paraphrase configuration
        self.para_cfg = config.get("paraphrase", {})
        self.variant = self.para_cfg.get("variant", "answer")
        self.k_rollouts = self.para_cfg.get("k_rollouts", 8)
        self.paraphrase_per_x = self.para_cfg.get("paraphrase_per_x", 1)
        self.target_solve_rate = self.para_cfg.get("target_solve_rate", 0.5)
        
        # Selection config
        self.select_cfg = self.para_cfg.get("select", {})
        self.correct_topk = self.select_cfg.get("correct_topk", 4)
        self.incorrect_fill = self.select_cfg.get("incorrect_fill", "high")
        self.seed_correct = self.select_cfg.get("seed_correct", "lowest_entropy")
        
        # Answer styles
        self.answer_styles = self.para_cfg.get("answer_styles", [
            "condense; keep only decisive steps",
            "number key steps and final check", 
            "self-verify then present final concise solution"
        ])
        
        # Generation config
        self.max_new_tokens = self.para_cfg.get("max_new_tokens", 512)
        
        # CE distillation
        ce_cfg = self.para_cfg.get("ce_distill", {})
        self.ce_enabled = ce_cfg.get("enabled", True)
        self.ce_weight = ce_cfg.get("weight", 0.1)
        self.ce_buffer_size = ce_cfg.get("buffer_size", 2048)
        self.ce_buffer = deque(maxlen=self.ce_buffer_size)
        
        # Transport config
        transport_cfg = self.para_cfg.get("transport", {})
        self.transport_enabled = transport_cfg.get("enabled", True)
        self.clip_ratio = transport_cfg.get("clip_ratio", 0.2)
        self.log_ratio_cap = transport_cfg.get("log_ratio_cap", 8.0)
        
        # Logging
        log_cfg = self.para_cfg.get("logging", {})
        self.debug_rows = log_cfg.get("debug_rows", True)
        self.write_every_steps = log_cfg.get("write_every_steps", 50)
        log_dir = log_cfg.get("dir", f"{config.trainer.default_local_dir}/paraphrase_logs")
        self.metrics_tracker = ParaphraseMetricsTracker(log_dir, self.write_every_steps)
        
        # Round counter
        self.round_id = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @torch.no_grad()
    def collect_batch(self, prompts: List[str]) -> Tuple[List[SampleRecord], List[tuple]]:
        """Collect batch with paraphrasing based on variant.
        
        Args:
            prompts: Input prompts
            
        Returns:
            (sample_records, ce_pairs) - Records and CE distillation pairs
        """
        # First, do standard GRPO rollouts
        plain_records = self._collect_plain_rollouts(prompts)
        
        # Fallback strategy: if no success exists in plain records, use question paraphrasing
        any_success = any(r.reward == 1 for r in plain_records)

        if self.variant == "answer" and any_success:
            return collect_answer_paraphrase_batch(
                model_or_wg=self.actor_rollout_wg,  # Use worker group
                tokenizer=self.tokenizer,
                verifier=self.reward_fn,
                plain_records=plain_records,
                config=self.para_cfg,
                device=self.device
            )
        elif self.variant == "question" or not any_success:
            return collect_question_paraphrase_batch(
                model_or_wg=self.actor_rollout_wg,  # Use worker group
                tokenizer=self.tokenizer,
                verifier=self.reward_fn,
                plain_records=plain_records,
                prompts=prompts,
                config=self.para_cfg,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown paraphrase variant: {self.variant}")
    
    def _collect_plain_rollouts(self, prompts: List[str]) -> List[SampleRecord]:
        """Collect standard GRPO rollouts on plain prompts.
        
        Args:
            prompts: Input prompts
            
        Returns:
            List of sample records
        """
        records = []
        
        # Tokenize prompts to get tensors
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # Create batch for generation
        batch_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "position_ids": torch.arange(tokenized["input_ids"].shape[1], dtype=torch.long).unsqueeze(0).expand(tokenized["input_ids"].shape[0], -1)
        }
        gen_batch = DataProto.from_single_dict(batch_dict)
        gen_batch = gen_batch.repeat(repeat_times=self.k_rollouts, interleave=True)
        
        # Add required meta info
        gen_batch.meta_info = {"eos_token_id": self.tokenizer.eos_token_id}
        
        # Generate sequences using actor rollout worker group
        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # Extract generated sequences
        responses = gen_output.batch["responses"]  # [batch_size * k_rollouts, seq_len]
        old_logprobs = gen_output.batch.get("log_probs", None)
        
        # Process each rollout
        for i, prompt in enumerate(prompts):
            qid = compute_stable_qid(prompt)
            group_id = f"{qid}:{self.round_id}:PLAIN"
            
            for j in range(self.k_rollouts):
                idx = i * self.k_rollouts + j
                response_tokens = responses[idx]
                
                # Decode response - ensure tokens are on CPU for decoding
                if isinstance(response_tokens, torch.Tensor):
                    response_tokens_cpu = response_tokens.cpu()
                else:
                    response_tokens_cpu = response_tokens
                response_text = self.tokenizer.decode(response_tokens_cpu, skip_special_tokens=True)
                
                # Calculate path entropy from generation (simplified)
                path_entropy = 0.0  # Would need to extract from generation process
                
                # Ensure response_tokens is a tensor
                if not isinstance(response_tokens, torch.Tensor):
                    response_tokens = torch.tensor(response_tokens, device=self.device)
                
                record = SampleRecord(
                    qid=qid,
                    round_id=self.round_id,
                    group_id=group_id,
                    ctx_type="PLAIN",
                    source_id="ON",
                    sid=f"{qid}:{self.round_id}:PLAIN:{j}",
                    parent_sid=None,
                    prompt_plain=prompt,
                    prompt_used=prompt,
                    tokens=response_tokens,
                    reward=0,  # Will be scored
                    lengths=len(response_tokens),
                    path_entropy=path_entropy,
                    old_logprobs_ctx=old_logprobs[idx] if old_logprobs is not None else torch.zeros_like(response_tokens, dtype=torch.float32),
                    text=response_text
                )
                
                # Score reward via reward manager
                record.reward = score_reward_under_plain(
                    verifier=self.reward_fn,
                    tokenizer=self.tokenizer,
                    prompt_plain=prompt,
                    response_text=response_text
                )
                
                records.append(record)
        
        return records
    
    def compute_losses(
        self,
        sample_records: List[SampleRecord]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GRPO loss with transport ratios and optional CE.
        
        Args:
            sample_records: Batch of sample records
            
        Returns:
            (total_loss, metrics) - Combined loss and metrics dict
        """
        # Compute transport ratios for off-policy samples
        for rec in sample_records:
            if rec.source_id.startswith('OFF'):
                # Re-score under plain prompt
                rec.cur_logprobs_plain = rescore_under_plain_prompt(
                    model=self.actor_model,
                    tokenizer=self.tokenizer,
                    prompt_plain=rec.prompt_plain,
                    tokens=rec.tokens,
                    device=self.device
                )
                
                # Compute transport ratio
                rec.transport_log_ratio = compute_transport_ratio(
                    cur_logprobs_plain=rec.cur_logprobs_plain,
                    old_logprobs_ctx=rec.old_logprobs_ctx,
                    log_ratio_cap=self.log_ratio_cap,
                    device=self.device
                )
            else:
                # Standard on-policy ratio
                rec.cur_logprobs_plain = self._compute_current_logprobs(
                    rec.prompt_plain,
                    rec.tokens
                )
                rec.transport_log_ratio = rec.cur_logprobs_plain - rec.old_logprobs_ctx
        
        # Compute GRPO advantages over mixed groups
        groups = group_by_group_id(sample_records)
        L_rl = 0.0
        metrics = {}
        
        for gid, group in groups.items():
            rewards = torch.tensor([r.reward for r in group], device=self.device, dtype=torch.float32)
            
            # GRPO advantage: normalize within group
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            
            # Token-wise PPO/GRPO loss
            for rec, adv in zip(group, advantages):
                # Get response mask
                response_mask = self._get_response_mask(rec.prompt_plain, rec.tokens)
                
                # Compute ratio
                rho = torch.exp(rec.transport_log_ratio)
                
                # Apply PPO clipping if enabled (though we set clip_coef=inf)
                if self.clip_ratio < float('inf'):
                    rho_clip = torch.clamp(rho, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    token_loss = -torch.minimum(rho * adv, rho_clip * adv)
                else:
                    token_loss = -rho * adv
                
                # Mask and average
                masked_loss = token_loss * response_mask
                L_rl += masked_loss.sum() / response_mask.sum()
        
        L_rl = L_rl / len(groups)  # Average over groups
        
        # Optional CE distillation
        L_ce = 0.0
        if self.ce_enabled and len(self.ce_buffer) > 0:
            L_ce = self._compute_ce_loss()
        
        # Total loss
        total_loss = L_rl + self.ce_weight * L_ce
        
        # Compute metrics
        metrics['L_rl'] = L_rl.item()
        metrics['L_ce'] = L_ce.item() if isinstance(L_ce, torch.Tensor) else L_ce
        metrics['total_loss'] = total_loss.item()
        
        # Transport statistics
        transport_stats = compute_transport_statistics(sample_records, self.clip_ratio)
        metrics.update(transport_stats)
        
        return total_loss, metrics
    
    def _compute_current_logprobs(
        self,
        prompt: str,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute current policy logprobs for on-policy samples.
        
        Args:
            prompt: Input prompt
            tokens: Response tokens
            
        Returns:
            Token-level log probabilities
        """
        return rescore_under_plain_prompt(
            model=self.actor_model,
            tokenizer=self.tokenizer,
            prompt_plain=prompt,
            tokens=tokens,
            device=self.device
        )
    
    def _get_response_mask(
        self,
        prompt: str,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """Get mask for response tokens.
        
        Args:
            prompt: Input prompt
            tokens: Response tokens
            
        Returns:
            Binary mask tensor
        """
        # Simple implementation - all tokens are response
        return torch.ones_like(tokens, dtype=torch.float32)
    
    def _compute_ce_loss(self) -> torch.Tensor:
        """Compute CE distillation loss on buffer.
        
        Returns:
            CE loss value
        """
        if not self.ce_buffer:
            return torch.tensor(0.0, device=self.device)
        
        # Sample from buffer
        batch_size = min(16, len(self.ce_buffer))
        sampled_pairs = random.sample(list(self.ce_buffer), batch_size)
        
        total_loss = 0.0
        for prompt, response in sampled_pairs:
            # Tokenize
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Compute CE loss
            full_ids = torch.tensor(prompt_ids + response_ids, device=self.device)
            
            with torch.no_grad():
                outputs = self.actor_model(
                    input_ids=full_ids.unsqueeze(0),
                    use_cache=False
                )
                logits = outputs.logits[0]
            
            # Extract response logits
            prompt_len = len(prompt_ids)
            response_logits = logits[prompt_len-1:-1]
            response_targets = full_ids[prompt_len:]
            
            # CE loss
            loss = F.cross_entropy(response_logits, response_targets)
            total_loss += loss
        
        return total_loss / batch_size
    
    def fit(self):
        """Training loop with paraphrasing.
        
        This overrides the base fit() method to inject paraphrasing logic
        into the standard PPO flow.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from copy import deepcopy
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # Progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # Start from step 1
        self.global_steps += 1
        self.round_id = self.global_steps
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                # Create base batch
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # Add uid for tracking
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                # Store reward_fn_key for later use
                self.reward_fn_key = self.config.data.get("reward_fn_key", "data_source")
                
                # Extract prompts for paraphrasing BEFORE _get_gen_batch pops the keys
                prompts = self.tokenizer.batch_decode(
                    batch.batch["input_ids"], skip_special_tokens=True
                )
                
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                with marked_timer("paraphrase_ppo_step", timing_raw):
                    # Collect batch with paraphrasing
                    with marked_timer("collect_paraphrase_batch", timing_raw):
                        sample_records, ce_pairs = self.collect_batch(prompts)
                        
                    # Update CE buffer
                    if self.ce_enabled:
                        self.ce_buffer.extend(ce_pairs)
                    
                    # Convert sample records to DataProto format
                    with marked_timer("convert_to_dataproto", timing_raw):
                        para_batch = self._convert_records_to_dataproto(sample_records, batch)
                    
                    # Compute rewards
                    with marked_timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(para_batch)
                            para_batch = para_batch.union(reward_tensor)
                        
                        # Use custom compute_reward that handles tensor indexing
                        reward_tensor, reward_extra_infos_dict = self._compute_reward_safe(para_batch)
                        para_batch.batch["token_level_scores"] = reward_tensor
                        
                        if reward_extra_infos_dict:
                            para_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                    
                    # Compute old log probs for the entire mixed batch to align with PPO expectations
                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(para_batch)
                        # Drop entropys if present (align with ray trainer behavior)
                        if "entropys" in old_log_prob.batch:
                            old_log_prob.batch.pop("entropys")
                        para_batch = para_batch.union(old_log_prob)
                    
                    # Reference policy if needed
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(para_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(para_batch)
                            para_batch = para_batch.union(ref_log_prob)
                    
                    # Compute values if using critic
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(para_batch)
                            para_batch = para_batch.union(values)
                    
                    # Apply KL penalty if configured
                    if self.config.algorithm.use_kl_in_reward:
                        para_batch, kl_metrics = _apply_kl_penalty(
                            para_batch, kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        para_batch.batch["token_level_rewards"] = para_batch.batch["token_level_scores"]
                    
                    # Compute advantages using standard GRPO over groups keyed by uid
                    with marked_timer("advantage", timing_raw):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        para_batch = _compute_advantage(
                            para_batch,
                            adv_estimator=AdvantageEstimator.GRPO,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.k_rollouts,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Compute global valid token counts for FLOPs/memory metrics in workers
                    # Matches logic in mix_src and PPO trainers
                    para_batch.meta_info["global_token_num"] = (
                        torch.sum(para_batch.batch["attention_mask"], dim=-1).tolist()
                    )
                    
                    # Update critic if used
                    if self.use_critic and self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(para_batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                    
                    # Update actor with standard PPO step
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(para_batch)
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    
                    # Compute CE loss if enabled
                    if self.ce_enabled and len(self.ce_buffer) > 0:
                        with marked_timer("ce_distill", timing_raw):
                            ce_loss = self._compute_and_apply_ce_loss()
                            metrics["loss/ce_distill"] = ce_loss
                
                # Logging
                if self.debug_rows:
                    self.metrics_tracker.log_debug_rows(sample_records)
                
                transport_stats = compute_transport_statistics(sample_records, self.clip_ratio)
                self.metrics_tracker.log_batch_metrics(
                    records=sample_records,
                    ce_pairs=ce_pairs,
                    transport_stats=transport_stats,
                    loss_dict=metrics,
                    global_step=self.global_steps
                )
                
                # Periodic validation
                is_last_step = self.global_steps >= self.total_training_steps
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    val_metrics = self._validate()
                    metrics.update(val_metrics)
                
                # Save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()
                
                # Update metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                
                # Log all metrics
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                self.round_id = self.global_steps
                
                if is_last_step:
                    progress_bar.close()
                    return
    
    def _compute_reward_safe(self, data: DataProto) -> tuple[torch.Tensor, dict]:
        """Compute reward with safe tensor indexing handling."""
        # Create a wrapper class that intercepts DataProtoItem access
        class SafeDataProto:
            def __init__(self, data):
                self._data = data
                
            def __len__(self):
                return len(self._data)
                
            def __getitem__(self, idx):
                item = self._data[idx]
                
                # Create a wrapper for the batch that handles tensor operations
                class SafeBatch:
                    def __init__(self, batch):
                        self._batch = batch
                        
                    def __getitem__(self, key):
                        value = self._batch[key]
                        if key == "attention_mask":
                            # Wrap attention mask to ensure sum() returns int
                            class AttentionMaskWrapper:
                                def __init__(self, tensor):
                                    self._tensor = tensor
                                    
                                def __getitem__(self, idx):
                                    sliced = self._tensor[idx]
                                    # Return wrapper for sliced tensor
                                    return AttentionMaskWrapper(sliced)
                                    
                                def sum(self):
                                    result = self._tensor.sum()
                                    # Convert to int for indexing
                                    if hasattr(result, 'item'):
                                        return int(result.item())
                                    return int(result)
                                    
                                # Delegate other operations to underlying tensor
                                def __getattr__(self, name):
                                    return getattr(self._tensor, name)
                                    
                                @property
                                def shape(self):
                                    return self._tensor.shape
                                    
                            return AttentionMaskWrapper(value)
                        return value
                        
                    def keys(self):
                        return self._batch.keys()
                
                # Return wrapped item
                from verl.protocol import DataProtoItem
                return DataProtoItem(
                    batch=SafeBatch(item.batch),
                    non_tensor_batch=item.non_tensor_batch,
                    meta_info=item.meta_info
                )
                
            @property
            def batch(self):
                return self._data.batch
                
            @property
            def non_tensor_batch(self):
                return self._data.non_tensor_batch
                
            @property
            def meta_info(self):
                return self._data.meta_info
        
        # Call compute_reward with wrapped data
        return compute_reward(SafeDataProto(data), self.reward_fn)
    
    def _convert_records_to_dataproto(self, records: List[SampleRecord], base_batch: DataProto) -> DataProto:
        """Convert SampleRecords to DataProto format expected by base trainer.
        
        Args:
            records: List of sample records
            base_batch: Original batch for structure reference
            
        Returns:
            DataProto with paraphrase data
        """
        # Group records by original prompt index
        batch_size = len(base_batch.batch)
        k_rollouts = self.k_rollouts
        
        # Initialize tensors
        all_prompts = []
        all_responses = []
        all_input_ids = []
        all_old_logprobs = []
        all_rewards = []
        all_uids = []
        all_source_ids = []
        
        # Initialize lists for non-tensor data from base batch
        all_reward_model_data = []
        all_data_sources = []
        all_extra_info = []
        
        # Process records in order
        for i, rec in enumerate(records):
            # Tokenize prompt if needed
            prompt_ids = self.tokenizer.encode(rec.prompt_plain, add_special_tokens=True)
            prompt_tensor = torch.tensor(prompt_ids, device=self.device)
            all_prompts.append(prompt_tensor)
            
            # Response tokens - ensure they're tensors on the correct device
            response_tensor = rec.tokens
            if not isinstance(response_tensor, torch.Tensor):
                response_tensor = torch.tensor(response_tensor, device=self.device)
            elif response_tensor.device != self.device:
                response_tensor = response_tensor.to(self.device)
            all_responses.append(response_tensor)
            
            # Create input_ids by concatenating prompt and response
            input_ids = torch.cat([prompt_tensor, response_tensor])
            all_input_ids.append(input_ids)
            
            # Ensure old_logprobs are on the correct device
            old_logprobs_tensor = rec.old_logprobs_ctx
            if not isinstance(old_logprobs_tensor, torch.Tensor):
                old_logprobs_tensor = torch.tensor(old_logprobs_tensor, device=self.device, dtype=torch.float32)
            elif old_logprobs_tensor.device != self.device:
                old_logprobs_tensor = old_logprobs_tensor.to(self.device)
            all_old_logprobs.append(old_logprobs_tensor)
            all_rewards.append(rec.reward)
            all_uids.append(rec.qid)
            all_source_ids.append(rec.source_id)
            
            # Extract original non-tensor data from base batch
            # Map back to original batch index
            orig_idx = i // k_rollouts  # Each original sample generates k_rollouts records
            
            # Copy non-tensor data from original batch
            if "reward_model" in base_batch.non_tensor_batch:
                all_reward_model_data.append(base_batch.non_tensor_batch["reward_model"][orig_idx])
            else:
                # Provide default structure if not present
                all_reward_model_data.append({"ground_truth": ""})
                
            if hasattr(self, 'reward_fn_key') and self.reward_fn_key in base_batch.non_tensor_batch:
                all_data_sources.append(base_batch.non_tensor_batch[self.reward_fn_key][orig_idx])
            else:
                # Provide default data source
                all_data_sources.append(rec.prompt_plain)
                
            if "extra_info" in base_batch.non_tensor_batch:
                all_extra_info.append(base_batch.non_tensor_batch["extra_info"][orig_idx])
            else:
                all_extra_info.append({})
        
        # Ensure consistent sequence alignment between prompts, responses, and input_ids.
        # We must right-pad input_ids and responses to the same total length so that
        # response_mask and attention_mask align like the core PPO trainer expects.
        # Pad sequences
        from torch.nn.utils.rnn import pad_sequence
        prompts_padded = pad_sequence(all_prompts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        responses_padded = pad_sequence(all_responses, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids_padded = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        old_logprobs_padded = pad_sequence(all_old_logprobs, batch_first=True, padding_value=0.0)
        
        # Create attention masks
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).float()
        # Response mask should be aligned to the tail of attention_mask with the same
        # response length (equal to responses_padded.shape[1]). The ray PPO trainer
        # uses response_mask = attention_mask[:, -response_length:].
        response_length = responses_padded.shape[1]
        response_mask = attention_mask[:, -response_length:]
        
        # Create position_ids
        seq_length = input_ids_padded.shape[1]
        position_ids = torch.arange(seq_length, device=self.device).unsqueeze(0).expand(len(records), -1)
        
        # Build batch dict
        batch_dict = {
            "prompts": prompts_padded,
            "responses": responses_padded,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "old_log_probs": old_logprobs_padded,
        }
        
        # Non-tensor data
        reward_fn_key = getattr(self, 'reward_fn_key', 'data_source')
        non_tensor_batch = {
            "uid": np.array(all_uids, dtype=object),
            "source_id": np.array(all_source_ids, dtype=object),
            "reward": np.array(all_rewards),
            "reward_model": np.array(all_reward_model_data, dtype=object),
            reward_fn_key: np.array(all_data_sources, dtype=object),
            "extra_info": np.array(all_extra_info, dtype=object),
        }
        
        # Copy any other non-tensor data that might be needed
        if "__num_turns__" in base_batch.non_tensor_batch:
            num_turns_data = []
            for i in range(len(records)):
                orig_idx = i // k_rollouts
                num_turns_data.append(base_batch.non_tensor_batch["__num_turns__"][orig_idx])
            non_tensor_batch["__num_turns__"] = np.array(num_turns_data)
        
        # Create DataProto
        para_batch = DataProto.from_dict(batch_dict)
        para_batch.non_tensor_batch = non_tensor_batch
        
        # Set meta_info with required fields for compute_log_prob
        meta_info = {"sample_records": records}
        
        # Add required fields for compute_log_prob from config if available
        if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
            rollout_cfg = self.config.actor_rollout_ref.rollout
            if hasattr(rollout_cfg, 'log_prob_micro_batch_size_per_gpu') and rollout_cfg.log_prob_micro_batch_size_per_gpu is not None:
                meta_info["micro_batch_size"] = int(rollout_cfg.log_prob_micro_batch_size_per_gpu)
            if hasattr(rollout_cfg, 'log_prob_max_token_len_per_gpu') and rollout_cfg.log_prob_max_token_len_per_gpu is not None:
                meta_info["max_token_len"] = int(rollout_cfg.log_prob_max_token_len_per_gpu)
            if hasattr(rollout_cfg, 'log_prob_use_dynamic_bsz'):
                meta_info["use_dynamic_bsz"] = rollout_cfg.log_prob_use_dynamic_bsz
            if hasattr(rollout_cfg, 'temperature'):
                meta_info["temperature"] = rollout_cfg.temperature
        
        para_batch.meta_info = meta_info
        
        return para_batch
    
    def _get_on_policy_mask(self, records: List[SampleRecord]) -> torch.Tensor:
        """Get mask for on-policy samples."""
        mask = torch.tensor([rec.source_id == "ON" for rec in records], dtype=torch.bool)
        return mask
    
    def _filter_batch_by_mask(self, batch: DataProto, mask: torch.Tensor) -> DataProto:
        """Filter batch by boolean mask."""
        # Create filtered tensors dictionary
        filtered_tensors = {}
        for key, value in batch.batch.items():
            if isinstance(value, torch.Tensor):
                filtered_tensors[key] = value[mask]
        
        # Create DataProto with filtered batch
        filtered_batch = DataProto()
        filtered_batch.batch = TensorDict(filtered_tensors, batch_size=filtered_tensors[list(filtered_tensors.keys())[0]].shape[0])
        
        # Filter non-tensor data
        if hasattr(batch, 'non_tensor_batch'):
            filtered_non_tensor = {}
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    filtered_non_tensor[key] = value[mask.cpu().numpy()]
            filtered_batch.non_tensor_batch = filtered_non_tensor
        
        # Preserve meta_info from original batch
        if hasattr(batch, 'meta_info'):
            filtered_batch.meta_info = batch.meta_info.copy() if hasattr(batch.meta_info, 'copy') else dict(batch.meta_info)
        
        return filtered_batch
    
    def _merge_log_probs(self, full_batch: DataProto, partial_batch: DataProto, mask: torch.Tensor) -> DataProto:
        """Merge computed log probs back into full batch."""
        if "old_log_probs" not in partial_batch.batch:
            return full_batch
        
        # Initialize if not exists
        if "old_log_probs" not in full_batch.batch:
            full_batch.batch["old_log_probs"] = torch.zeros_like(full_batch.batch["responses"], dtype=torch.float32)
        
        # Copy values where mask is True
        full_batch.batch["old_log_probs"][mask] = partial_batch.batch["old_log_probs"]
        
        return full_batch
    
    def _compute_paraphrase_advantages(self, batch: DataProto, records: List[SampleRecord]) -> DataProto:
        """Compute advantages with paraphrase-aware grouping."""
        # Group records by group_id
        groups = group_by_group_id(records)
        
        # Compute advantages per group
        all_advantages = torch.zeros(len(records), device=self.device)
        
        idx = 0
        for group_id, group_records in groups.items():
            group_size = len(group_records)
            group_rewards = torch.tensor([r.reward for r in group_records], device=self.device, dtype=torch.float32)
            
            # GRPO normalization within group
            advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-6)
            
            # Assign to correct positions
            all_advantages[idx:idx+group_size] = advantages
            idx += group_size
        
        # Broadcast advantages to token-level shape (batch_size, seq_len)
        # The response_mask will handle masking out non-response tokens
        response_mask = batch.batch["response_mask"]
        all_advantages = all_advantages.unsqueeze(-1) * response_mask
        
        batch.batch["advantages"] = all_advantages
        return batch
    
    def _update_actor_with_transport(self, batch: DataProto, records: List[SampleRecord]) -> DataProto:
        """Update actor with transport ratios."""
        # Deprecated: transport ratios path caused zero gradients in practice.
        # We now rely on standard PPO update with correctly formed masks/logprobs.
        return self.actor_rollout_wg.update_actor(batch)
    
    def _compute_and_apply_ce_loss(self) -> float:
        """Compute and apply CE distillation loss."""
        # This would need to be integrated with the actor update
        # For now, return placeholder
        return 0.0


# Import for backward compatibility
import random
from tqdm import tqdm
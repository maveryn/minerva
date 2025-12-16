# verl/trainer/para_distill_trainer.py
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

"""Trainer implementing RLVR + paraphrasing + self-distillation.

This module provides a thin wrapper around :class:`RayPPOTrainer` that
implements the three-phase procedure described in the project proposal.
The actual reinforcement learning update logic is delegated to the parent
class; this trainer focuses on the orchestration of paraphrasing and the
subsequent distillation pass.  The implementation here is intentionally
minimal and mainly serves as a reference implementation – many pieces such
as the reward computation or mixture sampling are simplified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward

from ..losses.distill_ce import DistillCELoss
from ..modules.paraphrase import build_paraphraser
from .ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    compute_advantage,
)


@dataclass
class ParaphraseResult:
    """Container storing paraphrase rollouts."""

    prompt: str
    answer_text: str
    reward: float
    entropy: float


class RayParaDistillTrainer(RayPPOTrainer):
    """RL trainer with paraphrasing and self-distillation phases.

    The class follows the algorithm sketch provided in the user instructions.
    It extends :class:`RayPPOTrainer` and therefore reuses the majority of the
    PPO/GRPO infrastructure.  Only the high level phase orchestration is
    implemented here – detailed implementation of mixture sampling, grouping
    and other optimisations is left as future work.
    """

    def __init__(self, config: DictConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # Hyper parameters for the additional phases
        self.lambda_distill = config.para_distill.lambda_distill
        self.gamma_mix = config.paraphrase.gamma_mix
        self.m = config.paraphrase.num_paraphrases_per_failure
        self.k = config.paraphrase.samples_per_prompt

        # Helper modules
        self.paraphraser = build_paraphraser(config.paraphrase)
        self.ce_loss = DistillCELoss()

        # Optional algorithm knobs
        self.clip_higher = config.algorithm.get("clip_higher", None)
        self.dynamic_sampling = config.algorithm.get("dynamic_sampling", None)

    # ------------------------------------------------------------------
    def _pick_highest_entropy(self, candidates: list[ParaphraseResult]) -> ParaphraseResult:  # pragma: no cover
        return max(candidates, key=lambda c: c.entropy)

    def _mk_dataproto(self, tensor_batch: dict, non_tensor_batch: dict) -> DataProto:
        """Build a DataProto like RLHFDataset + trainer do."""
        bsz = next(iter(tensor_batch.values())).shape[0]
        td = TensorDict(tensor_batch, batch_size=[bsz])
        return DataProto(batch=td, non_tensor_batch=non_tensor_batch)

    def _tokenize_prompt(self, text: str):
        """Chat-template a single-turn user message to prompt tokens."""
        messages = [{"role": "user", "content": text}]
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"], enc["attention_mask"]

    def _tokenize_response(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"]

    # ------------------------------------------------------------------
def fit(self) -> None:
    """Three-phase loop per training step."""
    import uuid
    from tqdm import tqdm

    # Match the base trainer preamble: set counters before using them.
    self.global_steps = 0
    self._load_checkpoint()

    # Fallback in case total_training_steps wasn't set (it normally is in _create_dataloader).
    if not hasattr(self, "total_training_steps"):
        self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

    progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="ParaDistill")

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # ---------- Phase 1: RL on original prompts ----------
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            if not self.async_rollout_mode:
                out = self.actor_rollout_wg.generate_sequences(gen_batch)
            else:
                out = self.async_rollout_manager.generate_sequences(gen_batch)

            batch = batch.union(out)

            # Compute log-probs (and entropies), reference log-probs, values, reward, advantage
            old_lp = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_lp.batch["entropys"]
            batch = batch.union(old_lp)

            if "response_mask" not in batch.batch.keys():
                # Base helpers will infer it if missing, but set it here for clarity.
                from verl.trainer.ppo.core_algos import compute_response_mask
                batch.batch["response_mask"] = compute_response_mask(batch)
            resp_mask = batch.batch["response_mask"]

            if self.use_reference_policy:
                ref_lp = (
                    self.ref_policy_wg.compute_ref_log_prob(batch)
                    if not self.ref_in_actor
                    else self.actor_rollout_wg.compute_ref_log_prob(batch)
                )
                batch = batch.union(ref_lp)

            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

            reward_tensor, reward_infos = compute_reward(self.reward_fn, batch)
            batch.batch["token_level_scores"] = reward_tensor

            if self.config.algorithm.use_kl_in_reward:
                batch, _ = self.apply_kl_penalty(
                    batch, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty
                )
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )

            if self.use_critic:
                self.critic_wg.update_critic(batch)
            self.actor_rollout_wg.update_actor(batch)

            # Identify failed originals (sum of rewards per uid == 0)
            tok_rewards = batch.batch["token_level_rewards"]
            sample_rewards = (tok_rewards * resp_mask).sum(-1).cpu().tolist()
            uids = batch.non_tensor_batch["uid"].tolist()
            prompts_txt = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            uid2sum = {}
            for uid, r in zip(uids, sample_rewards):
                uid2sum[uid] = uid2sum.get(uid, 0.0) + float(r)

            # Keep metadata we’ll need for reward on originals during paraphrase
            data_src = (
                batch.non_tensor_batch["data_source"].tolist()
                if "data_source" in batch.non_tensor_batch
                else [None] * len(uids)
            )
            gt_list = (
                [ri.get("ground_truth", None) for ri in reward_infos.get("reward_model", [])]
                if isinstance(reward_infos, dict) and "reward_model" in reward_infos
                else [None] * len(uids)
            )

            failed_infos = []
            for i, uid in enumerate(uids):
                if uid2sum[uid] == 0.0:
                    failed_infos.append(
                        {
                            "uid": uid,
                            "prompt_text": prompts_txt[i],
                            "data_source": data_src[i],
                            "ground_truth": gt_list[i] if i < len(gt_list) else None,
                        }
                    )

            # ---------- Phase 2: RL on paraphrases for failures ----------
            distill_pairs = []  # (original_x, best_y*)
            for f in failed_infos:
                x_primes = self.paraphraser.sample({"text": f["prompt_text"]}, m=self.m)
                best = self._paraphrase_step_and_update(f, x_primes, k=self.k)
                if best is not None:
                    distill_pairs.append((f["prompt_text"], best["answer_text"]))

            # ---------- Phase 3: CE-style distillation via GRPO ----------
            if distill_pairs:
                self._distill_via_grpo(distill_pairs, coef=self.lambda_distill)

            # Step accounting / progress
            self.global_steps += 1
            progress.update(1)
            if self.global_steps >= self.total_training_steps:
                progress.close()
                return


    # ------------------------------------------------------------------
    def _paraphrase_step_and_update(self, failed_info: dict, x_primes: list[str], k: int):
        """Run RL on paraphrases of a single original, return highest-entropy correct y* dict or None."""
        import uuid

        from verl.trainer.ppo.core_algos import agg_loss
        from verl.utils.model import compute_position_id_with_mask

        tensor_chunks, meta_list = [], []
        for xp in x_primes:
            p_ids, p_mask = self._tokenize_prompt(xp)
            pos_ids = compute_position_id_with_mask(p_mask)
            tensor_chunks.append(
                {
                    "input_ids": p_ids[0],
                    "attention_mask": p_mask[0],
                    "position_ids": pos_ids[0],
                }
            )
            meta_list.append(
                {
                    "uid": str(uuid.uuid4()),
                    "data_source": failed_info["data_source"],
                    "reward_model": {"ground_truth": failed_info["ground_truth"]},
                }
            )

        tb = {k: torch.stack([t[k] for t in tensor_chunks], dim=0) for k in tensor_chunks[0].keys()}
        ntb = {k: np.array([m[k] for m in meta_list], dtype=object) for k in meta_list[0].keys()}
        para_batch = self._mk_dataproto(tb, ntb)
        para_gen = self._get_gen_batch(para_batch).repeat(repeat_times=k, interleave=True)

        if self.gamma_mix > 0 and para_gen.batch.batch_size[0] > 1 and self.use_reference_policy:
            idx = torch.rand(para_gen.batch.batch_size[0]) < self.gamma_mix
            subA = para_gen.select(batch_keys=None, non_tensor_batch_keys=None)
            subB = para_gen.select(batch_keys=None, non_tensor_batch_keys=None)
            mask = idx.numpy()
            for key in para_gen.batch.keys():
                subA.batch[key] = para_gen.batch[key][~idx]
                subB.batch[key] = para_gen.batch[key][idx]
            for key in para_gen.non_tensor_batch.keys():
                subA.non_tensor_batch[key] = para_gen.non_tensor_batch[key][~mask]
                subB.non_tensor_batch[key] = para_gen.non_tensor_batch[key][mask]
            outA = self.actor_rollout_wg.generate_sequences(subA)
            outB = self.ref_policy_wg.generate_sequences(subB)
            out = DataProto.concat([outA, outB])
        else:
            out = self.actor_rollout_wg.generate_sequences(para_gen)

        batch = para_batch.union(out)
        old_lp = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = old_lp.batch["entropys"]
        resp_mask = batch.batch["response_mask"]
        batch = batch.union(old_lp)
        if self.use_reference_policy:
            ref_lp = (
                self.ref_policy_wg.compute_ref_log_prob(batch)
                if not self.ref_in_actor
                else self.actor_rollout_wg.compute_ref_log_prob(batch)
            )
            batch = batch.union(ref_lp)
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)
        reward_tensor, _ = compute_reward(self.reward_fn, batch)
        batch.batch["token_level_scores"] = reward_tensor
        if self.config.algorithm.use_kl_in_reward:
            batch, _ = self.apply_kl_penalty(batch, self.kl_ctrl_in_reward, self.config.algorithm.kl_penalty)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=k,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )
        if self.use_critic:
            self.critic_wg.update_critic(batch)
        self.actor_rollout_wg.update_actor(batch)

        sample_rewards = (batch.batch["token_level_rewards"] * resp_mask).sum(-1).cpu().tolist()
        ent_scalar = agg_loss(
            loss_mat=entropys,
            loss_mask=resp_mask,
            loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
        )
        ent_scalar = ent_scalar.detach().cpu().tolist()
        answers = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        best = None
        best_H = -1e9
        for r, H, ans in zip(sample_rewards, ent_scalar, answers, strict=False):
            if r > 0 and H > best_H:
                best = {"answer_text": ans, "entropy": H}
                best_H = H
        return best

    # ------------------------------------------------------------------
    def _distill_via_grpo(self, pairs: list[tuple[str, str]], coef: float):
        """Push (x,y*) pairs with constant positive reward to mimic CE."""
        from verl.utils.model import compute_position_id_with_mask

        prompts, prompt_masks, resps = [], [], []
        for x, y in pairs:
            p_ids, p_mask = self._tokenize_prompt(x)
            r_ids = self._tokenize_response(y)
            prompts.append(p_ids[0])
            prompt_masks.append(p_mask[0])
            resps.append(r_ids[0])

        prompts = torch.nn.utils.rnn.pad_sequence(prompts, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        prompt_masks = torch.nn.utils.rnn.pad_sequence(prompt_masks, batch_first=True, padding_value=0)
        resps = torch.nn.utils.rnn.pad_sequence(resps, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        ones_resp = torch.ones_like(resps).to(dtype=prompt_masks.dtype)
        attn = torch.cat([prompt_masks, ones_resp], dim=1)
        input_ids = torch.cat([prompts, resps], dim=1)
        pos_ids = compute_position_id_with_mask(attn)

        data = self._mk_dataproto(
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "position_ids": pos_ids,
            },
            {"uid": np.array([f"distill-{i}" for i in range(input_ids.shape[0])], dtype=object)},
        )

        old_lp = self.actor_rollout_wg.compute_log_prob(data)
        data = data.union(old_lp)
        data.batch["token_level_rewards"] = torch.ones_like(data.batch["old_log_probs"]) * float(coef)

        data = compute_advantage(
            data,
            adv_estimator=AdvantageEstimator.GRPO,
            config=self.config.algorithm,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        )
        self.actor_rollout_wg.update_actor(data)

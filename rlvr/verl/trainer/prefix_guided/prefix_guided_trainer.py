# verl/trainer/prefix_guided/prefix_guided_trainer.py
# Copyright 2024+
# Prefix-guided PPO built on top of RayPPOTrainer with veRL-compatible flow.

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer as _BasePPO,  # reuse PPO orchestration
    compute_advantage as _compute_advantage,
    compute_response_mask as _compute_response_mask,
    apply_kl_penalty as _apply_kl_penalty,
)
from verl.utils.tracking import Tracking  
from verl.trainer.ppo.reward import compute_reward as _compute_reward
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from .prefix_hint_samplers import FractionalStepBandit
from .prompting import (
    build_answer_guided_prompt,
    build_prefix_hint_messages,
    extract_boxed_answer,
    split_cot_to_steps,
)

ALL_SOLVE_THRESH = 0.95


class PrefixGuidedPPOTrainer(_BasePPO):
    """
    Prefix-guided variant of RayPPOTrainer.

    Key differences vs base PPO:
      * after the normal rollout + rewards, identify prompts with 0 successes (grouped by uid),
        generate answer-guided CoTs, cut a Thompson-sampled *fraction* of steps (curriculum),
        re-ask with the prefix only, and merge those guided samples back into the PPO update.
      * log prefix-guided metrics:
          - pg/zero_solve_frac, pg/zero_solve_count
          - pg/all_solve_frac, pg/all_solve_count
          - pg/zero_solve_post/reward_mean, pg/zero_solve_post/solve_rate
          - pg/ts_frac/mean|min|max, pg/ts_target_success
      * NEW: optional rescaling of off-policy guided scores to the on-policy scale
      * NEW: optional anchor supervision built from CoT remainder
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

        # ---- FORCE algorithmic knobs before BasePPO builds workers ----
        # Always GRPO
        if not hasattr(config, "algorithm"):
            raise ValueError("config.algorithm missing")
        config.algorithm.adv_estimator = "grpo"

        # Always no KL in reward
        config.algorithm.use_kl_in_reward = False

        # Disable PPO clipping globally (safe for guided rows and plain rows)
        # Make the clamp a no-op by setting +inf.
        if hasattr(config.algorithm, "clip_coef"):
            config.algorithm.clip_coef = float("inf")
        # Some stacks keep a shadow copy under actor config; set that too if present.
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


        pg_cfg = config.get("prefix_guided", {})
        self.pg_enable = bool(pg_cfg.get("enable", True))
        self.pg_ans_cot_samples = int(pg_cfg.get("ans_cot_samples", 8))
        self.pg_warmup = int(pg_cfg.get("warmup_prefix_steps", 50))
        self.pg_bins = int(pg_cfg.get("frac_bins", 7))
        self.pg_target = float(pg_cfg.get("target_success", 0.5))
        self.pg_minf = float(pg_cfg.get("min_prefix_frac", 0.1))
        self.pg_maxf = float(pg_cfg.get("max_prefix_frac", 0.9))
        self.pg_ans_template = str(pg_cfg.get("prompt", {}).get(
            "answer_guided_template",
            "Given a question and its final answer, write a clear, correct, step-by-step reasoning that leads to the answer.\n\n"
            "Question:\n{QUESTION}\n\nCorrect Answer:\n{ANSWER}\n\nReasoning:\n"
        ))
        self.pg_prefix_template = str(pg_cfg.get("prompt", {}).get(
            "prefix_hint_template",
            "Solve the following question. You are given partial intermediate steps to start from.\n\n"
            "Question:\n{QUESTION}\n\nPartial Steps:\n{PREFIX_STEPS}\n\nContinue the reasoning and give the final answer in \\boxed{...}."
        ))

        # Bandit over FRACTIONS of CoT steps
        self.pg_bandit = FractionalStepBandit(
            frac_bins=self.pg_bins,
            target_success=self.pg_target,
            min_frac=self.pg_minf,
            max_frac=self.pg_maxf,
            warmup_total_steps=self.pg_warmup,
        )

        # ---- rescaling config (off-policy -> on-policy) ----
        rescale_cfg = pg_cfg.get("rescale", {})
        self.pg_rescale_enable = bool(rescale_cfg.get("enable", True))
        self.pg_rescale_method = str(rescale_cfg.get("method", "affine_match"))  # 'affine_match' or 'none'
        self.pg_rescale_scope = str(rescale_cfg.get("scope", "global"))  # 'global' or 'per_uid'
        self.pg_rescale_clip_min = float(rescale_cfg.get("clip_min", math.exp(-2.0)))  # ≈ 0.1353
        self.pg_rescale_clip_max = float(rescale_cfg.get("clip_max", math.exp(2.0)))  # ≈ 7.389

        # ---- anchor supervision config ----
        anchor_cfg = pg_cfg.get("anchor", {})
        self.anchor_enable = bool(anchor_cfg.get("enable", False))
        self.anchor_weight = float(anchor_cfg.get("weight", 0.15))
        self.anchor_max_pairs_per_item = int(anchor_cfg.get("max_pairs_per_item", 1))

        self.pg_force_serial_gen = bool(pg_cfg.get("force_serial_gen", True))

        # ---- reward dithering ----
        rd_cfg = pg_cfg.get("reward_dither", {})
        self.rd_enable = bool(rd_cfg.get("enable", True))
        self.rd_first_n = int(rd_cfg.get("first_n_tokens", 32))
        self.rd_scale   = float(rd_cfg.get("scale", 0.03))
        self.rd_eps     = float(rd_cfg.get("eps", 1e-8))

        self._rng = random.Random(12345)

    # -------------------------- helpers --------------------------

    def _decode_texts(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _debug_log_guided_samples(
        self,
        records: Sequence[Dict[str, object]],
        guided_prompt_texts: Sequence[str],
        guided_resp_texts: Sequence[str],
    ) -> None:
        if not records:
            return

        for idx, rec in enumerate(records):
            ans_prompt = str(rec.get("answer_guided_prompt", "<NA>"))
            cot_text = str(rec.get("cot_text", "<NA>"))
            prefix_prompt = (
                str(guided_prompt_texts[idx])
                if idx < len(guided_prompt_texts)
                else "<NA>"
            )
            guided_resp = (
                str(guided_resp_texts[idx])
                if idx < len(guided_resp_texts)
                else "<NA>"
            )

            print("\n" + "=" * 120)
            print(f"[PG DEBUG] Sample {idx}")
            print("-" * 120)
            print("[1] Answer-guided PROMPT:\n")
            print(ans_prompt)
            print("\n[2] Answer-guided CoT (model output):\n")
            print(cot_text)
            print("\n[3] Hint-guided PROMPT (built from CoT prefix):\n")
            print(prefix_prompt)
            print("\n[4] Hint-guided RESPONSE (model output):\n")
            print(guided_resp)
            print("=" * 120 + "\n", flush=True)

    def _make_prompts_dataproto(self, items: List[Union[str, List[dict]]]) -> DataProto:
        """
        Build a DataProto for rollout.generate_sequences.

        Accepts either:
          - raw strings, or
          - chat 'messages' (list[{'role','content'}]) for tokenizer.apply_chat_template.

        Produces required tensors:
          batch['input_ids']      : (B, S)  torch.long
          batch['attention_mask'] : (B, S)  torch.long in {0,1}
          batch['position_ids']   : (B, S)  torch.long  (computed from attention_mask)

        Notes:
        - We set add_special_tokens=False since chat_template already injects specials.
        - For chat messages we pass add_generation_prompt=True so the assistant header is appended.
        """
        formatted_texts: List[str] = []
        for it in items:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], dict) and "role" in it[0] and "content" in it[
                0]:
                text = self.tokenizer.apply_chat_template(
                    it,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_texts.append(text)
            else:
                formatted_texts.append(str(it))

        enc = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"].long()
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min_(0).long()

        td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": input_ids.clone(),  # <-- ADD THIS: reward manager expects 'prompts'
            },
            batch_size=[input_ids.size(0)],
        )

        non_tensor = {"prompt_text": np.asarray(formatted_texts, dtype=object)}
        return DataProto(batch=td, non_tensor_batch=non_tensor)

    def _make_pr_dataproto(self, prompts: List[Union[str, List[dict]]], responses: List[str]) -> DataProto:
        """
        Build a DataProto for log-prob rescoring with BOTH concatenated inputs and split fields:
          batch['input_ids']      : (B, S_p+S_r)
          batch['attention_mask'] : (B, S_p+S_r)
          batch['position_ids']   : (B, S_p+S_r)
          batch['prompts']        : (B, S_p)
          batch['responses']      : (B, S_r)
          batch['response_mask']  : (B, S_r)  (1 where response tokens are real)
        """
        # 1) format prompts (support chat messages via apply_chat_template)
        formatted_prompts: List[str] = []
        for it in prompts:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], dict) and "role" in it[0] and "content" in it[
                0]:
                txt = self.tokenizer.apply_chat_template(
                    it,
                    tokenize=False,
                    add_generation_prompt=True,  # same header as rollout
                )
                formatted_prompts.append(txt)
            else:
                formatted_prompts.append(str(it))

        # 2) tokenize prompts and responses separately
        p_enc = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        r_enc = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=getattr(self.config.data, "max_response_length", 4096),
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompts_ids = p_enc["input_ids"]
        prompts_mask = p_enc["attention_mask"].long()
        responses_ids = r_enc["input_ids"]
        responses_mask = r_enc["attention_mask"].long()

        # 3) concatenate to build the full sequence fields expected by actor.compute_log_prob
        input_ids = torch.cat([prompts_ids, responses_ids], dim=1)
        attention_mask = torch.cat([prompts_mask, responses_mask], dim=1).long()
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min_(0).long()

        # 4) pack into TensorDict (keep split fields for RM and masking)
        td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "prompts": prompts_ids,  # RM uses this
                "responses": responses_ids,
                "response_mask": responses_mask,  # for selecting response token logprobs
            },
            batch_size=[input_ids.size(0)],
        )
        return DataProto(batch=td, non_tensor_batch={})

    def _right_pad_2d(self, x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if x is None or not torch.is_tensor(x) or x.dim() != 2:
            return x
        B, T = x.shape
        if T >= target_len:
            return x
        pad = x.new_full((B, target_len - T), pad_value)
        return torch.cat([x, pad], dim=1)

    def _pad_for_concat(self, dp_a: DataProto, dp_b: DataProto) -> Tuple[DataProto, DataProto]:
        """
        Right-pad ALL 2D [B, T] tensors so dp_a and dp_b can be concatenated along batch dim.

        We detect three time axes per DataProto:
          - full sequence length  : T_full  (from 'input_ids')
          - prompt length         : T_prompt (from 'prompts' if present)
          - response length       : T_resp  (from 'responses' if present)

        Any 2D tensor whose second dim equals one of these lengths is padded to the
        corresponding max across (dp_a, dp_b). Unknown 2D keys default to full-seq padding.
        """

        def lengths(dp):
            T_full = dp.batch["input_ids"].size(1) if "input_ids" in dp.batch.keys() else None
            T_prompt = dp.batch["prompts"].size(1) if "prompts" in dp.batch.keys() else None
            T_resp = dp.batch["responses"].size(1) if "responses" in dp.batch.keys() else None
            return T_full, T_prompt, T_resp

        Ta_full, Ta_prompt, Ta_resp = lengths(dp_a)
        Tb_full, Tb_prompt, Tb_resp = lengths(dp_b)

        T_full_max = max([t for t in (Ta_full, Tb_full) if t is not None], default=None)
        T_prompt_max = max([t for t in (Ta_prompt, Tb_prompt) if t is not None], default=None)
        T_resp_max = max([t for t in (Ta_resp, Tb_resp) if t is not None], default=None)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def _pad_tensor(t: torch.Tensor, target_len: int, pad_value: Union[int, float]) -> torch.Tensor:
            if t is None or not torch.is_tensor(t) or t.dim() != 2:
                return t
            B, T = t.shape
            if target_len is None or T >= target_len:
                return t
            if t.dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
                pad_fill = float(pad_value)
            else:
                pad_fill = int(pad_value)
            pad_block = t.new_full((B, target_len - T), pad_fill)
            return torch.cat([t, pad_block], dim=1)

        def _pad_dp(dp: DataProto, T_full_dp, T_prompt_dp, T_resp_dp) -> DataProto:
            # First pass: decide per-key target and pad values
            for key in list(dp.batch.keys()):
                t = dp.batch[key]
                if not (torch.is_tensor(t) and t.dim() == 2):
                    continue

                B, T = t.shape
                target_len = None
                pad_val = 0

                # Explicit categories (names win)
                if key in ("input_ids", "attention_mask", "position_ids"):
                    target_len = T_full_max
                    pad_val = pad_id if key == "input_ids" else 0
                elif key == "prompts":
                    target_len = T_prompt_max
                    pad_val = pad_id
                elif key in ("responses", "response_mask"):
                    target_len = T_resp_max
                    pad_val = pad_id if key == "responses" else 0
                elif key in ("old_log_probs", "ref_log_prob", "values",
                             "token_level_scores", "token_level_rewards",
                             "advantages", "returns"):
                    # These are response-time tensors in veRL
                    target_len = T_resp_max
                    pad_val = 0.0
                else:
                    # Heuristic: match by current length
                    if T_full_dp is not None and T == T_full_dp:
                        target_len = T_full_max
                        pad_val = 0
                    elif T_resp_dp is not None and T == T_resp_dp:
                        target_len = T_resp_max
                        pad_val = 0.0
                    elif T_prompt_dp is not None and T == T_prompt_dp:
                        target_len = T_prompt_max
                        pad_val = pad_id
                    else:
                        # Fall back to full-seq if we have it
                        target_len = T_full_max
                        pad_val = 0

                dp.batch[key] = _pad_tensor(t, target_len, pad_val)

            # Recompute position_ids from (possibly) padded attention_mask
            if "attention_mask" in dp.batch.keys():
                attn = dp.batch["attention_mask"]
                pos = (attn.cumsum(dim=1) - 1).clamp_min_(0).long()
                dp.batch["position_ids"] = pos
            return dp

        dp_a = _pad_dp(dp_a, Ta_full, Ta_prompt, Ta_resp)
        dp_b = _pad_dp(dp_b, Tb_full, Tb_prompt, Tb_resp)
        return dp_a, dp_b


    def _ensure_full_sequence_fields(self, dp: DataProto) -> None:
        """
        Ensure dp.batch has input_ids, attention_mask, position_ids.
    
        We try (in order):
          A) prompts + responses
          B) input_ids (prompt-only) + responses
    
        In both cases we compute attention_mask and position_ids from masks.
        """
        need_input = "input_ids" not in dp.batch.keys()
        need_attn  = "attention_mask" not in dp.batch.keys()
        need_pos   = "position_ids" not in dp.batch.keys()
        if not (need_input or need_attn or need_pos):
            return
    
        has_prompts   = "prompts"   in dp.batch.keys()
        has_responses = "responses" in dp.batch.keys()
        if not has_responses:
            # Nothing to do if there is no response yet
            return
    
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        responses_ids  = dp.batch["responses"]
        response_mask  = dp.batch.get("response_mask", (responses_ids != pad_id).long()).clone().long()
    
        if has_prompts:
            # Case A: prompts + responses
            prompts_ids   = dp.batch["prompts"]
            prompts_mask  = (prompts_ids != pad_id).long()
            input_ids     = torch.cat([prompts_ids, responses_ids], dim=1)
            attention_mask = torch.cat([prompts_mask, response_mask], dim=1).long()
        elif "input_ids" in dp.batch.keys():
            # Case B: only prompt in input_ids + responses from rollout
            prompts_ids    = dp.batch["input_ids"]              # treat as prompt-only
            prompts_mask   = (prompts_ids != pad_id).long()
            input_ids      = torch.cat([prompts_ids, responses_ids], dim=1)
            attention_mask = torch.cat([prompts_mask, response_mask], dim=1).long()
            # expose prompts for downstream (reward mgr / masking)
            dp.batch["prompts"] = prompts_ids
            dp.batch["response_mask"] = response_mask
        else:
            raise AssertionError("Cannot reconstruct full sequence fields: need either 'prompts' or 'input_ids' plus 'responses'.")
    
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min_(0).long()
    
        dp.batch["input_ids"]      = input_ids
        dp.batch["attention_mask"] = attention_mask
        dp.batch["position_ids"]   = position_ids




    def _align_non_tensor_for_concat(self, dps: Sequence[DataProto]) -> None:
        """
        Make non_tensor_batch schemas identical across DataProtos and ensure
        each value is a numpy.ndarray of length B (broadcast scalars if needed).
        In-place; returns None.
        """
        # union of keys
        all_keys = set()
        for dp in dps:
            all_keys |= set(dp.non_tensor_batch.keys())

        for dp in dps:
            B = dp.batch.batch_size[0]
            for k in all_keys:
                v = dp.non_tensor_batch.get(k, None)
                if v is None:
                    dp.non_tensor_batch[k] = np.empty(B, dtype=object)
                else:
                    arr = np.asarray(v, dtype=object)
                    if arr.ndim == 0:
                        arr = np.full((B,), arr, dtype=object)
                    elif arr.shape[0] != B:
                        # resize/broadcast to B (fill extras with None)
                        tmp = np.empty(B, dtype=object)
                        n = min(B, arr.shape[0])
                        tmp[:n] = arr[:n]
                        if n < B:
                            tmp[n:] = None
                        arr = tmp
                    dp.non_tensor_batch[k] = arr

    def _align_batch_keys_for_concat(self, dps: List[DataProto]) -> None:
        """
        Ensure all DataProtos in `dps` share the same batch keys by inserting
        zero-filled tensors with the correct shape/dtype for any missing keys.
        Operates in-place.
        """
        # union of batch keys
        all_keys = set()
        for dp in dps:
            all_keys |= set(dp.batch.keys())

        # length hints and dtypes
        def lens(dp: DataProto):
            B = dp.batch.batch_size[0]
            T_full = dp.batch["input_ids"].size(1) if "input_ids" in dp.batch.keys() else 0
            T_prompt = dp.batch["prompts"].size(1) if "prompts" in dp.batch.keys() else 0
            T_resp = dp.batch["responses"].size(1) if "responses" in dp.batch.keys() else 0
            dev = next(iter(dp.batch.values())).device
            return B, T_full, T_prompt, T_resp, dev

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for dp in dps:
            B, T_full, T_prompt, T_resp, dev = lens(dp)
            for k in all_keys:
                if k in dp.batch.keys():
                    continue

                # default shapes/dtypes per key
                if k == "input_ids":
                    dp.batch[k] = torch.full((B, T_full), pad_id, dtype=torch.long, device=dev)
                elif k == "attention_mask":
                    dp.batch[k] = torch.zeros((B, T_full), dtype=torch.long, device=dev)
                elif k == "position_ids":
                    pos = torch.arange(T_full, device=dev).unsqueeze(0).expand(B, -1).long()
                    dp.batch[k] = pos
                elif k == "prompts":
                    dp.batch[k] = torch.full((B, T_prompt), pad_id, dtype=torch.long, device=dev)
                elif k == "responses":
                    dp.batch[k] = torch.full((B, T_resp), pad_id, dtype=torch.long, device=dev)
                elif k == "response_mask":
                    dp.batch[k] = torch.zeros((B, T_resp), dtype=torch.long, device=dev)
                elif k in ("old_log_probs", "ref_log_prob", "values",
                           "token_level_scores", "token_level_rewards",
                           "advantages", "returns"):
                    dp.batch[k] = torch.zeros((B, T_resp), dtype=torch.float32, device=dev)
                else:
                    # unknown 2D key -> fall back to full-seq zeros (float)
                    dp.batch[k] = torch.zeros((B, T_full), dtype=torch.float32, device=dev)

    def _attach_uids(self, batch: DataProto) -> None:
        # one uid per *row* (NOT per key!)
        B = batch.batch.batch_size[0]
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(B)], dtype=object
        )


    def _ensure_response_mask(self, data: DataProto) -> None:
        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = _compute_response_mask(data)

    def _compute_rewards_and_adv(self, data: DataProto) -> None:
        # reward via the standard manager path (supports return_dict or tensor)
        reward_tensor, reward_extra = _compute_reward(data, self.reward_fn)
        data.batch["token_level_scores"] = reward_tensor

        # KL-in-reward (optional) mirrors the base trainer
        if self.config.algorithm.use_kl_in_reward:
            data, _ = _apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # advantages (GAE/GRPO/…)
        data = _compute_advantage(
            data,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

    # ---- NEW: utilities for rescaling guided scores ----------------

    @staticmethod
    def _seq_scores_from_tokens(token_level_scores: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        """Reduce token-level scores to per-sequence scores using the response mask."""
        if token_level_scores.dim() == 1:
            return token_level_scores
        # mask if provided; fall back to summing last T if shapes match
        T = token_level_scores.size(1)
        if response_mask is not None and response_mask.shape[-1] == T:
            return (token_level_scores * response_mask).sum(-1)
        return token_level_scores.sum(-1)

    @staticmethod
    def _seq_lengths_from_mask(response_mask: torch.Tensor) -> torch.Tensor:
        # sum as float and ensure min=1 to avoid divide-by-zero later
        return response_mask.to(torch.float32).sum(-1).clamp_min(1.0)


    def _affine_rescale_scores_inplace(self, guided: DataProto, ref: DataProto, scope: str = "global") -> Dict[
        str, float]:
        """
        Affine-match guided sequence scores to the reference (on-policy) distribution.
        Modifies guided.batch['token_level_scores'] in-place.
        """
        g_scores = self._seq_scores_from_tokens(guided.batch["token_level_scores"],
                                                guided.batch.get("response_mask", None))
        r_scores = self._seq_scores_from_tokens(ref.batch["token_level_scores"], ref.batch.get("response_mask", None))
        eps = 1e-8

        metrics = {}

        if scope == "per_uid" and "uid" in guided.non_tensor_batch and "uid" in ref.non_tensor_batch:
            # map each uid independently; fallback to global when missing
            g_uid = guided.non_tensor_batch["uid"]
            r_uid = ref.non_tensor_batch["uid"]
            unique = list({u for u in g_uid.tolist()})
            # precompute global
            mu_on, sigma_on = r_scores.mean(), r_scores.std()
            mu_off, sigma_off = g_scores.mean(), g_scores.std()
            a_global = sigma_on / torch.clamp(sigma_off, min=eps)
            b_global = mu_on - a_global * mu_off

            # build per-example a,b
            a = torch.full_like(g_scores, a_global)
            b = torch.full_like(g_scores, b_global)
            for u in unique:
                r_mask = torch.tensor(r_uid == u, device=r_scores.device)
                g_mask = torch.tensor(g_uid == u, device=g_scores.device)
                if not r_mask.any() or not g_mask.any():
                    continue
                r_sub = r_scores[r_mask]
                g_sub = g_scores[g_mask]
                mu_on_u, sigma_on_u = r_sub.mean(), r_sub.std()
                mu_off_u, sigma_off_u = g_sub.mean(), g_sub.std()
                a_u = sigma_on_u / torch.clamp(sigma_off_u, min=eps)
                b_u = mu_on_u - a_u * mu_off_u
                a[g_mask] = a_u
                b[g_mask] = b_u

            # apply to tokens
            L = self._seq_lengths_from_mask(guided.batch["response_mask"])
            guided.batch["token_level_scores"] = guided.batch["token_level_scores"] * a.unsqueeze(-1) + (
                    b / L).unsqueeze(-1)
            metrics.update({
                "pg/rescale_method": 0.0,
                "pg/rescale_mean_shift": float(b_global.detach().item()),
                "pg/rescale_scale": float(a_global.detach().item()),
            })
        else:
            # global affine match
            mu_on, sigma_on = r_scores.mean(), r_scores.std()
            mu_off, sigma_off = g_scores.mean(), g_scores.std()
            a = sigma_on / torch.clamp(sigma_off, min=eps)
            a = torch.clamp(a, min=self.pg_rescale_clip_min, max=self.pg_rescale_clip_max)
            b = mu_on - a * mu_off
            L = self._seq_lengths_from_mask(guided.batch["response_mask"])
            guided.batch["token_level_scores"] = guided.batch["token_level_scores"] * a + (b / L).unsqueeze(-1)
            metrics.update({
                "pg/rescale_method": 1.0,
                "pg/rescale_mean_shift": float(b.detach().item()),
                "pg/rescale_scale": float(a.detach().item()),
            })
        return metrics

    # -------------------------- main loop --------------------------

    def fit(self):
        """Training loop with a prefix-guided second rollout for 0-solve items."""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # (Optional) validation before training — reuse base validator and log
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            print(f"Initial validation metrics: {val_metrics}")
            if self.config.trainer.get("val_only", False):
                return

        total_steps = self.total_training_steps
        n_gpus = self.resource_pool_manager.get_n_gpus()

        progress_bar = tqdm(total=total_steps, initial=self.global_steps, desc="Training Progress")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: Dict[str, float] = {}
                timing_raw: Dict[str, float] = {}

                with marked_timer("step", timing_raw):
                    # ==== 1) prepare original batch like base PPO ====
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    self._attach_uids(batch)

                    # Form generation batch like base implementation
                    gen_batch = self._get_gen_batch(batch)
                    
                    # [ADD] carry uid into the rollout request so we can verify alignment after generation
                    if "uid" in batch.non_tensor_batch:
                        gen_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"].copy()
                    
                    # repeat request n times and roll out
                    n = self.config.actor_rollout_ref.rollout.n
                    
                    gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
                    gen_out   = self._generate_sequences_stable(gen_batch)
                    
                    # expand the original batch to match responses, then union
                    batch = batch.repeat(repeat_times=n, interleave=True)
                    batch = batch.union(gen_out)

                    if "prompts" not in batch.batch:
                        if "input_ids" in gen_batch.batch:
                            batch.batch["prompts"] = gen_batch.batch["input_ids"].clone()
                        elif "input_ids" in batch_dict:
                            # fall back to original dataloader field, if present
                            batch.batch["prompts"] = batch_dict["input_ids"].clone()

                    self._ensure_response_mask(batch)
                    self._ensure_full_sequence_fields(batch)
                    
                    # required by downstream components
                    self._ensure_response_mask(batch)
                    
                    # Accept possible reordering later (balance_batch), veRL does this.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # ==== 3) old log prob & entropy ====
                    # Minimal schema required by compute_log_prob
                    required = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask")
                    missing = [k for k in required if k not in batch.batch.keys()]
                    assert not missing, f"Missing fields before compute_log_prob: {missing}"

                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    ent = old_log_prob.batch.pop("entropys", None)
                    batch = batch.union(old_log_prob)
                    if ent is not None:
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy = agg_loss(loss_mat=ent, loss_mask=batch.batch["response_mask"],
                                           loss_agg_mode=loss_agg_mode)
                        metrics["actor/entropy"] = float(entropy.detach().item())

                    # ==== 4) ref log prob (optional) ====
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_out = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_out = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_out)

                    # ==== 5) critic values (optional) ====
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                    # ==== 6) reward & advantage (original rollout) ====
                    with marked_timer("reward+adv/orig", timing_raw):

                        if "data_source" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["data_source"] = np.asarray(
                                ["plain"] * len(batch), dtype=object
                            )
                          
                        if self.use_rm:
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        self._compute_rewards_and_adv(batch)

                    # mark token counts (required by workers)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ==== 7) prefix-guided branch for 0-solve items ====
                    guided_batch = None
                    anchor_batch = None
                    pg_metrics = {}
                    if self.pg_enable:
                        guided_batch, anchor_batch, pg_metrics = self._guided_branch(batch, n)

                    # ==== 8) merge & updates ====
                    if guided_batch is not None:
                        guided_uids = set(map(str, guided_batch.non_tensor_batch["uid"].tolist()))
                        # Keep only rows whose uid is NOT covered by guided
                        keep_mask = np.asarray([u not in guided_uids for u in batch.non_tensor_batch["uid"]], dtype=bool)
                        keep_idx = torch.as_tensor(np.nonzero(keep_mask)[0], dtype=torch.long)
                    
                        # Subselect plain batch
                        sub_plain = batch.select_idxs(keep_idx.tolist())
                    
                        # Align schemas and concat (critic stream first)
                        self._align_batch_keys_for_concat([sub_plain, guided_batch])
                        self._align_non_tensor_for_concat([sub_plain, guided_batch])
                        a, b = self._pad_for_concat(sub_plain, guided_batch)
                        train_batch_for_critic = DataProto.concat([a, b])
                    else:
                        train_batch_for_critic = batch

                    # actor concat (with optional anchor)
                    train_batch_for_actor = train_batch_for_critic
                    if anchor_batch is not None:
                        self._align_batch_keys_for_concat([train_batch_for_actor, anchor_batch])
                        self._align_non_tensor_for_concat([train_batch_for_actor, anchor_batch])
                        a, b = self._pad_for_concat(train_batch_for_actor, anchor_batch)
                        train_batch_for_actor = DataProto.concat([a, b])
                    # critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            crit_div = max(1, getattr(self.critic_wg, "world_size", 1))
                            tb_pad, _ = pad_dataproto_to_divisor(train_batch_for_critic, crit_div)
                            critic_output = self.critic_wg.update_critic(tb_pad)  # no unpad needed
                        metrics.update(reduce_metrics(critic_output.meta_info.get("metrics", {})))

                    # actor
                    if (not self.use_critic) or (self.config.trainer.critic_warmup <= self.global_steps):
                        with marked_timer("update_actor", timing_raw):
                            act_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
                            ta_pad, _ = pad_dataproto_to_divisor(train_batch_for_actor, act_div)
                            actor_output = self.actor_rollout_wg.update_actor(ta_pad)  # no unpad needed
                        metrics.update(reduce_metrics(actor_output.meta_info.get("metrics", {})))

                # ====== timing / throughput / data metrics (veRL style) ======
                metrics.update(pg_metrics)
                # use the largest batch we sent to workers for metrics
                metrics.update(compute_data_metrics(batch=train_batch_for_actor, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=train_batch_for_actor, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(batch=train_batch_for_actor, timing_raw=timing_raw, n_gpus=n_gpus))

                # ====== validation / save ======
                is_last_step = (self.global_steps + 1) >= total_steps
                if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or (self.global_steps + 1) % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # ====== log & step ======
                logger.log(data=metrics, step=self.global_steps + 1)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    return

                    

    # -------------------------- guided branch --------------------------
    def _generate_sequences_stable(self, dp: DataProto) -> DataProto:
        """
        veRL-compatible rollout:
          - pad to DP world size
          - generate
          - unpad
          - ensure response_mask
          - DROP any fields that would collide with inputs on union()
        """
        assert hasattr(dp, "batch") and hasattr(dp, "non_tensor_batch")
    
        size_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, size_div)
    
        out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        out = unpad_dataproto(out_pad, pad_size=pad_sz)
    
        # Ensure response_mask exists
        if "responses" in out.batch and "response_mask" not in out.batch:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            out.batch["response_mask"] = (out.batch["responses"] != pad_id).long()
    
        # --- CRITICAL: never let rollout-side sequence fields collide on union() ---
        # Keep only response-side fields (responses / response_mask / rollout_log_probs if present).
        for k in ("input_ids", "attention_mask", "position_ids", "prompts"):
            if k in out.batch:
                del out.batch[k]
    
        # Some backends may echo non-tensor prompt_text; keep original dp’s copy.
        # (Non-tensor union is more permissive, but we avoid surprises.)
        if "prompt_text" in out.non_tensor_batch:
            del out.non_tensor_batch["prompt_text"]
    
        return out



    def _rollout_and_union(self, dp: DataProto) -> DataProto:
        """
        Rollout helper that:
          - Pads to world size
          - Calls generate_sequences
          - Unpads
          - Ensures response_mask exists
          - Merges ONLY non-tensor metadata from `dp` into the rollout result
            (no batch-tensor union on input_ids/attention_mask/position_ids to avoid equality asserts)
          - Reconstructs full-sequence fields if the worker didn't return them
          - Fills meta_info['global_token_num'] for metrics
    
        NOTE: This mirrors the veRL/SvS pattern: keep the rollout output as the
        source of truth for sequence-side tensors; copy over non-tensor metadata only.
        """
        # 1) pad → generate → unpad
        size_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, size_div)
        out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        out = unpad_dataproto(out_pad, pad_size=pad_sz)
    
        # 2) ensure response_mask
        if "responses" in out.batch and "response_mask" not in out.batch:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            out.batch["response_mask"] = (out.batch["responses"] != pad_id).long()
    
        # 3) if worker didn't return 'prompts', fall back to the input prompts
        if "prompts" not in out.batch.keys() and "prompts" in dp.batch.keys():
            out.batch["prompts"] = dp.batch["prompts"]
    
        # 4) reconstruct full-sequence fields if needed
        self._ensure_full_sequence_fields(out)
    
        # 5) merge NON-TENSOR metadata from the input DP (length-align if necessary)
        #    (Do not touch batch tensors here to avoid equality assertions.)
        B = out.batch.batch_size[0]
        # Make sure values are 1-D arrays of len B
        for k, v in dp.non_tensor_batch.items():
            arr = np.asarray(v, dtype=object)
            if arr.ndim == 0:
                arr = np.full((B,), arr, dtype=object)
            elif arr.shape[0] != B:
                # best-effort align: truncate or pad with None
                tmp = np.empty((B,), dtype=object)
                n = min(B, arr.shape[0])
                tmp[:n] = arr[:n]
                if n < B:
                    tmp[n:] = None
                arr = tmp
            # prefer rollout's values if already present; otherwise take from dp
            if k not in out.non_tensor_batch:
                out.non_tensor_batch[k] = arr
    
        # 6) bookkeeping for throughput metrics
        out.meta_info["global_token_num"] = torch.sum(out.batch["attention_mask"], dim=-1).tolist()
        return out

    

    def _apply_reward_dithering(self, dp: DataProto, group_size: int) -> None:
        """
        A1 reward dithering:
          - Add N(0, sigma^2) noise to token-level scores on the first N response tokens.
          - sigma (per-row) = 0.03 * std_per_group, where std_per_group is computed over
            sequence scores (sum over response tokens) across the GRPO group for that prompt.

        NOTE: in our trainer, groups are contiguous blocks of size `group_size`.
        """
        if not getattr(self, "rd_enable", True) or self.rd_scale <= 0.0:
            return
        if "token_level_scores" not in dp.batch or "response_mask" not in dp.batch:
            return

        scores = dp.batch["token_level_scores"]          # [Bn, T] or [Bn]
        mask   = dp.batch["response_mask"]               # [Bn, T]  (0/1)
        if not (torch.is_tensor(scores) and scores.dim() == 2):
            # dithering is defined on token-level rewards; if we only have seq-level, skip
            return

        Bn, T = scores.shape
        if group_size <= 0 or (Bn % group_size) != 0:
            # cannot form groups reliably; skip
            return
        B = Bn // group_size

        with torch.no_grad():
            # sequence scores per row
            seq_scores = (scores * mask).sum(dim=1)      # [Bn]
            # group std over the group dimension
            seq_scores_g = seq_scores.view(B, group_size)                      # [B, n]
            std_per_group = seq_scores_g.float().std(dim=1, unbiased=False)    # [B]
            # per-row sigma = 0.03 * std_per_group
            sigma = (self.rd_scale * std_per_group).repeat_interleave(group_size)  # [Bn]
            sigma = sigma.clamp_min(self.rd_eps).unsqueeze(1)                      # [Bn, 1]

            # prefix mask: first rd_first_n valid response tokens
            N = min(int(self.rd_first_n), T)
            ar = torch.arange(T, device=scores.device).unsqueeze(0)            # [1, T]
            prefix_mask = (ar < N).to(mask.dtype) * mask                        # [Bn, T]

            noise = torch.randn_like(scores) * sigma
            dp.batch["token_level_scores"] = scores + noise * prefix_mask



    # --------------------------------------------------------------------------------

    def _guided_branch(
        self,
        normal_batch: DataProto,
        group_size: int,
    ) -> Tuple[Optional[DataProto], Optional[DataProto], Dict[str, float]]:
        """
        Build prefix-hint guided samples ONLY for the zero-solve questions (by uid),
        roll them out, compute log-probs/ref/values (if enabled), then compute
        reward -> (optional) affine rescale -> token_level_rewards -> advantages
        **on the guided batch itself**. No row reordering across DPs. Rewards and
        advs are fully self-contained on the guided DP.
    
        Returns:
          guided_batch: DataProto ready to be concatenated with the plain batch
                        for PPO updates (actor/critic).
          anchor_batch: (optional) lightweight supervision from CoT remainders.
          pg_metrics  : logging metrics for the guided branch.
        """
        # --- 0) Setup & convenience ---
        thr = float(self.config.prefix_guided.get("success_threshold", 0.5))
        pg_metrics: Dict[str, float] = {"pg/group_size": float(group_size)}
    
        # -- read per-row success on the *normal* batch (already computed upstream)
        tls = normal_batch.batch["token_level_scores"]
        resp_mask = normal_batch.batch.get("response_mask", None)
        if tls.dim() == 2:
            seq_scores_all = (tls * (resp_mask.to(tls.dtype) if resp_mask is not None else 1)).sum(-1)
        else:
            seq_scores_all = tls
    
        # canonical per-uid order (first occurrence)
        uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        uid_first_idx: Dict[str, int] = {}
        uid_order: List[str] = []
        uid_any_success: Dict[str, bool] = {}
        uid_all_success: Dict[str, bool] = {}
        for i, u in enumerate(uids_all.tolist()):
            if u not in uid_first_idx:
                uid_first_idx[u] = i
                uid_order.append(u)
            if u not in uid_all_success:
                uid_all_success[u] = True
            uid_any_success[u] = uid_any_success.get(u, False) or (float(seq_scores_all[i].item()) > thr)
            if float(seq_scores_all[i].item()) < ALL_SOLVE_THRESH:
                uid_all_success[u] = False
    
        # derive zero‑solve set
        zero_uids: List[str] = [u for u in uid_order if not uid_any_success[u]]
        B = len(uid_order)
        pg_metrics["pg/zero_solve_count"] = float(len(zero_uids))
        pg_metrics["pg/zero_solve_frac"] = float(len(zero_uids) / max(1, B))
        all_uids: List[str] = [u for u in uid_order if uid_all_success.get(u, False)]
        pg_metrics["pg/all_solve_count"] = float(len(all_uids))
        pg_metrics["pg/all_solve_frac"] = float(len(all_uids) / max(1, B))
    
        if not zero_uids:
            return None, None, pg_metrics
    
        # --- 1) Gather canonical plain question text + ground truth per uid ---
        prompts_text_all = self._decode_texts(normal_batch.batch["prompts"])  # [B*n]
        gt_all = self._extract_gt_answers(normal_batch)                       # [B*n]
    
        plain_q_text_by_uid: Dict[str, str] = {
            uid: prompts_text_all[uid_first_idx[uid]] for uid in uid_order
        }
        gt_by_uid: Dict[str, Optional[str]] = {
            uid: gt_all[uid_first_idx[uid]] for uid in uid_order
        }
    
        # --- 2) Build answer-guided prompts for zero-solve uids (only when GT exists) ---
        ans_guided_prompts: List[str] = []
        owner_uid_for_cot: List[str] = []
        for uid in zero_uids:
            gt = gt_by_uid.get(uid)
            if gt is None:
                continue
            guided_prompt = build_answer_guided_prompt(plain_q_text_by_uid[uid], gt, self.pg_ans_template)
            # we can sample multiple CoTs per uid to increase prefix variety
            ans_guided_prompts.extend([guided_prompt] * self.pg_ans_cot_samples)
            owner_uid_for_cot.extend([uid] * self.pg_ans_cot_samples)
    
        if not ans_guided_prompts:
            return None, None, pg_metrics
    
        # --- 3) Roll out CoTs (answer-guided) and UNION back (order preserved by union) ---
        cot_in = self._make_prompts_dataproto(ans_guided_prompts)
        cot_in.non_tensor_batch["uid"] = np.asarray(owner_uid_for_cot, dtype=object)
        cot_out = self._rollout_and_union(cot_in)  # responses live here; original order preserved
    
        # --- 4) From CoTs, construct prefix-hint prompts (TS on fraction of steps) ---
        cot_texts = self._decode_texts(cot_out.batch["responses"])
        cot_prompt_texts = cot_out.non_tensor_batch["prompt_text"].tolist()  # for debug/logging
    
        guided_records: List[Dict[str, object]] = []
        chosen_fracs: List[float] = []
        chosen_bins: List[int] = []
        per_uid_anchor_count: Dict[str, int] = defaultdict(int)
    
        for i, cot in enumerate(cot_texts):
            owner_uid = owner_uid_for_cot[i]
            steps = split_cot_to_steps(cot)
            final_answer = extract_boxed_answer(cot)
    
            # must have at least 2 steps and a boxed answer
            if len(steps) < 2 or final_answer is None:
                continue
    
            # filter out off-topic CoTs whose boxed answer disagrees with GT
            gt = gt_by_uid.get(owner_uid)
            if gt is not None and str(final_answer).strip() != str(gt).strip():
                continue
    
            k_steps, frac, bin_idx = self.pg_bandit.select_prefix_steps(
                total_steps=len(steps),
                rng=self._rng,
                trainer_step=self.global_steps,
            )
            k_steps = min(k_steps, max(1, len(steps) - 1))  # never reveal the final step
            prefix_steps = steps[:k_steps]
            remainder = "\n\n".join(steps[k_steps:])  # for optional anchor
            prefix_prompt = build_prefix_hint_messages(plain_q_text_by_uid[owner_uid], prefix_steps, self.pg_prefix_template)
    
            guided_records.append({
                "owner_uid": owner_uid,
                "answer_guided_prompt": cot_prompt_texts[i],
                "cot_text": cot,
                "cot_boxed_answer": final_answer,
                "prefix_prompt": prefix_prompt,
                "plain_question": plain_q_text_by_uid[owner_uid],
                "anchor_remainder": remainder,
                "frac": float(k_steps / float(len(steps))),
                "bin_idx": int(bin_idx),
            })
            chosen_fracs.append(float(k_steps / float(len(steps))))
            chosen_bins.append(int(bin_idx))
    
            if self.anchor_enable and self.anchor_weight > 0.0 and remainder.strip():
                if per_uid_anchor_count[owner_uid] < self.anchor_max_pairs_per_item:
                    per_uid_anchor_count[owner_uid] += 1
                    guided_records[-1]["use_for_anchor"] = True
    
        # bucket candidates by uid and sample exactly `group_size` per uid (with replacement)
        by_uid: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for rec in guided_records:
            by_uid[rec["owner_uid"]].append(rec)
    
        selected_uids = [uid for uid in zero_uids if uid in by_uid and len(by_uid[uid]) > 0]
        if not selected_uids:
            return None, None, pg_metrics
    
        prefix_prompts: List[Union[str, List[dict]]] = []
        prefix_owner_uids: List[str] = []
        picked_for_debug: List[Dict[str, object]] = []
        chosen_fracs_sel: List[float] = []
        chosen_bins_sel: List[int] = []
    
        for uid in selected_uids:
            cands = by_uid[uid]
            for _ in range(group_size):
                rec = self._rng.choice(cands)
                picked_for_debug.append(rec)
                prefix_prompts.append(rec["prefix_prompt"])
                prefix_owner_uids.append(uid)
                chosen_fracs_sel.append(float(rec["frac"]))
                chosen_bins_sel.append(int(rec["bin_idx"]))
    
        pg_metrics["pg/hint_guided_q_total"] = float(len(selected_uids))
        pg_metrics["pg/hint_guided_total"] = float(len(prefix_prompts))
        pg_metrics["pg/hint_guided_expected_total"] = float(len(selected_uids) * int(group_size))
    
        if not prefix_prompts:
            return None, None, pg_metrics
    
        # --- 5) Roll out the hint-guided prompts and UNION back (order preserved) ---
        guided_in = self._make_prompts_dataproto(prefix_prompts)
        guided_in.non_tensor_batch["uid"] = np.asarray(prefix_owner_uids, dtype=object)
    
        # carry reward model / GT for reward_fn (length aligns one-to-one with rows)
        reward_models = []
        for uid in prefix_owner_uids:
            reward_models.append({"ground_truth": gt_by_uid.get(uid)})
        guided_in.non_tensor_batch["reward_model"] = np.asarray(reward_models, dtype=object)


        # --- ensure non-tensor 'data_source' exists for the reward manager ---
        uid2ds = {}
        if "data_source" in normal_batch.non_tensor_batch:
            uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
            ds_all   = np.asarray(normal_batch.non_tensor_batch["data_source"], dtype=object)
            for u, ds in zip(uids_all.tolist(), ds_all.tolist()):
                if u not in uid2ds:
                    uid2ds[u] = ds
        default_ds = "prefix_guided"
        guided_in.non_tensor_batch["data_source"] = np.asarray(
            [uid2ds.get(uid, default_ds) for uid in prefix_owner_uids], dtype=object
        )

    
        guided_batch = self._rollout_and_union(guided_in)
        self._ensure_response_mask(guided_batch)

        guided_prompt_texts_raw = guided_batch.non_tensor_batch.get("prompt_text", [])
        if isinstance(guided_prompt_texts_raw, np.ndarray):
            guided_prompt_texts = guided_prompt_texts_raw.tolist()
        elif isinstance(guided_prompt_texts_raw, list):
            guided_prompt_texts = guided_prompt_texts_raw
        elif guided_prompt_texts_raw:
            guided_prompt_texts = [guided_prompt_texts_raw]
        else:
            guided_prompt_texts = []
    
        # --- 6) Old log probs / ref / critic on the guided batch itself (pad→compute→unpad) ---
        # old_log_probs
        a_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        gb_pad, pad_sz = pad_dataproto_to_divisor(guided_batch, a_div)
        old_lp_pad = self.actor_rollout_wg.compute_log_prob(gb_pad)
        old_lp = unpad_dataproto(old_lp_pad, pad_sz)
        if "entropys" in old_lp.batch:
            old_lp.batch.pop("entropys")
        guided_batch = guided_batch.union(old_lp)
    
        # reference policy (optional)
        if self.use_reference_policy:
            r_div = max(1, getattr(self.ref_policy_wg, "world_size", 1))
            gb_pad, pad_sz = pad_dataproto_to_divisor(guided_batch, r_div)
            ref_lp_pad = self.ref_policy_wg.compute_ref_log_prob(gb_pad)
            ref_lp = unpad_dataproto(ref_lp_pad, pad_sz)
            guided_batch = guided_batch.union(ref_lp)
    
        # critic values (optional)
        if self.use_critic:
            c_div = max(1, getattr(self.critic_wg, "world_size", 1))
            gb_pad, pad_sz = pad_dataproto_to_divisor(guided_batch, c_div)
            values_pad = self.critic_wg.compute_values(gb_pad)
            values = unpad_dataproto(values_pad, pad_sz)
            guided_batch = guided_batch.union(values)
    
        # rm scores (optional)
        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(guided_batch)
            guided_batch = guided_batch.union(rm_scores)
    
        # --- 7) Reward → (optional) affine rescale → token_level_rewards → advantages ---
        guided_scores, reward_extra_infos = _compute_reward(guided_batch, self.reward_fn)
        guided_batch.batch["token_level_scores"] = guided_scores
        # Optional dithering before computing advantages
        self._ensure_response_mask(guided_batch)
        self._apply_reward_dithering(guided_batch, group_size=group_size)
        # (optional) add any reward extra info (dict of lists) to non-tensor fields
        if reward_extra_infos:
            for k, v in reward_extra_infos.items():
                guided_batch.non_tensor_batch[k] = np.asarray(v, dtype=object)
    
        # affine rescale (order-independent)
        if self.pg_rescale_enable and self.pg_rescale_method == "affine_match":
            rescale_metrics = self._affine_rescale_scores_inplace(
                guided_batch, normal_batch, scope=self.pg_rescale_scope
            )
            pg_metrics.update({f"pg/rescale/{k.split('/')[-1]}": v for k, v in rescale_metrics.items()})
    
        # KL-in-reward disabled via __init__; otherwise call _apply_kl_penalty
        guided_batch.batch["token_level_rewards"] = guided_batch.batch["token_level_scores"]
    
        guided_batch = _compute_advantage(
            guided_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )
    
        # stamp token counts for logging
        guided_batch.meta_info["global_token_num"] = torch.sum(
            guided_batch.batch["attention_mask"], dim=-1
        ).tolist()
    
        # --- 8) Metrics: success of guided batch + bandit updates ---
        if guided_batch.batch["token_level_scores"].dim() == 2:
            g_seq = (guided_batch.batch["token_level_scores"] *
                     guided_batch.batch["response_mask"].to(guided_batch.batch["token_level_scores"].dtype)).sum(-1)
        else:
            g_seq = guided_batch.batch["token_level_scores"]
    
        pg_metrics["pg/zero_solve_post/reward_mean"] = float(g_seq.mean().item())
        solve = (g_seq > thr)
        pg_metrics["pg/zero_solve_post/solve_rate"] = float(solve.float().mean().item())
        if chosen_fracs_sel:
            pg_metrics["pg/ts_frac/mean"] = float(np.mean(chosen_fracs_sel))
            pg_metrics["pg/ts_frac/min"] = float(np.min(chosen_fracs_sel))
            pg_metrics["pg/ts_frac/max"] = float(np.max(chosen_fracs_sel))
        pg_metrics["pg/ts_target_success"] = self.pg_target
        for bin_idx, ok in zip(chosen_bins_sel, solve.tolist()):
            self.pg_bandit.update(bin_idx=bin_idx, success=bool(ok))
    
        # sanity text‑match success
        resp_txt = self._decode_texts(guided_batch.batch["responses"])
        self._debug_log_guided_samples(picked_for_debug, guided_prompt_texts, resp_txt)
        gt_for_row = [gt_by_uid[uid] for uid in prefix_owner_uids]
        succ_text = [str(extract_boxed_answer(t)).strip() == str(gt).strip()
                     for t, gt in zip(resp_txt, gt_for_row)]
        pg_metrics["pg/zero_solve_post/solve_rate_text"] = float(np.mean(succ_text))
    
        # group by uid: how many zero‑solve questions became solvable
        uid_any = defaultdict(bool)
        for ok, uid in zip(solve.tolist(), prefix_owner_uids):
            uid_any[uid] = uid_any[uid] or bool(ok)
        pg_metrics["pg/q_post_hint_pos_count"] = float(sum(1 for v in uid_any.values() if v))
        pg_metrics["pg/q_post_hint_pos_frac"] = float(
            pg_metrics["pg/q_post_hint_pos_count"] / max(1, len(zero_uids))
        )
    
        # --- 9) Optional ANCHOR batch (plain question → remainder) ---
        anchor_batch: Optional[DataProto] = None
        if self.anchor_enable and self.anchor_weight > 0.0:
            anchor_prompts: List[str] = []
            anchor_resps: List[str] = []
            for rec in picked_for_debug:
                if rec.get("use_for_anchor"):
                    anchor_prompts.append(rec["plain_question"])
                    anchor_resps.append(rec["anchor_remainder"])
    
            if anchor_prompts:
                ab = self._make_pr_dataproto(anchor_prompts, anchor_resps)
    
                # compute old_log_probs (pad→compute→unpad), then set fixed advantages
                a_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
                ab_pad, pad_sz = pad_dataproto_to_divisor(ab, a_div)
                lp_pad = self.actor_rollout_wg.compute_log_prob(ab_pad)
                lp = unpad_dataproto(lp_pad, pad_sz)
                if "entropys" in lp.batch:
                    lp.batch.pop("entropys")
                ab = ab.union(lp)
    
                self._ensure_response_mask(ab)
                mask = ab.batch["response_mask"].to(dtype=torch.float32)
                ab.batch["advantages"] = mask * self.anchor_weight
                ab.batch["returns"] = torch.zeros_like(ab.batch["advantages"])
                ab.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(anchor_prompts))], dtype=object)
    
                L = int(ab.batch["prompts"].shape[1] + ab.batch["responses"].shape[1])
                ab.meta_info["global_token_num"] = [L] * ab.batch.batch_size[0]
    
                anchor_batch = ab
                pg_metrics["pg/anchor/num_pairs"] = float(len(anchor_prompts))
    
        return guided_batch, anchor_batch, pg_metrics


    # -------------------- zero-solve utils & gt extraction -------------------
    def _zero_solve_group_mask(self, token_level_scores: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
        """
        Return (zero_group_mask, B) where zero_group_mask is a BOOL tensor of shape [B]:
          True  => the entire group (size `group_size`) has zero successes
          False => at least one success in the group
        """
        # token_level_scores is [B*n, T] or [B*n]
        if token_level_scores.dim() == 2:
            seq_scores = token_level_scores.sum(-1)  # [B*n]
        elif token_level_scores.dim() == 1:
            seq_scores = token_level_scores
        else:
            raise ValueError(f"Unexpected reward shape: {tuple(token_level_scores.shape)}")

        Bn = seq_scores.shape[0]
        assert Bn % group_size == 0, f"Total sequences ({Bn}) not divisible by group size ({group_size})."
        B = Bn // group_size

        scores = seq_scores.view(B, group_size)  # [B, n]
        success = (scores > 0.5)  # success if seq score > 0.5
        zero_group_mask = ~success.any(dim=1)  # [B] True where ALL n are failures
        return zero_group_mask, B

    def _extract_gt_answers(self, batch: DataProto) -> List[Optional[str]]:
        """Vectorized extraction of ground-truth answers for each row."""
        out: List[Optional[str]] = []
        rm = batch.non_tensor_batch.get("reward_model", None)
        if rm is None:
            # no ground truth on this batch
            return [None] * batch.batch.batch_size[0]
    
        rm_arr = np.asarray(rm, dtype=object)
        for r in rm_arr.tolist():
            gt = None
            if isinstance(r, dict):
                # common shapes: {'ground_truth': <str or dict>} or directly {'answer': ...}
                v = r.get("ground_truth", r)
                if isinstance(v, dict):
                    gt = v.get("answer", None)
                else:
                    gt = v
            out.append(gt)
        return out

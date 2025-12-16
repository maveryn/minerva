from __future__ import annotations

import math
import re
import uuid
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer as _BasePPO,
    compute_advantage as _compute_advantage,
    compute_response_mask as _compute_response_mask,
    apply_kl_penalty as _apply_kl_penalty,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward as _compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .prompting import (
    build_instruction_guided_messages,
    build_instruction_messages,
    configure_templates,
    _extract_user_text,
    _sanitize_instructions_text,
)


_BOXED_PATTERN = re.compile(r"\\boxed\{[^{}]*\}")

bad_line = re.compile(r"(\\boxed|≈|≥|≤|⇒|→|Answer|Final|Therefore|Thus|Option\s*\(|[°])",
                      re.IGNORECASE)

_ONLY_PUNCT = re.compile(r"^[\)\(\{\}\[\];:.,`'\"-]+\s*$")


class InstructionGuidedPPOTrainer(_BasePPO):
    """Instruction-guided variant of RayPPOTrainer.

    The core PPO loop matches the prefix-guided trainer. After the base rollout we
    identify zero-solve questions (by uid), ask the policy for high-level
    instructions without revealing answers, re-ask with those instructions, and
    merge the guided samples back into the PPO update.
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
        if not hasattr(config, "algorithm"):
            raise ValueError("config.algorithm missing")
        config.algorithm.adv_estimator = "grpo"
        config.algorithm.use_kl_in_reward = False

        if hasattr(config.algorithm, "clip_coef"):
            config.algorithm.clip_coef = float("inf")
        if hasattr(config, "actor_rollout_ref") and hasattr(
            config.actor_rollout_ref, "actor"
        ):
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

        ig_cfg = config.get("instruction_guided", {})
        self.ig_enable = bool(ig_cfg.get("enable", True))
        self.ig_instruction_samples = int(ig_cfg.get("instruction_samples", 3))
        self.ig_ans_per_instruction = int(
            ig_cfg.get("answer_rollouts_per_instruction", 1)
        )
        self.ig_temperature = float(ig_cfg.get("temperature", 0.7))
        self.ig_max_instruction_tokens = int(ig_cfg.get("max_instruction_tokens", 256))


        instrew_cfg = ig_cfg.get("instruction_reward", {})
        self.ir_enable = bool(instrew_cfg.get("enable", False))
        self.ir_weight = float(instrew_cfg.get("weight", 1.0))  # scale for advantages
        self.ir_len_penalty = float(instrew_cfg.get("length_penalty", 0.0))  # optional
        self.ir_success_pow = float(instrew_cfg.get("success_power", 1.0))   # optional shaping


        prompt_cfg = ig_cfg.get("prompt", {})

        # Gather templates from either the nested prompt block or top-level (both supported)
        tpl_inst = None
        tpl_guided = None
        if isinstance(prompt_cfg, dict):
            tpl_inst = prompt_cfg.get("instruction_template")
            tpl_guided = prompt_cfg.get("instruction_guided_template")
        
        tpl_inst = tpl_inst or ig_cfg.get("instruction_template")
        tpl_guided = tpl_guided or ig_cfg.get("instruction_guided_template")
        
        # Provide sturdy fallbacks if still missing
        if not tpl_inst:
            tpl_inst = (
                "You are given a math problem. Propose a high-level plan to solve it "
                "without performing calculations or symbolic manipulations. "
                "Use a single approach for the plan; do not mention or list alternatives. "
                "Write a short bullet list; start each bullet with an imperative verb "
                "(e.g., 'Define', 'Set up', 'Apply', 'Use', 'Conclude'). "
                "Keep it abstract: you may name variables, define expressions, or cite theorems/lemmas, "
                "but do not plug in values, estimate, or state the final answer. "
                "Limit each bullet to one sentence. "
                "End with a final 'Check' bullet describing how one would verify the result "
                "(units, bounds, substitution, special cases) without carrying it out. "
                "Output only the bullet list—no preamble or conclusion."
            )
        
        if not tpl_guided:
            tpl_guided = (
                "You are given a math problem. Your goal is to solve it by faithfully executing the provided high-level plan. "
                "Follow the instructions step by step, expanding each bullet into the necessary reasoning and computations. "
                "Carry out all steps carefully and completely, justifying each transition as needed. "
                "If the plan includes a 'Check' step, perform a concise verification immediately before stating the final answer. "
                "Present the final answer clearly inside \\boxed{}."
            )

        
        configure_templates(
            instruction_template=tpl_inst,
            instruction_guided_template=tpl_guided,
        )


        rd_cfg = ig_cfg.get("reward_dither", {})
        self.rd_enable = bool(rd_cfg.get("enable", True))
        self.rd_first_n = int(rd_cfg.get("first_n_tokens", 32))
        self.rd_scale = float(rd_cfg.get("scale", 0.03))
        self.rd_eps = float(rd_cfg.get("eps", 1e-8))

        rescale_cfg = ig_cfg.get("rescale", {})
        self.ig_rescale_enable = bool(rescale_cfg.get("enable", False))
        self.ig_rescale_scope = str(rescale_cfg.get("scope", "global"))
        self.ig_rescale_clip_min = float(
            rescale_cfg.get("clip_min", math.exp(-2.0))
        )
        self.ig_rescale_clip_max = float(
            rescale_cfg.get("clip_max", math.exp(2.0))
        )
        self.ig_success_threshold = float(
            ig_cfg.get("success_threshold", 0.5)
        )

    # ------------------------------------------------------------------
    # basic helpers reused from prefix-guided trainer
    # ------------------------------------------------------------------
    def _decode_texts(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _strip_fences(self, s: str) -> str:
        # Remove ``` blocks and leading/trailing whitespace; keep plain text bullets.
        s = str(s or "")
        if "```" in s:
            parts = []
            keep = True
            for line in s.splitlines():
                if line.strip().startswith("```"):
                    keep = not keep
                    continue
                if keep:
                    parts.append(line)
            s = "\n".join(parts)
        return s.strip()

    def _debug_log_guided_samples(
        self,
        records: Sequence[Dict[str, object]],
        guided_prompt_texts: Sequence[str],
        guided_resp_texts: Sequence[str],
    ) -> None:
        if not records:
            return

        for idx, rec in enumerate(records):
            inst_prompt = str(rec.get("instruction_prompt", "<NA>"))
            inst_raw = str(rec.get("instruction_raw", "<NA>"))
            inst_clean = str(rec.get("instruction_text", "<NA>"))
            question = str(rec.get("question", "<NA>"))

            guided_prompt = (
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
            print(f"[IG DEBUG] Sample {idx}")
            print("-" * 120)
            print("[Q] Plain Question:\n")
            print(question)
            print("-" * 120)
            print("\n[1] Instruction PROMPT:\n")
            print(inst_prompt)
            print("-" * 120)
            print("\n[2] Instruction RESPONSE (raw):\n")
            print(inst_raw)
            print("-" * 120)
            print("\n[3] Cleaned Instructions (used):\n")
            print(inst_clean)
            print("-" * 120)
            print("\n[4] Instruction-guided PROMPT:\n")
            print(guided_prompt)
            print("-" * 120)
            print("\n[5] Instruction-guided RESPONSE:\n")
            print(guided_resp)
            print("=" * 120 + "\n", flush=True)

    def _make_prompts_dataproto(
        self, items: List[Union[str, List[dict]]]
    ) -> DataProto:
        formatted_texts: List[str] = []
        for it in items:
            if (
                isinstance(it, (list, tuple))
                and it
                and isinstance(it[0], dict)
                and "role" in it[0]
                and "content" in it[0]
            ):
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
                "prompts": input_ids.clone(),
            },
            batch_size=[input_ids.size(0)],
        )

        non_tensor = {"prompt_text": np.asarray(formatted_texts, dtype=object)}
        return DataProto(batch=td, non_tensor_batch=non_tensor)

    def _repeat_with_sample_uids(
        self,
        dp: DataProto,
        repeat_times: int,
        *,
        uid_key: str = "uid",
        sample_key: str = "sample_uid",
    ) -> DataProto:
        """Repeat a DataProto while emitting deterministic sample-level identifiers.

        Each row that originates from the same ``uid`` receives a derived ``sample_uid``
        of the form ``"{uid}::r{rollout_idx}"`` so that downstream stages can align
        prompts/responses without relying on positional ordering.
        """

        repeated = dp.repeat(repeat_times=repeat_times, interleave=True)

        if uid_key in dp.non_tensor_batch:
            base_uids = np.asarray(dp.non_tensor_batch[uid_key], dtype=object).tolist()
            sample_ids: List[str] = []
            for uid in base_uids:
                for ridx in range(repeat_times):
                    sample_ids.append(f"{uid}::r{ridx}")
            repeated.non_tensor_batch[sample_key] = np.asarray(
                sample_ids, dtype=object
            )

        return repeated

    def _ensure_response_mask(self, data: DataProto) -> None:
        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = _compute_response_mask(data)

    def _compute_rewards_and_adv(self, data: DataProto) -> None:
        reward_tensor, reward_extra = _compute_reward(data, self.reward_fn)
        data.batch["token_level_scores"] = reward_tensor

        if self.config.algorithm.use_kl_in_reward:
            data, _ = _apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        data = _compute_advantage(
            data,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            ),
            config=self.config.algorithm,
        )

        if reward_extra:
            for k, v in reward_extra.items():
                data.non_tensor_batch[k] = np.asarray(v, dtype=object)

    def _ensure_full_sequence_fields(self, dp: DataProto) -> None:
        """
        Ensure dp.batch has input_ids, attention_mask, position_ids.

        We try (in order):
          A) prompts + responses
          B) input_ids (prompt-only) + responses

        In both cases we compute attention_mask and position_ids from masks.
        """

        need_input = "input_ids" not in dp.batch.keys()
        need_attn = "attention_mask" not in dp.batch.keys()
        need_pos = "position_ids" not in dp.batch.keys()
        if not (need_input or need_attn or need_pos):
            return

        has_prompts = "prompts" in dp.batch.keys()
        has_responses = "responses" in dp.batch.keys()
        if not has_responses:
            return

        pad_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        )
        responses_ids = dp.batch["responses"]
        response_mask = dp.batch.get(
            "response_mask", (responses_ids != pad_id).long()
        ).clone().long()

        if has_prompts:
            prompts_ids = dp.batch["prompts"]
            prompts_mask = (prompts_ids != pad_id).long()
            input_ids = torch.cat([prompts_ids, responses_ids], dim=1)
            attention_mask = torch.cat([prompts_mask, response_mask], dim=1).long()
        elif "input_ids" in dp.batch.keys():
            prompts_ids = dp.batch["input_ids"]
            prompts_mask = (prompts_ids != pad_id).long()
            input_ids = torch.cat([prompts_ids, responses_ids], dim=1)
            attention_mask = torch.cat([prompts_mask, response_mask], dim=1).long()
            dp.batch["prompts"] = prompts_ids
            dp.batch["response_mask"] = response_mask
        else:
            raise AssertionError(
                "Cannot reconstruct full sequence fields: need either 'prompts' or 'input_ids' plus 'responses'."
            )

        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min_(0).long()

        dp.batch["input_ids"] = input_ids
        dp.batch["attention_mask"] = attention_mask
        dp.batch["position_ids"] = position_ids

    def _generate_sequences_stable(self, dp: DataProto) -> DataProto:
        assert hasattr(dp, "batch") and hasattr(dp, "non_tensor_batch")

        size_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, size_div)

        out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        out = unpad_dataproto(out_pad, pad_size=pad_sz)

        if "responses" in out.batch and "response_mask" not in out.batch:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            out.batch["response_mask"] = (out.batch["responses"] != pad_id).long()

        for k in ("input_ids", "attention_mask", "position_ids", "prompts"):
            if k in out.batch:
                del out.batch[k]
        if "prompt_text" in out.non_tensor_batch:
            del out.non_tensor_batch["prompt_text"]

        return out

    def _rollout_and_union(self, dp: DataProto) -> DataProto:
        """Generate sequences for the given prompts *and* return an aligned DataProto.

        Critical invariant we enforce here (matching RaySvSTrainer behavior):
        - The *row order* of every tensor and non-tensor field in the returned DataProto
          must match the *row order of the generated responses* coming from the engine.
        - To achieve that, we use the engine-returned non-tensor 'prompt_text' as the
          source of truth for the output order and reorder the input meta/fields to match it.

        This fixes misalignment between prompts / uids / inst_ids and responses across calls.
        """
        assert hasattr(dp, "batch") and hasattr(dp, "non_tensor_batch")

        # pad to world_size, run generation, then unpad
        size_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, size_div)
        out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        out = unpad_dataproto(out_pad, pad_size=pad_sz)

        # Ensure response mask exists
        if "responses" in out.batch and "response_mask" not in out.batch:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            out.batch["response_mask"] = (out.batch["responses"] != pad_id).long()

        # Extract engine-side prompt_text (order-of-truth) *before* we touch non-tensor fields
        engine_pt = None
        if "prompt_text" in out.non_tensor_batch:
            engine_pt = np.asarray(out.non_tensor_batch["prompt_text"], dtype=object).tolist()

        def _build_perm(candidate: str) -> Optional[List[int]]:
            dp_vals = np.asarray(dp.non_tensor_batch.get(candidate, []), dtype=object).tolist()
            out_vals = np.asarray(out.non_tensor_batch.get(candidate, []), dtype=object).tolist()
            if not dp_vals or not out_vals or len(dp_vals) != len(out_vals):
                return None
            from collections import defaultdict

            buckets = defaultdict(list)
            for idx, key in enumerate(dp_vals):
                buckets[key].append(idx)

            perm_local: List[int] = []
            missed_local: List[str] = []
            for key in out_vals:
                lst = buckets.get(key)
                if lst:
                    perm_local.append(lst.pop(0))
                else:
                    missed_local.append(str(key))

            if missed_local:
                print(
                    f"[IG WARN] Could not align {len(missed_local)} rows using '{candidate}';"
                    " falling back to next strategy."
                )
                return None
            return perm_local

        perm: Optional[List[int]] = None
        for key in ("inst_id", "sample_uid", "uid"):
            perm = _build_perm(key)
            if perm is not None:
                break

        if perm is None and engine_pt is not None:
            dp_pt = (
                np.asarray(dp.non_tensor_batch.get("prompt_text", []), dtype=object).tolist()
                if "prompt_text" in dp.non_tensor_batch
                else None
            )
            if dp_pt is not None and len(engine_pt) == len(dp_pt):
                from collections import defaultdict

                idxs = defaultdict(list)
                for i, s in enumerate(dp_pt):
                    idxs[s].append(i)
                perm = []
                missed = []
                for s in engine_pt:
                    lst = idxs.get(s)
                    if lst:
                        perm.append(lst.pop(0))
                    else:
                        missed.append(s)
                if missed:
                    print(
                        f"[IG WARN] Could not find {len(missed)} engine prompts in dp.prompt_text;"
                        " using identity for remaining rows."
                    )
                    unused = [i for i in range(len(dp_pt)) if i not in perm]
                    need = len(engine_pt) - len(perm)
                    perm.extend(unused[:need])

        # Attach prompts (tokens) and non-tensor meta to *out* aligned to engine order
        if "prompts" in dp.batch:
            if perm is not None:
                out.batch["prompts"] = dp.batch["prompts"][perm].clone()
            else:
                out.batch["prompts"] = dp.batch["prompts"].clone()

        # Merge all non-tensor fields from dp, but reorder using perm; never overwrite engine prompt_text
        B = out.batch.batch_size[0]
        for k, v in dp.non_tensor_batch.items():
            if k == "prompt_text" and engine_pt is not None:
                # keep engine-side prompt_text to reflect true output order
                continue
            arr = np.asarray(v, dtype=object)
            if arr.ndim == 0:
                arr = np.full((B,), arr, dtype=object)
            elif arr.shape[0] != B:
                # pad/truncate carefully to B rows
                tmp = np.empty((B,), dtype=object)
                n = min(B, arr.shape[0])
                tmp[:n] = arr[:n]
                if n < B:
                    tmp[n:] = None
                arr = tmp
            if perm is not None and arr.shape[0] == len(perm):
                arr = arr[perm]
            out.non_tensor_batch[k] = arr

        # Build full-sequence fields from prompts + responses
        self._ensure_full_sequence_fields(out)

        # The generation output should be considered the source of truth for timing/length.
        out.meta_info["global_token_num"] = torch.sum(out.batch["attention_mask"], dim=-1).tolist()
        return out

    def _align_non_tensor_for_concat(self, dps: List[DataProto]) -> None:
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
                        tmp = np.empty(B, dtype=object)
                        n = min(B, arr.shape[0])
                        tmp[:n] = arr[:n]
                        if n < B:
                            tmp[n:] = None
                        arr = tmp
                    dp.non_tensor_batch[k] = arr

    def _align_batch_keys_for_concat(self, dps: List[DataProto]) -> None:
        all_keys = set()
        for dp in dps:
            all_keys |= set(dp.batch.keys())

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
                elif k in (
                    "old_log_probs",
                    "ref_log_prob",
                    "values",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                ):
                    dp.batch[k] = torch.zeros((B, T_resp), dtype=torch.float32, device=dev)
                else:
                    dp.batch[k] = torch.zeros((B, T_full), dtype=torch.float32, device=dev)

    def _right_pad_2d(self, x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if x is None or not torch.is_tensor(x) or x.dim() != 2:
            return x
        B, T = x.shape
        if T >= target_len or target_len is None:
            return x
        pad = x.new_full((B, target_len - T), pad_value)
        return torch.cat([x, pad], dim=1)

    def _pad_for_concat(
        self, dp_a: DataProto, dp_b: DataProto
    ) -> Tuple[DataProto, DataProto]:
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
            pad_block = t.new_full((B, target_len - T), pad_value)
            return torch.cat([t, pad_block], dim=1)

        response_level_keys = {
            "responses",
            "response_mask",
            "old_log_probs",
            "ref_log_prob",
            "rollout_log_probs",
            "token_level_scores",
            "token_level_rewards",
            "advantages",
            "returns",
            "values",
        }

        def _pad_dp(dp: DataProto, T_full_dp, T_prompt_dp, T_resp_dp) -> DataProto:
            for key in list(dp.batch.keys()):
                t = dp.batch[key]
                if not (torch.is_tensor(t) and t.dim() == 2):
                    continue

                target_len = None
                pad_val: Union[int, float] = 0

                if key in ("input_ids", "attention_mask", "position_ids"):
                    target_len = T_full_max
                    pad_val = pad_id if key == "input_ids" else 0
                elif key == "prompts":
                    target_len = T_prompt_max
                    pad_val = pad_id
                elif key in response_level_keys:
                    target_len = T_resp_max
                    pad_val = pad_id if key == "responses" else 0
                else:
                    target_len = T_full_max
                    pad_val = 0

                dp.batch[key] = _pad_tensor(t, target_len, pad_val)
            return dp

        return _pad_dp(dp_a, Ta_full, Ta_prompt, Ta_resp), _pad_dp(
            dp_b, Tb_full, Tb_prompt, Tb_resp
        )

    def _attach_uids(self, batch: DataProto) -> None:
        B = batch.batch.batch_size[0]
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(B)], dtype=object
        )

    # ------------------------------------------------------------------
    # Instruction branch helpers
    # ------------------------------------------------------------------
    def _uids_with_zero_success(
        self, normal_batch: DataProto, threshold: float = 0.5
    ) -> List[str]:
        tls = normal_batch.batch["token_level_scores"]
        resp_mask = normal_batch.batch.get("response_mask", None)
        if tls.dim() == 2 and resp_mask is not None:
            seq_scores = (tls * resp_mask.to(tls.dtype)).sum(-1)
        elif tls.dim() == 2:
            seq_scores = tls.sum(-1)
        else:
            seq_scores = tls

        uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        uid_first_idx: Dict[str, int] = {}
        uid_order: List[str] = []
        uid_any_success: Dict[str, bool] = {}
        for i, u in enumerate(uids_all.tolist()):
            if u not in uid_first_idx:
                uid_first_idx[u] = i
                uid_order.append(u)
            uid_any_success[u] = uid_any_success.get(u, False) or (
                float(seq_scores[i].item()) > threshold
            )
        zero = [u for u in uid_order if not uid_any_success.get(u, False)]
        return zero

    def _uid_to_plain_question_text(self, normal_batch: DataProto) -> Dict[str, str]:
        prompts_text_all = self._decode_texts(normal_batch.batch["prompts"])
        uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        mapping: Dict[str, str] = {}
        for idx, uid in enumerate(uids_all.tolist()):
            if uid not in mapping:
                raw = prompts_text_all[idx]
                mapping[uid] = _extract_user_text(raw)
                
        return mapping

    def _uid_to_prompt_tokens(self, normal_batch: DataProto) -> Dict[str, torch.Tensor]:
        prompts = normal_batch.batch["prompts"]
        uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        mapping: Dict[str, torch.Tensor] = {}
        for idx, uid in enumerate(uids_all.tolist()):
            if uid not in mapping:
                mapping[uid] = prompts[idx].clone()
        return mapping

    def _materialize_plain_prompts_for_uids(
        self, uids: Sequence[str], normal_batch: DataProto
    ) -> torch.Tensor:
        uid2tokens = self._uid_to_prompt_tokens(normal_batch)
        tensors = [uid2tokens[u].clone() for u in uids]
        return torch.stack(tensors, dim=0)

    def _score_with_old_ref_critic(self, dp: DataProto) -> DataProto:
        if dp is None:
            return dp

        a_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        gb_pad, pad_sz = pad_dataproto_to_divisor(dp, a_div)
        old_lp_pad = self.actor_rollout_wg.compute_log_prob(gb_pad)
        old_lp = unpad_dataproto(old_lp_pad, pad_sz)
        if "entropys" in old_lp.batch:
            old_lp.batch.pop("entropys")
        dp = dp.union(old_lp)

        if self.use_reference_policy:
            r_div = max(1, getattr(self.ref_policy_wg, "world_size", 1))
            gb_pad, pad_sz = pad_dataproto_to_divisor(dp, r_div)
            ref_lp_pad = self.ref_policy_wg.compute_ref_log_prob(gb_pad)
            ref_lp = unpad_dataproto(ref_lp_pad, pad_sz)
            dp = dp.union(ref_lp)

        if self.use_critic:
            c_div = max(1, getattr(self.critic_wg, "world_size", 1))
            gb_pad, pad_sz = pad_dataproto_to_divisor(dp, c_div)
            values_pad = self.critic_wg.compute_values(gb_pad)
            values = unpad_dataproto(values_pad, pad_sz)
            dp = dp.union(values)

        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(dp)
            dp = dp.union(rm_scores)

        return dp

    @staticmethod
    def _seq_scores_from_tokens(
        token_level_scores: torch.Tensor, response_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if token_level_scores.dim() == 1:
            return token_level_scores
        if response_mask is not None and response_mask.shape[-1] == token_level_scores.size(1):
            return (token_level_scores * response_mask).sum(-1)
        return token_level_scores.sum(-1)

    @staticmethod
    def _seq_lengths_from_mask(response_mask: torch.Tensor) -> torch.Tensor:
        return response_mask.to(torch.float32).sum(-1).clamp_min(1.0)

    def _affine_rescale_scores_inplace(
        self, guided: DataProto, ref: DataProto, scope: str = "global"
    ) -> Dict[str, float]:
        g_scores = self._seq_scores_from_tokens(
            guided.batch["token_level_scores"], guided.batch.get("response_mask", None)
        )
        r_scores = self._seq_scores_from_tokens(
            ref.batch["token_level_scores"], ref.batch.get("response_mask", None)
        )
        eps = 1e-8
        metrics: Dict[str, float] = {}

        if (
            scope == "per_uid"
            and "uid" in guided.non_tensor_batch
            and "uid" in ref.non_tensor_batch
        ):
            g_uid = guided.non_tensor_batch["uid"]
            r_uid = ref.non_tensor_batch["uid"]
            unique = list({u for u in g_uid.tolist()})
            mu_on, sigma_on = r_scores.mean(), r_scores.std()
            mu_off, sigma_off = g_scores.mean(), g_scores.std()
            a_global = sigma_on / torch.clamp(sigma_off, min=eps)
            b_global = mu_on - a_global * mu_off

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

            L = self._seq_lengths_from_mask(guided.batch["response_mask"])
            guided.batch["token_level_scores"] = (
                guided.batch["token_level_scores"] * a.unsqueeze(-1)
                + (b / L).unsqueeze(-1)
            )
            metrics.update(
                {
                    "ig/rescale/method": 0.0,
                    "ig/rescale/mean_shift": float(b_global.detach().item()),
                    "ig/rescale/scale": float(a_global.detach().item()),
                }
            )
        else:
            mu_on, sigma_on = r_scores.mean(), r_scores.std()
            mu_off, sigma_off = g_scores.mean(), g_scores.std()
            a = sigma_on / torch.clamp(sigma_off, min=eps)
            a = torch.clamp(a, min=self.ig_rescale_clip_min, max=self.ig_rescale_clip_max)
            b = mu_on - a * mu_off
            L = self._seq_lengths_from_mask(guided.batch["response_mask"])
            guided.batch["token_level_scores"] = (
                guided.batch["token_level_scores"] * a
                + (b / L).unsqueeze(-1)
            )
            metrics.update(
                {
                    "ig/rescale/method": 1.0,
                    "ig/rescale/mean_shift": float(b.detach().item()),
                    "ig/rescale/scale": float(a.detach().item()),
                }
            )
        return metrics

    def _apply_reward_dithering(self, dp: DataProto, group_size: int) -> None:
        if not getattr(self, "rd_enable", True) or self.rd_scale <= 0.0:
            return
        if "token_level_scores" not in dp.batch or "response_mask" not in dp.batch:
            return

        scores = dp.batch["token_level_scores"]
        mask = dp.batch["response_mask"]
        if not (torch.is_tensor(scores) and scores.dim() == 2):
            return

        Bn, T = scores.shape
        if group_size <= 0 or (Bn % group_size) != 0:
            return
        B = Bn // group_size

        with torch.no_grad():
            seq_scores = (scores * mask).sum(dim=1)
            seq_scores_g = seq_scores.view(B, group_size)
            std_per_group = seq_scores_g.float().std(dim=1, unbiased=False)
            sigma = (self.rd_scale * std_per_group).repeat_interleave(group_size)
            sigma = sigma.clamp_min(self.rd_eps).unsqueeze(1)

            N = min(int(self.rd_first_n), T)
            ar = torch.arange(T, device=scores.device).unsqueeze(0)
            prefix_mask = (ar < N).to(mask.dtype) * mask
            noise = torch.randn_like(scores) * sigma
            dp.batch["token_level_scores"] = scores + noise * prefix_mask

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            print(f"Initial validation metrics: {val_metrics}")
            if self.config.trainer.get("val_only", False):
                return

        total_steps = self.total_training_steps
        n_gpus = self.resource_pool_manager.get_n_gpus()

        progress_bar = tqdm(
            total=total_steps, initial=self.global_steps, desc="Training Progress"
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: Dict[str, float] = {}
                timing_raw: Dict[str, float] = {}

                with marked_timer("step", timing_raw):
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    self._attach_uids(batch)

                    gen_batch = self._get_gen_batch(batch)
                    if "uid" in batch.non_tensor_batch:
                        gen_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"].copy()

                    n = self.config.actor_rollout_ref.rollout.n
                    gen_batch = self._repeat_with_sample_uids(gen_batch, repeat_times=n)
                    gen_out = self._generate_sequences_stable(gen_batch)
                    if "uid" in gen_batch.non_tensor_batch:
                        gen_out.non_tensor_batch["uid"] = gen_batch.non_tensor_batch["uid"].copy()
                    if "sample_uid" in gen_batch.non_tensor_batch:
                        gen_out.non_tensor_batch["sample_uid"] = gen_batch.non_tensor_batch[
                            "sample_uid"
                        ].copy()

                    batch = self._repeat_with_sample_uids(batch, repeat_times=n)
                    batch = batch.union(gen_out)

                    if "prompts" not in batch.batch:
                        if "input_ids" in gen_batch.batch:
                            batch.batch["prompts"] = gen_batch.batch["input_ids"].clone()
                        elif "input_ids" in batch_dict:
                            batch.batch["prompts"] = batch_dict["input_ids"].clone()

                    self._ensure_response_mask(batch)
                    self._ensure_full_sequence_fields(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    required = (
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "responses",
                        "response_mask",
                    )
                    missing = [k for k in required if k not in batch.batch.keys()]
                    assert not missing, f"Missing fields before compute_log_prob: {missing}"

                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    ent = old_log_prob.batch.pop("entropys", None)
                    batch = batch.union(old_log_prob)
                    if ent is not None:
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy = agg_loss(
                            loss_mat=ent,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=loss_agg_mode,
                        )
                        metrics["actor/entropy"] = float(entropy.detach().item())

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_out = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_out = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_out)

                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                    with marked_timer("reward+adv/orig", timing_raw):
                        if "data_source" not in batch.non_tensor_batch:
                            batch.non_tensor_batch["data_source"] = np.asarray(
                                ["plain"] * len(batch), dtype=object
                            )

                        if self.use_rm:
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        self._compute_rewards_and_adv(batch)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    guided_batch = None
                    ig_metrics: Dict[str, float] = {}
                    if self.ig_enable:
                        guided_batch, ig_metrics = self._guided_branch(batch, n)

                    if guided_batch is not None:
                        guided_uids = set(
                            map(str, guided_batch.non_tensor_batch["uid"].tolist())
                        )
                        keep_mask = np.asarray(
                            [u not in guided_uids for u in batch.non_tensor_batch["uid"]],
                            dtype=bool,
                        )
                        keep_idx = torch.as_tensor(
                            np.nonzero(keep_mask)[0], dtype=torch.long
                        )
                        sub_plain = batch.select_idxs(keep_idx.tolist())

                        self._align_batch_keys_for_concat([sub_plain, guided_batch])
                        self._align_non_tensor_for_concat([sub_plain, guided_batch])
                        a, b = self._pad_for_concat(sub_plain, guided_batch)
                        train_batch_for_critic = DataProto.concat([a, b])
                    else:
                        train_batch_for_critic = batch

                    train_batch_for_actor = train_batch_for_critic

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            crit_div = max(
                                1, getattr(self.critic_wg, "world_size", 1)
                            )
                            tb_pad, _ = pad_dataproto_to_divisor(
                                train_batch_for_critic, crit_div
                            )
                            critic_output = self.critic_wg.update_critic(tb_pad)
                        metrics.update(
                            reduce_metrics(
                                critic_output.meta_info.get("metrics", {})
                            )
                        )

                    if (
                        not self.use_critic
                    ) or (
                        self.config.trainer.critic_warmup <= self.global_steps
                    ):
                        with marked_timer("update_actor", timing_raw):
                            act_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
                            ta_pad, _ = pad_dataproto_to_divisor(
                                train_batch_for_actor, act_div
                            )
                            actor_output = self.actor_rollout_wg.update_actor(ta_pad)
                        metrics.update(
                            reduce_metrics(actor_output.meta_info.get("metrics", {}))
                        )

                        # Second update: instruction tokens (if present this step)
                        if getattr(self, "_inst_train_batch", None) is not None:
                            with marked_timer("update_actor_instructions", timing_raw):
                                act_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
                                ta_pad, _ = pad_dataproto_to_divisor(self._inst_train_batch, act_div)
                                actor_output_inst = self.actor_rollout_wg.update_actor(ta_pad)
                            metrics.update(reduce_metrics(actor_output_inst.meta_info.get("metrics", {})))
                            # free buffer
                            self._inst_train_batch = None

                        
                metrics.update(ig_metrics)
                metrics.update(
                    compute_data_metrics(
                        batch=train_batch_for_actor, use_critic=self.use_critic
                    )
                )
                metrics.update(
                    compute_timing_metrics(
                        batch=train_batch_for_actor, timing_raw=timing_raw
                    )
                )
                metrics.update(
                    compute_throughout_metrics(
                        batch=train_batch_for_actor,
                        timing_raw=timing_raw,
                        n_gpus=n_gpus,
                    )
                )

                is_last_step = (self.global_steps + 1) >= total_steps
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last_step
                        or (self.global_steps + 1)
                        % self.config.trainer.test_freq
                        == 0
                    )
                ):
                    with marked_timer("testing", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or (self.global_steps+1) % self.config.trainer.save_freq == 0
                ):
                    with marked_timer(
                        "save_checkpoint", timing_raw, color="green"
                    ):
                        self._save_checkpoint()

                logger.log(data=metrics, step=self.global_steps + 1)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    return

    # ------------------------------------------------------------------
    # Instruction-guided branch
    # ------------------------------------------------------------------
    def _guided_branch(
        self, normal_batch: DataProto, group_size: int
    ) -> Tuple[Optional[DataProto], Dict[str, float]]:
        ig_metrics: Dict[str, float] = {"ig/group_size": float(group_size)}

        uids_all = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        total_questions = len({u for u in uids_all.tolist()})

        zero_uids = self._uids_with_zero_success(normal_batch)
        ig_metrics["ig/zero_solve_count"] = float(len(zero_uids))
        ig_metrics["ig/zero_solve_frac"] = float(
            len(zero_uids) / max(1, total_questions)
        )
        if not zero_uids:
            return None, ig_metrics

        uid2q = self._uid_to_plain_question_text(normal_batch)

        inst_prompts: List[Union[str, List[dict]]] = []
        inst_owner: List[str] = []
        inst_ids:   List[str] = []
        
        for uid in zero_uids:
            question = uid2q.get(uid)
            if question is None:
                continue
            for k in range(self.ig_instruction_samples):
                inst_prompts.append(build_instruction_messages(question))
                inst_owner.append(uid)
                inst_ids.append(f"{uid}#{k}")
        
        inst_dp = self._make_prompts_dataproto(inst_prompts)
        inst_dp.non_tensor_batch["uid"] = np.asarray(inst_owner, dtype=object)
        inst_dp.non_tensor_batch["inst_id"] = np.asarray(inst_ids, dtype=object)

        if not inst_prompts:
            return None, ig_metrics

        inst_dp.meta_info["temperature"] = self.ig_temperature
        inst_dp.meta_info["response_length"] = self.ig_max_instruction_tokens
        inst_out = self._rollout_and_union(inst_dp)

        # Build instruction records directly in *engine order* to keep perfect alignment
        inst_prompt_arr = inst_out.non_tensor_batch.get("prompt_text")
        inst_prompt_texts = inst_prompt_arr.tolist() if inst_prompt_arr is not None else ["" for _ in range(len(inst_out))]
        instructions_raw = self._decode_texts(inst_out.batch["responses"])

        # Collect aligned owner/meta from inst_out (already aligned to responses)
        uids_engine = np.asarray(inst_out.non_tensor_batch.get("uid"), dtype=object).tolist()
        inst_ids_engine = np.asarray(inst_out.non_tensor_batch.get("inst_id"), dtype=object).tolist()

        valid_instruction_records = []
        for i, (uid, instr_text) in enumerate(zip(uids_engine, instructions_raw)):
            question = uid2q.get(uid)
            if question is None:
                continue
            cleaned = _sanitize_instructions_text(self._strip_fences(_BOXED_PATTERN.sub("", instr_text or "")))
            if not cleaned:
                continue
            valid_instruction_records.append(
                {
                    "uid": uid,
                    "inst_id": inst_ids_engine[i],
                    "row_idx": i,  # index into inst_out
                    "question": question,
                    "instruction_prompt": inst_prompt_texts[i],
                    "instruction_raw": instr_text,
                    "instruction_text": cleaned,
                }
            )

        # when building guided prompts
        ig_prompts, ig_uids, ig_inst_ids, guided_records = [], [], [], []
        for rec in valid_instruction_records:
            ig_prompts.append(build_instruction_guided_messages(rec["question"], rec["instruction_text"]))
            ig_uids.append(rec["uid"])
            ig_inst_ids.append(rec["inst_id"])
            guided_records.append(rec)

        if not ig_prompts:
            return None, ig_metrics
        
        guided_in = self._make_prompts_dataproto(ig_prompts)
        guided_in.non_tensor_batch["uid"] = np.asarray(ig_uids, dtype=object)
        guided_in.non_tensor_batch["inst_id"] = np.asarray(ig_inst_ids, dtype=object)

        guided_in.meta_info["temperature"] = getattr(self, "ig_temperature", 0.7)
        guided_in.meta_info["response_length"] = int(getattr(self.config.data, "max_response_length", 4096))

        base_uid_arr = np.asarray(normal_batch.non_tensor_batch["uid"], dtype=object)
        uid2ds: Dict[str, object] = {}
        if "data_source" in normal_batch.non_tensor_batch:
            ds_all = np.asarray(normal_batch.non_tensor_batch["data_source"], dtype=object)
            for u, ds in zip(base_uid_arr.tolist(), ds_all.tolist()):
                if u not in uid2ds:
                    uid2ds[u] = ds
        default_ds = "instruction_guided"
        guided_in.non_tensor_batch["data_source"] = np.asarray(
            [uid2ds.get(uid, default_ds) for uid in ig_uids],
            dtype=object,
        )

        if "reward_model" in normal_batch.non_tensor_batch:
            uid2rm: Dict[str, object] = {}
            rm_all = np.asarray(normal_batch.non_tensor_batch["reward_model"], dtype=object)
            for u, rm in zip(base_uid_arr.tolist(), rm_all.tolist()):
                if u not in uid2rm:
                    uid2rm[u] = rm
            guided_in.non_tensor_batch["reward_model"] = np.asarray(
                [uid2rm.get(uid) for uid in ig_uids], dtype=object
            )
            
        guided_batch = self._rollout_and_union(guided_in)

        # ---- Reindex guided_batch to match our input order (by inst_id) ----
        inst_ids_out = np.asarray(guided_batch.non_tensor_batch.get("inst_id"), dtype=object).tolist()
        id2idx = {sid: i for i, sid in enumerate(inst_ids_out)}
        
        target_order = []
        missing = []
        for rec in guided_records:
            sid = rec["inst_id"]
            if sid in id2idx:
                target_order.append(id2idx[sid])
            else:
                missing.append(sid)
        
        if missing:
            print(f"[IG WARN] Missing {len(missing)} inst_id(s) in rollout: {missing[:5]} ...")
        
        if target_order and (len(target_order) != len(inst_ids_out)):
            print(f"[IG INFO] Selecting {len(target_order)} of {len(inst_ids_out)} guided rows to preserve order.")
        
        if target_order:
            guided_batch = guided_batch.select_idxs(target_order)
        
        # Use engine-order prompt_text for logging; it is aligned to guided_batch responses
        guided_prompt_arr = guided_batch.non_tensor_batch.get("prompt_text")
        guided_prompt_texts = guided_prompt_arr.tolist() if guided_prompt_arr is not None else ["" for _ in range(len(guided_records))]

        # Sanity: every guided prompt must contain its instruction text.
        try:
            guided_pt = guided_batch.non_tensor_batch.get("prompt_text")
            if isinstance(guided_pt, np.ndarray):
                guided_pt = guided_pt.tolist()
            for rec, pt in zip(guided_records, guided_pt or []):
                inst_txt = str(rec.get("instruction_text", "")).strip()
                if inst_txt and (inst_txt not in pt):
                    print("[IG ERROR] Missing instruction text inside guided prompt. inst_id=", rec.get("inst_id"))
        except Exception as _e:
            print("[IG WARN] Could not verify instruction injection:", _e)

        self._ensure_response_mask(guided_batch)
        guided_batch = self._score_with_old_ref_critic(guided_batch)

        guided_scores, reward_extra = _compute_reward(guided_batch, self.reward_fn)
        guided_batch.batch["token_level_scores"] = guided_scores
        self._ensure_response_mask(guided_batch)

        guided_uid_arr = np.asarray(guided_batch.non_tensor_batch["uid"], dtype=object)
        guided_uid_list = guided_uid_arr.tolist()
        uid_counts: Dict[str, int] = {}
        for uid in guided_uid_list:
            uid_counts[uid] = uid_counts.get(uid, 0) + 1
        unique_counts = set(uid_counts.values())
        guided_group_size = unique_counts.pop() if len(unique_counts) == 1 else None

        if guided_group_size and guided_group_size > 0:
            self._apply_reward_dithering(guided_batch, group_size=guided_group_size)
        if reward_extra:
            for k, v in reward_extra.items():
                guided_batch.non_tensor_batch[k] = np.asarray(v, dtype=object)

        if self.ig_rescale_enable:
            rescale_metrics = self._affine_rescale_scores_inplace(
                guided_batch, normal_batch, scope=self.ig_rescale_scope
            )
            ig_metrics.update(rescale_metrics)

        guided_batch.batch["token_level_rewards"] = guided_batch.batch[
            "token_level_scores"
        ]
        guided_batch = _compute_advantage(
            guided_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=guided_group_size or 1,
            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            ),
            config=self.config.algorithm,
        )

        guided_resp_texts = self._decode_texts(guided_batch.batch["responses"])
        self._debug_log_guided_samples(
            guided_records, guided_prompt_texts, guided_resp_texts
        )

        if guided_batch.batch["token_level_scores"].dim() == 2:
            seq_scores = (
                guided_batch.batch["token_level_scores"]
                * guided_batch.batch["response_mask"].to(
                    guided_batch.batch["token_level_scores"].dtype
                )
            ).sum(-1)
        else:
            seq_scores = guided_batch.batch["token_level_scores"]

        solve = seq_scores > self.ig_success_threshold

        inst_reward_map = {}
        if self.ir_enable:
            gb_inst_ids = np.asarray(guided_batch.non_tensor_batch.get("inst_id"), dtype=object).tolist()
            solve_np = np.asarray(solve.to(torch.float32).cpu())
            # mean success per instruction
            tmp = {}
            for sid, s in zip(gb_inst_ids, solve_np.tolist()):
                tmp.setdefault(sid, []).append(s)
            for sid, vals in tmp.items():
                r = float(np.mean(vals))
                # optional shaping: length penalty or power
                if self.ir_success_pow != 1.0:
                    r = r ** self.ir_success_pow
                inst_reward_map[sid] = r
            ig_metrics["ig/inst/mean_reward"] = float(np.mean(list(inst_reward_map.values()))) if inst_reward_map else 0.0

        
        ig_metrics["ig/zero_solve_post/reward_mean"] = float(seq_scores.mean().item())
        ig_metrics["ig/zero_solve_post/solve_rate"] = float(
            solve.float().mean().item()
        )

        uid_any: Dict[str, bool] = {}
        for ok, uid in zip(solve.tolist(), guided_uid_list):
            uid_any[uid] = uid_any.get(uid, False) or bool(ok)
        ig_metrics["ig/q_post_hint_pos_count"] = float(
            sum(1 for v in uid_any.values() if v)
        )
        ig_metrics["ig/q_post_hint_pos_frac"] = float(
            ig_metrics["ig/q_post_hint_pos_count"] / max(1, len(zero_uids))
        )

        if "response_mask" in guided_batch.batch:
            resp_lens = guided_batch.batch["response_mask"].sum(dim=-1).float()
            ig_metrics["ig/guided_response_tokens/mean"] = float(resp_lens.mean().item())
            ig_metrics["ig/guided_response_tokens/std"] = float(resp_lens.std().item())

        # ---- Instruction-token update (separate on-policy PPO/GRPO) ----
        if self.ir_enable and inst_reward_map:
            # Subset inst_out to the rows we actually used
            inst_ids_used = set(inst_reward_map.keys())
            inst_ids_all = np.asarray(inst_out.non_tensor_batch["inst_id"], dtype=object).tolist()
            keep_idx = [i for i, sid in enumerate(inst_ids_all) if sid in inst_ids_used]
            if keep_idx:
                inst_train = inst_out.select_idxs(keep_idx)
        
                # Old log-probs / ref / values (on-policy for instructions)
                inst_train = self._score_with_old_ref_critic(inst_train)
        
                # Token-level scores: uniform per token to sum to r^{inst}
                mask = inst_train.batch["response_mask"].to(torch.float32)
                lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                row_inst_ids = np.asarray(inst_train.non_tensor_batch["inst_id"], dtype=object).tolist()
                row_rewards = torch.tensor([inst_reward_map[sid] for sid in row_inst_ids],
                                           device=mask.device, dtype=torch.float32).unsqueeze(1)
                scores = (row_rewards / lengths) * mask
        
                inst_train.batch["token_level_scores"]  = scores
                inst_train.batch["token_level_rewards"] = scores
        
                # Grouping for GRPO: number of instructions per uid (assumed uniform if no filtering)
                inst_uids = np.asarray(inst_train.non_tensor_batch["uid"], dtype=object).tolist()
                from collections import Counter
                cnts = Counter(inst_uids)
                unique_counts = set(cnts.values())
                inst_group_size = unique_counts.pop() if len(unique_counts) == 1 else 1
        
                inst_train = _compute_advantage(
                    inst_train,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=inst_group_size,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm,
                )
        
                # Scale advantages to control influence of instruction head
                inst_train.batch["advantages"] = inst_train.batch["advantages"] * self.ir_weight
        
                # Stash for a second actor update in fit()
                self._inst_train_batch = inst_train

        return guided_batch, ig_metrics

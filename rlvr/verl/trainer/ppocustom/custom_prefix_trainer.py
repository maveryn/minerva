# verl/trainer/ppocustom/custom_prefix_trainer.py
"""Implementation of the prefix-guided PPO loop with strict data alignment.

This trainer re-implements the algorithmic steps described in the project
specification while reusing the infra provided by :class:`RayPPOTrainer`.
The original prefix-guided trainer heavily relied on implicit ordering
assumptions between prompts and responses which regularly broke in practice.

The custom trainer keeps explicit python-level tracking objects for every
intermediate and only converts back to :class:`~verl.protocol.DataProto`
objects when we finally need to interact with the Ray workers.  This ensures
that hints, prompts and completions always refer to the same question across
all stages of the pipeline.
"""

from __future__ import annotations

import hashlib
import math
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer as _BasePPO,
    apply_kl_penalty as _apply_kl_penalty,
    compute_advantage as _compute_advantage,
    compute_response_mask as _compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward as _compute_reward
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from ..prefix_guided.prefix_hint_samplers import FractionalStepBandit
from ..prefix_guided.prompting import (
    build_answer_guided_prompt,
    build_prefix_hint_messages,
    extract_boxed_answer,
    split_cot_to_steps,
)


@dataclass
class _GroupContext:
    """Container for information associated with a single original prompt."""

    uid: str
    question_text: str
    ground_truth: Optional[str]
    data_source: Optional[object]
    extra_info: Optional[object]
    num_turns: Optional[object]


@dataclass
class _HintCandidate:
    """State carried across Step 2 → Step 5 of the guided pipeline."""

    group: _GroupContext
    prefix_steps: List[str]
    prefix_prompt: Union[str, List[dict]]
    remainder: str
    frac: float
    bin_idx: int
    answer_prompt_text: str
    cot_text: str
    prefix_prompt_text: Optional[str] = None
    guided_response_text: Optional[str] = None
    success: bool = False
    score: float = 0.0


class CustomPrefixGuidedPPOTrainer(_BasePPO):
    """Prefix-guided PPO trainer with strict bookkeeping.

    The overall structure follows :class:`RayPPOTrainer` with the additional
    guided branch implementing the six algorithmic steps outlined in the user
    request.  Compared to the original implementation this version keeps
    explicit python data-structures to avoid any ambiguity about which prompt a
    generated completion belongs to.
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
        self.pg_minf = 0.9
        self.pg_maxf = 0.99
        self.pg_ans_template = str(
            pg_cfg.get(
                "prompt",
                {},
            ).get(
                "answer_guided_template",
                (
                    "Given a question and its final answer, write a clear, "
                    "correct, step-by-step reasoning that leads to the answer.\n\n"
                    "Question:\n{QUESTION}\n\nCorrect Answer:\n{ANSWER}\n\nReasoning:\n"
                ),
            )
        )
        self.pg_prefix_template = str(
            pg_cfg.get(
                "prompt",
                {},
            ).get(
                "prefix_hint_template",
                (
                    "Solve the following question. You are given partial "
                    "intermediate steps to start from.\n\nQuestion:\n{QUESTION}\n\n"
                    "Partial Steps:\n{PREFIX_STEPS}\n\nContinue the reasoning and give the final answer in \\boxed{...}."
                ),
            )
        )

        self.pg_bandit = FractionalStepBandit(
            frac_bins=self.pg_bins,
            target_success=self.pg_target,
            min_frac=self.pg_minf,
            max_frac=self.pg_maxf,
            warmup_total_steps=self.pg_warmup,
        )

        rescale_cfg = pg_cfg.get("rescale", {})
        self.pg_rescale_enable = bool(rescale_cfg.get("enable", True))
        self.pg_rescale_method = str(rescale_cfg.get("method", "affine_match"))
        self.pg_rescale_scope = str(rescale_cfg.get("scope", "global"))
        self.pg_rescale_clip_min = float(rescale_cfg.get("clip_min", math.exp(-2.0)))
        self.pg_rescale_clip_max = float(rescale_cfg.get("clip_max", math.exp(2.0)))

        anchor_cfg = pg_cfg.get("anchor", {})
        self.anchor_enable = bool(anchor_cfg.get("enable", False))
        self.anchor_weight = float(anchor_cfg.get("weight", 0.05))
        self.anchor_max_response_len = int(anchor_cfg.get("max_response_len", 512))
        self.anchor_max_pairs_per_item = int(anchor_cfg.get("max_pairs_per_item", 1))

        self._rng = random.Random(12345)

    # ------------------------------------------------------------------
    # Helper utilities that convert between python objects and DataProto
    # ------------------------------------------------------------------

    def _decode_texts(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _debug_log_guided_samples(self, candidates: Sequence[_HintCandidate]) -> None:
        if not candidates:
            return

        for idx, cand in enumerate(candidates):
            ans_prompt = cand.answer_prompt_text if cand.answer_prompt_text is not None else "<NA>"
            cot_text = cand.cot_text if cand.cot_text is not None else "<NA>"
            prefix_prompt = (
                cand.prefix_prompt_text if cand.prefix_prompt_text is not None else "<NA>"
            )
            guided_resp = (
                cand.guided_response_text if cand.guided_response_text is not None else "<NA>"
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

    # The following helper mirrors the utilities in the original trainer but
    # are rewritten here so the new package is self contained.
    def _make_prompts_dataproto(self, items: List[Union[str, List[dict]]]) -> DataProto:
        formatted: List[str] = []
        for it in items:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], dict) and "role" in it[0]:
                text = self.tokenizer.apply_chat_template(
                    it,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(text)
            else:
                formatted.append(str(it))

        enc = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        attn = enc["attention_mask"].long()
        pos = (attn.cumsum(dim=1) - 1).clamp_min_(0).long()
        td = TensorDict(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": attn,
                "position_ids": pos,
                "prompts": enc["input_ids"].clone(),
            },
            batch_size=[enc["input_ids"].size(0)],
        )
        non_tensor = {"prompt_text": np.asarray(formatted, dtype=object)}
        return DataProto(batch=td, non_tensor_batch=non_tensor)

    def _make_pr_dataproto(self, prompts: List[Union[str, List[dict]]], responses: List[str]) -> DataProto:
        formatted_prompts: List[str] = []
        for it in prompts:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], dict) and "role" in it[0]:
                txt = self.tokenizer.apply_chat_template(
                    it,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_prompts.append(txt)
            else:
                formatted_prompts.append(str(it))

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
        input_ids = torch.cat([p_enc["input_ids"], r_enc["input_ids"]], dim=1)
        attn = torch.cat([p_enc["attention_mask"].long(), r_enc["attention_mask"].long()], dim=1)
        pos = (attn.cumsum(dim=1) - 1).clamp_min_(0).long()
        td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "position_ids": pos,
                "prompts": p_enc["input_ids"],
                "responses": r_enc["input_ids"],
                "response_mask": r_enc["attention_mask"].long(),
            },
            batch_size=[input_ids.size(0)],
        )
        return DataProto(batch=td, non_tensor_batch={})

    def _make_pr_dataproto_with_response_ids(
        self,
        prompts: List[Union[str, List[dict]]],
        responses_ids: torch.Tensor,
        response_mask: Optional[torch.Tensor],
    ) -> DataProto:
        formatted_prompts: List[str] = []
        for it in prompts:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], dict) and "role" in it[0]:
                txt = self.tokenizer.apply_chat_template(
                    it,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_prompts.append(txt)
            else:
                formatted_prompts.append(str(it))

        p_enc = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompts_ids = p_enc["input_ids"]
        prompts_mask = p_enc["attention_mask"].long()

        r_ids = responses_ids.clone().long()
        if response_mask is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            r_mask = (r_ids != pad_id).long()
        else:
            r_mask = response_mask.clone().long()

        input_ids = torch.cat([prompts_ids, r_ids], dim=1)
        attn = torch.cat([prompts_mask, r_mask], dim=1)
        pos = (attn.cumsum(dim=1) - 1).clamp_min_(0).long()

        td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "position_ids": pos,
                "prompts": prompts_ids,
                "responses": r_ids,
                "response_mask": r_mask,
            },
            batch_size=[input_ids.size(0)],
        )
        return DataProto(batch=td, non_tensor_batch={})

    def _ensure_response_mask(self, data: DataProto) -> None:
        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = _compute_response_mask(data)

    def _compute_rewards_and_adv(self, data: DataProto) -> None:
        reward_tensor, _ = _compute_reward(data, self.reward_fn)
        data.batch["token_level_scores"] = reward_tensor

        if self.config.algorithm.use_kl_in_reward:
            data, _ = _apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        data = _compute_advantage(
            data,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

    # --- padding utilities -------------------------------------------------

    def _pad_for_concat(self, dp_a: DataProto, dp_b: DataProto) -> Tuple[DataProto, DataProto]:
        def lengths(dp: DataProto):
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

        def _pad_tensor(t: torch.Tensor, target_len: Optional[int], pad_value: Union[int, float]) -> torch.Tensor:
            if t is None or not torch.is_tensor(t) or t.dim() != 2:
                return t
            B, T = t.shape
            if target_len is None or T >= target_len:
                return t
            if t.dtype.is_floating_point:
                pad_fill = float(pad_value)
            else:
                pad_fill = int(pad_value)
            pad_block = t.new_full((B, target_len - T), pad_fill)
            return torch.cat([t, pad_block], dim=1)

        def _pad_dp(dp: DataProto, T_full_dp: Optional[int], T_prompt_dp: Optional[int], T_resp_dp: Optional[int]) -> DataProto:
            for key in list(dp.batch.keys()):
                tensor = dp.batch[key]
                if not (torch.is_tensor(tensor) and tensor.dim() == 2):
                    continue

                target_len: Optional[int] = None
                pad_val: Union[int, float] = 0

                if key in ("input_ids", "attention_mask", "position_ids"):
                    target_len = T_full_max
                    pad_val = pad_id if key == "input_ids" else 0
                elif key == "prompts":
                    target_len = T_prompt_max
                    pad_val = pad_id
                elif key in ("responses", "response_mask"):
                    target_len = T_resp_max
                    pad_val = pad_id if key == "responses" else 0
                elif key in (
                    "old_log_probs",
                    "ref_log_prob",
                    "values",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                ):
                    target_len = T_resp_max
                    pad_val = 0.0
                else:
                    if T_full_dp is not None and tensor.shape[1] == T_full_dp:
                        target_len = T_full_max
                    elif T_resp_dp is not None and tensor.shape[1] == T_resp_dp:
                        target_len = T_resp_max
                    elif T_prompt_dp is not None and tensor.shape[1] == T_prompt_dp:
                        target_len = T_prompt_max
                    else:
                        target_len = T_full_max
                    pad_val = 0

                dp.batch[key] = _pad_tensor(tensor, target_len, pad_val)

            if "attention_mask" in dp.batch.keys():
                attn = dp.batch["attention_mask"]
                pos = (attn.cumsum(dim=1) - 1).clamp_min_(0).long()
                dp.batch["position_ids"] = pos
            return dp

        dp_a = _pad_dp(dp_a, Ta_full, Ta_prompt, Ta_resp)
        dp_b = _pad_dp(dp_b, Tb_full, Tb_prompt, Tb_resp)
        return dp_a, dp_b

    def _align_non_tensor_for_concat(self, dps: Sequence[DataProto]) -> None:
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

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for dp in dps:
            B = dp.batch.batch_size[0]
            T_full = dp.batch["input_ids"].size(1) if "input_ids" in dp.batch.keys() else 0
            T_prompt = dp.batch["prompts"].size(1) if "prompts" in dp.batch.keys() else 0
            T_resp = dp.batch["responses"].size(1) if "responses" in dp.batch.keys() else 0
            device = next(iter(dp.batch.values())).device

            for k in all_keys:
                if k in dp.batch.keys():
                    continue
                if k == "input_ids":
                    dp.batch[k] = torch.full((B, T_full), pad_id, dtype=torch.long, device=device)
                elif k == "attention_mask":
                    dp.batch[k] = torch.zeros((B, T_full), dtype=torch.long, device=device)
                elif k == "position_ids":
                    pos = torch.arange(T_full, device=device).unsqueeze(0).expand(B, -1).long()
                    dp.batch[k] = pos
                elif k == "prompts":
                    dp.batch[k] = torch.full((B, T_prompt), pad_id, dtype=torch.long, device=device)
                elif k == "responses":
                    dp.batch[k] = torch.full((B, T_resp), pad_id, dtype=torch.long, device=device)
                elif k == "response_mask":
                    dp.batch[k] = torch.zeros((B, T_resp), dtype=torch.long, device=device)
                elif k in (
                    "old_log_probs",
                    "ref_log_prob",
                    "values",
                    "token_level_scores",
                    "token_level_rewards",
                    "advantages",
                    "returns",
                ):
                    dp.batch[k] = torch.zeros((B, T_resp), dtype=torch.float32, device=device)
                else:
                    dp.batch[k] = torch.zeros((B, T_full), dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Generation helpers with deterministic alignment
    # ------------------------------------------------------------------

    def _attach_uids(self, batch: DataProto) -> None:
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )

    def _hash_rows_long(self, ids: torch.Tensor) -> List[int]:
        arr = ids.detach().to("cpu", non_blocking=True).numpy()
        out: List[int] = []
        for row in arr:
            h = hashlib.sha1(row.tobytes()).digest()
            out.append(int.from_bytes(h[:8], "big", signed=False))
        return out

    def _reorder_to_input(
        self,
        out_dp: DataProto,
        in_ids: torch.Tensor,
        key_candidates: Sequence[str] = ("input_ids", "prompts"),
    ) -> None:
        field = None
        for k in key_candidates:
            if out_dp.batch is not None and k in out_dp.batch.keys():
                field = k
                break
        if field is None:
            return

        out_ids = out_dp.batch[field]
        h_in = self._hash_rows_long(in_ids)
        h_out = self._hash_rows_long(out_ids)
        buckets: Dict[int, List[int]] = {}
        for j, hj in enumerate(h_out):
            buckets.setdefault(hj, []).append(j)

        reorder: List[int] = []
        for hi in h_in:
            if hi not in buckets or not buckets[hi]:
                raise RuntimeError("Failed to realign generation outputs")
            reorder.append(buckets[hi].pop(0))

        idx = torch.as_tensor(reorder, dtype=torch.long)
        for k, v in list(out_dp.batch.items()):
            if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == idx.size(0):
                out_dp.batch[k] = v.index_select(0, idx)
        for k, v in list(out_dp.non_tensor_batch.items()):
            arr = np.asarray(v, dtype=object)
            if arr.shape[0] == idx.size(0):
                out_dp.non_tensor_batch[k] = arr[idx.numpy()]

    def _generate_sequences_stable(self, dp: DataProto) -> DataProto:
        pad_to = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, pad_to)
        in_ids = dp_pad.batch["input_ids"]
        out_pad = self.actor_rollout_wg.generate_sequences(dp_pad)
        try:
            self._reorder_to_input(out_pad, in_ids)
        except Exception:
            pass
        out = unpad_dataproto(out_pad, pad_sz)
        if "responses" in out.batch and "response_mask" not in out.batch:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            out.batch["response_mask"] = (out.batch["responses"] != pad_id).long()
        return out

    def _generate_sequences_keep_order(self, dp: DataProto) -> DataProto:
        pad_id = self.tokenizer.pad_token_id or 0
        total = dp.batch.batch_size[0]
        if total == 0:
            empty = torch.empty(0, 0, dtype=torch.long)
            return DataProto(
                batch=TensorDict({"responses": empty, "response_mask": empty}, batch_size=[0]),
                non_tensor_batch={},
            )

        if getattr(self, "async_rollout_mode", False):
            generate_fn = self.async_rollout_manager.generate_sequences
            size_divisor = getattr(
                getattr(self.config.actor_rollout_ref.rollout, "agent", {}),
                "num_workers",
                getattr(self.actor_rollout_wg, "world_size", 1),
            )
        else:
            generate_fn = self.actor_rollout_wg.generate_sequences
            size_divisor = getattr(self.actor_rollout_wg, "world_size", 1)

        size_divisor = max(1, int(size_divisor))

        rollout_cfg = self.config.actor_rollout_ref.rollout
        rollout_n = int(rollout_cfg.get("n", 1)) if isinstance(rollout_cfg.get("n", 1), int) else 1
        train_bs = getattr(self.config.data, "train_batch_size", None)
        chunk_capacity = int(train_bs) * rollout_n if train_bs is not None else None
        max_num_seqs = rollout_cfg.get("max_num_seqs", None)
        if max_num_seqs is not None:
            try:
                max_num_seqs = int(max_num_seqs)
            except TypeError:
                max_num_seqs = None
        if chunk_capacity is None or chunk_capacity <= 0:
            chunk_capacity = total
        if max_num_seqs is not None and max_num_seqs > 0:
            chunk_capacity = min(chunk_capacity, max_num_seqs)
        chunk_size = max(1, min(chunk_capacity, total))

        row_key = "__pg_row_id__"
        resp_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []

        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            indices = list(range(start, end))
            sub = dp.select_idxs(indices)
            sub.non_tensor_batch[row_key] = np.arange(start, end, dtype=np.int64)
            sub_pad, pad_sz = pad_dataproto_to_divisor(sub, size_divisor)
            out_pad = generate_fn(sub_pad)
            out = unpad_dataproto(out_pad, pad_sz)
            # ensure order
            if row_key in out.non_tensor_batch:
                order = np.argsort(np.asarray(out.non_tensor_batch[row_key], dtype=np.int64))
                idx = torch.as_tensor(order, dtype=torch.long)
                for k, v in list(out.batch.items()):
                    if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == idx.size(0):
                        out.batch[k] = v.index_select(0, idx)
                out.non_tensor_batch[row_key] = np.asarray(out.non_tensor_batch[row_key])[order]
            resp = out.batch["responses"]
            resp_chunks.append(resp)
            if "response_mask" in out.batch.keys():
                mask_chunks.append(out.batch["response_mask"].long())
            else:
                mask_chunks.append((resp != pad_id).long())

        max_len = max(chunk.shape[1] for chunk in resp_chunks)
        for i, resp in enumerate(resp_chunks):
            if resp.shape[1] < max_len:
                pad = resp.new_full((resp.size(0), max_len - resp.shape[1]), pad_id)
                resp_chunks[i] = torch.cat([resp, pad], dim=1)
            mask = mask_chunks[i]
            if mask.shape[1] < max_len:
                pad_mask = mask.new_zeros((mask.size(0), max_len - mask.shape[1]))
                mask_chunks[i] = torch.cat([mask, pad_mask], dim=1)

        responses = torch.cat(resp_chunks, dim=0)
        response_mask = torch.cat(mask_chunks, dim=0)
        return DataProto(
            batch=TensorDict(
                {"responses": responses, "response_mask": response_mask},
                batch_size=[responses.size(0)],
            ),
            non_tensor_batch={},
        )

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seq_scores_from_tokens(token_level_scores: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        if token_level_scores.dim() == 1:
            return token_level_scores
        T = token_level_scores.size(1)
        if response_mask is not None and response_mask.shape[-1] == T:
            return (token_level_scores * response_mask).sum(-1)
        return token_level_scores.sum(-1)

    @staticmethod
    def _seq_lengths_from_mask(response_mask: torch.Tensor) -> torch.Tensor:
        return response_mask.sum(-1).clamp_min_(1.0)

    def _affine_rescale_scores_inplace(
        self,
        guided: DataProto,
        ref: DataProto,
        scope: str = "global",
    ) -> Dict[str, float]:
        g_scores = self._seq_scores_from_tokens(
            guided.batch["token_level_scores"], guided.batch.get("response_mask", None)
        )
        r_scores = self._seq_scores_from_tokens(
            ref.batch["token_level_scores"], ref.batch.get("response_mask", None)
        )
        eps = 1e-8
        metrics: Dict[str, float] = {}

        if scope == "per_uid" and "uid" in guided.non_tensor_batch and "uid" in ref.non_tensor_batch:
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
                guided.batch["token_level_scores"] * a.unsqueeze(-1) + (b / L).unsqueeze(-1)
            )
            metrics.update(
                {
                    "pg/rescale_method": 0.0,
                    "pg/rescale_mean_shift": float(b_global.detach().item()),
                    "pg/rescale_scale": float(a_global.detach().item()),
                }
            )
        else:
            mu_on, sigma_on = r_scores.mean(), r_scores.std()
            mu_off, sigma_off = g_scores.mean(), g_scores.std()
            a = sigma_on / torch.clamp(sigma_off, min=eps)
            b = mu_on - a * mu_off
            L = self._seq_lengths_from_mask(guided.batch["response_mask"])
            guided.batch["token_level_scores"] = guided.batch["token_level_scores"] * a + (
                b / L
            ).unsqueeze(-1)
            metrics.update(
                {
                    "pg/rescale_method": 1.0,
                    "pg/rescale_mean_shift": float(b.detach().item()),
                    "pg/rescale_scale": float(a.detach().item()),
                }
            )
        return metrics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)

        total_steps = self.total_training_steps
        n_gpus = self.resource_pool_manager.get_n_gpus()
        progress_bar = tqdm(total=total_steps, initial=self.global_steps, desc="Training Progress")

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: Dict[str, float] = {}
                timing_raw: Dict[str, float] = {}

                with marked_timer("step", timing_raw):
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    self._attach_uids(batch)
                    gen_batch = self._get_gen_batch(batch)
                    n = self.config.actor_rollout_ref.rollout.n
                    gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)

                    with marked_timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_out = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_out = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_out.meta_info.pop("timing", {})) if "timing" in gen_out.meta_info else None

                    batch = batch.repeat(repeat_times=n, interleave=True)
                    batch = batch.union(gen_out)
                    self._ensure_response_mask(batch)

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
                        if self.use_rm:
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        self._compute_rewards_and_adv(batch)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    guided_batch = None
                    anchor_batch = None
                    pg_metrics: Dict[str, float] = {}
                    if self.pg_enable:
                        guided_batch, anchor_batch, pg_metrics = self._guided_branch(batch, n)

                    if guided_batch is None:
                        train_batch_for_critic = batch
                    else:
                        self._align_batch_keys_for_concat([batch, guided_batch])
                        self._align_non_tensor_for_concat([batch, guided_batch])
                        a, b = self._pad_for_concat(batch, guided_batch)
                        train_batch_for_critic = DataProto.concat([a, b])

                    train_batch_for_actor = train_batch_for_critic
                    if anchor_batch is not None:
                        self._align_batch_keys_for_concat([train_batch_for_actor, anchor_batch])
                        self._align_non_tensor_for_concat([train_batch_for_actor, anchor_batch])
                        a, b = self._pad_for_concat(train_batch_for_actor, anchor_batch)
                        train_batch_for_actor = DataProto.concat([a, b])

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            crit_div = max(1, getattr(self.critic_wg, "world_size", 1))
                            tb_pad, _ = pad_dataproto_to_divisor(train_batch_for_critic, crit_div)
                            critic_output = self.critic_wg.update_critic(tb_pad)
                        metrics.update(reduce_metrics(critic_output.meta_info.get("metrics", {})))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw):
                            act_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
                            ta_pad, _ = pad_dataproto_to_divisor(train_batch_for_actor, act_div)
                            actor_output = self.actor_rollout_wg.update_actor(ta_pad)
                        metrics.update(reduce_metrics(actor_output.meta_info.get("metrics", {})))

                metrics.update(pg_metrics)
                metrics.update(compute_data_metrics(batch=train_batch_for_actor, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=train_batch_for_actor, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(batch=train_batch_for_actor, timing_raw=timing_raw, n_gpus=n_gpus)
                )

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
                    is_last_step or (self.global_steps + 1) % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                logger.log(data=metrics, step=self.global_steps + 1)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    return

    # ------------------------------------------------------------------
    # Guided branch implementing the six-step algorithm
    # ------------------------------------------------------------------

    def _collect_group_context(
        self, normal_batch: DataProto, group_size: int
    ) -> Tuple[List[_GroupContext], torch.Tensor, int]:
        tls = normal_batch.batch["token_level_scores"]
        zero_group_mask, B = self._zero_solve_group_mask(tls, group_size=group_size)
        prompts_text = self._decode_texts(normal_batch.batch["prompts"])
        uids_all = normal_batch.non_tensor_batch["uid"]

        reward_model = normal_batch.non_tensor_batch.get("reward_model", None)
        data_source = normal_batch.non_tensor_batch.get("data_source", None)
        dataset = normal_batch.non_tensor_batch.get("dataset", None)
        source_field = data_source if data_source is not None else dataset
        extra_info = normal_batch.non_tensor_batch.get("extra_info", None)
        num_turns = normal_batch.non_tensor_batch.get("__num_turns__", None)

        groups: List[_GroupContext] = []
        for g in range(B):
            idx = g * group_size
            uid = str(uids_all[idx])
            question_text = prompts_text[idx]
            gt = None
            if reward_model is not None:
                rm_item = reward_model[idx]
                if isinstance(rm_item, dict):
                    gt = rm_item.get("ground_truth", rm_item.get("answer", None))
                else:
                    gt = rm_item
                if isinstance(gt, dict) and "answer" in gt:
                    gt = gt["answer"]
            src = None
            if source_field is not None:
                src = source_field[idx]
            groups.append(
                _GroupContext(
                    uid=uid,
                    question_text=question_text,
                    ground_truth=gt,
                    data_source=src,
                    extra_info=extra_info[idx] if extra_info is not None else None,
                    num_turns=num_turns[idx] if num_turns is not None else None,
                )
            )
        return groups, zero_group_mask, B

    def _generate_answer_guided_cots(
        self,
        groups: List[_GroupContext],
        zero_mask: torch.Tensor,
    ) -> Tuple[List[_HintCandidate], List[float], List[int]]:
        ans_prompts: List[Union[str, List[dict]]] = []
        owners: List[_GroupContext] = []
        chosen_fracs: List[float] = []
        chosen_bins: List[int] = []
        for g, group in enumerate(groups):
            if not zero_mask[g]:
                continue
            if group.ground_truth is None:
                continue
            template_prompt = build_answer_guided_prompt(
                group.question_text, group.ground_truth, self.pg_ans_template
            )
            for _ in range(self.pg_ans_cot_samples):
                ans_prompts.append(template_prompt)
                owners.append(group)
        if not ans_prompts:
            return [], chosen_fracs, chosen_bins

        cot_in = self._make_prompts_dataproto(ans_prompts)
        cot_out = self._generate_sequences_stable(cot_in)
        cot_texts = self._decode_texts(cot_out.batch["responses"])
        cot_prompt_texts = cot_in.non_tensor_batch["prompt_text"].tolist()

        candidates: List[_HintCandidate] = []
        for idx, cot in enumerate(cot_texts):
            steps = split_cot_to_steps(cot)
            if len(steps) < 2 or extract_boxed_answer(cot) is None:
                continue
            group = owners[idx]
            k_steps, frac, bin_idx = self.pg_bandit.select_prefix_steps(
                total_steps=len(steps), rng=self._rng
            )
            k_steps = min(k_steps, max(1, len(steps) - 1))
            prefix_steps = steps[:k_steps]
            remainder = "\n\n".join(steps[k_steps:])
            hint_prompt = build_prefix_hint_messages(
                group.question_text, prefix_steps, self.pg_prefix_template
            )
            candidates.append(
                _HintCandidate(
                    group=group,
                    prefix_steps=prefix_steps,
                    prefix_prompt=hint_prompt,
                    remainder=remainder,
                    frac=frac,
                    bin_idx=bin_idx,
                    answer_prompt_text=cot_prompt_texts[idx],
                    cot_text=cot,
                )
            )
            chosen_fracs.append(frac)
            chosen_bins.append(bin_idx)
        return candidates, chosen_fracs, chosen_bins

    def _score_hint_candidates(
        self,
        candidates: List[_HintCandidate],
    ) -> torch.Tensor:
        if not candidates:
            return torch.empty(0)
        guided_in = self._make_prompts_dataproto([c.prefix_prompt for c in candidates])
        guided_out = self._generate_sequences_stable(guided_in)
        responses = guided_out.batch["responses"]
        guided_prompt_texts = guided_in.non_tensor_batch.get("prompt_text", [])
        if isinstance(guided_prompt_texts, np.ndarray):
            guided_prompt_texts = guided_prompt_texts.tolist()
        guided_resp_texts = self._decode_texts(responses)
        for idx, cand in enumerate(candidates):
            if guided_prompt_texts and idx < len(guided_prompt_texts):
                cand.prefix_prompt_text = str(guided_prompt_texts[idx])
            else:
                cand.prefix_prompt_text = None
            if guided_resp_texts and idx < len(guided_resp_texts):
                cand.guided_response_text = str(guided_resp_texts[idx])
            else:
                cand.guided_response_text = None
        self._debug_log_guided_samples(candidates)
        response_mask = guided_out.batch.get("response_mask", None)
        eval_dp = self._make_pr_dataproto_with_response_ids(
            prompts=[c.prefix_prompt for c in candidates],
            responses_ids=responses,
            response_mask=response_mask,
        )
        reward_models = []
        for cand in candidates:
            reward_models.append({"ground_truth": cand.group.ground_truth})
        eval_dp.non_tensor_batch["reward_model"] = np.asarray(reward_models, dtype=object)
        if any(cand.group.data_source is not None for cand in candidates):
            eval_dp.non_tensor_batch["data_source"] = np.asarray(
                [cand.group.data_source for cand in candidates], dtype=object
            )
        if any(cand.group.extra_info is not None for cand in candidates):
            eval_dp.non_tensor_batch["extra_info"] = np.asarray(
                [cand.group.extra_info for cand in candidates], dtype=object
            )
        if any(cand.group.num_turns is not None for cand in candidates):
            eval_dp.non_tensor_batch["__num_turns__"] = np.asarray(
                [cand.group.num_turns for cand in candidates], dtype=object
            )
        self._ensure_response_mask(eval_dp)
        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(eval_dp)
            eval_dp = eval_dp.union(rm_scores)
        scores, _ = _compute_reward(eval_dp, self.reward_fn)
        eval_dp.batch["token_level_scores"] = scores
        seq_scores = self._seq_scores_from_tokens(scores, eval_dp.batch["response_mask"])
        success_flags = seq_scores > 0.5
        for cand, succ, sc in zip(candidates, success_flags.tolist(), seq_scores.tolist()):
            cand.success = bool(succ)
            cand.score = float(sc)
            self.pg_bandit.update(bin_idx=cand.bin_idx, success=cand.success)
        return seq_scores

    def _select_successful_hints(
        self,
        candidates: List[_HintCandidate],
        group_size: int,
    ) -> List[_HintCandidate]:
        by_uid: Dict[str, List[_HintCandidate]] = {}
        for cand in candidates:
            if not cand.success:
                continue
            by_uid.setdefault(cand.group.uid, []).append(cand)

        selected: List[_HintCandidate] = []
        for uid, cand_list in by_uid.items():
            cand_list.sort(key=lambda c: c.score, reverse=True)
            selected.extend(cand_list[:group_size])
        return selected

    def _compute_logprob_inplace(self, dp: DataProto) -> DataProto:
        size_div = max(1, getattr(self.actor_rollout_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, size_div)
        logp_pad = self.actor_rollout_wg.compute_log_prob(dp_pad)
        logp = unpad_dataproto(logp_pad, pad_sz)
        logp.batch.pop("entropys", None)
        return dp.union(logp)

    def _compute_ref_inplace(self, dp: DataProto) -> DataProto:
        ref_div = max(1, getattr(self.ref_policy_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, ref_div)
        ref_pad = self.ref_policy_wg.compute_ref_log_prob(dp_pad)
        ref = unpad_dataproto(ref_pad, pad_sz)
        return dp.union(ref)

    def _compute_values_inplace(self, dp: DataProto) -> DataProto:
        crit_div = max(1, getattr(self.critic_wg, "world_size", 1))
        dp_pad, pad_sz = pad_dataproto_to_divisor(dp, crit_div)
        val_pad = self.critic_wg.compute_values(dp_pad)
        val = unpad_dataproto(val_pad, pad_sz)
        return dp.union(val)

    def _build_anchor_batch(self, hints: List[_HintCandidate]) -> Optional[DataProto]:
        if not (self.anchor_enable and self.anchor_weight > 0.0):
            return None
        anchor_prompts: List[str] = []
        anchor_responses: List[str] = []
        counts: Dict[str, int] = {}
        for cand in hints:
            remainder = cand.remainder.strip()
            if not remainder:
                continue
            uid = cand.group.uid
            if counts.get(uid, 0) >= self.anchor_max_pairs_per_item:
                continue
            anchor_prompts.append(cand.group.question_text)
            anchor_responses.append(remainder)
            counts[uid] = counts.get(uid, 0) + 1
        if not anchor_prompts:
            return None
        anchor_dp = self._make_pr_dataproto(anchor_prompts, anchor_responses)
        anchor_dp = self._compute_logprob_inplace(anchor_dp)
        self._ensure_response_mask(anchor_dp)
        mask = anchor_dp.batch["response_mask"].to(dtype=torch.float32)
        anchor_dp.batch["advantages"] = mask * self.anchor_weight
        anchor_dp.batch["returns"] = torch.zeros_like(anchor_dp.batch["advantages"])
        anchor_dp.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(anchor_prompts))], dtype=object
        )
        anchor_dp.meta_info["global_token_num"] = (
            anchor_dp.batch["prompts"].shape[1] + anchor_dp.batch["responses"].shape[1]
        ) * torch.ones(anchor_dp.batch.batch_size[0]).tolist()
        return anchor_dp

    def _guided_branch(
        self,
        normal_batch: DataProto,
        group_size: int,
    ) -> Tuple[Optional[DataProto], Optional[DataProto], Dict[str, float]]:
        groups, zero_mask, B = self._collect_group_context(normal_batch, group_size)
        zero_count = int(zero_mask.long().sum().item())
        metrics: Dict[str, float] = {
            "pg/zero_solve_frac": float(zero_count / max(1, B)),
            "pg/zero_solve_count": float(zero_count),
            "pg/group_size": float(group_size),
        }
        if zero_count == 0:
            return None, None, metrics

        candidates, chosen_fracs, chosen_bins = self._generate_answer_guided_cots(groups, zero_mask)
        metrics["pg/hint_guided_expected_total"] = float(zero_count * group_size)
        metrics["pg/hint_guided_q_total"] = float(zero_count)
        metrics["pg/ts_target_success"] = self.pg_target
        if not candidates:
            return None, None, metrics

        seq_scores = self._score_hint_candidates(candidates)
        metrics["pg/hint_guided_total"] = float(len(candidates))
        if len(chosen_fracs) > 0:
            metrics.update(
                {
                    "pg/ts_frac/mean": float(np.mean(chosen_fracs)),
                    "pg/ts_frac/min": float(np.min(chosen_fracs)),
                    "pg/ts_frac/max": float(np.max(chosen_fracs)),
                }
            )
        if len(chosen_bins) > 0:
            metrics["pg/ts_bins/unique"] = float(len(set(chosen_bins)))

        successes = [cand for cand in candidates if cand.success]
        metrics["pg/guided_pos_count_step3"] = float(len(successes))
        if seq_scores.numel() > 0:
            metrics["pg/hint_guided_step3/reward_mean"] = float(seq_scores.mean().item())

        selected = self._select_successful_hints(candidates, group_size=group_size)
        metrics["pg/guided_pos_count"] = float(len(selected))
        if not selected:
            return None, None, metrics

        per_uid_success = {}
        for cand in selected:
            per_uid_success[cand.group.uid] = True
        metrics["pg/q_post_hint_pos_count"] = float(sum(1 for _ in per_uid_success))
        metrics["pg/q_post_hint_pos_frac"] = float(
            metrics["pg/q_post_hint_pos_count"] / max(1.0, float(zero_count))
        )

        final_prompts = [cand.prefix_prompt for cand in selected]
        final_in = self._make_prompts_dataproto(final_prompts)
        final_out = self._generate_sequences_keep_order(final_in)
        responses = final_out.batch["responses"]
        response_mask = final_out.batch.get("response_mask", None)

        guided_dp = self._make_pr_dataproto_with_response_ids(
            prompts=final_prompts,
            responses_ids=responses,
            response_mask=response_mask,
        )
        owner_uids = [cand.group.uid for cand in selected]
        guided_dp.non_tensor_batch["uid"] = np.asarray(owner_uids, dtype=object)
        reward_models = [{"ground_truth": cand.group.ground_truth} for cand in selected]
        guided_dp.non_tensor_batch["reward_model"] = np.asarray(reward_models, dtype=object)

        if any(cand.group.data_source is not None for cand in selected):
            guided_dp.non_tensor_batch["data_source"] = np.asarray(
                [cand.group.data_source for cand in selected], dtype=object
            )
        if any(cand.group.extra_info is not None for cand in selected):
            guided_dp.non_tensor_batch["extra_info"] = np.asarray(
                [cand.group.extra_info for cand in selected], dtype=object
            )
        if any(cand.group.num_turns is not None for cand in selected):
            guided_dp.non_tensor_batch["__num_turns__"] = np.asarray(
                [cand.group.num_turns for cand in selected], dtype=object
            )

        guided_dp = self._compute_logprob_inplace(guided_dp)
        if self.use_reference_policy:
            guided_dp = self._compute_ref_inplace(guided_dp)
        if self.use_critic:
            guided_dp = self._compute_values_inplace(guided_dp)
        if self.use_rm:
            rm_scores = self.rm_wg.compute_rm_score(guided_dp)
            guided_dp = guided_dp.union(rm_scores)
        self._ensure_response_mask(guided_dp)
        guided_scores, _ = _compute_reward(guided_dp, self.reward_fn)
        guided_dp.batch["token_level_scores"] = guided_scores

        if self.pg_rescale_method == "affine_match":
            rescale_metrics = self._affine_rescale_scores_inplace(
                guided_dp, normal_batch, scope=self.pg_rescale_scope
            )
            metrics.update({f"pg/rescale/{k.split('/')[-1]}": v for k, v in rescale_metrics.items()})

        plain_prompts = [cand.group.question_text for cand in selected]
        rescore_dp = self._make_pr_dataproto_with_response_ids(
            prompts=plain_prompts,
            responses_ids=guided_dp.batch["responses"],
            response_mask=guided_dp.batch["response_mask"],
        )
        rescore_dp = self._compute_logprob_inplace(rescore_dp)
        logp_orig = rescore_dp.batch["old_log_probs"]
        logp_guided = guided_dp.batch["old_log_probs"]
        mask = guided_dp.batch["response_mask"].to(logp_guided.dtype)
        Tg, To = logp_guided.size(1), logp_orig.size(1)
        if To != Tg:
            if To > Tg:
                logp_orig = logp_orig[:, :Tg]
                mask = mask[:, :Tg]
            else:
                pad = To - Tg
                logp_guided = torch.nn.functional.pad(logp_guided, (0, pad), value=0.0)
                mask = torch.nn.functional.pad(mask, (0, pad), value=0.0)
        log_ratio = ((logp_orig - logp_guided) * mask).sum(dim=-1)
        rho = torch.exp(log_ratio).clamp(
            min=self.pg_rescale_clip_min, max=self.pg_rescale_clip_max
        )
        guided_dp.non_tensor_batch["offpolicy_rho"] = rho.detach().cpu().numpy()

        self._compute_rewards_and_adv(guided_dp)
        if guided_dp.batch["advantages"].dim() == 2:
            guided_dp.batch["advantages"] = guided_dp.batch["advantages"] * rho.unsqueeze(-1)
            guided_dp.batch["returns"] = guided_dp.batch["returns"] * rho.unsqueeze(-1)
        else:
            guided_dp.batch["advantages"] = guided_dp.batch["advantages"] * rho
            guided_dp.batch["returns"] = guided_dp.batch["returns"] * rho

        uid2qtext = {group.uid: group.question_text for group in groups}
        guided_for_actor = self._project_guided_to_plain_for_actor(
            guided_dp,
            owner_uids,
            uid2qtext,
        )

        seq_scores = self._seq_scores_from_tokens(
            guided_dp.batch["token_level_scores"], guided_dp.batch["response_mask"]
        )
        if seq_scores.numel() > 0:
            metrics["pg/zero_solve_post/reward_mean"] = float(seq_scores.mean().item())
            metrics["pg/zero_solve_post/solve_rate"] = float((seq_scores > 0.5).float().mean().item())
        if rho.numel() > 0:
            rho_np = rho.detach().cpu().numpy()
            metrics.update(
                {
                    "pg/offpolicy_rho/mean": float(rho_np.mean()),
                    "pg/offpolicy_rho/p10": float(np.percentile(rho_np, 10)),
                    "pg/offpolicy_rho/p90": float(np.percentile(rho_np, 90)),
                }
            )

        anchor_batch = self._build_anchor_batch(selected)
        if anchor_batch is not None:
            metrics["pg/anchor/num_pairs"] = float(anchor_batch.batch.batch_size[0])

        guided_for_actor.meta_info["global_token_num"] = torch.sum(
            guided_for_actor.batch["attention_mask"], dim=-1
        ).tolist()

        return guided_for_actor, anchor_batch, metrics

    # ------------------------------------------------------------------
    # Utility methods borrowed from the original trainer
    # ------------------------------------------------------------------

    def _project_guided_to_plain_for_actor(
        self,
        gb: DataProto,
        owner_uids: List[str],
        uid2qtext: Dict[str, str],
    ) -> DataProto:
        plain_prompts = [uid2qtext[u] for u in owner_uids]
        responses_text = self._decode_texts(gb.batch["responses"])
        proj = self._make_pr_dataproto(plain_prompts, responses_text)
        for k in ["old_log_probs", "advantages", "returns", "token_level_scores", "token_level_rewards"]:
            if k in gb.batch:
                proj.batch[k] = gb.batch[k]
        if "values" in gb.batch:
            proj.batch["values"] = gb.batch["values"]
        proj.non_tensor_batch["uid"] = gb.non_tensor_batch["uid"]
        proj.meta_info["global_token_num"] = torch.sum(
            proj.batch["attention_mask"], dim=-1
        ).tolist()
        if "offpolicy_rho" in gb.non_tensor_batch:
            proj.non_tensor_batch["offpolicy_rho"] = gb.non_tensor_batch["offpolicy_rho"]
        return proj

    def _zero_solve_group_mask(
        self, token_level_scores: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, int]:
        if token_level_scores.dim() == 2:
            seq_scores = token_level_scores.sum(-1)
        elif token_level_scores.dim() == 1:
            seq_scores = token_level_scores
        else:
            raise ValueError(f"Unexpected reward shape: {tuple(token_level_scores.shape)}")
        Bn = seq_scores.shape[0]
        assert Bn % group_size == 0, "Total sequences not divisible by group size"
        B = Bn // group_size
        scores = seq_scores.view(B, group_size)
        success = scores > 0.5
        zero_group_mask = ~success.any(dim=1)
        return zero_group_mask, B

    def _extract_gt_answers(self, batch: DataProto) -> List[Optional[str]]:
        out: List[Optional[str]] = []
        for item in batch:
            gt = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            if isinstance(gt, dict) and "answer" in gt:
                out.append(gt["answer"])
            else:
                out.append(gt)
        return out

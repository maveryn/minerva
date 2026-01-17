# verl/trainer/ppo/ray_trainer.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pprint import pprint
from typing import Any, Optional, Tuple

import numpy as np
import ray
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.reward_manager.naive import NaiveRewardManager

ZERO_SOLVE_THRESH = 1e-3
ALL_SOLVE_THRESH = 0.95


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


class _RefJudgeRunner:
    def __init__(self, ref_worker_group: RayWorkerGroup):
        self._ref_worker_group = ref_worker_group

    def generate(self, prompts: list[str], *, max_new_tokens: int, temperature: float, top_p: float, batch_size: int):
        outputs = self._ref_worker_group.generate_judge_outputs(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            return outputs[0]
        if isinstance(outputs, list):
            return outputs
        return []


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # ``uid`` indexes may be absent if the dataloader does not provide them.
        # Fallback to using a unique index for each sample so that the GRPO
        # advantage computation still functions without grouping.
        index = data.non_tensor_batch.get("uid")
        if index is None:
            index = np.arange(grpo_calculation_mask.shape[0])

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self._expanduser_in_config(self.config)
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._init_distill_state()
        self._init_acr_state()
        self._best_val_metric = None
        self._best_val_step = None

    def _expanduser_in_config(self, cfg):
        """Recursively expand user home (~) in string paths within the config."""

        if isinstance(cfg, DictConfig):
            with open_dict(cfg):
                for key, value in cfg.items():
                    cfg[key] = self._expanduser_in_config(value)
            return cfg

        if isinstance(cfg, ListConfig):
            for idx, value in enumerate(cfg):
                cfg[idx] = self._expanduser_in_config(value)
            return cfg

        if isinstance(cfg, str) and cfg.startswith("~"):
            return os.path.expanduser(cfg)

        return cfg

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, *, val_dataloader=None, metric_prefix: str = "val", skip_global_avg: bool = False):
        val_debug = os.getenv("VERL_VAL_DEBUG", "0").lower() in {"1", "true", "yes"}
        val_log_every = int(os.getenv("VERL_VAL_LOG_EVERY", "10"))
        val_max_batches = int(os.getenv("VERL_VAL_MAX_BATCHES", "0"))
        val_start_time = time.time()
        last_batch_time = val_start_time
        if val_dataloader is None:
            val_dataloader = self.val_dataloader
        if val_debug:
            print(
                f"[VAL] start step={self.global_steps} "
                f"val_len={len(val_dataloader)} val_batch_size={getattr(val_dataloader, 'batch_size', None)}"
            )
        data_source_lst = []
        source_file_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for batch_idx, test_data in enumerate(val_dataloader):
            if val_debug and batch_idx == 0:
                print(f"[VAL] first batch after {time.time() - val_start_time:.2f}s")
            elif val_debug and (batch_idx % max(val_log_every, 1) == 0):
                print(
                    f"[VAL] batch {batch_idx}/{len(val_dataloader)} "
                    f"after {time.time() - last_batch_time:.2f}s"
                )
                last_batch_time = time.time()
            if val_max_batches > 0 and batch_idx >= val_max_batches:
                if val_debug:
                    print(f"[VAL] stopping early at batch {batch_idx}/{len(val_dataloader)}")
                break
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            gen_start = time.time()
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            if val_debug:
                print(f"[VAL] generation done in {time.time() - gen_start:.2f}s")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            if "source_file" in test_batch.non_tensor_batch:
                source_file_lst.append(test_batch.non_tensor_batch["source_file"])
            else:
                source_file_lst.append(["unknown"] * reward_tensor.shape[0])

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        source_files = np.concatenate(source_file_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Aggregate reward per source_file when it spans multiple data sources
        if (
            not skip_global_avg
            and "reward" in reward_extra_infos_dict
            and len(source_files) == len(reward_extra_infos_dict["reward"])
        ):
            rewards = np.array(reward_extra_infos_dict["reward"])
            file_means = {}
            for sf in np.unique(source_files):
                mask = source_files == sf
                if mask.sum() == 0:
                    continue
                file_means[str(sf)] = float(rewards[mask].mean())
                # skip files with a single data_source
                if len(np.unique(data_sources[mask])) <= 1:
                    continue
                metric_dict[f"val-core/{sf}/reward/mean"] = float(rewards[mask].mean())
            minerva_file = None
            if "minerva-lhc-dev.jsonl" in file_means:
                minerva_file = "minerva-lhc-dev.jsonl"
            else:
                for sf in file_means:
                    if "minerva" in sf and "dev" in sf and sf.endswith(".jsonl"):
                        minerva_file = sf
                        break
            if minerva_file is not None:
                minerva_mean = file_means.get(minerva_file)
                athena_means = [val for sf, val in file_means.items() if sf.startswith("athena-cti-")]
                ifeval_mean = None
                if data_sources is not None:
                    ifeval_mask = data_sources == "reward_instruction_following"
                    if ifeval_mask.any():
                        ifeval_mean = float(rewards[ifeval_mask].mean())
                        metric_dict["val-core/ifeval/reward/mean"] = ifeval_mean
                if athena_means:
                    athena_mean = float(np.mean(athena_means))
                    metric_dict["val-core/athena-bench/reward/mean"] = athena_mean
                    if minerva_mean is not None:
                        if ifeval_mean is not None:
                            metric_dict["val-core/global-val/reward/mean"] = float(
                                0.4 * minerva_mean + 0.4 * athena_mean + 0.2 * ifeval_mean
                            )
                        else:
                            metric_dict["val-core/global-val/reward/mean"] = float(
                                (minerva_mean + athena_mean) / 2.0
                            )
                if minerva_mean is not None:
                    metric_dict["val-core/minerva-dev/reward/mean"] = float(minerva_mean)
                if athena_means and minerva_mean is not None:
                    metric_dict.setdefault(
                        "val-core/global-val/reward/mean",
                        float((minerva_mean + athena_mean) / 2.0),
                    )

        # Fallback aggregation when file-level labels are missing or incomplete.
        if (
            not skip_global_avg
            and "reward" in reward_extra_infos_dict
            and len(data_sources) == len(reward_extra_infos_dict["reward"])
        ):
            rewards = np.array(reward_extra_infos_dict["reward"])
            if (
                "val-core/minerva-dev/reward/mean" not in metric_dict
                or "val-core/athena-bench/reward/mean" not in metric_dict
            ):
                data_sources_arr = np.array([str(ds) for ds in data_sources])
                athena_mask = np.array([ds.startswith("athena-cti-") for ds in data_sources_arr])
                ifeval_mask = data_sources_arr == "reward_instruction_following"
                minerva_mask = ~athena_mask
                if "val-core/minerva-dev/reward/mean" not in metric_dict and minerva_mask.any():
                    metric_dict["val-core/minerva-dev/reward/mean"] = float(rewards[minerva_mask].mean())
                if "val-core/athena-bench/reward/mean" not in metric_dict and athena_mask.any():
                    metric_dict["val-core/athena-bench/reward/mean"] = float(rewards[athena_mask].mean())
                if "val-core/ifeval/reward/mean" not in metric_dict and ifeval_mask.any():
                    metric_dict["val-core/ifeval/reward/mean"] = float(rewards[ifeval_mask].mean())
            if "val-core/global-val/reward/mean" not in metric_dict:
                minerva_mean = metric_dict.get("val-core/minerva-dev/reward/mean")
                athena_mean = metric_dict.get("val-core/athena-bench/reward/mean")
                ifeval_mean = metric_dict.get("val-core/ifeval/reward/mean")
                if minerva_mean is not None and athena_mean is not None:
                    if ifeval_mean is not None:
                        metric_dict["val-core/global-val/reward/mean"] = float(
                            0.4 * minerva_mean + 0.4 * athena_mean + 0.2 * ifeval_mean
                        )
                    else:
                        metric_dict["val-core/global-val/reward/mean"] = float((minerva_mean + athena_mean) / 2.0)

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if metric_prefix and metric_prefix != "val":
            prefixed = {}
            for key, val in metric_dict.items():
                if key.startswith("val-"):
                    prefixed[f"{metric_prefix}{key[3:]}"] = val
                else:
                    prefixed[key] = val
            metric_dict = prefixed

        return metric_dict

    def _build_val_dataloader(self, *, val_files: list[str], data_cfg) -> StatefulDataLoader:
        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        dataset = create_rl_dataset(val_files, data_cfg, self.tokenizer, self.processor, is_train=False)
        val_batch_size = data_cfg.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(dataset)
        return StatefulDataLoader(
            dataset=dataset,
            batch_size=val_batch_size,
            num_workers=data_cfg["dataloader_num_workers"],
            shuffle=data_cfg.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=default_collate_fn,
        )

    def _run_extra_validations(self) -> dict:
        extra_runs = self.config.trainer.get("extra_val_runs", None)
        if not extra_runs:
            return {}
        if isinstance(extra_runs, str):
            try:
                extra_runs = json.loads(extra_runs)
            except json.JSONDecodeError:
                return {}
        try:
            extra_runs = list(extra_runs)
        except TypeError:
            return {}

        from omegaconf import OmegaConf, open_dict

        metrics: dict = {}
        for idx, run in enumerate(extra_runs):
            if isinstance(run, DictConfig):
                run = OmegaConf.to_container(run, resolve=True)
            if not isinstance(run, dict):
                continue
            val_files = run.get("val_files") or []
            if isinstance(val_files, str):
                val_files = [val_files]
            elif isinstance(val_files, ListConfig):
                val_files = list(val_files)
            if not val_files:
                continue
            name = str(run.get("name") or f"extra_{idx}")
            metric_prefix = str(run.get("metric_prefix") or f"val-{name}")
            tarba_eval_mode = run.get("tarba_eval_mode")
            tarba_eval_budget_B = run.get("tarba_eval_budget_B")
            skip_global_avg = bool(run.get("skip_global_avg", False))

            data_cfg = OmegaConf.create(OmegaConf.to_container(self.config.data, resolve=True))
            with open_dict(data_cfg):
                data_cfg.val_files = val_files
                if tarba_eval_mode is not None or tarba_eval_budget_B is not None:
                    tarba_cfg = data_cfg.get("tarba", {}) or {}
                    if not isinstance(tarba_cfg, dict):
                        tarba_cfg = {}
                    if tarba_eval_mode is not None:
                        tarba_cfg["eval_mode"] = tarba_eval_mode
                    if tarba_eval_budget_B is not None:
                        tarba_cfg["eval_budget_B"] = int(tarba_eval_budget_B)
                    data_cfg["tarba"] = tarba_cfg

            val_dataloader = self._build_val_dataloader(val_files=val_files, data_cfg=data_cfg)
            run_metrics = self._validate(
                val_dataloader=val_dataloader,
                metric_prefix=metric_prefix,
                skip_global_avg=skip_global_avg,
            )
            metrics.update(run_metrics)

        return metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create judge worker if requested (separate GPU-backed model)
        if self._acr_judge_backend_requested == "worker":
            judge_model = self._acr_judge_model
            if not judge_model:
                print("ACR judge backend 'worker' requested but judge_model is missing; using HF judge.")
                self._acr_judge_backend_requested = None
            else:
                judge_cfg = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
                with open_dict(judge_cfg):
                    ref_cfg = judge_cfg.get("ref")
                    if ref_cfg is None:
                        judge_cfg["ref"] = {}
                    ref_cfg = judge_cfg["ref"]
                    if ref_cfg.get("model") is None:
                        ref_cfg["model"] = {}
                    ref_cfg["model"]["path"] = judge_model
                judge_worker_cls = self.role_worker_mapping.get(Role.RefPolicy) or self.role_worker_mapping.get(
                    Role.ActorRollout
                )
                if judge_worker_cls is None:
                    print("ACR judge backend 'worker' unavailable; using HF judge.")
                    self._acr_judge_backend_requested = None
                else:
                    resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
                    judge_cls = RayClassWithInitArgs(judge_worker_cls, config=judge_cfg, role="ref")
                    self.resource_pool_to_cls[resource_pool]["judge"] = judge_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        if "judge" in all_wg:
            self.judge_wg = all_wg["judge"]
            self.judge_wg.init_model()

        self._maybe_init_acr_judge_runner()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _maybe_init_acr_judge_runner(self) -> None:
        if not self._acr_enabled or not self._acr_judge_backend_requested:
            return
        if self._acr_judge_backend_requested not in {"ref", "worker"}:
            return
        judge_backend = self._acr_judge_backend_requested
        if judge_backend == "ref":
            if not self.use_reference_policy:
                print("ACR judge backend 'ref' requested but reference policy is disabled; using HF judge.")
                return
            if not hasattr(self, "ref_policy_wg") or self.ref_policy_wg is None:
                print("ACR judge backend 'ref' requested but ref worker is unavailable; using HF judge.")
                return
            if not hasattr(self.ref_policy_wg, "generate_judge_outputs"):
                print("ACR judge backend 'ref' not supported by ref worker; using HF judge.")
                return
            judge_wg = self.ref_policy_wg
        else:
            if self.judge_wg is None:
                print("ACR judge backend 'worker' requested but judge worker is unavailable; using HF judge.")
                return
            if not hasattr(self.judge_wg, "generate_judge_outputs"):
                print("ACR judge backend 'worker' not supported by judge worker; using HF judge.")
                return
            judge_wg = self.judge_wg
        try:
            from verl.utils.reward_score.reward_acr_batch import reward_acr_batch
        except Exception as exc:
            print(f"ACR judge backend '{judge_backend}' unavailable: {exc}")
            return

        self._acr_judge_runner = _RefJudgeRunner(judge_wg)
        reward_kwargs = dict(self._acr_reward_kwargs)
        reward_kwargs["judge_backend"] = judge_backend
        reward_kwargs["judge_runner"] = self._acr_judge_runner
        self._acr_reward_fn.compute_score = partial(reward_acr_batch, **reward_kwargs)
        self._acr_judge_backend_requested = None

    def _save_checkpoint(
        self,
        *,
        folder_name: Optional[str] = None,
        update_tracker: bool = True,
        purge_dir: bool = False,
        skip_ckpt_rotation: bool = False,
    ):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        folder_name = folder_name or f"global_step_{self.global_steps}"
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, folder_name)

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        if skip_ckpt_rotation:
            max_actor_ckpt_to_keep = None
            max_critic_ckpt_to_keep = None

        if purge_dir and local_global_step_folder:
            shutil.rmtree(local_global_step_folder, ignore_errors=True)

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        step_marker_path = os.path.join(local_global_step_folder, "checkpoint_step.txt")
        with open(step_marker_path, "w") as f:
            f.write(str(self.global_steps))

        # latest checkpointed iteration tracker (for atomic usage)
        if update_tracker:
            local_latest_checkpointed_iteration = os.path.join(
                self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
            )
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(folder_name))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        if "global_step_" in global_step_folder:
            self.global_steps = int(global_step_folder.split("global_step_")[-1])
        else:
            step_marker_path = os.path.join(global_step_folder, "checkpoint_step.txt")
            if os.path.exists(step_marker_path):
                with open(step_marker_path, "r") as f:
                    try:
                        self.global_steps = int(f.read().strip())
                    except (TypeError, ValueError):
                        self.global_steps = 0
            else:
                print(f"Warning: missing checkpoint_step.txt in {global_step_folder}, setting global_step=0")
                self.global_steps = 0

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _zero_solve_group_mask(
        self, token_level_scores: torch.Tensor, group_size: int
    ) -> Tuple[torch.Tensor, int]:
        """Return mask over prompt groups whose responses all failed."""

        if token_level_scores.dim() == 2:
            seq_scores = token_level_scores.sum(-1)
        elif token_level_scores.dim() == 1:
            seq_scores = token_level_scores
        else:
            raise ValueError(
                f"Unexpected reward shape: {tuple(token_level_scores.shape)}"
            )

        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")

        total = seq_scores.shape[0]
        if total % group_size != 0:
            raise ValueError(
                f"Total sequences ({total}) not divisible by group size ({group_size})."
            )

        num_groups = total // group_size
        scores = seq_scores.view(num_groups, group_size)
        success = scores > ZERO_SOLVE_THRESH
        zero_group_mask = ~success.any(dim=1)
        return zero_group_mask, num_groups

    def _compute_zero_solve_metrics(
        self, batch: DataProto, token_level_scores: torch.Tensor
    ) -> dict[str, float]:
        """Compute zero-solve metrics mirroring the prefix-guided trainer logic."""

        if token_level_scores.dim() == 2:
            seq_scores = token_level_scores.sum(-1)
        elif token_level_scores.dim() == 1:
            seq_scores = token_level_scores
        else:
            raise ValueError(
                f"Unexpected reward shape: {tuple(token_level_scores.shape)}"
            )

        uids = batch.non_tensor_batch.get("uid")
        if uids is not None:
            uid_list = np.asarray(uids, dtype=object).tolist()
            if len(uid_list) != seq_scores.shape[0]:
                raise ValueError(
                    "UID count does not match number of token-level scores."
                )

            uid_index: dict[object, int] = {}
            group_success: list[bool] = []
            group_all_success: list[bool] = []
            seq_scores_np = seq_scores.detach().cpu().numpy()

            for idx, uid in enumerate(uid_list):
                pos = uid_index.get(uid)
                if pos is None:
                    pos = len(group_success)
                    uid_index[uid] = pos
                    group_success.append(False)
                    group_all_success.append(True)
                if seq_scores_np[idx] > ZERO_SOLVE_THRESH:
                    group_success[pos] = True
                if seq_scores_np[idx] < ALL_SOLVE_THRESH:
                    group_all_success[pos] = False

            zero_count = group_success.count(False)
            all_count = group_all_success.count(True)
            num_groups = len(group_success)
            return {
                "pg/zero_solve_frac": float(zero_count / max(1, num_groups)),
                "pg/zero_solve_count": float(zero_count),
                "pg/all_solve_frac": float(all_count / max(1, num_groups)),
                "pg/all_solve_count": float(all_count),
            }

        group_size = int(getattr(self.config.actor_rollout_ref.rollout, "n", 1))
        zero_group_mask, num_groups = self._zero_solve_group_mask(
            seq_scores, group_size=group_size
        )
        zero_count = int(zero_group_mask.long().sum().item())
        scores = seq_scores.view(num_groups, group_size)
        all_group_mask = (scores >= ALL_SOLVE_THRESH).all(dim=1)
        all_count = int(all_group_mask.long().sum().item())
        return {
            "pg/zero_solve_frac": float(zero_count / max(1, num_groups)),
            "pg/zero_solve_count": float(zero_count),
            "pg/all_solve_frac": float(all_count / max(1, num_groups)),
            "pg/all_solve_count": float(all_count),
        }

    def _init_distill_state(self) -> None:
        slhc_cfg = self.config.data.get("stochastic_slhc", {}) if self.config is not None else {}
        tarba_cfg = self.config.data.get("tarba", {}) if self.config is not None else {}
        acr_cfg = self.config.data.get("acr", {}) if self.config is not None else {}
        distill_cfg = None
        distill_source = None

        if isinstance(slhc_cfg, DictConfig):
            distill_cfg = slhc_cfg.get("distill", None)
            distill_source = "slhc" if distill_cfg is not None else None
        elif isinstance(slhc_cfg, dict):
            distill_cfg = slhc_cfg.get("distill", None)
            distill_source = "slhc" if distill_cfg is not None else None

        if distill_cfg is None:
            if isinstance(tarba_cfg, DictConfig):
                distill_cfg = tarba_cfg.get("distill", None)
                distill_source = "tarba" if distill_cfg is not None else distill_source
            elif isinstance(tarba_cfg, dict):
                distill_cfg = tarba_cfg.get("distill", None)
                distill_source = "tarba" if distill_cfg is not None else distill_source

        if distill_cfg is None:
            if isinstance(acr_cfg, DictConfig):
                distill_cfg = acr_cfg.get("distill", None)
                distill_source = "acr" if distill_cfg is not None else distill_source
            elif isinstance(acr_cfg, dict):
                distill_cfg = acr_cfg.get("distill", None)
                distill_source = "acr" if distill_cfg is not None else distill_source

        if distill_cfg is None:
            distill_cfg = {}
        dedup_by_uid_present = False
        user_max_buffer = False
        user_batch_size = False
        user_buffer_mode = False
        if isinstance(distill_cfg, DictConfig):
            dedup_by_uid_present = "dedup_by_uid" in distill_cfg
            user_max_buffer = "max_buffer" in distill_cfg
            user_batch_size = "batch_size" in distill_cfg
            user_buffer_mode = "buffer_mode" in distill_cfg
            distill_cfg = OmegaConf.to_container(distill_cfg, resolve=True)
        if not isinstance(distill_cfg, dict):
            distill_cfg = {}
        if isinstance(distill_cfg, dict) and "dedup_by_uid" in distill_cfg:
            dedup_by_uid_present = True
        if isinstance(distill_cfg, dict):
            user_max_buffer = user_max_buffer or ("max_buffer" in distill_cfg)
            user_batch_size = user_batch_size or ("batch_size" in distill_cfg)
            user_buffer_mode = user_buffer_mode or ("buffer_mode" in distill_cfg)

        defaults = {
            "enabled": False,
            "interval": 10,
            "reward_threshold": 0.5,
            "max_buffer": 4096,
            "batch_size": None,
            "max_seq_len": None,
            "entropy_tiebreak": "mean_nll",
            "selection_mode": "top_reward",
            "degenerate_filter": False,
            "degenerate_min_tokens": 40,
            "degenerate_rep_3_max": 0.85,
            "degenerate_rep_4_max": 0.9,
            "entropy_sampling": False,
            "entropy_beta": 1.0,
            "lr_scale": 1.0,
            "drop_last": True,
            "dedup_by_uid": True,
            "buffer_mode": "rolling",
        }
        if distill_source == "acr":
            defaults["interval"] = 10
            defaults["reward_threshold"] = 0.99
            defaults["entropy_tiebreak"] = "mean_nll"
            defaults["selection_mode"] = "random"
            defaults["degenerate_filter"] = True
            defaults["entropy_sampling"] = True
            defaults["max_buffer"] = 1024
            defaults["batch_size"] = 256
        for key, value in defaults.items():
            distill_cfg.setdefault(key, value)

        buffer_mode = str(distill_cfg.get("buffer_mode", "rolling")).lower().strip()
        if buffer_mode not in {"rolling", "flush"}:
            buffer_mode = "rolling"
        distill_cfg["buffer_mode"] = buffer_mode
        if distill_source == "acr" and buffer_mode == "flush":
            if not user_max_buffer:
                distill_cfg["max_buffer"] = 512
            if not user_batch_size:
                distill_cfg["batch_size"] = 512

        distill_method = str(distill_cfg.get("method", "sft")).lower().strip()
        if distill_method not in {"sft", "dpo"}:
            distill_method = "sft"
        distill_cfg["method"] = distill_method

        dpo_cfg = distill_cfg.get("dpo", {})
        if not isinstance(dpo_cfg, dict):
            dpo_cfg = {}
        dpo_defaults = {
            "beta": 0.1,
            "require_rejected_parses": True,
        }
        for key, value in dpo_defaults.items():
            dpo_cfg.setdefault(key, value)
        distill_cfg["dpo"] = dpo_cfg

        if distill_method == "dpo" and distill_source == "acr" and not dedup_by_uid_present:
            distill_cfg["dedup_by_uid"] = False

        self._distill_cfg = distill_cfg
        self._distill_source = distill_source
        self._distill_buffer_local = []
        self._distill_last_run_step = -1
        self._distill_actor_dp_size = None

    def _init_acr_state(self) -> None:
        self._acr_enabled = False
        self._acr_cfg: dict = {}
        self._acr_reward_fn = None
        self._acr_details_store = None
        self._acr_dedupe_labels = None
        self._acr_extract_gold_labels = None
        self._acr_try_build_messages = None
        self._acr_get_task_spec = None
        self._acr_normalize_label = None
        self._acr_extract_labels_from_truth = None
        self._acr_task_spec_cls = None
        self._acr_task_reasoning_hints = {}
        self._acr_entity_reasoning_hints = {}
        self._acr_update_actor = False
        self._acr_judge_runner = None
        self._acr_judge_backend_requested = None
        self._acr_reward_kwargs = {}
        self._acr_judge_model = None
        self._acr_skip_task_keys: set[str] = set()
        self.judge_wg = None
        self._acr_reward_fn_key = self.config.data.get("reward_fn_key", "data_source")
        self._acr_apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        self._acr_truncation = self.config.data.get("truncation", "error")
        self._acr_debug_samples = max(0, int(os.getenv("ACRD_DEBUG_SAMPLES", "0")))
        self._acr_details_debug_samples = max(0, int(os.getenv("ACRD_DETAILS_DEBUG_SAMPLES", "2")))

        acr_cfg = self.config.data.get("acr", {}) if self.config is not None else {}
        if isinstance(acr_cfg, DictConfig):
            acr_cfg = OmegaConf.to_container(acr_cfg, resolve=True)
        if not isinstance(acr_cfg, dict):
            return
        if not acr_cfg.get("per_batch", False):
            return
        if self.processor is not None:
            print("ACR per-batch disabled: multimodal processor is not supported.")
            return

        try:
            from minerva.acr_prompt import dedupe_labels, extract_gold_labels, try_build_acr_messages
            from minerva.cti_task_specs import get_task_spec
            from minerva.label_details_store import LabelDetailsStore
            from minerva.retrieval.task_specs import TaskSpec as RetrievalTaskSpec
            from minerva.retrieval.task_specs import extract_labels_from_truth, normalize_label
        except Exception as exc:
            print(f"ACR per-batch disabled: {exc}")
            return
        try:
            from verl.utils.dataset.minerva_acr_dataset import (
                DEFAULT_ENTITY_REASONING_HINTS,
                DEFAULT_TASK_REASONING_HINTS,
            )
        except Exception:
            DEFAULT_TASK_REASONING_HINTS = {}
            DEFAULT_ENTITY_REASONING_HINTS = {}

        def apply_reasoning_hints(defaults, override):
            if not isinstance(defaults, dict):
                defaults = {}
            merged = dict(defaults)
            if not isinstance(override, dict):
                return merged
            for key, value in override.items():
                hint_key = str(key or "").strip()
                if not hint_key:
                    continue
                if value is None:
                    merged.pop(hint_key, None)
                    continue
                hint_text = str(value).strip()
                if not hint_text:
                    merged.pop(hint_key, None)
                    continue
                merged[hint_key] = hint_text
            return merged

        reward_manager_name = str(acr_cfg.get("reward_manager", "naive")).strip().lower()
        reward_kwargs = dict(acr_cfg.get("reward_kwargs", {}))
        reward_kwargs.setdefault("enforce_no_id_in_reasoning", bool(acr_cfg.get("enforce_no_id_in_reasoning", True)))
        if reward_manager_name != "batch":
            reward_kwargs = {k: v for k, v in reward_kwargs.items() if not str(k).startswith("judge_")}
        else:
            judge_backend = str(reward_kwargs.get("judge_backend", "hf")).strip().lower()
            if judge_backend in {"ref", "worker"}:
                self._acr_judge_backend_requested = judge_backend
                reward_kwargs["judge_backend"] = "hf"
                reward_kwargs.pop("judge_runner", None)
                self._acr_judge_model = reward_kwargs.get("judge_model")
        self._acr_reward_kwargs = dict(reward_kwargs)
        try:
            if reward_manager_name == "batch":
                from verl.utils.reward_score.reward_acr_batch import reward_acr_batch
            else:
                from verl.utils.reward_score.reward_acr import reward_acr
        except Exception as exc:
            print(f"ACR per-batch disabled: {exc}")
            return

        label_details_dir = acr_cfg.get("label_details_dir")
        self._acr_details_store = LabelDetailsStore(label_details_dir)
        self._acr_dedupe_labels = dedupe_labels
        self._acr_extract_gold_labels = extract_gold_labels
        self._acr_try_build_messages = try_build_acr_messages
        self._acr_get_task_spec = get_task_spec
        self._acr_normalize_label = normalize_label
        self._acr_extract_labels_from_truth = extract_labels_from_truth
        self._acr_task_spec_cls = RetrievalTaskSpec
        self._acr_task_reasoning_hints = apply_reasoning_hints(
            DEFAULT_TASK_REASONING_HINTS, acr_cfg.get("task_reasoning_hints")
        )
        self._acr_entity_reasoning_hints = apply_reasoning_hints(
            DEFAULT_ENTITY_REASONING_HINTS, acr_cfg.get("entity_reasoning_hints")
        )

        self._acr_cfg = acr_cfg
        self._acr_enabled = True
        skip_task_keys = acr_cfg.get("skip_task_keys")
        if skip_task_keys is None:
            skip_task_keys = acr_cfg.get("skip_data_sources")
        if isinstance(skip_task_keys, (list, tuple, set, ListConfig)):
            skip_task_keys = list(skip_task_keys)
        elif isinstance(skip_task_keys, str):
            skip_task_keys = [item.strip() for item in skip_task_keys.split(",") if item.strip()]
        else:
            skip_task_keys = []
        skip_set = {str(item).strip().lower() for item in skip_task_keys if str(item).strip()}
        if bool(acr_cfg.get("skip_cvss", False)):
            skip_set.update({"reward_cvss_v31", "reward_cvss_v40", "cve_to_cvss_v31", "cve_to_cvss_v40"})
        self._acr_skip_task_keys = skip_set
        self._acr_max_details_chars = int(acr_cfg.get("max_details_chars", 4048))
        self._acr_enforce_no_id = bool(acr_cfg.get("enforce_no_id_in_reasoning", True))
        self._acr_max_prompt_length = int(
            acr_cfg.get("max_prompt_length", self.config.data.get("max_prompt_length", 1024))
        )
        self._acr_rollout_n = max(1, int(acr_cfg.get("rollout_n", 4)))
        self._acr_weight = float(acr_cfg.get("rl_weight", 0.3))
        self._acr_update_actor = bool(acr_cfg.get("update_actor", False))
        hard_reward_threshold = acr_cfg.get("hard_reward_threshold", None)
        hard_reward_threshold_set = "hard_reward_threshold" in acr_cfg and hard_reward_threshold is not None
        if hard_reward_threshold_set:
            try:
                self._acr_hard_reward_threshold = float(hard_reward_threshold)
            except (TypeError, ValueError):
                self._acr_hard_reward_threshold = None
                hard_reward_threshold_set = False
        else:
            self._acr_hard_reward_threshold = None
        self._acr_hard_reward_threshold_set = hard_reward_threshold_set

        self._acr_hard_reward_no_perfect = False
        hard_reward_mode = str(acr_cfg.get("hard_reward_mode", "no_perfect")).lower().strip()
        if hard_reward_mode in {"mean", "avg", "average", "mean_reward"}:
            hard_reward_mode = "mean"
        elif hard_reward_mode in {"max", "max_reward"}:
            hard_reward_mode = "max"
        elif hard_reward_mode in {"no_perfect", "no_perfect_reward", "no_perfect_rollout"}:
            hard_reward_mode = "max"
            self._acr_hard_reward_no_perfect = True
        else:
            hard_reward_mode = "max"
            self._acr_hard_reward_no_perfect = True
        self._acr_hard_reward_mode = hard_reward_mode
        if reward_manager_name == "batch":
            from verl.workers.reward_manager.batch import BatchRewardManager

            compute_score = partial(reward_acr_batch, **reward_kwargs)
            self._acr_reward_fn = BatchRewardManager(
                tokenizer=self.tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=self._acr_reward_fn_key,
            )
        else:
            compute_score = partial(reward_acr, **reward_kwargs)
            self._acr_reward_fn = NaiveRewardManager(
                tokenizer=self.tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=self._acr_reward_fn_key,
            )

    def _acr_should_skip_task(self, data_source: Any, extra_info: Any = None) -> bool:
        skip_keys = getattr(self, "_acr_skip_task_keys", None)
        if not skip_keys:
            return False

        def _match(value: Any) -> bool:
            if value is None:
                return False
            text = str(value).strip().lower()
            return bool(text) and text in skip_keys

        if _match(data_source):
            return True
        if isinstance(extra_info, dict):
            if _match(extra_info.get("task")):
                return True
            if _match(extra_info.get("acr_task_key")):
                return True
            source_file = extra_info.get("source_file")
            if source_file:
                stem = os.path.splitext(os.path.basename(str(source_file)))[0]
                if _match(stem):
                    return True
        return False

    def _debug_log_acr_samples(
        self,
        acr_batch: DataProto,
        reward_scalar: Optional[torch.Tensor],
        reward_threshold_scalar: Optional[torch.Tensor],
        reward_extra_infos: dict,
    ) -> None:
        max_samples = int(getattr(self, "_acr_debug_samples", 0) or 0)
        if max_samples <= 0 or reward_scalar is None:
            return

        uid_arr = acr_batch.non_tensor_batch.get("uid")
        extra_arr = acr_batch.non_tensor_batch.get("extra_info")
        data_source_arr = acr_batch.non_tensor_batch.get(self._acr_reward_fn_key)
        if uid_arr is None or extra_arr is None or data_source_arr is None:
            return

        responses = acr_batch.batch.get("responses")
        if responses is None:
            return
        responses = responses.detach().cpu()

        response_mask = acr_batch.batch.get("response_mask")
        if response_mask is not None:
            response_mask = response_mask.detach().cpu()

        log_probs = acr_batch.batch.get("rollout_log_probs")
        if log_probs is None:
            log_probs = acr_batch.batch.get("old_log_probs")
        if log_probs is not None:
            log_probs = log_probs.detach().cpu()

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        extra_list = np.asarray(extra_arr, dtype=object).tolist()
        data_source_list = np.asarray(data_source_arr, dtype=object).tolist()

        reward_list = reward_scalar.detach().cpu().tolist()
        reward_threshold_list = reward_list
        if reward_threshold_scalar is not None:
            try:
                threshold_list = reward_threshold_scalar.detach().cpu().tolist()
            except Exception:
                threshold_list = None
            if isinstance(threshold_list, list) and len(threshold_list) == len(reward_list):
                reward_threshold_list = threshold_list

        base_scores = reward_extra_infos.get("acr_base_score")
        if isinstance(base_scores, np.ndarray):
            base_scores = base_scores.tolist()
        if not isinstance(base_scores, list) or len(base_scores) != len(reward_list):
            base_scores = [None] * len(reward_list)

        leak_hits = reward_extra_infos.get("acr_leak_hit")
        extracted = reward_extra_infos.get("acr_extracted")
        if isinstance(extracted, np.ndarray):
            extracted = extracted.tolist()
        if not isinstance(extracted, list) or len(extracted) != len(reward_list):
            extracted = [None] * len(reward_list)
        if isinstance(leak_hits, np.ndarray):
            leak_hits = leak_hits.tolist()
        if not isinstance(leak_hits, list) or len(leak_hits) != len(reward_list):
            leak_hits = [None] * len(reward_list)

        threshold = float(self._distill_cfg.get("reward_threshold", 0.5))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        response_ids_cache: dict[int, list[int]] = {}

        def extract_response_ids(idx: int) -> list[int]:
            cached = response_ids_cache.get(idx)
            if cached is not None:
                return cached
            if response_mask is not None:
                valid_len = int(response_mask[idx].sum().item())
                response_ids = responses[idx][:valid_len].tolist()
            else:
                response_ids = responses[idx].tolist()
                while response_ids and response_ids[-1] == pad_id:
                    response_ids.pop()
            response_ids_cache[idx] = response_ids
            return response_ids

        def render_messages(messages: Optional[list]) -> str:
            if not isinstance(messages, list):
                return ""
            try:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self._acr_apply_chat_template_kwargs
                )
            except Exception:
                return json.dumps(messages, ensure_ascii=True)

        def truncate_text(text: str, max_chars: int = 4000) -> str:
            if not text:
                return ""
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "\n...[truncated]"

        def mean_nll(idx: int) -> Optional[float]:
            if log_probs is None or response_mask is None:
                return None
            mask = response_mask[idx]
            denom = mask.sum()
            if denom.item() <= 0:
                return None
            return float(-(log_probs[idx] * mask).sum().item() / denom.item())

        def response_len(idx: int) -> int:
            if response_mask is not None:
                return int(response_mask[idx].sum().item())
            resp = responses[idx]
            if resp.numel() == 0:
                return 0
            if pad_id is None:
                return int(resp.numel())
            non_pad = (resp != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                return 0
            return int(non_pad[-1].item() + 1)

        def response_len(idx: int) -> int:
            if response_mask is not None:
                return int(response_mask[idx].sum().item())
            resp = responses[idx]
            if resp.numel() == 0:
                return 0
            if pad_id is None:
                return int(resp.numel())
            non_pad = (resp != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                return 0
            return int(non_pad[-1].item() + 1)

        degenerate_filter = bool(self._distill_cfg.get("degenerate_filter", False))
        degenerate_min_tokens = int(self._distill_cfg.get("degenerate_min_tokens", 0) or 0)
        try:
            degenerate_rep_3_max = float(self._distill_cfg.get("degenerate_rep_3_max", 1.0))
        except (TypeError, ValueError):
            degenerate_rep_3_max = 1.0
        try:
            degenerate_rep_4_max = float(self._distill_cfg.get("degenerate_rep_4_max", 1.0))
        except (TypeError, ValueError):
            degenerate_rep_4_max = 1.0
        degenerate_rep_3_max = min(max(degenerate_rep_3_max, 0.0), 1.0)
        degenerate_rep_4_max = min(max(degenerate_rep_4_max, 0.0), 1.0)

        def distinct_ngram_ratio(tokens: list[int], n: int) -> float:
            total = len(tokens) - n + 1
            if total <= 0:
                return 1.0
            ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
            return len(ngrams) / total

        def is_degenerate(tokens: list[int]) -> bool:
            if not degenerate_filter or len(tokens) < degenerate_min_tokens:
                return False
            rep_3 = 1.0 - distinct_ngram_ratio(tokens, 3)
            rep_4 = 1.0 - distinct_ngram_ratio(tokens, 4)
            return rep_3 >= degenerate_rep_3_max or rep_4 >= degenerate_rep_4_max

        grouped: dict[str, list[int]] = {}
        skipped_task = 0
        for idx, uid in enumerate(uid_list):
            if self._acr_should_skip_task(data_source_list[idx], extra_list[idx]):
                skipped_task += 1
                continue
            grouped.setdefault(str(uid), []).append(idx)

        tiebreak_mode = str(self._distill_cfg.get("entropy_tiebreak", "mean_nll"))
        selection_mode = str(self._distill_cfg.get("selection_mode", "random")).lower().strip()
        if selection_mode in {"max_reward", "top_reward", "best", "reward"}:
            selection_mode = "top_reward"
        elif selection_mode in {"random", "rand", "uniform"}:
            selection_mode = "random"
        else:
            selection_mode = "random"
        entropy_sampling = bool(self._distill_cfg.get("entropy_sampling", False))
        entropy_beta = self._distill_cfg.get("entropy_beta", 1.0)
        try:
            entropy_beta = float(entropy_beta)
        except (TypeError, ValueError):
            entropy_beta = 1.0
        rng = np.random.default_rng()
        shown = 0
        for uid, idxs in grouped.items():
            if shown >= max_samples:
                break
            if not idxs:
                continue
            idxs = list(idxs)
            ref_idx = idxs[0]
            extra_info = extra_list[ref_idx] if isinstance(extra_list[ref_idx], dict) else {}
            data_source = data_source_list[ref_idx]

            orig_prompt = render_messages(extra_info.get("acr_orig_prompt"))
            acr_prompt = render_messages(extra_info.get("acr_prompt"))

            if self._distill_source == "acr":
                eligible = []
                for i in idxs:
                    base_val = reward_threshold_list[i]
                    if base_scores[i] is not None:
                        base_val = base_scores[i]
                    if base_val is None or base_val < threshold:
                        continue
                    if leak_hits[i] is not None and bool(leak_hits[i]):
                        continue
                    if degenerate_filter:
                        response_ids = extract_response_ids(i)
                        if not response_ids or is_degenerate(response_ids):
                            continue
                    eligible.append(i)
            else:
                eligible = [i for i in idxs if reward_threshold_list[i] >= threshold]
            chosen_idx = None
            tiebreak_proxy = None
            if eligible:
                if selection_mode == "random":
                    chosen_idx = int(rng.choice(eligible))
                    if tiebreak_mode in {"response_len", "length"}:
                        tiebreak_proxy = response_len(chosen_idx)
                    else:
                        tiebreak_proxy = mean_nll(chosen_idx)
                else:
                    max_reward = max(reward_list[i] for i in eligible)
                    top = [i for i in eligible if reward_list[i] >= (max_reward - 1e-6)]
                    top.sort()
                    chosen_idx = top[0]
                    if len(top) > 1:
                        if tiebreak_mode in {"response_len", "length"}:
                            length_map = {i: response_len(i) for i in top}
                            chosen_idx = max(top, key=lambda i: (length_map.get(i, -1), -i))
                            tiebreak_proxy = length_map.get(chosen_idx)
                        else:
                            entropy_map = {i: mean_nll(i) for i in top}
                            valid = [(i, val) for i, val in entropy_map.items() if val is not None]
                            if not valid:
                                chosen_idx = top[0]
                                tiebreak_proxy = entropy_map.get(chosen_idx)
                            elif entropy_sampling:
                                idxs_only = [item[0] for item in valid]
                                vals = np.array([item[1] for item in valid], dtype=np.float64)
                                if not np.isfinite(vals).all():
                                    vals = np.nan_to_num(vals, nan=-1e6, posinf=1e6, neginf=-1e6)
                                vals = vals * entropy_beta
                                vals = vals - np.max(vals)
                                weights = np.exp(vals)
                                total = float(weights.sum())
                                if total <= 0 or not np.isfinite(total):
                                    probs = np.ones_like(weights) / max(1, len(weights))
                                else:
                                    probs = weights / total
                                chosen_idx = int(rng.choice(idxs_only, p=probs))
                                tiebreak_proxy = entropy_map.get(chosen_idx)
                            else:
                                chosen_idx = max(
                                    [item[0] for item in valid],
                                    key=lambda i: (entropy_map.get(i, float("-inf")), -i),
                                )
                                tiebreak_proxy = entropy_map.get(chosen_idx)
                    else:
                        if tiebreak_mode in {"response_len", "length"}:
                            tiebreak_proxy = response_len(chosen_idx)
                        else:
                            tiebreak_proxy = mean_nll(chosen_idx)

            print(f"\n[ACRD DEBUG] uid={uid} data_source={data_source} eligible={len(eligible)}/{len(idxs)}")
            if orig_prompt:
                print("[ACRD DEBUG] original prompt:\n" + truncate_text(orig_prompt))
            if acr_prompt:
                print("[ACRD DEBUG] modified prompt:\n" + truncate_text(acr_prompt))

            for j, idx in enumerate(idxs, start=1):
                response_ids = extract_response_ids(idx)
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True) if response_ids else ""
                reward_val = reward_list[idx]
                base_score_val = base_scores[idx]
                leak_hit_val = leak_hits[idx]
                if self._distill_source == "acr":
                    base_val = reward_threshold_list[idx]
                    if base_scores[idx] is not None:
                        base_val = base_scores[idx]
                    is_eligible = (
                        base_val is not None
                        and base_val >= threshold
                        and (leak_hits[idx] is None or not bool(leak_hits[idx]))
                    )
                else:
                    is_eligible = reward_threshold_list[idx] >= threshold
                base_str = f"{float(base_score_val):.3f}" if base_score_val is not None else "NA"
                leak_str = "NA" if leak_hit_val is None else str(bool(leak_hit_val))
                extracted_val = extracted[idx] if extracted[idx] is not None else None
                extracted_str = "NA" if extracted_val is None else str(bool(extracted_val))
                degenerate_str = "NA"
                if degenerate_filter and response_ids:
                    degenerate_str = str(is_degenerate(response_ids))
                print(
                    "[ACRD DEBUG] response "
                    f"{j}/{len(idxs)} reward={reward_val:.3f} base={base_str} eligible={is_eligible} "
                    f"extracted={extracted_str} leak={leak_str} degenerate={degenerate_str}\n"
                    + truncate_text(response_text)
                )

            if chosen_idx is not None:
                chosen_ids = extract_response_ids(chosen_idx)
                chosen_text = self.tokenizer.decode(chosen_ids, skip_special_tokens=True) if chosen_ids else ""
                if selection_mode == "random":
                    label = "random"
                elif tiebreak_mode in {"response_len", "length"}:
                    label = "response_len"
                else:
                    label = "entropy_proxy"
                    if entropy_sampling:
                        label = "entropy_sample"
                proxy_str = f"{tiebreak_proxy:.3f}" if tiebreak_proxy is not None else "NA"
                print("[ACRD DEBUG] selected response (" + label + "=" + proxy_str + "):\n" + truncate_text(chosen_text))
            else:
                print(f"[ACRD DEBUG] selected response: none (no response >= {threshold:.2f})")
            shown += 1

    def _get_actor_dp_size(self) -> int:
        if self._distill_actor_dp_size is not None:
            return self._distill_actor_dp_size
        dp_size = 1
        try:
            dp_rank_mapping = self.actor_rollout_wg._query_dispatch_info("actor")
            if isinstance(dp_rank_mapping, list) and dp_rank_mapping:
                dp_size = max(int(rank) for rank in dp_rank_mapping) + 1
        except Exception:
            dp_size = 1
        self._distill_actor_dp_size = max(1, dp_size)
        return self._distill_actor_dp_size

    def _acr_prompt_too_long(self, messages: list[dict]) -> bool:
        if not self._acr_enabled:
            return False
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self._acr_apply_chat_template_kwargs
            )
            input_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            return len(input_ids) > self._acr_max_prompt_length
        except Exception:
            return False

    def _acr_tokenize_messages(self, messages: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self._acr_apply_chat_template_kwargs
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        input_ids, attention_mask = postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self._acr_max_prompt_length,
            pad_token_id=pad_id,
            left_pad=True,
            truncation=self._acr_truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        return input_ids[0], attention_mask[0], position_ids[0]

    def _build_acr_batch(self, batch: DataProto) -> tuple[Optional[DataProto], dict[str, float]]:
        metrics: dict[str, float] = {}
        if not self._acr_enabled:
            return None, metrics

        raw_prompt_arr = batch.non_tensor_batch.get("raw_prompt")
        extra_arr = batch.non_tensor_batch.get("extra_info")
        reward_arr = batch.non_tensor_batch.get("reward_model")
        data_source_arr = batch.non_tensor_batch.get(self._acr_reward_fn_key)
        uid_arr = batch.non_tensor_batch.get("uid")

        total = len(batch)
        if reward_arr is None or data_source_arr is None or uid_arr is None:
            metrics["acr/skip_missing_meta"] = float(total)
            return None, metrics

        raw_prompts = np.asarray(raw_prompt_arr, dtype=object).tolist() if raw_prompt_arr is not None else [None] * total
        extras = np.asarray(extra_arr, dtype=object).tolist() if extra_arr is not None else [None] * total
        rewards = np.asarray(reward_arr, dtype=object).tolist()
        data_sources = np.asarray(data_source_arr, dtype=object).tolist()
        uids = np.asarray(uid_arr, dtype=object).tolist()

        optional_keys = []
        for key in ("tools_kwargs", "interaction_kwargs", "need_tools_kwargs", "__num_turns__"):
            if key in batch.non_tensor_batch:
                optional_keys.append(key)
        optional_values = {
            key: np.asarray(batch.non_tensor_batch[key], dtype=object).tolist() for key in optional_keys
        }

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        non_tensor_lists: dict[str, list] = {
            self._acr_reward_fn_key: [],
            "reward_model": [],
            "extra_info": [],
            "uid": [],
            "raw_prompt": [],
        }
        for key in optional_keys:
            non_tensor_lists[key] = []

        skipped_missing_prompt = 0
        skipped_missing_labels = 0
        skipped_prompt_too_long = 0
        skipped_tokenize_error = 0
        skipped_task = 0
        details_missing = 0
        details_omitted = 0
        details_truncated = 0
        details_not_applicable = 0
        details_logged = 0
        details_log_limit = int(getattr(self, "_acr_details_debug_samples", 0) or 0)

        def log_details_issue(reason: str, *, details_text: str, used_details: str, option_text: str = "") -> None:
            nonlocal details_logged
            if details_log_limit <= 0 or details_logged >= details_log_limit:
                return
            source_file = extra_info.get("source_file") if isinstance(extra_info, dict) else None
            task_name = extra_info.get("task") if isinstance(extra_info, dict) else None
            details_len = len(details_text or "")
            used_len = len(used_details or "")
            label_str = ",".join(gold_norm)
            print(
                "[ACRD DETAILS] "
                f"reason={reason} uid={uids[idx]} data_source={data_source} "
                f"task_key={spec.task_key or ''} entity_type={entity_type or ''} "
                f"labels={label_str} option_text={option_text or ''} source_file={source_file or ''} task={task_name or ''} "
                f"details_len={details_len} used_details_len={used_len} "
                f"max_details_chars={self._acr_max_details_chars} max_prompt_len={self._acr_max_prompt_length}"
            )
            details_logged += 1

        def extract_option_map(messages: list[dict]) -> dict[str, str]:
            text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    text = msg["content"]
                    break
            if not text:
                return {}
            marker = "Options:"
            if marker not in text:
                return {}
            tail = text.split(marker, 1)[1]
            options: dict[str, str] = {}
            for line in tail.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^([A-Z])\s*[\.\)]\s*(.+)$", line)
                if not match:
                    continue
                options[match.group(1).upper()] = match.group(2).strip()
            return options

        for idx in range(total):
            extra_info = extras[idx] if isinstance(extras[idx], dict) else {}
            messages = raw_prompts[idx]
            if not isinstance(messages, list):
                fallback = extra_info.get("acr_orig_prompt") if isinstance(extra_info, dict) else None
                messages = fallback if isinstance(fallback, list) else None
            if not isinstance(messages, list):
                skipped_missing_prompt += 1
                continue

            reward_model = rewards[idx] if isinstance(rewards[idx], dict) else {}
            ground_truth = reward_model.get("ground_truth")
            data_source = data_sources[idx]

            if self._acr_should_skip_task(data_source, extra_info):
                skipped_task += 1
                continue

            spec = self._acr_get_task_spec(data_source, ground_truth, extra_info)
            entity_type = spec.entity_type
            reasoning_hint = None
            task_hints = getattr(self, "_acr_task_reasoning_hints", None)
            if spec.task_key and isinstance(task_hints, dict):
                reasoning_hint = task_hints.get(spec.task_key)
            if not reasoning_hint:
                entity_hints = getattr(self, "_acr_entity_reasoning_hints", None)
                if entity_type and isinstance(entity_hints, dict):
                    reasoning_hint = entity_hints.get(entity_type)
            gold_norm = []
            if self._acr_extract_labels_from_truth is not None and self._acr_task_spec_cls is not None:
                try:
                    retrieval_spec = self._acr_task_spec_cls(
                        label_type=entity_type,
                        is_multilabel=bool(spec.is_multilabel),
                        label_id_regex=spec.label_id_regex,
                    )
                    gold_norm = self._acr_extract_labels_from_truth(ground_truth, retrieval_spec)
                except Exception:
                    gold_norm = []
            if not gold_norm:
                gold_labels = self._acr_extract_gold_labels(ground_truth)
                gold_norm = [
                    self._acr_normalize_label(entity_type, label) if entity_type else str(label).strip()
                    for label in gold_labels
                ]
                gold_norm = self._acr_dedupe_labels(gold_norm)
            if not gold_norm:
                skipped_missing_labels += 1
                continue

            details_labels = gold_norm
            option_text = ""
            if (
                entity_type
                and len(gold_norm) == 1
                and len(gold_norm[0]) == 1
                and isinstance(messages, list)
            ):
                options = extract_option_map(messages)
                option_text = options.get(gold_norm[0].upper(), "")
                if option_text:
                    if self._acr_extract_labels_from_truth is not None and self._acr_task_spec_cls is not None:
                        try:
                            retrieval_spec = self._acr_task_spec_cls(
                                label_type=entity_type,
                                is_multilabel=bool(spec.is_multilabel),
                                label_id_regex=spec.label_id_regex,
                            )
                            extracted = self._acr_extract_labels_from_truth(option_text, retrieval_spec)
                            if extracted:
                                details_labels = extracted
                        except Exception:
                            pass
                    if details_labels == gold_norm:
                        details_labels = [option_text]

            details_applicable = bool(entity_type)
            details_text = None
            if details_applicable:
                details_text = self._acr_details_store.get_details(entity_type, details_labels)
            acr_messages, skipped, used_details = self._acr_try_build_messages(
                messages,
                gold_norm,
                details_text,
                max_details_chars=self._acr_max_details_chars,
                prompt_too_long=self._acr_prompt_too_long,
                enforce_no_id=self._acr_enforce_no_id,
                reasoning_hint=reasoning_hint,
            )
            if skipped:
                skipped_prompt_too_long += 1
                continue

            if details_applicable:
                if not details_text:
                    details_missing += 1
                    log_details_issue(
                        "details_missing",
                        details_text=details_text or "",
                        used_details=used_details or "",
                        option_text=option_text,
                    )
                elif not used_details:
                    details_omitted += 1
                    log_details_issue(
                        "details_omitted",
                        details_text=details_text or "",
                        used_details=used_details or "",
                        option_text=option_text,
                    )
                elif used_details and details_text and used_details.endswith("...") and len(used_details) < len(details_text):
                    details_truncated += 1
            else:
                details_not_applicable += 1

            try:
                input_ids, attention_mask, position_ids = self._acr_tokenize_messages(acr_messages)
            except Exception:
                skipped_tokenize_error += 1
                continue

            extra = dict(extra_info) if isinstance(extra_info, dict) else {}
            extra.setdefault("acr_orig_prompt", deepcopy(messages))
            extra["acr_prompt"] = acr_messages
            extra["acr_skipped"] = False
            if option_text:
                extra["acr_option_text"] = option_text
            if details_labels != gold_norm:
                extra["acr_details_labels"] = details_labels
            extra["acr_entity_type"] = entity_type or ""
            extra["acr_gold_labels"] = gold_norm
            extra["acr_gold_keys"] = [f"{entity_type}:{label}" if entity_type else label for label in gold_norm]
            extra["acr_mode"] = "acr"
            if spec.task_key:
                extra["acr_task_key"] = spec.task_key
            if used_details:
                extra["acr_details_chars"] = int(len(used_details))

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            non_tensor_lists[self._acr_reward_fn_key].append(data_source)
            non_tensor_lists["reward_model"].append(reward_model)
            non_tensor_lists["extra_info"].append(extra)
            non_tensor_lists["uid"].append(uids[idx])
            non_tensor_lists["raw_prompt"].append(acr_messages)
            for key in optional_keys:
                non_tensor_lists[key].append(optional_values[key][idx])

        used = len(input_ids_list)
        metrics["acr/batch_total"] = float(total)
        metrics["acr/batch_used"] = float(used)
        if skipped_missing_prompt:
            metrics["acr/skip_missing_prompt"] = float(skipped_missing_prompt)
        if skipped_missing_labels:
            metrics["acr/skip_missing_labels"] = float(skipped_missing_labels)
        if skipped_prompt_too_long:
            metrics["acr/skip_prompt_long"] = float(skipped_prompt_too_long)
        if skipped_tokenize_error:
            metrics["acr/skip_tokenize_error"] = float(skipped_tokenize_error)
        if skipped_task:
            metrics["acr/skip_task"] = float(skipped_task)
        if details_missing:
            metrics["acr/details_missing"] = float(details_missing)
        if details_omitted:
            metrics["acr/details_omitted"] = float(details_omitted)
        if details_truncated:
            metrics["acr/details_truncated"] = float(details_truncated)
        if details_not_applicable:
            metrics["acr/details_not_applicable"] = float(details_not_applicable)
        if used:
            metrics["acr/details_missing_frac"] = float(details_missing / used)
            metrics["acr/details_omitted_frac"] = float(details_omitted / used)
            metrics["acr/details_truncated_frac"] = float(details_truncated / used)

        if used == 0:
            return None, metrics

        tensors = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "position_ids": torch.stack(position_ids_list, dim=0),
        }
        non_tensors = {key: np.array(vals, dtype=object) for key, vals in non_tensor_lists.items()}
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors), metrics

    def _compute_acr_hard_uids(
        self, batch: DataProto, reward_scalar: Optional[torch.Tensor]
    ) -> tuple[Optional[set[str]], dict[str, float]]:
        metrics: dict[str, float] = {}
        if not self._acr_enabled or reward_scalar is None:
            return None, metrics

        uid_arr = batch.non_tensor_batch.get("uid")
        if uid_arr is None:
            return None, metrics

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        reward_list = reward_scalar.detach().cpu().tolist()
        if len(uid_list) != len(reward_list):
            return None, metrics

        reward_sum_by_uid: dict[str, float] = {}
        reward_count_by_uid: dict[str, int] = {}
        reward_max_by_uid: dict[str, float] = {}
        for uid, reward in zip(uid_list, reward_list):
            key = str(uid)
            reward_val = float(reward)
            reward_sum_by_uid[key] = reward_sum_by_uid.get(key, 0.0) + reward_val
            reward_count_by_uid[key] = reward_count_by_uid.get(key, 0) + 1
            prev = reward_max_by_uid.get(key)
            if prev is None or reward_val > prev:
                reward_max_by_uid[key] = reward_val

        hard_mode = str(getattr(self, "_acr_hard_reward_mode", "max")).lower().strip()
        if hard_mode in {"mean", "avg", "average", "mean_reward"}:
            hard_mode = "mean"
        elif hard_mode in {"max", "max_reward", "no_perfect", "no_perfect_reward", "no_perfect_rollout"}:
            hard_mode = "max"
        else:
            hard_mode = "max"

        if hard_mode == "mean":
            mean_reward_by_uid = {
                uid: (reward_sum_by_uid[uid] / max(1, reward_count_by_uid.get(uid, 0)))
                for uid in reward_sum_by_uid
            }
            total = len(mean_reward_by_uid)
            if getattr(self, "_acr_hard_reward_threshold_set", False):
                threshold = float(getattr(self, "_acr_hard_reward_threshold", 0.5))
            else:
                threshold = 0.5
            hard_uids = {uid for uid, score in mean_reward_by_uid.items() if score < threshold}
        else:
            total = len(reward_max_by_uid)
            if getattr(self, "_acr_hard_reward_no_perfect", False):
                threshold = 1.0
            elif getattr(self, "_acr_hard_reward_threshold_set", False):
                threshold = float(getattr(self, "_acr_hard_reward_threshold", 1.0))
            else:
                threshold = 1.0
            hard_uids = {uid for uid, score in reward_max_by_uid.items() if score < (threshold - 1e-6)}

        if total > 0:
            metrics["acr/hard_uid_total"] = float(total)
            metrics["acr/hard_uid_count"] = float(len(hard_uids))
            metrics["acr/hard_uid_frac"] = float(len(hard_uids) / total)

        return hard_uids, metrics

    def _compute_uid_max_rewards(
        self, batch: DataProto, reward_scalar: Optional[torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, float]]:
        metrics: dict[str, float] = {}
        if reward_scalar is None:
            return {}, metrics

        uid_arr = batch.non_tensor_batch.get("uid")
        if uid_arr is None:
            return {}, metrics

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        reward_list = reward_scalar.detach().cpu().tolist()
        if len(uid_list) != len(reward_list):
            return {}, metrics

        max_reward_by_uid: dict[str, float] = {}
        for uid, reward in zip(uid_list, reward_list):
            key = str(uid)
            reward_val = float(reward)
            prev = max_reward_by_uid.get(key)
            if prev is None or reward_val > prev:
                max_reward_by_uid[key] = reward_val

        if max_reward_by_uid:
            metrics["acr/uid_max_reward_mean"] = float(np.mean(list(max_reward_by_uid.values())))
        return max_reward_by_uid, metrics

    def _filter_acr_batch_by_uid_max_reward(
        self, acr_batch: Optional[DataProto], uid_max_rewards: Optional[dict[str, float]]
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        metrics: dict[str, float] = {}
        if acr_batch is None or len(acr_batch) == 0:
            return acr_batch, metrics

        if not uid_max_rewards:
            metrics["acr/dpo_prompt_uid_used"] = 0.0
            metrics["acr/dpo_prompt_uid_filtered"] = float(len(acr_batch))
            return None, metrics

        uid_arr = acr_batch.non_tensor_batch.get("uid")
        if uid_arr is None:
            return acr_batch, metrics

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        target_uids = {uid for uid, score in uid_max_rewards.items() if score < 1.0 - 1e-6}
        total = len(uid_list)

        if not target_uids:
            metrics["acr/dpo_prompt_uid_used"] = 0.0
            metrics["acr/dpo_prompt_uid_filtered"] = float(total)
            return None, metrics

        mask = np.array([str(uid) in target_uids for uid in uid_list], dtype=bool)
        used = int(mask.sum())
        metrics["acr/dpo_prompt_uid_used"] = float(used)
        metrics["acr/dpo_prompt_uid_filtered"] = float(total - used)
        metrics["acr/dpo_prompt_uid_used_frac"] = float(used / max(1, total))

        if used == 0:
            return None, metrics
        if used < total:
            acr_batch = acr_batch.select_idxs(mask)

        return acr_batch, metrics

    def _filter_acr_batch_by_uids(
        self, acr_batch: Optional[DataProto], hard_uids: Optional[set[str]]
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        metrics: dict[str, float] = {}
        if acr_batch is None or len(acr_batch) == 0:
            return acr_batch, metrics

        if hard_uids is None:
            return acr_batch, metrics

        uid_arr = acr_batch.non_tensor_batch.get("uid")
        if uid_arr is None:
            return acr_batch, metrics

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        total = len(uid_list)
        if not hard_uids:
            metrics["acr/hard_uid_used"] = 0.0
            metrics["acr/hard_uid_used_frac"] = 0.0
            metrics["acr/hard_uid_filtered"] = float(total)
            return None, metrics

        mask = np.array([str(uid) in hard_uids for uid in uid_list], dtype=bool)
        used = int(mask.sum())
        metrics["acr/hard_uid_used"] = float(used)
        metrics["acr/hard_uid_used_frac"] = float(used / max(1, len(hard_uids)))
        metrics["acr/hard_uid_filtered"] = float(total - used)

        if used == 0:
            return None, metrics
        if used < total:
            acr_batch = acr_batch.select_idxs(mask)

        return acr_batch, metrics

    def _tokenize_prompt_nohint(self, messages: list[dict]) -> Optional[list[int]]:
        if not isinstance(messages, list):
            return None
        kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **kwargs
            )
            return self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        except Exception:
            return None

    def _collect_distill_candidates(
        self,
        batch: Optional[DataProto],
        reward_scalar: Optional[torch.Tensor],
        reward_threshold_scalar: Optional[torch.Tensor] = None,
        reward_extra_infos: Optional[dict] = None,
        rollout_batch: Optional[DataProto] = None,
        rollout_reward_scalar: Optional[torch.Tensor] = None,
        uid_max_rewards: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        cfg = self._distill_cfg
        if str(cfg.get("method", "sft")).lower().strip() == "dpo" and self._distill_source == "acr":
            return self._collect_distill_candidates_dpo(
                batch=batch,
                reward_scalar=reward_scalar,
                reward_threshold_scalar=reward_threshold_scalar,
                reward_extra_infos=reward_extra_infos,
                rollout_batch=rollout_batch,
                rollout_reward_scalar=rollout_reward_scalar,
                uid_max_rewards=uid_max_rewards,
            )
        if not cfg.get("enabled", False) or reward_scalar is None:
            return {}

        uid_arr = batch.non_tensor_batch.get("uid")
        extra_arr = batch.non_tensor_batch.get("extra_info")
        if uid_arr is None or extra_arr is None:
            return {}

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        extra_list = np.asarray(extra_arr, dtype=object).tolist()
        task_arr = batch.non_tensor_batch.get("data_source")
        if task_arr is not None:
            task_list = np.asarray(task_arr, dtype=object).tolist()
        else:
            task_list = [None] * len(uid_list)

        reward_list = reward_scalar.detach().cpu().tolist()
        if len(uid_list) != len(reward_list):
            return {}
        reward_threshold_list = reward_list
        if reward_threshold_scalar is not None:
            try:
                threshold_list = reward_threshold_scalar.detach().cpu().tolist()
            except Exception:
                threshold_list = None
            if isinstance(threshold_list, list) and len(threshold_list) == len(reward_list):
                reward_threshold_list = threshold_list

        base_scores_list = None
        extracted_list = None
        leak_hits_list = None
        if isinstance(reward_extra_infos, dict):
            base_scores_list = reward_extra_infos.get("acr_base_score")
            extracted_list = reward_extra_infos.get("acr_extracted")
            leak_hits_list = reward_extra_infos.get("acr_leak_hit")
        if isinstance(base_scores_list, np.ndarray):
            base_scores_list = base_scores_list.tolist()
        if not isinstance(base_scores_list, list) or len(base_scores_list) != len(reward_list):
            base_scores_list = None
        if isinstance(extracted_list, np.ndarray):
            extracted_list = extracted_list.tolist()
        if not isinstance(extracted_list, list) or len(extracted_list) != len(reward_list):
            extracted_list = None
        if isinstance(leak_hits_list, np.ndarray):
            leak_hits_list = leak_hits_list.tolist()
        if not isinstance(leak_hits_list, list) or len(leak_hits_list) != len(reward_list):
            leak_hits_list = None

        responses = batch.batch.get("responses")
        if responses is None:
            return {}
        responses = responses.detach().cpu()

        response_mask = batch.batch.get("response_mask")
        if response_mask is not None:
            response_mask = response_mask.detach().cpu()

        log_probs = batch.batch.get("rollout_log_probs")
        if log_probs is None:
            log_probs = batch.batch.get("old_log_probs")
        if log_probs is not None:
            log_probs = log_probs.detach().cpu()

        threshold = float(cfg.get("reward_threshold", 0.5))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        response_ids_cache: dict[int, list[int]] = {}

        def extract_response_ids(idx: int) -> list[int]:
            cached = response_ids_cache.get(idx)
            if cached is not None:
                return cached
            if response_mask is not None:
                valid_len = int(response_mask[idx].sum().item())
                response_ids = responses[idx][:valid_len].tolist()
            else:
                response_ids = responses[idx].tolist()
                while response_ids and response_ids[-1] == pad_id:
                    response_ids.pop()
            response_ids_cache[idx] = response_ids
            return response_ids

        def mean_nll(idx: int) -> Optional[float]:
            if log_probs is None or response_mask is None:
                return None
            mask = response_mask[idx]
            denom = mask.sum()
            if denom.item() <= 0:
                return None
            return float(-(log_probs[idx] * mask).sum().item() / denom.item())

        def response_len(idx: int) -> int:
            if response_mask is not None:
                return int(response_mask[idx].sum().item())
            resp = responses[idx]
            if resp.numel() == 0:
                return 0
            if pad_id is None:
                return int(resp.numel())
            non_pad = (resp != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                return 0
            return int(non_pad[-1].item() + 1)

        degenerate_filter = bool(cfg.get("degenerate_filter", False))
        degenerate_min_tokens = int(cfg.get("degenerate_min_tokens", 0) or 0)
        try:
            degenerate_rep_3_max = float(cfg.get("degenerate_rep_3_max", 1.0))
        except (TypeError, ValueError):
            degenerate_rep_3_max = 1.0
        try:
            degenerate_rep_4_max = float(cfg.get("degenerate_rep_4_max", 1.0))
        except (TypeError, ValueError):
            degenerate_rep_4_max = 1.0
        degenerate_rep_3_max = min(max(degenerate_rep_3_max, 0.0), 1.0)
        degenerate_rep_4_max = min(max(degenerate_rep_4_max, 0.0), 1.0)

        def distinct_ngram_ratio(tokens: list[int], n: int) -> float:
            total = len(tokens) - n + 1
            if total <= 0:
                return 1.0
            ngrams = {tuple(tokens[i : i + n]) for i in range(total)}
            return len(ngrams) / total

        def is_degenerate(tokens: list[int]) -> bool:
            if not degenerate_filter or len(tokens) < degenerate_min_tokens:
                return False
            rep_3 = 1.0 - distinct_ngram_ratio(tokens, 3)
            rep_4 = 1.0 - distinct_ngram_ratio(tokens, 4)
            return rep_3 >= degenerate_rep_3_max or rep_4 >= degenerate_rep_4_max

        grouped: dict[str, list[int]] = {}
        skip_uids: set[str] = set()
        skipped_task = 0
        for idx, uid in enumerate(uid_list):
            uid_str = str(uid)
            if self._acr_should_skip_task(task_list[idx], extra_list[idx]):
                skip_uids.add(uid_str)
                skipped_task += 1
                continue
            grouped.setdefault(uid_str, []).append(idx)

        total_groups = len(grouped)
        selected_groups = 0
        selected_rewards: list[float] = []
        selected_tiebreak: list[float] = []
        hint_count = 0
        nohint_count = 0
        records = []
        degenerate_checked = 0
        degenerate_filtered = 0
        entropy_sampling = bool(cfg.get("entropy_sampling", False))
        entropy_beta = cfg.get("entropy_beta", 1.0)
        try:
            entropy_beta = float(entropy_beta)
        except (TypeError, ValueError):
            entropy_beta = 1.0
        rng = np.random.default_rng()

        tiebreak_mode = str(cfg.get("entropy_tiebreak", "mean_nll"))
        selection_mode = str(cfg.get("selection_mode", "random")).lower().strip()
        if selection_mode in {"max_reward", "top_reward", "best", "reward"}:
            selection_mode = "top_reward"
        elif selection_mode in {"random", "rand", "uniform"}:
            selection_mode = "random"
        else:
            selection_mode = "random"

        for uid, idxs in grouped.items():
            if self._distill_source == "acr":
                eligible = []
                for i in idxs:
                    base_score = reward_threshold_list[i]
                    if base_scores_list is not None:
                        base_score = base_scores_list[i]
                    if base_score is None:
                        continue
                    if base_score < threshold:
                        continue
                    if leak_hits_list is not None and bool(leak_hits_list[i]):
                        continue
                    if degenerate_filter:
                        degenerate_checked += 1
                        response_ids = extract_response_ids(i)
                        if not response_ids or is_degenerate(response_ids):
                            degenerate_filtered += 1
                            continue
                    eligible.append(i)
            else:
                eligible = [i for i in idxs if reward_threshold_list[i] >= threshold]
                if degenerate_filter:
                    degenerate_checked += len(eligible)
                    kept = []
                    for i in eligible:
                        response_ids = extract_response_ids(i)
                        if not response_ids or is_degenerate(response_ids):
                            degenerate_filtered += 1
                            continue
                        kept.append(i)
                    eligible = kept
            if not eligible:
                continue
            chosen_idx = None
            tiebreak_proxy = None
            if selection_mode == "random":
                chosen_idx = int(rng.choice(eligible))
                if tiebreak_mode in {"response_len", "length"}:
                    tiebreak_proxy = response_len(chosen_idx)
                else:
                    tiebreak_proxy = mean_nll(chosen_idx)
            else:
                max_reward = max(reward_list[i] for i in eligible)
                top = [i for i in eligible if reward_list[i] >= (max_reward - 1e-6)]
                top.sort()

                entropy_mode = tiebreak_mode
                chosen_idx = top[0]
                if len(top) > 1:
                    if entropy_mode in {"response_len", "length"}:
                        length_map = {i: response_len(i) for i in top}
                        chosen_idx = max(top, key=lambda i: (length_map.get(i, -1), -i))
                        tiebreak_proxy = length_map.get(chosen_idx)
                    else:
                        entropy_map = {i: mean_nll(i) for i in top}
                        valid = [(i, val) for i, val in entropy_map.items() if val is not None]
                        if not valid:
                            chosen_idx = top[0]
                            tiebreak_proxy = entropy_map.get(chosen_idx)
                        elif entropy_sampling:
                            idxs_only = [item[0] for item in valid]
                            vals = np.array([item[1] for item in valid], dtype=np.float64)
                            if not np.isfinite(vals).all():
                                vals = np.nan_to_num(vals, nan=-1e6, posinf=1e6, neginf=-1e6)
                            vals = vals * entropy_beta
                            vals = vals - np.max(vals)
                            weights = np.exp(vals)
                            total = float(weights.sum())
                            if total <= 0 or not np.isfinite(total):
                                probs = np.ones_like(weights) / max(1, len(weights))
                            else:
                                probs = weights / total
                            chosen_idx = int(rng.choice(idxs_only, p=probs))
                            tiebreak_proxy = entropy_map.get(chosen_idx)
                        else:
                            chosen_idx = max(
                                [item[0] for item in valid],
                                key=lambda i: (entropy_map.get(i, float("-inf")), -i),
                            )
                            tiebreak_proxy = entropy_map.get(chosen_idx)
                else:
                    if entropy_mode in {"response_len", "length"}:
                        tiebreak_proxy = response_len(chosen_idx)
                    else:
                        tiebreak_proxy = mean_nll(chosen_idx)

            extra_info = extra_list[chosen_idx]
            prompt_nohint = None
            hint_used = False
            if isinstance(extra_info, dict):
                if self._distill_source == "acr":
                    prompt_nohint = extra_info.get("acr_orig_prompt")
                    if not isinstance(prompt_nohint, list):
                        prompt_nohint = extra_info.get("orig_prompt")
                    hint_used = True
                else:
                    prompt_nohint = extra_info.get("slhc_prompt_nohint")
                    if not isinstance(prompt_nohint, list):
                        prompt_nohint = extra_info.get("tarba_prompt_no_tool")
                    if "slhc_hint_used" in extra_info:
                        hint_used = bool(extra_info.get("slhc_hint_used", False))
                    else:
                        hint_used = bool(extra_info.get("tarba_allow_retrieval", False))

            if not isinstance(prompt_nohint, list):
                continue

            response_ids = extract_response_ids(chosen_idx)

            if not response_ids:
                continue

            record = {
                "uid": uid,
                "task_key": str(task_list[chosen_idx]) if task_list[chosen_idx] is not None else "unknown",
                "prompt_nohint": deepcopy(prompt_nohint),
                "response_ids": response_ids,
                "response_text": None,
                "reward": float(reward_list[chosen_idx]),
                "entropy_proxy": tiebreak_proxy,
                "hint_used": hint_used,
            }
            records.append(record)
            selected_groups += 1
            selected_rewards.append(record["reward"])
            if tiebreak_proxy is not None:
                selected_tiebreak.append(float(tiebreak_proxy))
            if hint_used:
                hint_count += 1
            else:
                nohint_count += 1

        if records:
            self._distill_buffer_local.extend(records)
            max_buffer = int(cfg.get("max_buffer", 0) or 0)
            if max_buffer > 0:
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                max_local = max(1, int(math.ceil(max_buffer / world_size)))
                if len(self._distill_buffer_local) > max_local:
                    buffer_mode = str(cfg.get("buffer_mode", "rolling")).lower().strip()
                    if buffer_mode == "flush":
                        rng = np.random.default_rng(int(getattr(self, "global_steps", 0)))
                        idxs = rng.choice(len(self._distill_buffer_local), size=max_local, replace=False)
                        self._distill_buffer_local = [self._distill_buffer_local[i] for i in idxs]
                    else:
                        self._distill_buffer_local = self._distill_buffer_local[-max_local:]

        metrics: dict[str, float] = {}
        if total_groups > 0:
            metrics["distill/uid_group_frac"] = float(selected_groups / max(1, total_groups))
            metrics["distill/uid_group_count"] = float(selected_groups)
            metrics["distill/uid_group_total"] = float(total_groups)
        if degenerate_filter and degenerate_checked > 0:
            metrics["distill/degenerate_filtered"] = float(degenerate_filtered)
            metrics["distill/degenerate_filtered_frac"] = float(degenerate_filtered / max(1, degenerate_checked))
        if selected_rewards:
            metrics["distill/selected_reward_mean"] = float(np.mean(selected_rewards))
        if selected_tiebreak:
            if tiebreak_mode in {"response_len", "length"}:
                metrics["distill/response_len_mean"] = float(np.mean(selected_tiebreak))
                metrics["distill/response_len_max"] = float(np.max(selected_tiebreak))
            else:
                metrics["distill/entropy_proxy_mean"] = float(np.mean(selected_tiebreak))
                metrics["distill/entropy_proxy_max"] = float(np.max(selected_tiebreak))
        metrics["distill/selected_hint_count"] = float(hint_count)
        metrics["distill/selected_nohint_count"] = float(nohint_count)
        metrics["distill/buffer_size_local"] = float(len(self._distill_buffer_local))
        if skipped_task:
            metrics["distill/skip_task"] = float(skipped_task)
        return metrics

    def _collect_distill_candidates_dpo(
        self,
        batch: Optional[DataProto],
        reward_scalar: Optional[torch.Tensor],
        reward_threshold_scalar: Optional[torch.Tensor] = None,
        reward_extra_infos: Optional[dict] = None,
        rollout_batch: Optional[DataProto] = None,
        rollout_reward_scalar: Optional[torch.Tensor] = None,
        uid_max_rewards: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        cfg = self._distill_cfg
        if not cfg.get("enabled", False) or rollout_batch is None or rollout_reward_scalar is None:
            return {}

        uid_arr = rollout_batch.non_tensor_batch.get("uid")
        extra_arr = rollout_batch.non_tensor_batch.get("extra_info")
        if uid_arr is None or extra_arr is None:
            return {}

        uid_list = np.asarray(uid_arr, dtype=object).tolist()
        extra_list = np.asarray(extra_arr, dtype=object).tolist()
        task_arr = rollout_batch.non_tensor_batch.get("data_source")
        if task_arr is not None:
            task_list = np.asarray(task_arr, dtype=object).tolist()
        else:
            task_list = [None] * len(uid_list)

        rollout_reward_list = rollout_reward_scalar.detach().cpu().tolist()
        if len(uid_list) != len(rollout_reward_list):
            return {}

        if uid_max_rewards is None:
            uid_max_rewards, _ = self._compute_uid_max_rewards(rollout_batch, rollout_reward_scalar)
        if not uid_max_rewards:
            return {}

        eps = 1e-6
        threshold = float(cfg.get("reward_threshold", 1.0))
        perfect_uids = {uid for uid, score in uid_max_rewards.items() if score >= 1.0 - eps}
        if skip_uids:
            perfect_uids = {uid for uid in perfect_uids if uid not in skip_uids}

        dpo_cfg = cfg.get("dpo", {})
        if not isinstance(dpo_cfg, dict):
            dpo_cfg = {}

        rollout_n = int(getattr(self, "_acr_rollout_n", 1) or 1)

        def tiebreak(uid: str, idx: int, tag: str) -> int:
            token = f"{uid}-{tag}-{idx}".encode("utf-8")
            return int.from_bytes(hashlib.md5(token).digest()[:8], "big")

        def listify(value: Any, length: int) -> Optional[list]:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, list) and len(value) == length:
                return value
            return None

        def resolve_prompt_nohint(extra_info: Any, raw_prompt: Any) -> Optional[list[dict]]:
            if isinstance(extra_info, dict):
                for key in ("acr_orig_prompt", "orig_prompt", "slhc_prompt_nohint", "tarba_prompt_no_tool"):
                    val = extra_info.get(key)
                    if isinstance(val, list):
                        return val
            if isinstance(raw_prompt, list):
                return raw_prompt
            return None

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        def extract_response_ids(
            responses: torch.Tensor, response_mask: Optional[torch.Tensor], idx: int
        ) -> Optional[list[int]]:
            if response_mask is not None:
                valid_len = int(response_mask[idx].sum().item())
                response_ids = responses[idx][:valid_len].tolist()
            else:
                response_ids = responses[idx].tolist()
                while response_ids and response_ids[-1] == pad_id:
                    response_ids.pop()
            return response_ids or None

        grouped: dict[str, list[int]] = {}
        for idx, uid in enumerate(uid_list):
            grouped.setdefault(str(uid), []).append(idx)

        selected_rollout_idxs: list[int] = []
        for uid, idxs in grouped.items():
            if uid not in perfect_uids:
                continue
            idxs_sorted = sorted(
                idxs,
                key=lambda i: (-float(rollout_reward_list[i]), tiebreak(uid, i, "rollout")),
            )
            selected_rollout_idxs.extend(idxs_sorted[: min(rollout_n, len(idxs_sorted))])

        rollout_rubric_list: Optional[list] = None
        rollout_extracted_list: Optional[list] = None
        if selected_rollout_idxs:
            rollout_subset = rollout_batch.select_idxs(selected_rollout_idxs)
            extra_infos = rollout_subset.non_tensor_batch.get("extra_info")
            if extra_infos is None:
                extra_list_subset = [{} for _ in range(len(rollout_subset))]
            else:
                extra_list_subset = [
                    dict(info) if isinstance(info, dict) else {} for info in np.asarray(extra_infos, dtype=object)
                ]
            raw_prompt_arr = rollout_subset.non_tensor_batch.get("raw_prompt")
            raw_prompt_list = (
                np.asarray(raw_prompt_arr, dtype=object).tolist() if raw_prompt_arr is not None else None
            )
            prompt_texts = None
            prompts = rollout_subset.batch.get("prompts") if rollout_subset.batch is not None else None
            if raw_prompt_list is None and prompts is not None:
                prompt_texts = self.tokenizer.batch_decode(prompts.detach().cpu(), skip_special_tokens=True)
            for i, info in enumerate(extra_list_subset):
                if not isinstance(info.get("acr_orig_prompt"), list):
                    msg = None
                    if raw_prompt_list is not None and isinstance(raw_prompt_list[i], list):
                        msg = raw_prompt_list[i]
                    elif prompt_texts is not None:
                        msg = [{"role": "user", "content": str(prompt_texts[i])}]
                    if msg is not None:
                        info["acr_orig_prompt"] = msg
                extra_list_subset[i] = info
            rollout_subset.non_tensor_batch["extra_info"] = np.array(extra_list_subset, dtype=object)

            _, rollout_reward_extra = compute_reward(rollout_subset, self._acr_reward_fn)
            rollout_rubric_list = rollout_reward_extra.get("acr_rubric_score")
            if rollout_rubric_list is None:
                rollout_rubric_list = rollout_reward_extra.get("acr_rubric")
            rollout_extracted_list = rollout_reward_extra.get("acr_extracted")
            rollout_rubric_list = listify(rollout_rubric_list, len(selected_rollout_idxs))
            rollout_extracted_list = listify(rollout_extracted_list, len(selected_rollout_idxs))

        rollout_responses = rollout_batch.batch.get("responses")
        if rollout_responses is None:
            return {}
        rollout_responses = rollout_responses.detach().cpu()
        rollout_response_mask = rollout_batch.batch.get("response_mask")
        if rollout_response_mask is not None:
            rollout_response_mask = rollout_response_mask.detach().cpu()
        raw_prompt_arr = rollout_batch.non_tensor_batch.get("raw_prompt")
        raw_prompt_list = np.asarray(raw_prompt_arr, dtype=object).tolist() if raw_prompt_arr is not None else None

        candidates_by_uid: dict[str, list[dict]] = {}

        def add_candidate(uid: str, candidate: dict) -> None:
            candidates_by_uid.setdefault(uid, []).append(candidate)

        for pos, idx in enumerate(selected_rollout_idxs):
            uid = str(uid_list[idx])
            if uid not in perfect_uids:
                continue
            response_ids = extract_response_ids(rollout_responses, rollout_response_mask, idx)
            if not response_ids:
                continue
            extra_info = extra_list[idx]
            raw_prompt = raw_prompt_list[idx] if raw_prompt_list is not None else None
            prompt_nohint = resolve_prompt_nohint(extra_info, raw_prompt)
            if not isinstance(prompt_nohint, list):
                continue
            rubric = float(rollout_rubric_list[pos]) if rollout_rubric_list is not None else 0.0
            extracted = bool(rollout_extracted_list[pos]) if rollout_extracted_list is not None else False
            reward_val = float(rollout_reward_list[idx])
            task_key = task_list[idx]
            add_candidate(
                uid,
                {
                    "idx": idx,
                    "source": "rollout",
                    "response_ids": response_ids,
                    "reward": reward_val,
                    "base_score": reward_val,
                    "extracted": extracted,
                    "rubric": rubric,
                    "response_len": len(response_ids),
                    "prompt_nohint": prompt_nohint,
                    "task_key": str(task_key) if task_key is not None else "unknown",
                },
            )

        if batch is not None and len(batch) > 0 and isinstance(reward_extra_infos, dict):
            acr_uid_arr = batch.non_tensor_batch.get("uid")
            acr_extra_arr = batch.non_tensor_batch.get("extra_info")
            if acr_uid_arr is not None and acr_extra_arr is not None:
                acr_uid_list = np.asarray(acr_uid_arr, dtype=object).tolist()
                acr_extra_list = np.asarray(acr_extra_arr, dtype=object).tolist()
                acr_task_arr = batch.non_tensor_batch.get("data_source")
                if acr_task_arr is not None:
                    acr_task_list = np.asarray(acr_task_arr, dtype=object).tolist()
                else:
                    acr_task_list = [None] * len(acr_uid_list)

                base_scores_list = listify(reward_extra_infos.get("acr_base_score"), len(acr_uid_list))
                rubric_scores_list = listify(reward_extra_infos.get("acr_rubric_score"), len(acr_uid_list))
                if rubric_scores_list is None:
                    rubric_scores_list = listify(reward_extra_infos.get("acr_rubric"), len(acr_uid_list))
                extracted_list = listify(reward_extra_infos.get("acr_extracted"), len(acr_uid_list))

                acr_responses = batch.batch.get("responses")
                if acr_responses is not None:
                    acr_responses = acr_responses.detach().cpu()
                    acr_response_mask = batch.batch.get("response_mask")
                    if acr_response_mask is not None:
                        acr_response_mask = acr_response_mask.detach().cpu()
                else:
                    acr_response_mask = None

                if acr_responses is not None and base_scores_list is not None:
                    for i, uid_raw in enumerate(acr_uid_list):
                        uid = str(uid_raw)
                        if uid in skip_uids or self._acr_should_skip_task(acr_task_list[i], acr_extra_list[i]):
                            continue
                        uid_max = uid_max_rewards.get(uid)
                        if uid_max is None or uid_max >= 1.0 - eps:
                            continue
                        response_ids = extract_response_ids(acr_responses, acr_response_mask, i)
                        if not response_ids:
                            continue
                        base_score = base_scores_list[i]
                        if base_score is None:
                            continue
                        extra_info = acr_extra_list[i]
                        prompt_nohint = resolve_prompt_nohint(extra_info, None)
                        if not isinstance(prompt_nohint, list):
                            continue
                        rubric = float(rubric_scores_list[i]) if rubric_scores_list is not None else 0.0
                        extracted = bool(extracted_list[i]) if extracted_list is not None else False
                        reward_val = float(base_score)
                        task_key = acr_task_list[i]
                        add_candidate(
                            uid,
                            {
                                "idx": i,
                                "source": "acr",
                                "response_ids": response_ids,
                                "reward": reward_val,
                                "base_score": reward_val,
                                "extracted": extracted,
                                "rubric": rubric,
                                "response_len": len(response_ids),
                                "prompt_nohint": prompt_nohint,
                                "task_key": str(task_key) if task_key is not None else "unknown",
                            },
                        )

        total_groups = len(candidates_by_uid)
        selected_groups = 0
        selected_rewards: list[float] = []
        total_pairs = 0
        skipped_pairs = 0
        records = []

        for uid, candidates in candidates_by_uid.items():
            chosen_pool = [c for c in candidates if c["reward"] >= threshold - eps]
            if not chosen_pool:
                skipped_pairs += 1
                continue
            rejected_pool = [c for c in candidates if c["reward"] < threshold - eps]

            def pick_tiebreak(item: dict, tag: str) -> int:
                return tiebreak(uid, item["idx"], f"{tag}-{item['source']}")

            if rejected_pool:
                chosen = max(chosen_pool, key=lambda c: (c["rubric"], pick_tiebreak(c, "chosen")))
                rejected = max(
                    rejected_pool, key=lambda c: (c["rubric"], c["reward"], pick_tiebreak(c, "rejected"))
                )
            else:
                fallback_pool = chosen_pool
                if len(fallback_pool) < 2:
                    skipped_pairs += 1
                    continue
                chosen = max(fallback_pool, key=lambda c: (c["rubric"], pick_tiebreak(c, "chosen")))
                rejected = min(fallback_pool, key=lambda c: (c["rubric"], pick_tiebreak(c, "rejected")))
                if rejected["idx"] == chosen["idx"]:
                    skipped_pairs += 1
                    continue

            record = {
                "uid": uid,
                "task_key": chosen["task_key"],
                "prompt_nohint": deepcopy(chosen["prompt_nohint"]),
                "chosen_ids": chosen["response_ids"],
                "rejected_ids": rejected["response_ids"],
                "chosen_meta": {
                    "acr_score": float(chosen["reward"]),
                    "acr_base_score": float(chosen["base_score"]) if chosen["base_score"] is not None else None,
                    "rubric_score": float(chosen["rubric"]),
                },
                "rejected_meta": {
                    "acr_score": float(rejected["reward"]),
                    "acr_base_score": float(rejected["base_score"]) if rejected["base_score"] is not None else None,
                    "rubric_score": float(rejected["rubric"]),
                },
                "pair_weight": 1.0,
            }
            records.append(record)
            total_pairs += 1
            selected_rewards.append(float(chosen["reward"]))
            selected_groups += 1

        if records:
            self._distill_buffer_local.extend(records)
            max_buffer = int(cfg.get("max_buffer", 0) or 0)
            if max_buffer > 0:
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                max_local = max(1, int(math.ceil(max_buffer / world_size)))
                if len(self._distill_buffer_local) > max_local:
                    buffer_mode = str(cfg.get("buffer_mode", "rolling")).lower().strip()
                    if buffer_mode == "flush":
                        rng = np.random.default_rng(int(getattr(self, "global_steps", 0)))
                        idxs = rng.choice(len(self._distill_buffer_local), size=max_local, replace=False)
                        self._distill_buffer_local = [self._distill_buffer_local[i] for i in idxs]
                    else:
                        self._distill_buffer_local = self._distill_buffer_local[-max_local:]

        metrics: dict[str, float] = {}
        if total_groups > 0:
            metrics["distill/uid_group_frac"] = float(selected_groups / max(1, total_groups))
            metrics["distill/uid_group_count"] = float(selected_groups)
            metrics["distill/uid_group_total"] = float(total_groups)
        if selected_rewards:
            metrics["distill/selected_reward_mean"] = float(np.mean(selected_rewards))
        metrics["distill/dpo_pairs_added"] = float(total_pairs)
        metrics["distill/dpo_pairs_skipped"] = float(skipped_pairs)
        metrics["distill/buffer_size_local"] = float(len(self._distill_buffer_local))
        if skipped_task:
            metrics["distill/skip_task"] = float(skipped_task)
        return metrics

    def _dedup_distill_records(self, records: list[dict]) -> list[dict]:
        deduped: dict[str, dict] = {}
        for record in records:
            uid = record.get("uid")
            if uid is None:
                continue
            reward = float(record.get("reward", 0.0))
            entropy = record.get("entropy_proxy")
            entropy_val = float(entropy) if entropy is not None else float("-inf")
            prev = deduped.get(uid)
            if prev is None:
                deduped[uid] = record
                continue
            prev_reward = float(prev.get("reward", 0.0))
            prev_entropy = prev.get("entropy_proxy")
            prev_entropy_val = float(prev_entropy) if prev_entropy is not None else float("-inf")
            if reward > prev_reward + 1e-6 or (abs(reward - prev_reward) <= 1e-6 and entropy_val > prev_entropy_val):
                deduped[uid] = record
        return list(deduped.values())

    def _dedup_distill_pairs(self, records: list[dict]) -> list[dict]:
        deduped: dict[str, dict] = {}
        for record in records:
            uid = record.get("uid")
            if uid is None:
                continue
            chosen_meta = record.get("chosen_meta") or {}
            chosen_score = float(chosen_meta.get("acr_score", 0.0) or 0.0)
            prev = deduped.get(uid)
            if prev is None:
                deduped[uid] = record
                continue
            prev_meta = prev.get("chosen_meta") or {}
            prev_score = float(prev_meta.get("acr_score", 0.0) or 0.0)
            if chosen_score > prev_score + 1e-6:
                deduped[uid] = record
        return list(deduped.values())

    def _build_distill_dataproto(
        self, records: list[dict], max_seq_len: Optional[int]
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        stats: dict[str, float] = {}
        if max_seq_len is not None:
            try:
                max_seq_len = int(max_seq_len)
            except (TypeError, ValueError):
                max_seq_len = None

        prompt_limit = self.config.data.get("max_prompt_length", None)
        response_limit = self.config.data.get("max_response_length", None)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        examples = []
        rewards = []
        tiebreak_vals = []
        hint_count = 0
        nohint_count = 0
        skipped = 0
        tiebreak_mode = str(self._distill_cfg.get("entropy_tiebreak", "mean_nll"))

        for record in records:
            prompt_nohint = record.get("prompt_nohint")
            prompt_ids = self._tokenize_prompt_nohint(prompt_nohint)
            if not prompt_ids:
                skipped += 1
                continue
            if prompt_limit is not None and len(prompt_ids) > prompt_limit:
                prompt_ids = prompt_ids[-int(prompt_limit) :]

            response_ids = record.get("response_ids")
            if response_ids is None:
                response_text = record.get("response_text")
                if response_text:
                    response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            if isinstance(response_ids, torch.Tensor):
                response_ids = response_ids.tolist()
            if not response_ids:
                skipped += 1
                continue
            if response_limit is not None and len(response_ids) > response_limit:
                response_ids = response_ids[: int(response_limit)]

            if max_seq_len is not None:
                total_len = len(prompt_ids) + len(response_ids)
                if total_len > max_seq_len:
                    if len(response_ids) >= max_seq_len:
                        response_ids = response_ids[-max_seq_len:]
                        prompt_ids = []
                    else:
                        keep_prompt = max_seq_len - len(response_ids)
                        prompt_ids = prompt_ids[-keep_prompt:]

            examples.append({"prompt_ids": prompt_ids, "response_ids": response_ids})
            rewards.append(float(record.get("reward", 0.0)))
            tiebreak = record.get("entropy_proxy")
            if tiebreak is not None:
                tiebreak_vals.append(float(tiebreak))
            if record.get("hint_used"):
                hint_count += 1
            else:
                nohint_count += 1

        if not examples:
            stats["sft_skipped"] = float(skipped)
            return None, stats

        max_prompt_len = max(len(ex["prompt_ids"]) for ex in examples)
        max_resp_len = max(len(ex["response_ids"]) for ex in examples)

        input_ids = []
        attention_mask = []
        responses = []
        response_mask = []

        for ex in examples:
            prompt_ids = ex["prompt_ids"]
            response_ids = ex["response_ids"]
            prompt_pad = max_prompt_len - len(prompt_ids)
            resp_pad = max_resp_len - len(response_ids)

            input_ids.append([pad_id] * prompt_pad + prompt_ids + response_ids + [pad_id] * resp_pad)
            attention_mask.append(
                [0] * prompt_pad + [1] * len(prompt_ids) + [1] * len(response_ids) + [0] * resp_pad
            )
            responses.append(response_ids + [pad_id] * resp_pad)
            response_mask.append([1] * len(response_ids) + [0] * resp_pad)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        responses = torch.tensor(responses, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        meta_info = {
            "sft_mode": True,
            "temperature": float(self.config.actor_rollout_ref.rollout.get("temperature", 1.0)),
            "global_token_num": attention_mask.sum(dim=-1).tolist(),
        }

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            meta_info=meta_info,
        )

        stats["sft_examples"] = float(len(examples))
        stats["sft_reward_mean"] = float(np.mean(rewards)) if rewards else 0.0
        stats["sft_hint_count"] = float(hint_count)
        stats["sft_nohint_count"] = float(nohint_count)
        if tiebreak_vals:
            if tiebreak_mode in {"response_len", "length"}:
                stats["sft_response_len_mean"] = float(np.mean(tiebreak_vals))
                stats["sft_response_len_max"] = float(np.max(tiebreak_vals))
            else:
                stats["sft_entropy_mean"] = float(np.mean(tiebreak_vals))
                stats["sft_entropy_max"] = float(np.max(tiebreak_vals))
        stats["sft_skipped"] = float(skipped)
        return data, stats

    def _build_distill_dpo_dataproto(
        self, records: list[dict], max_seq_len: Optional[int], beta: float
    ) -> tuple[Optional[DataProto], dict[str, float]]:
        stats: dict[str, float] = {}
        if max_seq_len is not None:
            try:
                max_seq_len = int(max_seq_len)
            except (TypeError, ValueError):
                max_seq_len = None

        prompt_limit = self.config.data.get("max_prompt_length", None)
        response_limit = self.config.data.get("max_response_length", None)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        examples = []
        pair_weights: list[float] = []
        skipped = 0
        for record in records:
            prompt_nohint = record.get("prompt_nohint")
            prompt_ids = self._tokenize_prompt_nohint(prompt_nohint)
            if not prompt_ids:
                skipped += 1
                continue
            if prompt_limit is not None and len(prompt_ids) > prompt_limit:
                prompt_ids = prompt_ids[-int(prompt_limit) :]

            chosen_ids = record.get("chosen_ids")
            rejected_ids = record.get("rejected_ids")
            if isinstance(chosen_ids, torch.Tensor):
                chosen_ids = chosen_ids.tolist()
            if isinstance(rejected_ids, torch.Tensor):
                rejected_ids = rejected_ids.tolist()
            if not chosen_ids or not rejected_ids:
                skipped += 1
                continue

            if response_limit is not None:
                chosen_ids = chosen_ids[: int(response_limit)]
                rejected_ids = rejected_ids[: int(response_limit)]

            if max_seq_len is not None:
                total_len = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
                if total_len > max_seq_len:
                    if len(chosen_ids) >= max_seq_len or len(rejected_ids) >= max_seq_len:
                        chosen_ids = chosen_ids[-max_seq_len:]
                        rejected_ids = rejected_ids[-max_seq_len:]
                        prompt_ids = []
                    else:
                        keep_prompt = max_seq_len - max(len(chosen_ids), len(rejected_ids))
                        prompt_ids = prompt_ids[-keep_prompt:]

            pair_weight = float(record.get("pair_weight", 1.0))
            examples.append({"prompt_ids": prompt_ids, "response_ids": chosen_ids})
            examples.append({"prompt_ids": prompt_ids, "response_ids": rejected_ids})
            pair_weights.append(pair_weight)

        if not examples:
            stats["dpo_skipped"] = float(skipped)
            return None, stats

        max_prompt_len = max(len(ex["prompt_ids"]) for ex in examples)
        max_resp_len = max(len(ex["response_ids"]) for ex in examples)

        input_ids = []
        attention_mask = []
        responses = []
        response_mask = []

        for ex in examples:
            prompt_ids = ex["prompt_ids"]
            response_ids = ex["response_ids"]
            prompt_pad = max_prompt_len - len(prompt_ids)
            resp_pad = max_resp_len - len(response_ids)

            input_ids.append([pad_id] * prompt_pad + prompt_ids + response_ids + [pad_id] * resp_pad)
            attention_mask.append(
                [0] * prompt_pad + [1] * len(prompt_ids) + [1] * len(response_ids) + [0] * resp_pad
            )
            responses.append(response_ids + [pad_id] * resp_pad)
            response_mask.append([1] * len(response_ids) + [0] * resp_pad)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        responses = torch.tensor(responses, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        dpo_pair_weight = []
        for weight in pair_weights:
            dpo_pair_weight.extend([weight, weight])
        dpo_pair_weight = torch.tensor(dpo_pair_weight, dtype=torch.float)

        meta_info = {
            "dpo_mode": True,
            "dpo_beta": float(beta),
            "temperature": float(self.config.actor_rollout_ref.rollout.get("temperature", 1.0)),
            "global_token_num": attention_mask.sum(dim=-1).tolist(),
            "dpo_pair_count": int(len(pair_weights)),
        }

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "dpo_pair_weight": dpo_pair_weight,
            },
            meta_info=meta_info,
        )

        stats["dpo_pairs"] = float(len(pair_weights))
        stats["dpo_skipped"] = float(skipped)
        return data, stats

    def _maybe_run_distill_sft(self, global_step: int) -> dict[str, float]:
        cfg = self._distill_cfg
        if not cfg.get("enabled", False):
            return {}
        if str(cfg.get("method", "sft")).lower().strip() != "sft":
            return {}

        interval = int(cfg.get("interval", 0) or 0)
        if interval <= 0:
            return {}
        if (global_step + 1) % interval != 0:
            return {}
        if global_step == self._distill_last_run_step:
            return {}

        local_records = list(self._distill_buffer_local)
        buffer_mode = str(cfg.get("buffer_mode", "rolling")).lower().strip()
        keep_buffer = self._distill_source == "acr" and buffer_mode != "flush"
        if not local_records:
            return {}

        target_batch_size = cfg.get("batch_size", None)
        if target_batch_size is None:
            target_batch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
            if target_batch_size is None:
                target_batch_size = getattr(self.config.actor_rollout_ref.actor, "ppo_mini_batch_size", None)
        if target_batch_size is not None:
            try:
                target_batch_size = int(target_batch_size)
            except (TypeError, ValueError):
                target_batch_size = None
            if target_batch_size is not None and target_batch_size <= 0:
                target_batch_size = None

        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_records)
            all_records: list[dict] = []
            for chunk in gathered:
                if chunk:
                    all_records.extend(chunk)
            if dist.get_rank() == 0:
                records = all_records
                if cfg.get("dedup_by_uid", True):
                    records = self._dedup_distill_records(records)
                pool_size = len(records)
                rng = np.random.default_rng(int(global_step))
                if target_batch_size is not None and len(records) > target_batch_size:
                    idxs = rng.choice(len(records), size=target_batch_size, replace=False)
                    records = [records[i] for i in idxs]
                payload = {"records": records, "pool_size": pool_size}
            else:
                payload = None
            obj_list = [payload]
            dist.broadcast_object_list(obj_list, src=0)
            payload = obj_list[0] or {}
            records = payload.get("records") or []
            pool_size = int(payload.get("pool_size") or 0)
        else:
            records = local_records
            if cfg.get("dedup_by_uid", True):
                records = self._dedup_distill_records(records)
            pool_size = len(records)
            rng = np.random.default_rng(int(global_step))
            if target_batch_size is not None and len(records) > target_batch_size:
                idxs = rng.choice(len(records), size=target_batch_size, replace=False)
                records = [records[i] for i in idxs]

        if not records:
            if not keep_buffer:
                self._distill_buffer_local = []
            return {}

        dp_size = self._get_actor_dp_size()
        sampled_count = len(records)

        max_seq_len = cfg.get("max_seq_len", None)
        if max_seq_len is None:
            max_prompt = self.config.data.get("max_prompt_length", None)
            max_response = self.config.data.get("max_response_length", None)
            if max_prompt is not None and max_response is not None:
                max_seq_len = int(max_prompt) + int(max_response)
            elif max_prompt is not None:
                max_seq_len = int(max_prompt)
            elif max_response is not None:
                max_seq_len = int(max_response)

        data, stats = self._build_distill_dataproto(records, max_seq_len)
        if data is None:
            if not keep_buffer:
                self._distill_buffer_local = []
            return {}

        divisor = max(1, dp_size)
        pad_size = 0
        if divisor > 1:
            data, pad_size = pad_dataproto_to_divisor(data, divisor)

        data.meta_info["sft_mini_batch_size"] = int(len(data))
        if max_seq_len is not None:
            data.meta_info["sft_max_token_len"] = int(max_seq_len)
        lr_scale = cfg.get("lr_scale", 1.0)
        if lr_scale is not None:
            try:
                lr_scale = float(lr_scale)
            except (TypeError, ValueError):
                lr_scale = 1.0
            data.meta_info["sft_lr_scale"] = lr_scale
        data.meta_info["sft_epochs"] = 1
        data.meta_info["sft_shuffle"] = True

        actor_output = self.actor_rollout_wg.update_actor(data)
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])

        metrics: dict[str, float] = {
            "distill/ran": 1.0,
            "distill/buffer_size_global": float(pool_size),
            "distill/sft_sampled_count": float(sampled_count),
        }
        for key, val in stats.items():
            metrics[f"distill/{key}"] = float(val)
        for key, val in actor_metrics.items():
            mapped = key[len("actor/") :] if key.startswith("actor/") else key
            metrics[f"distill_sft/{mapped}"] = float(val)

        if not keep_buffer:
            self._distill_buffer_local = []
        self._distill_last_run_step = global_step
        return metrics

    def _maybe_run_distill_dpo(self, global_step: int) -> dict[str, float]:
        cfg = self._distill_cfg
        if not cfg.get("enabled", False):
            return {}
        if str(cfg.get("method", "sft")).lower().strip() != "dpo":
            return {}

        interval = int(cfg.get("interval", 0) or 0)
        if interval <= 0:
            return {}
        if (global_step + 1) % interval != 0:
            return {}
        if global_step == self._distill_last_run_step:
            return {}

        local_records = list(self._distill_buffer_local)
        if not local_records:
            return {}

        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_records)
            all_records: list[dict] = []
            for chunk in gathered:
                if chunk:
                    all_records.extend(chunk)
            if dist.get_rank() == 0:
                records = all_records
                if cfg.get("dedup_by_uid", False):
                    records = self._dedup_distill_pairs(records)
                records.sort(
                    key=lambda r: (
                        -float((r.get("chosen_meta") or {}).get("acr_score", 0.0)),
                        str(r.get("uid", "")),
                    )
                )
                max_buffer = int(cfg.get("max_buffer", 0) or 0)
                if max_buffer > 0:
                    records = records[:max_buffer]
            else:
                records = None
            obj_list = [records]
            dist.broadcast_object_list(obj_list, src=0)
            records = obj_list[0] or []
        else:
            records = local_records
            if cfg.get("dedup_by_uid", False):
                records = self._dedup_distill_pairs(records)
            records.sort(
                key=lambda r: (
                    -float((r.get("chosen_meta") or {}).get("acr_score", 0.0)),
                    str(r.get("uid", "")),
                )
            )
            max_buffer = int(cfg.get("max_buffer", 0) or 0)
            if max_buffer > 0:
                records = records[:max_buffer]

        if not records:
            self._distill_buffer_local = []
            return {}

        dpo_cfg = cfg.get("dpo", {})
        if not isinstance(dpo_cfg, dict):
            dpo_cfg = {}
        beta = dpo_cfg.get("beta", 0.1)
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            beta = 0.1

        dp_size = self._get_actor_dp_size()
        drop_last = bool(cfg.get("drop_last", True))

        dpo_batch_size = cfg.get("batch_size", None)
        if dpo_batch_size is not None:
            dpo_batch_size = int(dpo_batch_size)
            if dpo_batch_size <= 0:
                dpo_batch_size = None
            elif len(records) < dpo_batch_size:
                dpo_batch_size = len(records)
        if dpo_batch_size is None and records:
            dpo_batch_size = len(records)

        pair_divisor = max(1, math.lcm(dp_size, 2) // 2)
        if dpo_batch_size is not None and dpo_batch_size > 0:
            pair_divisor = math.lcm(pair_divisor, dpo_batch_size)

        if drop_last and pair_divisor > 1:
            usable_pairs = (len(records) // pair_divisor) * pair_divisor
            records = records[:usable_pairs]
            if not records:
                self._distill_buffer_local = []
                return {}

        max_seq_len = cfg.get("max_seq_len", None)
        if max_seq_len is None:
            max_prompt = self.config.data.get("max_prompt_length", None)
            max_response = self.config.data.get("max_response_length", None)
            if max_prompt is not None and max_response is not None:
                max_seq_len = int(max_prompt) + int(max_response)

        data, stats = self._build_distill_dpo_dataproto(records, max_seq_len, beta=beta)
        if data is None or len(data) == 0:
            self._distill_buffer_local = []
            return {}

        lr_scale = cfg.get("lr_scale", 1.0)
        try:
            lr_scale = float(lr_scale)
        except (TypeError, ValueError):
            lr_scale = 1.0
        data.meta_info["dpo_lr_scale"] = lr_scale
        data.meta_info["dpo_epochs"] = 1
        data.meta_info["dpo_shuffle"] = False
        dpo_mini_batch_size = int(dpo_batch_size) if dpo_batch_size is not None else len(records)
        dpo_mini_batch_size = max(1, dpo_mini_batch_size) * 2
        if dpo_mini_batch_size % 2 != 0:
            dpo_mini_batch_size = max(2, dpo_mini_batch_size - 1)
        data.meta_info["dpo_mini_batch_size"] = dpo_mini_batch_size

        actor_cfg = self.config.actor_rollout_ref.actor
        if isinstance(actor_cfg, DictConfig):
            micro_bsz = actor_cfg.get("ppo_micro_batch_size_per_gpu", None)
            if micro_bsz is None:
                micro_bsz = actor_cfg.get("ppo_mini_batch_size", None)
        else:
            micro_bsz = getattr(actor_cfg, "ppo_micro_batch_size_per_gpu", None)
            if micro_bsz is None:
                micro_bsz = getattr(actor_cfg, "ppo_mini_batch_size", None)
        if micro_bsz is None:
            micro_bsz = dpo_mini_batch_size
        micro_bsz = int(micro_bsz)
        if micro_bsz % 2 != 0:
            micro_bsz = max(2, micro_bsz - 1)
        if micro_bsz > dpo_mini_batch_size:
            micro_bsz = dpo_mini_batch_size
        data.meta_info["dpo_micro_batch_size"] = int(micro_bsz)

        ref_worker = self._get_dpo_ref_worker()

        divisor = math.lcm(dp_size, 2)
        pad_size = 0
        if not drop_last and divisor > 1:
            data, pad_size = pad_dataproto_to_divisor(data, divisor)

        ref_batch, ref_pad = pad_dataproto_to_divisor(data, dp_size)
        ref_log_prob = ref_worker.compute_ref_log_prob(ref_batch)
        if ref_pad:
            ref_log_prob = unpad_dataproto(ref_log_prob, pad_size=ref_pad)
        data = data.union(ref_log_prob)

        actor_output = self.actor_rollout_wg.update_actor(data)
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])

        metrics: dict[str, float] = {
            "distill/ran": 1.0,
            "distill/buffer_size_global": float(len(records)),
        }
        for key, val in stats.items():
            metrics[f"distill/{key}"] = float(val)
        for key, val in actor_metrics.items():
            mapped = key[len("actor/") :] if key.startswith("actor/") else key
            metrics[f"distill_dpo/{mapped}"] = float(val)
        metrics["distill_dpo/beta"] = float(beta)

        self._distill_buffer_local = []
        self._distill_last_run_step = global_step
        return metrics

    def _get_dpo_ref_worker(self):
        if not self.use_reference_policy:
            raise RuntimeError("DPO distill requires a reference policy worker. Enable ref policy for DPO runs.")
        if self.ref_in_actor:
            return self.actor_rollout_wg
        return self.ref_policy_wg

    def _run_acr_phase(
        self,
        acr_batch: Optional[DataProto],
        timing_raw: dict,
        *,
        rollout_batch: Optional[DataProto] = None,
        rollout_reward_scalar: Optional[torch.Tensor] = None,
        rollout_uid_max_rewards: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        if not self._acr_enabled:
            return {}

        distill_method = str(self._distill_cfg.get("method", "sft")).lower().strip()
        if acr_batch is None or len(acr_batch) == 0:
            metrics: dict[str, float] = {}
            if self._distill_source == "acr" and distill_method == "dpo":
                distill_metrics = self._collect_distill_candidates(
                    batch=acr_batch,
                    reward_scalar=None,
                    reward_threshold_scalar=None,
                    reward_extra_infos=None,
                    rollout_batch=rollout_batch,
                    rollout_reward_scalar=rollout_reward_scalar,
                    uid_max_rewards=rollout_uid_max_rewards,
                )
                if distill_metrics:
                    metrics.update({f"acr_{k}": float(v) for k, v in distill_metrics.items()})

                distill_metrics = self._maybe_run_distill_dpo(self.global_steps)
                if distill_metrics:
                    metrics.update({f"acr_{k}": float(v) for k, v in distill_metrics.items()})
            return metrics

        metrics: dict[str, float] = {}

        rollout_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )
        actor_divisor = self.actor_rollout_wg.world_size

        acr_gen_batch = self._get_gen_batch(acr_batch)
        acr_gen_batch.meta_info["global_steps"] = self.global_steps
        acr_gen_batch = acr_gen_batch.repeat(repeat_times=self._acr_rollout_n, interleave=True)

        with marked_timer("acr_gen", timing_raw, color="magenta"):
            acr_gen_batch_padded, pad_size = pad_dataproto_to_divisor(acr_gen_batch, rollout_divisor)
            if not self.async_rollout_mode:
                acr_gen_output_padded = self.actor_rollout_wg.generate_sequences(acr_gen_batch_padded)
            else:
                acr_gen_output_padded = self.async_rollout_manager.generate_sequences(acr_gen_batch_padded)
            timing_raw.update(acr_gen_output_padded.meta_info.get("timing", {}))
            acr_gen_output_padded.meta_info.pop("timing", None)
            acr_gen_output = unpad_dataproto(acr_gen_output_padded, pad_size=pad_size)

        acr_batch = acr_batch.repeat(repeat_times=self._acr_rollout_n, interleave=True)
        acr_batch = acr_batch.union(acr_gen_output)

        if "response_mask" not in acr_batch.batch.keys():
            acr_batch.batch["response_mask"] = compute_response_mask(acr_batch)

        if self.config.trainer.balance_batch:
            if len(acr_batch) % actor_divisor == 0:
                self._balance_batch(acr_batch, metrics=metrics, logging_prefix="acr_seqlen")
            else:
                metrics["acr/balance_skipped"] = 1.0
                metrics["acr/balance_skip_size"] = float(len(acr_batch))
                metrics["acr/balance_skip_world_size"] = float(actor_divisor)

        acr_batch.meta_info["global_token_num"] = torch.sum(acr_batch.batch["attention_mask"], dim=-1).tolist()

        with marked_timer("acr_reward", timing_raw, color="yellow"):
            reward_tensor_raw, reward_extra_infos_dict = compute_reward(acr_batch, self._acr_reward_fn)

            for key, values in reward_extra_infos_dict.items():
                if key == "score":
                    continue
                try:
                    this_val = np.asarray(values, dtype=float)
                except (TypeError, ValueError):
                    continue
                if this_val.size == 0:
                    continue
                metrics[f"acr/rewards/{key}"] = float(np.mean(this_val))

        reward_scalar_raw = None
        if reward_tensor_raw.dim() == 2:
            reward_scalar_raw = reward_tensor_raw.max(dim=-1).values
        elif reward_tensor_raw.dim() == 1:
            reward_scalar_raw = reward_tensor_raw

        do_acr_update = bool(self._acr_update_actor) or (
            self.use_critic and bool(self._acr_cfg.get("update_critic", False))
        )

        reward_threshold_scalar = None
        base_scores = reward_extra_infos_dict.get("acr_base_score")
        if base_scores is not None:
            try:
                base_scores_arr = np.asarray(base_scores, dtype=float)
            except (TypeError, ValueError):
                base_scores_arr = None
            if base_scores_arr is not None and base_scores_arr.size == len(acr_batch):
                reward_threshold_scalar = torch.tensor(base_scores_arr)

        if do_acr_update:
            reward_tensor = reward_tensor_raw
            if self._acr_weight != 1.0:
                reward_tensor = reward_tensor * float(self._acr_weight)
            acr_batch.batch["token_level_scores"] = reward_tensor

        with marked_timer("acr_old_log_prob", timing_raw, color="blue"):
            acr_batch_lp, pad_size = pad_dataproto_to_divisor(acr_batch, actor_divisor)
            old_log_prob = self.actor_rollout_wg.compute_log_prob(acr_batch_lp)
            if pad_size:
                old_log_prob = unpad_dataproto(old_log_prob, pad_size=pad_size)
            entropys = old_log_prob.batch["entropys"]
            response_masks = acr_batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["acr/actor_entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            acr_batch = acr_batch.union(old_log_prob)

        self._debug_log_acr_samples(
            acr_batch=acr_batch,
            reward_scalar=reward_scalar_raw,
            reward_threshold_scalar=reward_threshold_scalar,
            reward_extra_infos=reward_extra_infos_dict,
        )

        if do_acr_update:
            if self.use_reference_policy:
                with marked_timer("acr_ref", timing_raw, color="olive"):
                    acr_batch_ref, pad_size = pad_dataproto_to_divisor(acr_batch, actor_divisor)
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(acr_batch_ref)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(acr_batch_ref)
                    if pad_size:
                        ref_log_prob = unpad_dataproto(ref_log_prob, pad_size=pad_size)
                    acr_batch = acr_batch.union(ref_log_prob)

            if self.use_critic:
                with marked_timer("acr_values", timing_raw, color="cyan"):
                    acr_batch_val, pad_size = pad_dataproto_to_divisor(acr_batch, actor_divisor)
                    values = self.critic_wg.compute_values(acr_batch_val)
                    if pad_size:
                        values = unpad_dataproto(values, pad_size=pad_size)
                    acr_batch = acr_batch.union(values)

            with marked_timer("acr_adv", timing_raw, color="brown"):
                if self.config.algorithm.use_kl_in_reward:
                    acr_batch, kl_metrics = apply_kl_penalty(
                        acr_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    for key, val in kl_metrics.items():
                        metrics[f"acr/{key}"] = float(val)
                else:
                    acr_batch.batch["token_level_rewards"] = acr_batch.batch["token_level_scores"]

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                acr_batch = compute_advantage(
                    acr_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self._acr_rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            if self.use_critic and self._acr_cfg.get("update_critic", False):
                with marked_timer("acr_update_critic", timing_raw, color="pink"):
                    acr_batch_update, _ = pad_dataproto_to_divisor(acr_batch, actor_divisor)
                    critic_output = self.critic_wg.update_critic(acr_batch_update)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                for key, val in critic_output_metrics.items():
                    metrics[f"acr/{key}"] = float(val)

            if self._acr_update_actor:
                with marked_timer("acr_update_actor", timing_raw, color="red"):
                    acr_batch_update, _ = pad_dataproto_to_divisor(acr_batch, actor_divisor)
                    acr_batch_update.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(acr_batch_update)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                for key, val in actor_output_metrics.items():
                    metrics[f"acr/{key}"] = float(val)
        else:
            metrics["acr/ppo_update_skipped"] = 1.0

        if self._distill_source == "acr":
            distill_metrics = None
            if distill_method == "dpo":
                distill_metrics = self._collect_distill_candidates(
                    batch=acr_batch,
                    reward_scalar=reward_scalar_raw,
                    reward_threshold_scalar=reward_threshold_scalar,
                    reward_extra_infos=reward_extra_infos_dict,
                    rollout_batch=rollout_batch,
                    rollout_reward_scalar=rollout_reward_scalar,
                    uid_max_rewards=rollout_uid_max_rewards,
                )
            elif reward_scalar_raw is not None:
                distill_metrics = self._collect_distill_candidates(
                    batch=acr_batch,
                    reward_scalar=reward_scalar_raw,
                    reward_threshold_scalar=reward_threshold_scalar,
                    reward_extra_infos=reward_extra_infos_dict,
                )
            if distill_metrics:
                metrics.update({f"acr_{k}": float(v) for k, v in distill_metrics.items()})

        if self._distill_source == "acr":
            if distill_method == "dpo":
                distill_metrics = self._maybe_run_distill_dpo(self.global_steps)
            else:
                distill_metrics = self._maybe_run_distill_sft(self.global_steps)
            if distill_metrics:
                metrics.update({f"acr_{k}": float(v) for k, v in distill_metrics.items()})

        return metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            extra_val_metrics = self._run_extra_validations()
            if extra_val_metrics:
                val_metrics.update(extra_val_metrics)
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                distill_method = str(self._distill_cfg.get("method", "sft")).lower().strip()

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                acr_batch = None
                if self._acr_enabled:
                    acr_batch, acr_build_metrics = self._build_acr_batch(batch)
                    if acr_build_metrics:
                        metrics.update(acr_build_metrics)
                acr_hard_uids = None
                acr_uid_max_rewards = None

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                            for key, values in reward_extra_infos_dict.items():
                                if key == "score":
                                    continue
                                try:
                                    this_val = np.asarray(values, dtype=float)
                                except (TypeError, ValueError):
                                    # Skip non-numeric extra info (e.g., dicts for debugging).
                                    continue
                                if this_val.size == 0:
                                    continue
                                metrics.update({f"critic/rewards/{key}": float(np.mean(this_val))})

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        reward_scalar = None
                        if reward_tensor.dim() == 2:
                            reward_scalar = reward_tensor.max(dim=-1).values
                        elif reward_tensor.dim() == 1:
                            reward_scalar = reward_tensor

                        if self._acr_enabled:
                            if self._distill_source == "acr" and distill_method == "dpo":
                                acr_uid_max_rewards, acr_uid_metrics = self._compute_uid_max_rewards(
                                    batch=batch, reward_scalar=reward_scalar
                                )
                                if acr_uid_metrics:
                                    metrics.update(acr_uid_metrics)
                            else:
                                acr_hard_uids, acr_hard_metrics = self._compute_acr_hard_uids(
                                    batch=batch, reward_scalar=reward_scalar
                                )
                                if acr_hard_metrics:
                                    metrics.update(acr_hard_metrics)

                        # Optional: update adaptive label-hint curriculum on the training dataset.
                        if hasattr(self.train_dataset, "update_option_curriculum_from_rollout"):
                            sample_score = None
                            if reward_tensor.dim() == 2:
                                sample_score = reward_tensor.sum(-1)
                            elif reward_tensor.dim() == 1:
                                sample_score = reward_tensor
                            if sample_score is not None:
                                data_source = batch.non_tensor_batch.get("data_source")
                                uids = batch.non_tensor_batch.get("uid")
                                if data_source is not None and uids is not None:
                                    if "is_correct" in batch.non_tensor_batch:
                                        is_correct = batch.non_tensor_batch.get("is_correct")
                                    else:
                                        adaptive_cfg = self.config.data.get("adaptive_options", {})
                                        score_threshold = adaptive_cfg.get("score_threshold", 0.5)
                                        is_correct = (sample_score > score_threshold).detach().cpu().numpy()
                                    curriculum_metrics = self.train_dataset.update_option_curriculum_from_rollout(
                                        data_source_arr=data_source,
                                        uid_arr=uids,
                                        is_correct_arr=is_correct,
                                        global_step=self.global_steps,
                                    )
                                    if isinstance(curriculum_metrics, dict):
                                        metrics.update(curriculum_metrics)
                                    option_controller = getattr(self.train_dataset, "option_controller", None)
                                    if option_controller is not None:
                                        metrics.update(option_controller.get_metrics(prefix="option_curriculum/"))

                        # Optional: update stochastic SLHC controller on the training dataset.
                        if hasattr(self.train_dataset, "update_slhc_controller_from_groups"):
                            slhc_cfg = self.config.data.get("stochastic_slhc", {})
                            if slhc_cfg.get("enabled", False):
                                if reward_scalar is not None:
                                    success_threshold = slhc_cfg.get("success_threshold", 0.1)
                                    success = (reward_scalar >= success_threshold).float()
                                    uid_arr = batch.non_tensor_batch.get("uid")
                                    task_arr = batch.non_tensor_batch.get("data_source")
                                    extra_arr = batch.non_tensor_batch.get("extra_info")
                                    if uid_arr is not None and task_arr is not None and extra_arr is not None:
                                        uid_list = list(uid_arr)
                                        task_list = list(task_arr)
                                        extra_list = list(extra_arr)
                                        success_list = success.detach().cpu().tolist()
                                        grouped = {}
                                        for idx, uid in enumerate(uid_list):
                                            grouped.setdefault(str(uid), []).append(idx)

                                        group_summaries = []
                                        for _, idxs in grouped.items():
                                            if not idxs:
                                                continue
                                            success_count = int(sum(success_list[i] for i in idxs))
                                            group_size = len(idxs)
                                            acc_g = success_count / group_size
                                            zero_g = 1.0 if success_count == 0 else 0.0
                                            all_g = 1.0 if success_count == group_size else 0.0
                                            task_key = str(task_list[idxs[0]])
                                            extra_info = extra_list[idxs[0]]
                                            ctrl_active = False
                                            if isinstance(extra_info, dict):
                                                ctrl_active = bool(extra_info.get("slhc_ctrl_active", False))
                                            group_summaries.append(
                                                {
                                                    "task_key": task_key,
                                                    "ctrl_active": ctrl_active,
                                                    "acc_g": acc_g,
                                                    "zero_g": zero_g,
                                                    "all_g": all_g,
                                                }
                                            )

                                        total_steps = getattr(self, "total_training_steps", None)
                                        if total_steps is not None:
                                            try:
                                                total_steps = int(total_steps)
                                            except (TypeError, ValueError):
                                                total_steps = None
                                        if total_steps is None or total_steps <= 0:
                                            total_steps = self.config.trainer.get("total_training_steps", None)
                                            if total_steps is not None:
                                                try:
                                                    total_steps = int(total_steps)
                                                except (TypeError, ValueError):
                                                    total_steps = None
                                        if total_steps is None or total_steps <= 0:
                                            total_steps = slhc_cfg.get("total_steps", None)

                                        slhc_metrics = self.train_dataset.update_slhc_controller_from_groups(
                                            group_summaries=group_summaries,
                                            global_step=self.global_steps,
                                            total_steps=total_steps,
                                        )
                                        if isinstance(slhc_metrics, dict):
                                            metrics.update(slhc_metrics)

                        # Optional: update TARBA retrieval controller on the training dataset.
                        if hasattr(self.train_dataset, "update_tarba_controller_from_groups"):
                            tarba_cfg = self.config.data.get("tarba", {})
                            if tarba_cfg.get("enabled", False) and reward_scalar is not None:
                                uid_arr = batch.non_tensor_batch.get("uid")
                                task_arr = batch.non_tensor_batch.get("data_source")
                                extra_arr = batch.non_tensor_batch.get("extra_info")
                                if uid_arr is not None and task_arr is not None and extra_arr is not None:
                                    uid_list = list(uid_arr)
                                    task_list = list(task_arr)
                                    extra_list = list(extra_arr)
                                    if "is_correct" in batch.non_tensor_batch:
                                        is_correct_raw = batch.non_tensor_batch.get("is_correct")
                                        is_correct_list = [bool(x) for x in list(is_correct_raw)]
                                    else:
                                        score_threshold = tarba_cfg.get("score_threshold", 0.5)
                                        is_correct = (reward_scalar >= score_threshold).float()
                                        is_correct_list = is_correct.detach().cpu().tolist()

                                    grouped = {}
                                    for idx, uid in enumerate(uid_list):
                                        grouped.setdefault(str(uid), []).append(idx)

                                    group_summaries = []
                                    for _, idxs in grouped.items():
                                        if not idxs:
                                            continue
                                        success_count = int(sum(1 for i in idxs if is_correct_list[i]))
                                        group_size = len(idxs)
                                        acc_g = success_count / group_size
                                        task_key = str(task_list[idxs[0]])
                                        extra_info = extra_list[idxs[0]]
                                        allow_retrieval = False
                                        if isinstance(extra_info, dict):
                                            allow_retrieval = bool(extra_info.get("tarba_allow_retrieval", False))
                                        group_summaries.append(
                                            {
                                                "task_key": task_key,
                                                "allow_retrieval": allow_retrieval,
                                                "acc_g": acc_g,
                                            }
                                        )

                                    tarba_metrics = self.train_dataset.update_tarba_controller_from_groups(
                                        group_summaries=group_summaries
                                    )
                                    if isinstance(tarba_metrics, dict):
                                        metrics.update(tarba_metrics)

                        if self._distill_source != "acr":
                            distill_metrics = self._collect_distill_candidates(batch=batch, reward_scalar=reward_scalar)
                            if isinstance(distill_metrics, dict) and distill_metrics:
                                metrics.update(distill_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        if self._distill_source != "acr":
                            distill_metrics = self._maybe_run_distill_sft(self.global_steps)
                            if distill_metrics:
                                metrics.update(distill_metrics)

                        if self._acr_enabled:
                            if self._distill_source == "acr" and distill_method == "dpo":
                                if acr_uid_max_rewards is not None:
                                    acr_batch, acr_dpo_filter_metrics = self._filter_acr_batch_by_uid_max_reward(
                                        acr_batch, acr_uid_max_rewards
                                    )
                                    if acr_dpo_filter_metrics:
                                        metrics.update(acr_dpo_filter_metrics)
                            else:
                                if acr_hard_uids is not None:
                                    acr_batch, acr_hard_filter_metrics = self._filter_acr_batch_by_uids(
                                        acr_batch, acr_hard_uids
                                    )
                                    if acr_hard_filter_metrics:
                                        metrics.update(acr_hard_filter_metrics)
                            acr_metrics = self._run_acr_phase(
                                acr_batch,
                                timing_raw,
                                rollout_batch=batch,
                                rollout_reward_scalar=reward_scalar,
                                rollout_uid_max_rewards=acr_uid_max_rewards,
                            )
                            if acr_metrics:
                                metrics.update(acr_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                val_metrics = None
                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        extra_val_metrics = self._run_extra_validations()
                        if extra_val_metrics:
                            val_metrics.update(extra_val_metrics)
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                save_due = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )
                if save_due:
                    save_best_only = bool(self.config.trainer.get("save_best_only", False))
                    if save_best_only:
                        if val_metrics is not None:
                            metric_key = str(
                                self.config.trainer.get("save_best_metric", "val-core/global-val/reward/mean")
                            )
                            metric_mode = str(self.config.trainer.get("save_best_mode", "max")).lower()
                            metric_val = None
                            if isinstance(val_metrics, dict):
                                metric_val = val_metrics.get(metric_key)
                            if metric_val is None:
                                metric_val = metrics.get(metric_key)
                            if metric_val is None:
                                print(f"Skip save_best_only: metric {metric_key} not found.")
                            else:
                                try:
                                    metric_val = float(metric_val)
                                except (TypeError, ValueError):
                                    metric_val = None
                                if metric_val is None:
                                    print(f"Skip save_best_only: metric {metric_key} is not numeric.")
                                else:
                                    improved = False
                                    if self._best_val_metric is None:
                                        improved = True
                                    elif metric_mode in {"min", "lower"}:
                                        improved = metric_val < (self._best_val_metric - 1e-12)
                                    else:
                                        improved = metric_val > (self._best_val_metric + 1e-12)
                                    if improved:
                                        self._best_val_metric = metric_val
                                        self._best_val_step = self.global_steps
                                        best_dir = str(self.config.trainer.get("save_best_dir", "best"))
                                        if esi_close_to_expiration:
                                            print("Force saving checkpoint: ESI instance expiration approaching.")
                                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                                            self._save_checkpoint(
                                                folder_name=best_dir,
                                                update_tracker=False,
                                                purge_dir=True,
                                                skip_ckpt_rotation=True,
                                            )
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            last_dir = str(self.config.trainer.get("save_last_dir", "last"))
                            self._save_checkpoint(
                                folder_name=last_dir,
                                update_tracker=True,
                                purge_dir=True,
                                skip_ckpt_rotation=True,
                            )
                    else:
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                tool_topk_vals = batch.non_tensor_batch.get("tarba_tool_topk")
                if tool_topk_vals is not None:
                    try:
                        topk_arr = np.asarray(tool_topk_vals, dtype=float)
                    except (TypeError, ValueError):
                        topk_arr = None
                    if topk_arr is not None and topk_arr.size > 0:
                        used = topk_arr[topk_arr > 0]
                        metrics["tarba/tool_topk/mean"] = float(used.mean()) if used.size > 0 else 0.0
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                token_level_scores = batch.batch.get("token_level_scores")
                if token_level_scores is not None:
                    metrics.update(self._compute_zero_solve_metrics(batch, token_level_scores))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

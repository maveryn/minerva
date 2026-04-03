from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import logging

import msgspec
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer
from vllm import LLM as _VllmLLM
from vllm.outputs import RequestOutput

logger = logging.getLogger(__name__)


class WeightSyncPayload(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,
):
    actor_weights: Dict[str, torch.Tensor]
    load_format: str


def _normalize_load_format(load_format: str) -> str:
    load_format = str(load_format).lower()
    if load_format.startswith("dummy_"):
        load_format = load_format[len("dummy_"):]
    return load_format


def _sync_model_weights_impl(
    model: nn.Module,
    actor_weights: Dict[str, torch.Tensor],
    load_format: str,
) -> bool:
    normalized = _normalize_load_format(load_format)
    if normalized == "hf":
        from verl.third_party.vllm.vllm_v_0_6_3.hf_weight_loader import load_hf_weights

        load_hf_weights(dict(actor_weights), model)
        return True
    if normalized == "dtensor":
        from verl.third_party.vllm.vllm_v_0_6_3.dtensor_weight_loaders import load_dtensor_weights

        load_dtensor_weights(actor_weights, model)
        return True
    if normalized in {"megatron", "auto"}:
        from verl.third_party.vllm.vllm_v_0_6_3.megatron_weight_loaders import load_megatron_weights

        load_megatron_weights(actor_weights, model)
        return True
    raise ValueError(f"Unsupported rollout load_format for vLLM 0.13 sync: {load_format}")


def _sync_model_weights_on_worker(
    worker,
    payload: WeightSyncPayload,
) -> bool:
    return _sync_model_weights_impl(
        worker.get_model(),
        payload.actor_weights,
        payload.load_format,
    )


class LLMEngine:
    """Placeholder to preserve the older import surface."""


class LLM:
    """Small compatibility wrapper for vLLM 0.13.

    LUFFY's rollout path only needs:
    - generate(prompt_token_ids=..., sampling_params=...)
    - init_cache_engine()
    - free_cache_engine()
    - sync_model_weights(...)
    - offload_model_weights()

    This wrapper restores the minimal sync/offload surface that LUFFY's
    FSDP-vLLM bridge expects, using vLLM 0.13's apply_model RPC path.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_format: str = "auto",
        **kwargs,
    ) -> None:
        del model, max_context_len_to_capture, max_seq_len_to_capture, disable_custom_all_reduce, load_format

        # vLLM 0.13 only serializes Python callables for collective_rpc when
        # insecure serialization is enabled. We need that path to ship a small
        # top-level sync function while keeping the tensor payload separate.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be one of "
                "PreTrainedTokenizer, PreTrainedTokenizerFast, or HybridEngineBaseTokenizer."
            )

        model_path = getattr(model_hf_config, "_name_or_path", None) or getattr(model_hf_config, "name_or_path", None)
        if not model_path:
            raise ValueError("model_hf_config is missing _name_or_path/name_or_path; cannot initialize vLLM 0.13")

        self.tokenizer = tokenizer
        self._is_sleeping = False
        self._needs_kv_wake = False
        self.supports_weight_sync = True
        self.prefers_full_weight_sync = True
        self.tensor_parallel_size = tensor_parallel_size

        llm_kwargs = dict(
            model=model_path,
            tokenizer=model_path if not skip_tokenizer_init else None,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            enable_sleep_mode=True,
            disable_log_stats=True,
        )
        llm_kwargs.update(kwargs)

        self._llm = _VllmLLM(**llm_kwargs)

    def init_cache_engine(self):
        if self._is_sleeping:
            self._llm.wake_up()
            self._is_sleeping = False
            self._needs_kv_wake = False

    def free_cache_engine(self):
        self.offload_model_weights()

    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:
            for output in request_output.outputs:
                output_token_ids.append(torch.tensor(output.token_ids))
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, token_id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[token_id].logprob)
                    logprobs.append(torch.tensor(logprob))

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        else:
            logprobs = torch.empty((len(output_token_ids), 0), dtype=torch.float32)
        return output_token_ids, logprobs

    def generate(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=False, **kwargs):
        del prompts, kwargs
        if self._needs_kv_wake:
            self._llm.wake_up(tags=["kv_cache"])
            self._needs_kv_wake = False
        elif self._is_sleeping:
            self._llm.wake_up()
            self._is_sleeping = False
            self._needs_kv_wake = False
        if prompt_token_ids is None:
            raise ValueError("prompt_token_ids must be provided on the vLLM 0.13 compatibility path")

        inputs = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
        outputs = self._llm.generate(inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)
        return self._post_process_outputs(outputs)

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        if self._is_sleeping:
            self._llm.wake_up(tags=["weights"])
            self._is_sleeping = False
            self._needs_kv_wake = True
        payload = WeightSyncPayload(
            actor_weights=actor_weights,
            load_format=load_format,
        )
        self._llm.llm_engine.engine_core.call_utility(
            "sync_model_weights",
            payload,
        )

    def offload_model_weights(self) -> None:
        if not self._is_sleeping:
            try:
                # LUFFY always reloads fresh actor weights before the next
                # generation step, so we can discard vLLM's current weights
                # instead of backing them up in host RAM.
                self._llm.sleep(level=2)
                self._is_sleeping = True
                self._needs_kv_wake = False
            except Exception:
                logger.exception("vLLM sleep() failed; continuing without offloading the inference engine")

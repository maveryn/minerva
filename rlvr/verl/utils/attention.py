"""Attention backend selection helpers."""

from __future__ import annotations

import logging

from transformers.utils import is_flash_attn_2_available

from verl.utils.device import is_cuda_available

logger = logging.getLogger(__name__)


def resolve_attn_implementation(requested: str | None = None) -> str:
    requested = requested or "flash_attention_2"
    if requested != "flash_attention_2":
        return requested
    if is_flash_attn_2_available():
        return requested
    fallback = "sdpa" if is_cuda_available else "eager"
    logger.warning("flash_attention_2 requested but unavailable; falling back to %s", fallback)
    return fallback


__all__ = ["resolve_attn_implementation"]

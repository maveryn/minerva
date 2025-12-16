# verl/trainer/paraphrase_ppo/paraphrase_utils.py
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

"""Utilities for transport ratio computation, validation, and group assembly."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import torch
import torch.nn.functional as F
from tensordict import TensorDict


@dataclass
class SampleRecord:
    """Record for tracking a single sample through the pipeline."""
    qid: str                          # Stable question id
    round_id: int                     # Training step
    group_id: str                     # Unique per (qid, round_id, ctx_type)
    ctx_type: str                     # PLAIN | APARA | QPARA
    source_id: str                    # ON | OFF_APARA | OFF_QPARA
    sid: str                          # Unique sample id within group
    parent_sid: Optional[str]         # Parent sample id (for derived samples)
    prompt_plain: str                 # Original prompt x
    prompt_used: str                  # Prompt used for generation (x, x', or (x,s))
    tokens: torch.Tensor              # Token ids [T]
    reward: int                       # 0/1 under plain x
    lengths: int                      # Sequence length
    path_entropy: float               # Average token entropy during generation
    old_logprobs_ctx: torch.Tensor    # log π_old(y_t | prompt_used, y_<t)
    # Filled during update:
    cur_logprobs_plain: Optional[torch.Tensor] = None
    transport_log_ratio: Optional[torch.Tensor] = None
    # Additional metadata
    text: Optional[str] = None
    parse_error: bool = False
    truncated: bool = False
    
    
def compute_transport_ratio(
    cur_logprobs_plain: torch.Tensor,      # [T] log π_θ(y_t | x, y_<t)
    old_logprobs_ctx: torch.Tensor,        # [T] log π_old(y_t | c, y_<t) 
    log_ratio_cap: float = 8.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Compute cross-context transport ratio with stability caps.
    
    Args:
        cur_logprobs_plain: Current policy logprobs under plain prompt x
        old_logprobs_ctx: Old policy logprobs under context c (x', (x,s), etc)
        log_ratio_cap: Maximum absolute value for log ratio
        device: Target device
        
    Returns:
        log_ratio: Capped log transport ratio [T]
    """
    if device is not None:
        cur_logprobs_plain = cur_logprobs_plain.to(device)
        old_logprobs_ctx = old_logprobs_ctx.to(device)
    
    # Compute raw log ratio
    log_ratio = cur_logprobs_plain - old_logprobs_ctx
    
    # Apply stability cap
    log_ratio = torch.clamp(log_ratio, -log_ratio_cap, log_ratio_cap)
    
    return log_ratio


def rescore_under_plain_prompt(
    model_or_wg,
    tokenizer,
    prompt_plain: str,
    tokens: torch.Tensor,
    batch_size: int = 1,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Re-score tokens under plain prompt using teacher forcing.
    
    Args:
        model_or_wg: Policy model or worker group
        tokenizer: Tokenizer
        prompt_plain: Plain prompt x
        tokens: Token sequence to score [T]
        batch_size: Batch size for inference
        device: Target device
        
    Returns:
        logprobs: Token-level log probabilities [T]
    """
    # Use worker group interface if available
    if hasattr(model_or_wg, 'compute_log_prob'):
        from verl.protocol import DataProto
        
        # Create batch for log prob computation
        prompt_ids = tokenizer.encode(prompt_plain, add_special_tokens=True)
        batch_dict = {
            "prompts": torch.tensor([prompt_ids], device=device),
            "responses": tokens.unsqueeze(0).to(device),
            "attention_mask": torch.ones(1, len(prompt_ids) + len(tokens), device=device),
            "response_mask": torch.ones(1, len(tokens), device=device)
        }
        batch = DataProto.from_dict(batch_dict)
        
        # Compute log probs
        output = model_or_wg.compute_log_prob(batch)
        return output.batch["log_probs"][0]
    
    # Fallback to direct model interface
    model = model_or_wg
    model.eval()
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt_plain, add_special_tokens=True)
    prompt_tensor = torch.tensor(prompt_ids, device=device)
    
    # Concatenate prompt and response tokens
    full_seq = torch.cat([prompt_tensor, tokens.to(device)])
    
    # Create attention mask
    attention_mask = torch.ones_like(full_seq)
    
    # Forward pass with teacher forcing
    with torch.no_grad():
        outputs = model(
            input_ids=full_seq.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            use_cache=False
        )
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    # Extract logprobs for response tokens
    prompt_len = len(prompt_ids)
    response_logits = logits[prompt_len-1:-1]  # [T, vocab_size]
    response_logprobs = F.log_softmax(response_logits, dim=-1)
    
    # Gather logprobs for actual tokens
    token_logprobs = response_logprobs.gather(
        dim=1, 
        index=tokens.to(device).unsqueeze(1)
    ).squeeze(1)
    
    return token_logprobs


def compute_stable_qid(prompt: str) -> str:
    """Compute stable question ID from prompt text.
    
    Args:
        prompt: Question/prompt text
        
    Returns:
        qid: Stable hash-based ID
    """
    return hashlib.md5(prompt.encode('utf-8')).hexdigest()[:16]


def group_by_qid(records: List[SampleRecord]) -> Dict[str, List[SampleRecord]]:
    """Group sample records by question ID.
    
    Args:
        records: List of sample records
        
    Returns:
        Dictionary mapping qid to list of records
    """
    groups = {}
    for rec in records:
        if rec.qid not in groups:
            groups[rec.qid] = []
        groups[rec.qid].append(rec)
    return groups


def group_by_group_id(records: List[SampleRecord]) -> Dict[str, List[SampleRecord]]:
    """Group sample records by group ID.
    
    Args:
        records: List of sample records
        
    Returns:
        Dictionary mapping group_id to list of records
    """
    groups = {}
    for rec in records:
        if rec.group_id not in groups:
            groups[rec.group_id] = []
        groups[rec.group_id].append(rec)
    return groups


def select_seed_success(
    records: List[SampleRecord],
    criterion: str = "lowest_entropy"
) -> Optional[SampleRecord]:
    """Select seed success from PLAIN samples.
    
    Args:
        records: List of sample records (should be from PLAIN context)
        criterion: Selection criterion ("lowest_entropy" or "shortest")
        
    Returns:
        Selected seed record or None if no successes
    """
    # Filter for successful PLAIN samples
    successes = [r for r in records if r.ctx_type == "PLAIN" and r.reward == 1]
    
    if not successes:
        return None
    
    # Sort by criterion
    if criterion == "lowest_entropy":
        # Primary: lowest entropy, Secondary: shortest length
        successes.sort(key=lambda r: (r.path_entropy, r.lengths))
    elif criterion == "shortest":
        # Primary: shortest length, Secondary: lowest entropy  
        successes.sort(key=lambda r: (r.lengths, r.path_entropy))
    else:
        raise ValueError(f"Unknown selection criterion: {criterion}")
    
    return successes[0]


def select_group(
    records: List[SampleRecord],
    target_solve_rate: float = 0.5,
    correct_topk: int = 4,
    incorrect_fill: str = "high",
    k_rollouts: int = 8
) -> List[SampleRecord]:
    """Select balanced group for question paraphrasing variant.
    
    Args:
        records: Combined ON + OFF_QPARA records
        target_solve_rate: Target success rate
        correct_topk: Maximum correct samples to include
        incorrect_fill: Strategy for filling with incorrect ("high" or "low" entropy)
        k_rollouts: Total group size
        
    Returns:
        Selected group of k_rollouts samples
    """
    # Separate correct and incorrect
    correct = [r for r in records if r.reward == 1]
    incorrect = [r for r in records if r.reward == 0]
    
    # Sort correct by path entropy (descending - highest entropy first)
    correct.sort(key=lambda r: r.path_entropy, reverse=True)
    
    # Sort incorrect based on fill strategy
    if incorrect_fill == "high":
        incorrect.sort(key=lambda r: r.path_entropy, reverse=True)
    else:  # "low"
        incorrect.sort(key=lambda r: r.path_entropy, reverse=False)
    
    # Select up to correct_topk correct samples
    selected = correct[:correct_topk]
    
    # Fill remainder with incorrect
    remaining_slots = k_rollouts - len(selected)
    if remaining_slots > 0:
        selected.extend(incorrect[:remaining_slots])
    
    # If still not enough samples, add any remaining samples
    if len(selected) < k_rollouts:
        all_remaining = [r for r in records if r not in selected]
        selected.extend(all_remaining[:k_rollouts - len(selected)])
    
    # Ensure we don't exceed k_rollouts
    return selected[:k_rollouts]


def replace_one(
    records: List[SampleRecord],
    replace_target: SampleRecord,
    with_rec: SampleRecord
) -> List[SampleRecord]:
    """Replace one record in the list with another.
    
    Args:
        records: Original list of records
        replace_target: Record to replace
        with_rec: Replacement record
        
    Returns:
        New list with replacement done
    """
    new_records = []
    replaced = False
    
    for rec in records:
        if not replaced and rec.sid == replace_target.sid:
            new_records.append(with_rec)
            replaced = True
        else:
            new_records.append(rec)
    
    # If target not found, just append at end
    if not replaced:
        new_records.append(with_rec)
    
    return new_records


def validate_transport_ratio(
    log_ratio: torch.Tensor,
    threshold: float = 10.0
) -> Tuple[bool, Dict[str, float]]:
    """Validate transport ratio for numerical stability.
    
    Args:
        log_ratio: Log transport ratio tensor
        threshold: Maximum acceptable absolute value
        
    Returns:
        (is_valid, stats) - validation result and statistics
    """
    abs_max = log_ratio.abs().max().item()
    mean_val = log_ratio.mean().item()
    std_val = log_ratio.std().item()
    
    stats = {
        "abs_max": abs_max,
        "mean": mean_val, 
        "std": std_val,
        "frac_capped": (log_ratio.abs() > threshold).float().mean().item()
    }
    
    is_valid = abs_max < threshold * 1.5  # Allow some margin
    
    return is_valid, stats


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format.
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer or None if not found
    """
    import re
    
    # Find last \boxed{...} in text
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = list(re.finditer(pattern, text))
    
    if matches:
        return matches[-1].group(1)
    
    return None
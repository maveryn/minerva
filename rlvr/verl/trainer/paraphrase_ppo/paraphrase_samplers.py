# verl/trainer/paraphrase_ppo/paraphrase_samplers.py
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

"""Sampling utilities for paraphrase generation with entropy tracking."""

import random
import uuid
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.utils.debug import log_gpu_memory_usage
from verl.protocol import DataProto
from verl.trainer.ppo.reward import compute_reward as _compute_reward

from .paraphrase_prompts import (
    build_answer_rewrite_prompt,
    build_question_paraphrase_prompt,
    extract_paraphrased_question,
    format_answer_style_context
)
from .paraphrase_utils import (
    SampleRecord,
    compute_stable_qid,
    extract_boxed_answer,
    select_seed_success,
    select_group,
    replace_one
)


def compute_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute entropy for each token position.
    
    Args:
        logits: Logits tensor [seq_len, vocab_size]
        temperature: Temperature for scaling
        
    Returns:
        entropies: Entropy per token [seq_len]
    """
    # Apply temperature
    logits = logits / temperature
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    log_probs = F.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)
    
    return entropies


def rollout_with_tracking(
    model_or_wg,
    tokenizer,
    prompt: str,
    k: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: Optional[torch.device] = None,
    track_entropy: bool = True
) -> List[Dict]:
    """Generate k rollouts with entropy tracking.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        k: Number of rollouts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature  
        top_p: Nucleus sampling threshold
        device: Target device
        track_entropy: Whether to track per-token entropy
        
    Returns:
        List of rollout dictionaries with:
            - tokens: Token ids
            - text: Decoded text
            - logprobs: Token log probabilities
            - entropies: Per-token entropies (if track_entropy)
            - path_entropy: Average entropy
    """
    # Handle both direct model and worker group interfaces
    if hasattr(model_or_wg, 'generate_sequences'):
        # Use worker group interface
        # Tokenize prompt to get tensors
        tokenized = tokenizer(
            [prompt] * k,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        batch_dict = {
            "prompts": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
        gen_batch = DataProto.from_single_dict(batch_dict)
        gen_batch.meta_info["temperature"] = temperature
        gen_batch.meta_info["top_p"] = top_p
        gen_batch.meta_info["max_new_tokens"] = max_new_tokens
        
        gen_output = model_or_wg.generate_sequences(gen_batch)
        
        rollouts = []
        for i in range(k):
            response_tokens = gen_output.batch["responses"][i]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Extract log probs if available
            logprobs = gen_output.batch.get("log_probs", [None] * k)[i]
            if logprobs is None:
                logprobs = torch.zeros_like(response_tokens, dtype=torch.float32)
            
            rollouts.append({
                'tokens': response_tokens,
                'text': response_text,
                'logprobs': logprobs,
                'entropies': None,  # TODO: extract from generation
                'path_entropy': 0.0  # TODO: compute from entropies
            })
        return rollouts
    
    # Fallback to direct model interface (for local testing)
    model = model_or_wg
    model.eval()
    rollouts = []
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_tensor = torch.tensor(prompt_ids, device=device)
    
    for _ in range(k):
        generated_tokens = []
        token_logprobs = []
        token_entropies = []
        
        # Initialize with prompt
        input_ids = prompt_tensor.unsqueeze(0)
        past_key_values = None
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[0, -1, :] / temperature  # [vocab_size]
                past_key_values = outputs.past_key_values
            
            # Track entropy if requested
            if track_entropy:
                entropy = compute_token_entropy(logits.unsqueeze(0), temperature=1.0)[0]
                token_entropies.append(entropy.item())
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Store token and logprob
            generated_tokens.append(next_token.item())
            token_logprobs.append(F.log_softmax(logits, dim=-1)[next_token].item())
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Update input for next step
            input_ids = next_token.unsqueeze(0)
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Compute average entropy
        path_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
        
        rollouts.append({
            'tokens': torch.tensor(generated_tokens, device=device),
            'text': generated_text,
            'logprobs': torch.tensor(token_logprobs, device=device),
            'entropies': torch.tensor(token_entropies, device=device) if track_entropy else None,
            'path_entropy': path_entropy
        })
    
    return rollouts


def score_reward_under_plain(
    verifier,
    tokenizer,
    prompt_plain: str,
    response_text: str,
    gold_answer: Optional[str] = None,
) -> int:
    """Score response under plain prompt using the reward manager.
    
    This builds a minimal DataProto with `prompts` and `responses` and calls the
    configured reward manager to obtain a token-level reward tensor. A sequence
    is considered correct if the summed reward > 0.5.
    
    Args:
        verifier: Reward manager (AbstractRewardManager)
        tokenizer: Tokenizer to encode texts
        prompt_plain: Original prompt x
        response_text: Generated response  
        gold_answer: Optional gold answer for verification (unused here)
        
    Returns:
        reward: 0 or 1
    """
    try:
        # Encode prompt and response
        prompt_ids = tokenizer.encode(prompt_plain, add_special_tokens=True)
        resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if len(resp_ids) == 0:
            return 0

        prompts = torch.tensor([prompt_ids], dtype=torch.long)
        responses = torch.tensor([resp_ids], dtype=torch.long)
        attention_mask = torch.ones(1, len(prompt_ids) + len(resp_ids), dtype=torch.float32)

        batch = DataProto.from_dict({
            'prompts': prompts,
            'responses': responses,
            'attention_mask': attention_mask,
        })

        reward_tensor, _ = _compute_reward(batch, verifier)
        seq_score = reward_tensor.sum(-1).item()
        return int(seq_score > 0.5)
    except Exception:
        # Fallback to simple heuristic using boxed answer if reward manager fails
        extracted = extract_boxed_answer(response_text)
        if extracted is None:
            return 0
        if gold_answer and extracted:
            return int(extracted.strip() == gold_answer.strip())
        return 0


def generate_answer_paraphrase(
    model_or_wg,
    tokenizer,
    seed_record: SampleRecord,
    answer_styles: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[torch.device] = None
) -> Optional[SampleRecord]:
    """Generate answer paraphrase from a seed success.
    
    Args:
        model: Language model
        tokenizer: Tokenizer  
        seed_record: Seed success record
        answer_styles: List of style instructions
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        device: Target device
        
    Returns:
        New SampleRecord with paraphrased answer or None if generation fails
    """
    # Randomly select a style
    style = random.choice(answer_styles)
    
    # Build answer rewrite prompt
    prompt = build_answer_rewrite_prompt(
        question_plain=seed_record.prompt_plain,
        solution_text=seed_record.text,
        style_instruction=style
    )
    
    # Generate paraphrase
    rollouts = rollout_with_tracking(
        model_or_wg=model_or_wg,
        tokenizer=tokenizer,
        prompt=prompt,
        k=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device
    )
    
    if not rollouts:
        return None
    
    rollout = rollouts[0]
    
    # Create context used for generation
    context = format_answer_style_context(seed_record.prompt_plain, style)
    
    # Create new sample record
    new_record = SampleRecord(
        qid=seed_record.qid,
        round_id=seed_record.round_id,
        group_id=seed_record.group_id,  # Same group as seed
        ctx_type="APARA",
        source_id="OFF_APARA", 
        sid=str(uuid.uuid4())[:8],
        parent_sid=seed_record.sid,
        prompt_plain=seed_record.prompt_plain,
        prompt_used=context,
        tokens=rollout['tokens'],
        reward=0,  # Will be scored later
        lengths=len(rollout['tokens']),
        path_entropy=rollout['path_entropy'],
        old_logprobs_ctx=rollout['logprobs'],
        text=rollout['text']
    )
    
    return new_record


def generate_question_paraphrases(
    model_or_wg,
    tokenizer,
    prompt_plain: str,
    num_paraphrases: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    device: Optional[torch.device] = None
) -> List[str]:
    """Generate question paraphrases that preserve the answer.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt_plain: Original question
        num_paraphrases: Number of paraphrases to generate
        max_new_tokens: Maximum tokens per paraphrase
        temperature: Generation temperature
        device: Target device
        
    Returns:
        List of paraphrased questions
    """
    # Build paraphrase prompt
    prompt = build_question_paraphrase_prompt(prompt_plain)
    
    paraphrases = []
    
    for _ in range(num_paraphrases):
        rollouts = rollout_with_tracking(
            model_or_wg=model_or_wg,
            tokenizer=tokenizer,
            prompt=prompt,
            k=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            track_entropy=False  # Don't need entropy for paraphrase generation
        )
        
        if rollouts:
            # Extract paraphrased question from response
            paraphrase = extract_paraphrased_question(rollouts[0]['text'])
            if paraphrase and paraphrase != prompt_plain:
                paraphrases.append(paraphrase)
    
    return paraphrases


def collect_answer_paraphrase_batch(
    model_or_wg,
    tokenizer, 
    verifier,
    plain_records: List[SampleRecord],
    config: DictConfig,
    device: Optional[torch.device] = None
) -> Tuple[List[SampleRecord], List[tuple]]:
    """Collect batch with answer paraphrasing for zero-solve prompts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        verifier: Reward verifier
        plain_records: Plain GRPO rollout records
        config: Paraphrase configuration
        device: Target device
        
    Returns:
        Mixed batch of plain + paraphrased records
    """
    from .paraphrase_utils import group_by_qid
    
    mixed_records = []
    ce_pairs = []
    
    # Group by question ID
    qid_groups = group_by_qid(plain_records)
    
    for qid, records in qid_groups.items():
        # Check if any success exists
        has_success = any(r.reward == 1 for r in records)
        
        if not has_success:
            # No success - keep plain records only
            mixed_records.extend(records)
            continue
        
        # Select seed success
        seed = select_seed_success(records, criterion=config.select.seed_correct)
        
        if seed is None:
            mixed_records.extend(records)
            continue
        
        # Generate answer paraphrase
        paraphrase = generate_answer_paraphrase(
            model_or_wg=model_or_wg,
            tokenizer=tokenizer,
            seed_record=seed,
            answer_styles=config.answer_styles,
            max_new_tokens=config.max_new_tokens,
            temperature=0.7,
            device=device
        )
        
        if paraphrase is None:
            mixed_records.extend(records)
            continue
        
        # Score paraphrase under plain prompt
        paraphrase.reward = score_reward_under_plain(
            verifier=verifier,
            tokenizer=tokenizer,
            prompt_plain=seed.prompt_plain,
            response_text=paraphrase.text,
        )
        
        if paraphrase.reward == 1:
            # Replace seed with paraphrase
            records = replace_one(records, seed, paraphrase)
            
            # Add to CE buffer if enabled
            if config.ce_distill.enabled:
                ce_pairs.append((seed.prompt_plain, paraphrase.text))
        
        mixed_records.extend(records)
    
    return mixed_records, ce_pairs


def collect_question_paraphrase_batch(
    model_or_wg,
    tokenizer,
    verifier,
    plain_records: List[SampleRecord],
    prompts: List[str],
    config: DictConfig,
    device: Optional[torch.device] = None
) -> Tuple[List[SampleRecord], List[tuple]]:
    """Collect batch with question paraphrasing.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        verifier: Reward verifier
        plain_records: Plain GRPO rollout records
        prompts: Original prompts
        config: Paraphrase configuration
        device: Target device
        
    Returns:
        Selected mixed batch with balanced solve rate
    """
    all_records = plain_records[:]
    ce_pairs = []
    
    # Generate paraphrases for each prompt
    for prompt in prompts:
        qid = compute_stable_qid(prompt)
        round_id = plain_records[0].round_id if plain_records else 0
        
        # Generate question paraphrases
        paraphrases = generate_question_paraphrases(
            model_or_wg=model_or_wg,
            tokenizer=tokenizer,
            prompt_plain=prompt,
            num_paraphrases=config.paraphrase_per_x,
            max_new_tokens=256,
            temperature=0.8,
            device=device
        )
        
        # Generate rollouts for each paraphrase
        for x_prime in paraphrases:
            rollouts = rollout_with_tracking(
                model_or_wg=model_or_wg,
                tokenizer=tokenizer,
                prompt=x_prime,
                k=config.k_rollouts,
                max_new_tokens=config.max_new_tokens,
                temperature=0.7,
                device=device
            )
            
            # Create records for paraphrased rollouts
            for i, rollout in enumerate(rollouts):
                record = SampleRecord(
                    qid=qid,
                    round_id=round_id,
                    group_id=f"{qid}:{round_id}:QPARA",
                    ctx_type="QPARA",
                    source_id="OFF_QPARA",
                    sid=str(uuid.uuid4())[:8],
                    parent_sid=None,
                    prompt_plain=prompt,
                    prompt_used=x_prime,
                    tokens=rollout['tokens'],
                    reward=0,  # Will be scored
                    lengths=len(rollout['tokens']),
                    path_entropy=rollout['path_entropy'],
                    old_logprobs_ctx=rollout['logprobs'],
                    text=rollout['text']
                )
                
                # Score under plain prompt
                record.reward = score_reward_under_plain(
                    verifier=verifier,
                    tokenizer=tokenizer,
                    prompt_plain=prompt,
                    response_text=rollout['text'],
                )
                
                all_records.append(record)
                
                # Add correct samples to CE buffer
                if record.reward == 1 and config.ce_distill.enabled:
                    ce_pairs.append((prompt, rollout['text']))
    
    # Group by qid and select balanced groups
    from .paraphrase_utils import group_by_qid
    qid_groups = group_by_qid(all_records)
    
    selected_records = []
    for qid, records in qid_groups.items():
        group = select_group(
            records=records,
            target_solve_rate=config.target_solve_rate,
            correct_topk=config.select.correct_topk,
            incorrect_fill=config.select.incorrect_fill,
            k_rollouts=config.k_rollouts
        )
        selected_records.extend(group)
    
    return selected_records, ce_pairs
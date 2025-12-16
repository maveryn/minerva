# verl/trainer/paraphrase_ppo/paraphrase_logging.py
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

"""Logging utilities for paraphrase PPO training."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import numpy as np

from .paraphrase_utils import SampleRecord, group_by_qid, group_by_group_id


class ParaphraseMetricsTracker:
    """Track and log metrics for paraphrase PPO training."""
    
    def __init__(self, log_dir: str, write_every_steps: int = 50):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.write_every_steps = write_every_steps
        self.step_count = 0
        self.metrics_buffer = []
        self.debug_rows_buffer = []
    
    def log_batch_metrics(
        self,
        records: List[SampleRecord],
        ce_pairs: List[tuple],
        transport_stats: Dict[str, float],
        loss_dict: Dict[str, float],
        global_step: int
    ):
        """Log metrics for a training batch.
        
        Args:
            records: Sample records from the batch
            ce_pairs: CE distillation pairs
            transport_stats: Transport ratio statistics
            loss_dict: Loss components
            global_step: Global training step
        """
        metrics = {}
        
        # Compute pass@1 by context type
        ctx_groups = defaultdict(list)
        for r in records:
            ctx_groups[r.ctx_type].append(r)
        
        for ctx_type, ctx_records in ctx_groups.items():
            rewards = [r.reward for r in ctx_records]
            metrics[f'rollout_pass@1/{ctx_type}'] = np.mean(rewards) if rewards else 0.0
            metrics[f'rollout_count/{ctx_type}'] = len(ctx_records)
        
        # Mixed group pass@1
        all_rewards = [r.reward for r in records]
        metrics['rollout_pass@1/mixed'] = np.mean(all_rewards) if all_rewards else 0.0
        
        # Group-level statistics
        group_stats = self._compute_group_stats(records)
        metrics.update(group_stats)
        
        # Entropy statistics by context type
        for ctx_type, ctx_records in ctx_groups.items():
            entropies = [r.path_entropy for r in ctx_records]
            if entropies:
                metrics[f'entropy/path_{ctx_type}_mean'] = np.mean(entropies)
                metrics[f'entropy/path_{ctx_type}_std'] = np.std(entropies)
        
        # Transport ratio statistics
        if transport_stats:
            metrics['transport/mean_log_ratio'] = transport_stats.get('mean', 0.0)
            metrics['transport/std_log_ratio'] = transport_stats.get('std', 0.0)
            metrics['transport/frac_capped'] = transport_stats.get('frac_capped', 0.0)
            metrics['transport/frac_clipped'] = transport_stats.get('frac_clipped', 0.0)
        
        # CE buffer statistics
        metrics['ce_distill/buffer_size'] = len(ce_pairs)
        
        # Loss components
        metrics.update({f'loss/{k}': v for k, v in loss_dict.items()})
        
        # Add global step
        metrics['global_step'] = global_step
        
        self.metrics_buffer.append(metrics)
        
        # Write debug rows
        if self.debug_rows_buffer:
            self._write_debug_rows(records, global_step)
        
        # Periodic write
        self.step_count += 1
        if self.step_count % self.write_every_steps == 0:
            self.flush()
    
    def log_debug_rows(self, records: List[SampleRecord]):
        """Add records to debug row buffer.
        
        Args:
            records: Sample records to log
        """
        for rec in records:
            debug_row = {
                'qid': rec.qid,
                'round_id': rec.round_id,
                'group_id': rec.group_id,
                'ctx_type': rec.ctx_type,
                'source_id': rec.source_id,
                'sid': rec.sid,
                'parent_sid': rec.parent_sid,
                'reward': rec.reward,
                'path_entropy': rec.path_entropy,
                'length': rec.lengths,
                'prompt_plain_hash': hash(rec.prompt_plain) % (10**8),
                'prompt_used_hash': hash(rec.prompt_used) % (10**8),
                'parse_error': rec.parse_error,
                'truncated': rec.truncated,
            }
            
            # Add transport ratio stats if available
            if rec.transport_log_ratio is not None:
                debug_row['mean_log_ratio'] = rec.transport_log_ratio.mean().item()
                debug_row['max_abs_log_ratio'] = rec.transport_log_ratio.abs().max().item()
            
            # Add truncated text snippets
            if rec.text:
                debug_row['text_start'] = rec.text[:200]
                debug_row['text_end'] = rec.text[-200:] if len(rec.text) > 200 else ''
            
            self.debug_rows_buffer.append(debug_row)
    
    def _compute_group_stats(self, records: List[SampleRecord]) -> Dict[str, float]:
        """Compute group-level statistics.
        
        Args:
            records: Sample records
            
        Returns:
            Dictionary of group statistics
        """
        stats = {}
        
        # Group solve rates
        groups = group_by_group_id(records)
        solve_rates = []
        group_sizes = []
        
        for group_id, group_records in groups.items():
            rewards = [r.reward for r in group_records]
            solve_rate = np.mean(rewards) if rewards else 0.0
            solve_rates.append(solve_rate)
            group_sizes.append(len(group_records))
        
        if solve_rates:
            stats['group/solve_rate_mean'] = np.mean(solve_rates)
            stats['group/solve_rate_std'] = np.std(solve_rates)
            stats['group/size_mean'] = np.mean(group_sizes)
        
        # Question-level statistics
        qid_groups = group_by_qid(records)
        unique_solves = sum(1 for qid, recs in qid_groups.items() 
                           if any(r.reward == 1 for r in recs))
        stats['unique_solves'] = unique_solves
        stats['unique_questions'] = len(qid_groups)
        
        return stats
    
    def _write_debug_rows(self, records: List[SampleRecord], global_step: int):
        """Write debug rows to file.
        
        Args:
            records: Sample records
            global_step: Current global step
        """
        if not self.debug_rows_buffer:
            return
        
        filename = self.log_dir / f'step_{global_step:07d}.jsonl'
        
        with open(filename, 'w') as f:
            for row in self.debug_rows_buffer:
                f.write(json.dumps(row) + '\n')
        
        self.debug_rows_buffer.clear()
    
    def flush(self):
        """Write buffered metrics to disk."""
        if not self.metrics_buffer:
            return
        
        metrics_file = self.log_dir / 'metrics.jsonl'
        
        # Append to existing file
        with open(metrics_file, 'a') as f:
            for metrics in self.metrics_buffer:
                f.write(json.dumps(metrics) + '\n')
        
        self.metrics_buffer.clear()
    
    def log_epoch_summary(self, epoch: int, epoch_metrics: Dict[str, float]):
        """Log epoch-level summary.
        
        Args:
            epoch: Epoch number
            epoch_metrics: Aggregated epoch metrics
        """
        summary = {
            'epoch': epoch,
            'type': 'epoch_summary',
            **epoch_metrics
        }
        
        summary_file = self.log_dir / 'epoch_summaries.jsonl'
        
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')


def compute_transport_statistics(
    records: List[SampleRecord],
    clip_ratio: float = 0.2
) -> Dict[str, float]:
    """Compute statistics on transport ratios.
    
    Args:
        records: Sample records with transport ratios
        clip_ratio: PPO clip ratio
        
    Returns:
        Dictionary of statistics
    """
    all_ratios = []
    
    for rec in records:
        if rec.transport_log_ratio is not None:
            all_ratios.append(rec.transport_log_ratio)
    
    if not all_ratios:
        return {}
    
    # Concatenate all ratios
    log_ratios = torch.cat([r.flatten() for r in all_ratios])
    ratios = torch.exp(log_ratios)
    
    stats = {
        'mean': log_ratios.mean().item(),
        'std': log_ratios.std().item(),
        'max': log_ratios.max().item(),
        'min': log_ratios.min().item(),
        'frac_capped': (log_ratios.abs() > 8.0).float().mean().item(),
        'frac_clipped': torch.logical_or(
            ratios < (1 - clip_ratio),
            ratios > (1 + clip_ratio)
        ).float().mean().item()
    }
    
    return stats


def log_paraphrase_examples(
    records: List[SampleRecord],
    num_examples: int = 5,
    log_file: Optional[str] = None
):
    """Log example paraphrases for inspection.
    
    Args:
        records: Sample records
        num_examples: Number of examples to log
        log_file: Optional file to write examples
    """
    # Find pairs of original and paraphrased
    examples = []
    
    # Group by qid
    qid_groups = group_by_qid(records)
    
    for qid, qid_records in list(qid_groups.items())[:num_examples]:
        example = {'qid': qid}
        
        # Find PLAIN and paraphrased samples
        plain_samples = [r for r in qid_records if r.ctx_type == "PLAIN"]
        apara_samples = [r for r in qid_records if r.ctx_type == "APARA"]
        qpara_samples = [r for r in qid_records if r.ctx_type == "QPARA"]
        
        if plain_samples:
            example['prompt_plain'] = plain_samples[0].prompt_plain
            example['plain_responses'] = [
                {'text': r.text[:500], 'reward': r.reward, 'entropy': r.path_entropy}
                for r in plain_samples[:2]
            ]
        
        if apara_samples:
            example['answer_paraphrases'] = [
                {
                    'context': r.prompt_used,
                    'text': r.text[:500],
                    'reward': r.reward,
                    'entropy': r.path_entropy
                }
                for r in apara_samples[:2]
            ]
        
        if qpara_samples:
            example['question_paraphrases'] = [
                {
                    'question': r.prompt_used,
                    'text': r.text[:500],
                    'reward': r.reward,
                    'entropy': r.path_entropy
                }
                for r in qpara_samples[:2]
            ]
        
        examples.append(example)
    
    # Log or write to file
    if log_file:
        with open(log_file, 'w') as f:
            json.dump(examples, f, indent=2)
    else:
        for ex in examples:
            print(f"\n{'='*80}")
            print(f"QID: {ex['qid']}")
            print(f"Prompt: {ex.get('prompt_plain', 'N/A')[:200]}...")
            if 'plain_responses' in ex:
                print("\nPlain responses:")
                for i, resp in enumerate(ex['plain_responses']):
                    print(f"  [{i}] Reward={resp['reward']}, Entropy={resp['entropy']:.3f}")
            if 'answer_paraphrases' in ex:
                print("\nAnswer paraphrases:")
                for i, para in enumerate(ex['answer_paraphrases']):
                    print(f"  [{i}] Reward={para['reward']}, Entropy={para['entropy']:.3f}")
            if 'question_paraphrases' in ex:
                print("\nQuestion paraphrases:")
                for i, para in enumerate(ex['question_paraphrases']):
                    print(f"  [{i}] Reward={para['reward']}, Entropy={para['entropy']:.3f}")
                    print(f"      Q: {para['question'][:100]}...")
            print(f"{'='*80}\n")
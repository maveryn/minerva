# Verification of Paraphrase PPO Fix

## Problem Analysis

The error occurred in `verl/utils/seqlen_balancing.py` at line 112:
```
TypeError: 'float' object cannot be interpreted as an integer
```

This happened when creating a `State` object with `k=k_partitions` where `k_partitions` was a float instead of an integer.

## Root Cause

The issue traced back through:
1. `paraphrase_trainer.py` calls `compute_log_prob` on a filtered batch
2. `fsdp_workers.py` forwards to `dp_actor.py`
3. `dp_actor.py` calls `prepare_dynamic_batch` 
4. `prepare_dynamic_batch` calls `rearrange_micro_batches`
5. `rearrange_micro_batches` calculates `num_micro_batches` which could be a float
6. This float is passed to `get_seqlen_balanced_partitions` as `k_partitions`

## Fixes Applied

### Fix 1: Preserve meta_info in _filter_batch_by_mask
**File**: `/workspace/verl/trainer/paraphrase_ppo/paraphrase_trainer.py`

The `_filter_batch_by_mask` method was not preserving the `meta_info` from the original batch. This caused the filtered batch passed to `compute_log_prob` to be missing critical configuration parameters.

**Change**: Added code to preserve meta_info:
```python
# Preserve meta_info from original batch
if hasattr(batch, 'meta_info'):
    filtered_batch.meta_info = batch.meta_info.copy() if hasattr(batch.meta_info, 'copy') else dict(batch.meta_info)
```

### Fix 2: Set required meta_info fields
**File**: `/workspace/verl/trainer/paraphrase_ppo/paraphrase_trainer.py`

The `_convert_records_to_dataproto` method now sets the required meta_info fields for `compute_log_prob`:

```python
# Add required fields for compute_log_prob from config if available
if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
    rollout_cfg = self.config.actor_rollout_ref.rollout
    if hasattr(rollout_cfg, 'log_prob_micro_batch_size_per_gpu'):
        meta_info["micro_batch_size"] = int(rollout_cfg.log_prob_micro_batch_size_per_gpu)
    if hasattr(rollout_cfg, 'log_prob_max_token_len_per_gpu'):
        meta_info["max_token_len"] = int(rollout_cfg.log_prob_max_token_len_per_gpu)
    if hasattr(rollout_cfg, 'log_prob_use_dynamic_bsz'):
        meta_info["use_dynamic_bsz"] = rollout_cfg.log_prob_use_dynamic_bsz
    if hasattr(rollout_cfg, 'temperature'):
        meta_info["temperature"] = rollout_cfg.temperature
```

Note the explicit `int()` conversion for numeric fields to ensure they are integers.

## Why This Fixes the Issue

1. By preserving `meta_info` in `_filter_batch_by_mask`, we ensure the batch passed to `compute_log_prob` has all necessary configuration.

2. By explicitly converting numeric fields to integers in `_convert_records_to_dataproto`, we prevent float values from propagating through the system.

3. The existing safeguards in `seqlen_balancing.py` (lines 30, 181, 304) will now receive properly typed values.

## Validation

The fixes ensure that:
- `meta_info` is always present when calling `compute_log_prob`
- Numeric configuration values are properly typed as integers
- The error "TypeError: 'float' object cannot be interpreted as an integer" should no longer occur

The changes are minimal and focused only within the `verl/trainer/paraphrase_ppo` package as requested.
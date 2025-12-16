# Paraphrase Trainer Fixes

## Issue
The paraphrase training script was throwing a `KeyError: 'key "prompts" not found in TensorDict with keys []'` error when trying to run.

## Root Cause Analysis
1. The dataset returns `input_ids` as the key for tokenized prompts, not `prompts`
2. The `_get_gen_batch()` method (inherited from base PPO trainer) pops keys from the batch, including `input_ids`
3. The paraphrase trainer was trying to access `batch.batch["prompts"]` (should be `input_ids`) AFTER `_get_gen_batch()` had already removed the key

## Fixes Applied

### Fix 1: Change key from "prompts" to "input_ids"
**Location**: Line 498 in `paraphrase_trainer.py`
```python
# Before:
prompts = self.tokenizer.batch_decode(
    batch.batch["prompts"], skip_special_tokens=True
)

# After:
prompts = self.tokenizer.batch_decode(
    batch.batch["input_ids"], skip_special_tokens=True
)
```

### Fix 2: Extract prompts BEFORE calling _get_gen_batch
**Location**: Lines 493-499 in `paraphrase_trainer.py`
```python
# Before:
gen_batch = self._get_gen_batch(batch)
gen_batch.meta_info["global_steps"] = self.global_steps

# Extract prompts for paraphrasing
prompts = self.tokenizer.batch_decode(
    batch.batch["input_ids"], skip_special_tokens=True
)

# After:
# Extract prompts for paraphrasing BEFORE _get_gen_batch pops the keys
prompts = self.tokenizer.batch_decode(
    batch.batch["input_ids"], skip_special_tokens=True
)

gen_batch = self._get_gen_batch(batch)
gen_batch.meta_info["global_steps"] = self.global_steps
```

## Key Insights
- The base PPO trainer's `_get_gen_batch()` method modifies the original batch by popping keys
- Dataset returns `input_ids`, not `prompts` as the key for tokenized text
- Order of operations matters when working with DataProto objects that get modified

## Testing
To run the fixed paraphrase training:
```bash
bash rlvrdistilscripts/qwen7b-paraphrase-answer.sh
```
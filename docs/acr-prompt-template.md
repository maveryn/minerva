# ACR Prompt Template (Training)

Source of truth:
- `minerva/acr_prompt.py` (`build_acr_block`), appended via `try_build_acr_messages`.
- Used in per-batch ACRD inside `rlvr/verl/trainer/ppo/ray_trainer.py` and in standalone ACR datasets via `rlvr/verl/utils/dataset/minerva_acr_dataset.py`.

The ACR block is appended to the last user message of the original RLVR prompt.

## Template (verbatim text; placeholders in <> )

```text
You are generating a reasoning trace for training.

GROUND_TRUTH_LABELS:
- <LABEL_1>
- <LABEL_2>

LABEL_REFERENCE:
<details_text>

Instructions:
- Write a short reasoning that would justify selecting the correct label(s) from the input.
- <TASK_OR_ENTITY_REASONING_HINT>
- Do NOT say or imply that the answer was provided (no phrases like "given the answer", "based on the provided label", "ground truth", etc.).
- End with the final answer in the same format required by the original task.
```

## Notes

- `LABEL_REFERENCE` is included only when `details_text` is provided by the label-details store. If the store is missing a value, the literal placeholder `(details omitted)` is inserted.
- The task/entity reasoning hint is optional; it is pulled from `data.acr.task_reasoning_hints` or `data.acr.entity_reasoning_hints` when configured.
- Prompt-length handling (truncating/omitting `details_text` or skipping ACR entirely) is implemented in `try_build_acr_messages`.

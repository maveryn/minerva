# ACR Leakage Filtering: Optional ML Classifier

This note explains why we may add a small classifier for ACR leak detection and how
to build a training set. It is an optional complement to the current normalization
and rule-based filters.

## Why consider ML?
- Rule/regex filters can miss obfuscated leak phrases (e.g., punctuation variants like `label(s)`).
- Verbatim/overlap checks catch copying but not paraphrased “label hint” language.
- A lightweight classifier can improve recall on subtle or paraphrased leak cues.

## Recommended model
- Small transformer classifier (e.g., MiniLM/DistilBERT) to keep inference cost low.
- Input can be response-only (meta-leak focus) or `LABEL_REFERENCE + RESPONSE` (paraphrase focus).
- Favor high recall; tune threshold on a held-out set.

## Dataset construction (LLM-assisted)
Use real outputs as the base and generate positives with targeted leakage:

1) Hybrid augmentation
   - Negatives: real ACR responses that pass current filters.
   - Positives: LLM-injected leaks into those same responses (add “label(s)”, “reference”, etc.).

2) Paired generation
   - For each prompt, ask the LLM for a clean response and a leaky response.
   - This yields balanced pairs with matched content.

3) Weak supervision + judge
   - Auto-label with current rules (easy positives/negatives).
   - Use an LLM judge only on uncertain cases to expand coverage.

4) Adversarial edge cases
   - Ask the LLM to generate variants that bypass rules (`label(s)`, `label[s]`, hyphenation).
   - Use these as hard positives.

## Evaluation guidance
- Split by prompt/label to prevent leakage across train/test.
- Measure precision/recall against a small human-verified set.
- Prefer recall; false positives can be limited via thresholding.

## Integration idea (high level)
- Run normalization + rules first (fast, deterministic).
- Apply the classifier to the remaining responses; reject if predicted leak.

This keeps costs low while improving robustness to phrasing variants and paraphrases.*** End Patch}Probably not; the tool is FREEFORM; use apply_patch accordingly. Let's craft patch. Use apply_patch with proper patch format. Need to escape backticks? not necessary. Keep ASCII. Let's do. Let's ensure apply_patch content correct. Make sure final line newline. We'll do. Continue. 

# Minerva SFT Trace Builder

This folder builds SFT traces over the full Minerva training split (32k rows)
by generating model responses on the **default Minerva prompt** and retrying
with an answer‑conditioned ACRD prompt if needed.

High‑level flow per sample:

1) Generate with the default prompt (CTI system + user prompt from split JSONL).
2) If reward < 1, generate a second time using the ACRD prompt block
   (GROUND_TRUTH_LABELS + optional CANONICAL_LABEL_DETAILS).
3) If still incorrect, fall back to the dataset ground‑truth answer.

The output is a single JSONL file with all metadata + the final response.

## Output files

- `minerva-sft/data/sft-trace.jsonl` (default): all rows with attempts + final response

## Dependencies

- Python deps already used in the repo: `transformers`, `torch`, `tqdm`
- Hugging Face auth: set `HF_TOKEN` for gated models
- OpenAI judge-style models are supported via `athena_eval.models` if needed

## Run

```bash
python minerva-sft/scripts/build_sft_trace.py \
  --input dataset/minerva_base_split/minerva-base-train.jsonl \
  --output minerva-sft/data/sft-trace.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct
```

### Testing on a small subset

```bash
python minerva-sft/scripts/build_sft_trace.py \
  --input dataset/minerva_base_split/minerva-base-train.jsonl \
  --output minerva-sft/data/sft-trace_sample.jsonl \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --limit 100
```

## Notes

- The first attempt uses the default Minerva prompt (system + user) and the
  model’s chat template (if available).
- The retry attempt uses the same ACRD prompt template as the ACRD pipeline.
- If a sample is already present in the output file, it is skipped (resume).

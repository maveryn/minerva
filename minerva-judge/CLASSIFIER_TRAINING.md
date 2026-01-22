# Classifier Training (TextCNN + ModernBERT)

This note documents how we built the binary GOOD/BAD classifier dataset and trained the
two main model families we use today: TextCNN and ModernBERT.

## Dataset construction

Inputs:
- Judge outputs: `minerva-judge/judge_data/judged_*.jsonl` (excluding `judged_gpt_oss_120b.jsonl`).
- Model responses: `minerva-judge/data/responses_*.jsonl`.
- Synthetic BADs: `minerva-judge/judge_data/synthetic_bad.jsonl` (optional; only rows with `correct=true`).

Builder:
- Script: `minerva-judge/classifier/build_dataset.py`.
- It joins judge rows to response rows by a stable key (`uid:model:variant[:attempt]`).
- Only rows with `judge_label` in {GOOD, BAD} are kept.
- Only rows with `correct=true` are kept (so all labels are on correct answers).
- Synthetic rows are included only if `correct=true`.
- `gpt_oss_120b` judge outputs are excluded by default.

Balancing and split:
- We downsample to balance GOOD/BAD (config: `--balance downsample`).
- We sample an equal number per class with `--sample-per-class` (32k each in the main run).
- Split is 80/20 within each class (`--train-ratio 0.8`).
- Output: `minerva-judge/classifier/data/train.jsonl`, `val.jsonl`, plus `summary.json`.

Example:
```bash
python minerva-judge/classifier/build_dataset.py \
  --judged-dir minerva-judge/judge_data \
  --responses-dir minerva-judge/data \
  --synthetic minerva-judge/judge_data/synthetic_bad.jsonl \
  --output-dir minerva-judge/classifier/data \
  --balance downsample \
  --sample-per-class 32000 \
  --train-ratio 0.8 \
  --seed 1337
```

## Input formatting

All models use prompt+response pairs and are designed to keep the full response intact.

ModernBERT (transformer):
- Implemented in `minerva-judge/classifier/train_longformer.py`.
- Tokenization uses paired inputs: `(prompt, response)`.
- Truncation: `truncation="only_first"` so the prompt is truncated first and the response is preserved.
- This matches the training intent: response is primary signal, prompt supplies context.

TextCNN:
- Implemented in `minerva-judge/classifier/baselines/train_textcnn.py`.
- Tokenization uses a simple regex (`[A-Za-z0-9_]+`) and lowercasing.
- `text_mode=response_prompt` builds tokens as:
  - response tokens first,
  - then a literal `sep` token,
  - then prompt tokens.
- If `max_tokens` is set:
  - response tokens are kept in full if they fit,
  - prompt tokens are truncated to remaining budget,
  - if response exceeds the budget, prompt tokens are dropped.

## ModernBERT training

Model:
- Base: `answerdotai/ModernBERT-base` (HF seq-classification).
- Tokenizer: same as model.

Common settings (2k runs):
- `max_length=2048`
- `batch_size=64`
- `eval_batch_size=64`
- `epochs=2`
- `bf16`

Hyperparameter sweep (2k):
- Learning rate: `1e-06`, `1e-05`, `2e-05`, `3e-05`, `5e-05`
- Scripts: `minerva-judge/classifier/scripts/train_modernbert_2k_lr*.sh`

4k runs:
- `max_length=4096`
- `batch_size=32` (memory-constrained)
- Scripts: `minerva-judge/classifier/scripts/train_modernbert_4k_lr*.sh`

Training entrypoint:
```bash
bash minerva-judge/classifier/scripts/train_modernbert_2k_lr5e-05.sh
```

Metrics:
- Saved to `outputs/<run_tag>/eval_metrics.json`.
- Results are appended to `minerva-judge/classifier/experiments/results.jsonl`
  by `minerva-judge/classifier/scripts/log_results.py`.

## TextCNN training

Architecture:
- Embedding layer (default 200d) + 1D convs over tokens + max pooling + FC.
- Kernels: {3,4,5} by default.
- Filters: 256 per kernel (default).

Default run:
- Script: `minerva-judge/classifier/scripts/train_textcnn_lr5e-4.sh`
- `max_tokens=3072`
- `max_vocab=100000`
- `dropout=0.3`
- `batch_size=128`
- `epochs=6`
- `lr=5e-4`

Pretrained embeddings (optional):
- Script: `minerva-judge/classifier/scripts/train_textcnn_lr5e-4_pretrained.sh`
- Set `TEXTCNN_EMB_PATH` to a word-vector file (e.g., GloVe 6B 200d).
- The embedding matrix is loaded into the model and fine-tuned by default.

Metrics:
- Saved to `outputs/baselines/<run_tag>/metrics.json`.

## Threshold sweeps

We sweep decision thresholds to tune precision/recall trade-offs:
- TextCNN: `minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py`
- HF models: `minerva-judge/classifier/scripts/eval_threshold_sweep.py`

Example:
```bash
python minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py \
  --model-dir minerva-judge/classifier/outputs/baselines/textcnn_lr5e-4 \
  --val-file minerva-judge/classifier/data/val.jsonl \
  --out minerva-judge/classifier/experiments/threshold_sweep_textcnn_lr5e-4.jsonl
```

## Validation results

Metrics below are from the saved `eval_metrics.json` (HF models) or `metrics.json`
(TextCNN) in `minerva-judge/classifier/outputs`.

ModernBERT (2k, prompt+response):

| run_tag | max_len | lr | prec | rec | f1 |
| --- | --- | --- | --- | --- | --- |
| modernbert-2k-lr1e-06 | 2048 | 1e-06 | 0.7627 | 0.7816 | 0.7720 |
| modernbert-2k-lr1e-05 | 2048 | 1e-05 | 0.8140 | 0.8934 | 0.8518 |
| modernbert-2k-lr3e-05 | 2048 | 3e-05 | 0.8232 | 0.9048 | 0.8621 |
| modernbert-2k-lr5e-05 | 2048 | 5e-05 | 0.8282 | 0.9000 | 0.8626 |
| modernbert-2k-lr5e-05-focal | 2048 | 5e-05 | 0.8975 | 0.7523 | 0.8185 |

TextCNN (prompt+response unless noted):

| run_tag | max_tokens | notes | prec | rec | f1 |
| --- | --- | --- | --- | --- | --- |
| textcnn_lr5e-4 | 3072 | default (k=3,4,5; 256 filters; 200d) | 0.8122 | 0.9041 | 0.8557 |
| textcnn_lr5e-4_pretrained | 3072 | pretrained word vecs | 0.8483 | 0.8030 | 0.8250 |
| textcnn_k2345 | 2048 | k=2,3,4,5 | 0.8153 | 0.8889 | 0.8505 |
| textcnn_wide | 2048 | 300d, 384 filters | 0.7970 | 0.9086 | 0.8492 |
| textcnn_lr1e-3 | 2048 | lr=1e-3 | 0.8469 | 0.8194 | 0.8329 |
| textcnn_lr3e-3 | 2048 | lr=3e-3 | 0.8369 | 0.8073 | 0.8219 |
| textcnn_response_only | 1024 | response-only | 0.8220 | 0.8366 | 0.8292 |

TextCNN threshold sweep (default `textcnn_lr5e-4`):

| threshold | prec | rec | f1 |
| --- | --- | --- | --- |
| 0.50 | 0.8122 | 0.9041 | 0.8557 |
| 0.55 | 0.8209 | 0.8892 | 0.8537 |
| 0.60 | 0.8286 | 0.8647 | 0.8462 |
| 0.65 | 0.8392 | 0.8398 | 0.8395 |
| 0.70 | 0.8545 | 0.8059 | 0.8295 |
| 0.75 | 0.8688 | 0.7588 | 0.8101 |
| 0.80 | 0.8839 | 0.7009 | 0.7819 |
| 0.85 | 0.8996 | 0.6162 | 0.7315 |
| 0.90 | 0.9235 | 0.5053 | 0.6532 |
| 0.95 | 0.9512 | 0.3259 | 0.4855 |

## Notes

- The classifier is trained only on responses that already pass the Minerva reward checks.
- This keeps labels aligned with “answer-correct, then GOOD/BAD by judge rubric.”
- For deployment in ACRD, we default to the TextCNN model with a high threshold
  to bias for precision.

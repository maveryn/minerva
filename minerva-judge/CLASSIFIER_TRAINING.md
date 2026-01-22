# Classifier Training (TextCNN)

This note documents how we build the GOOD/BAD classifier dataset and train the
TextCNN models.

## Dataset construction

Inputs:
- Judge outputs: `minerva-judge/judge_data/judged_*.jsonl` (excluding `judged_gpt_oss_120b.jsonl`).
- Model responses: `minerva-judge/data/responses_*.jsonl`.
- Synthetic BADs: `minerva-judge/judge_data/synthetic_bad.jsonl` (optional; only rows with `correct=true`).

Builder:
- Script: `minerva-judge/classifier/build_dataset.py`.
- Joins judge rows to response rows by a stable key (`uid:model:variant[:attempt]`).
- Keeps only `judge_label` in {GOOD, BAD}.
- Keeps only `correct=true` (labels are on correct answers).
- For responses, prefers `response_final` if present; otherwise strips final markers
  like `assistantfinal` from the raw response, then falls back to the raw response.
- Synthetic rows are included only if `correct=true`.
- `gpt_oss_120b` judge outputs are excluded by default, but synthetic BADs from
  gpt-oss-120b are kept.

Balancing and split:
- Downsample to balance GOOD/BAD.
- Sample equal number per class with `--sample-per-class` (32k each in the main run).
- Split 80/20 within each class.
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

## Input formatting (TextCNN)

Implementation: `minerva-judge/classifier/baselines/train_textcnn.py`.

Tokenization:
- Regex: `[A-Za-z0-9_]+` + lowercasing.
- `text_mode=response_prompt` builds tokens as:
  - response tokens first,
  - then a literal `sep`,
  - then prompt tokens.
- With `max_tokens` set:
  - response tokens are kept in full if they fit,
  - prompt tokens are truncated to the remaining budget,
  - if response exceeds the budget, prompt tokens are dropped.

## TextCNN training

Architecture:
- Embedding layer + 1D convs over tokens + max pooling + FC.
- Kernels: {3,4,5} (default).

Common settings:
- `text_mode=response_prompt`
- `max_vocab=100000`
- `batch_size=128`
- `epochs=5`

Outputs:
- `metrics.json` and `model.pt` under the run output directory.

## Hyperparameter sweep (current)

Script:
- `minerva-judge/classifier/scripts/run_all_textcnn_parallel.sh`
- Runs up to 4 jobs in parallel (one per GPU from `GPU_LIST`).

Grid:
- Learning rate: `3e-4`, `6e-4`, `1e-3`, `2e-3`, `4e-3`
- Kernel sizes: `3,4,5`
- Filters per kernel: `256`, `384`
- Dropout: `0`, `0.25`
- Embedding dim: `200`, `300`
- Max tokens: `2048`, `3072`

Outputs:
- Models: `minerva-judge/classifier/outputs/baselines/textcnn_sweep_<RUN_TAG>/...`
- Logs + aggregated results: `minerva-judge/classifier/experiments/logs/textcnn_sweep_<RUN_TAG>/`
  - `results.jsonl` (one JSON line per run with hyperparams + final metrics)

Run:
```bash
RUN_TAG=jan22_textcnn_v1 GPU_LIST=0,1,2,3 \
  bash minerva-judge/classifier/scripts/run_all_textcnn_parallel.sh
```

## Threshold sweep (best TextCNN so far)

Model:
- `textcnn_lr6e-4_k345_f384_e200_t3072_d0p25`

Threshold metrics (val):
```
threshold  precision  recall    f1
0.50       0.7814     0.9113    0.8413
0.55       0.7940     0.8962    0.8420
0.60       0.8060     0.8758    0.8394
0.65       0.8184     0.8548    0.8362
0.70       0.8314     0.8203    0.8258
0.75       0.8444     0.7742    0.8078
0.80       0.8564     0.7108    0.7768
0.85       0.8724     0.6258    0.7288
0.90       0.8950     0.4969    0.6390
0.95       0.9326     0.2787    0.4292
```

Operational note:
- We currently use `filter_threshold=0.75` for ACRD ML filtering.

## Threshold sweeps

We sweep decision thresholds to tune precision/recall trade-offs:
- `minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py`

Example:
```bash
python minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py \
  --model-dir minerva-judge/classifier/outputs/baselines/textcnn_sweep_<RUN_TAG>/<RUN_NAME> \
  --val-file minerva-judge/classifier/data/val.jsonl \
  --out minerva-judge/classifier/experiments/threshold_sweep_textcnn.jsonl
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

- The classifier is trained only on reward-correct responses.
- This keeps labels aligned with "answer-correct, then GOOD/BAD by judge rubric."

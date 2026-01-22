# Classifier Training (TextCNN)

This note documents how we build the GOOD/BAD classifier dataset and train the
TextCNN model used for ACRD ML filtering.

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

## Model input (TextCNN)

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

Vocabulary:
- Built from the training split.
- `max_vocab=100000`, `min_freq=2`.
- Special tokens: `<pad>` and `<unk>`.

## Model architecture

- Embedding layer (random init unless pretrained vectors are supplied).
- 1D convs over embeddings, ReLU, max-over-time pooling.
- Concatenate pooled features, apply dropout, then a 2-way linear classifier.
- Default kernel sizes: `3,4,5` (sweeped, not fixed).

## Training setup

- Optimizer: AdamW.
- Loss: cross-entropy over GOOD/BAD.
- Batch size: 128.
- Epochs: 5.
- Learning rate: swept (see below).
- Metrics: accuracy/precision/recall/f1 on the validation split each epoch.
- Outputs: `metrics.json` + `model.pt` per run.

## Hyperparameter sweep

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

Best model (used for ACRD ML filtering):
- `textcnn_lr6e-4_k345_f384_e200_t3072_d0p25`
- lr=6e-4, kernels=3/4/5, filters=384, embed_dim=200, max_tokens=3072, dropout=0.25

## Threshold sweep + selected threshold

Script:
- `minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py`

Example:
```bash
python minerva-judge/classifier/scripts/eval_textcnn_threshold_sweep.py \
  --model-dir minerva-judge/classifier/outputs/baselines/textcnn_sweep_<RUN_TAG>/<RUN_NAME> \
  --data-file minerva-judge/classifier/data/val.jsonl \
  --output minerva-judge/classifier/experiments/threshold_sweep_textcnn.jsonl
```

Threshold metrics (val) for `textcnn_lr6e-4_k345_f384_e200_t3072_d0p25`:
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

Selected threshold:
- We use `filter_threshold=0.75` for ACRD ML filtering.

## Notes

- The classifier is trained only on reward-correct responses.
- This keeps labels aligned with "answer-correct, then GOOD/BAD by judge rubric."

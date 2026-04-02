# DART-CTI v1-30k Dataset Note

## Scope

This note records how the `v1_30k` DART-CTI dataset was produced from the stopped plain-prompt collection run.

Artifacts:

- `dart-artifacts/dart_cti/datasets/train/attempts_v1.jsonl`: full stopped plain-generation attempt log
- `dart-artifacts/dart_cti/datasets/train/accepted_v1_30k.jsonl`: recovered first `30,000` accepted plain traces
- `dart-artifacts/dart_cti/datasets/train/train_v1_30k.parquet`: SFT parquet built from `accepted_v1_30k.jsonl`
- `dart-artifacts/dart_cti/datasets/train/summary_v1_30k.json`: machine-readable summary for the `30k` cutoff

## Generation Setup

Source train set:

- `rlvr/mydata/minerva_base/minerva_base_train.parquet`

Teacher and decoding setup from `dart/configs/dart_cti.yaml`:

- teacher model: `openai/gpt-oss-120b`
- backend: `vllm`
- batch size: `1024`
- `gpu_memory_utilization`: `0.95`
- `max_new_tokens`: `1024`
- temperature: `0.7`
- top-p: `0.95`

Plain `v1` collection rules:

- original prompt only, no answer-guided fill
- target accepted traces per question: `k_u = 2`
- safety cap: `100` attempts per question
- acceptance rule: keep a trace only when the Minerva verifier returns success
- training responses are cleaned before SFT export to remove `gpt-oss` channel markers such as `analysis` and `assistantfinal`

Run command:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python -m dart.run_dart_cti --config dart/configs/dart_cti.yaml --skip-v2
```

## Recovery Rule For v1-30k

The full stopped run did not finish the plain-stage `v1` cap. Instead, it produced a partial attempt log:

- total attempts in stopped run: `632,832`
- total accepted traces in stopped run: `30,045`

To construct `v1_30k`, we recovered the dataset chronologically from `attempts_v1.jsonl`:

1. Scan `attempts_v1.jsonl` from the beginning.
2. Keep only rows with `verifier_success = true`.
3. Stop at the first `30,000` accepted rows.
4. Assign `accepted_rank` per question in arrival order.
5. Set `trace_source = plain`.
6. Build the SFT parquet from those `30,000` recovered accepted traces.

This means `v1_30k` is the first `30,000` verifier-correct plain traces seen during the run, not a fully completed `k=2`, `cap=100` plain collection.

Excluded tail after the `30k` cutoff:

- excluded attempts: `11,726`
- excluded accepted traces: `45`

## Cutoff Statistics For v1-30k

Exact cutoff used for `v1_30k`:

- accepted traces kept: `30,000`
- total attempts required to reach `30,000` accepted traces: `621,106`
- average attempts per accepted trace: `20.7035`
- questions seen by the cutoff: `32,000`
- cutoff row UID: `minerva-base-train.jsonl:12884:reward_tactic_ids`

Accepted-trace coverage across the `32,000` train questions at the `30k` cutoff:

- `16,348` questions had `0` accepted traces
- `1,304` questions had `1` accepted trace
- `14,348` questions had `2` accepted traces

So, at the `30k` cutoff:

- questions with at least one accepted trace: `15,652`
- questions with two accepted traces: `14,348`

## Attempts Per Question

Maximum attempts observed for a single question at the `30k` cutoff:

- max attempt count: `32`
- questions at that max: `4,718`

Average attempts per seen question:

- `19.4096`

Frequency distribution of attempt counts per question at the `30k` cutoff:

```text
2  -> 8638
3  -> 1417
4  -> 738
5  -> 511
6  -> 396
7  -> 323
8  -> 240
9  -> 202
10 -> 186
11 -> 146
12 -> 162
13 -> 140
14 -> 99
15 -> 96
16 -> 100
17 -> 92
18 -> 86
19 -> 81
20 -> 71
21 -> 73
22 -> 57
23 -> 65
24 -> 69
25 -> 52
26 -> 55
27 -> 43
28 -> 49
29 -> 55
30 -> 42
31 -> 12998
32 -> 4718
```

Accepted-trace distribution per question at the `30k` cutoff:

```text
1 accepted trace  -> 1304 questions
2 accepted traces -> 14348 questions
```

## Output Files

Final recovered `v1_30k` artifacts:

- `dart-artifacts/dart_cti/datasets/train/accepted_v1_30k.jsonl`
- `dart-artifacts/dart_cti/datasets/train/train_v1_30k.parquet`
- `dart-artifacts/dart_cti/datasets/train/summary_v1_30k.json`

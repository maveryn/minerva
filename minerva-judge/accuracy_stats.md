# Minerva Judge Accuracy Stats (Overall)

Updated judge rubric summary (excluding `judged_gpt_oss_120b.jsonl`).
"Other/Invalid" means missing or unparseable judge labels.

## Model summary

| Model | GOOD | BAD | Other/Invalid | Total |
| --- | --- | --- | --- | --- |
| gpt_oss_20b | 17853 | 1415 | 14839 | 34107 |
| llama3_1_8b | 9973 | 6173 | 13441 | 29587 |
| llama3_2_3b | 6850 | 5080 | 12276 | 24206 |
| qwen3_4b | 12958 | 2035 | 17397 | 32390 |
| qwen3_8b | 5594 | 13539 | 5251 | 24384 |

Totals (5 models):
- GOOD: 53228
- BAD: 28242
- Other/Invalid: 63204
- Total: 144674

## BAD category share (percent within BAD)

| Model | 1:Leakage | 2:Incoherent | 3:Ungrounded | 4:Mismatch | 5:Unsupported | 6:Other |
| --- | --- | --- | --- | --- | --- | --- |
| gpt_oss_20b | 47.28% | 23.82% | 27.21% | 1.70% | 0.00% | 0.00% |
| llama3_1_8b | 79.25% | 0.97% | 6.88% | 12.89% | 0.00% | 0.00% |
| llama3_2_3b | 73.68% | 0.49% | 14.51% | 11.32% | 0.00% | 0.00% |
| qwen3_4b | 57.54% | 2.36% | 12.68% | 27.42% | 0.00% | 0.00% |
| qwen3_8b | 97.63% | 1.08% | 1.12% | 0.18% | 0.00% | 0.00% |

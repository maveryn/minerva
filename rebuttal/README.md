# Rebuttal Workspace

This directory is organized by artifact type.

- `source/`: submitted paper PDF and extracted paper text.
- `reviews/`: reviewer text files.
- `scripts/`: analysis/report generation scripts.
- `results/`: generated statistics, CI packets, and artifact summaries.
- `response/`: working notes, plans, and rebuttal drafts.

Useful regeneration commands from the repository root:

```bash
python rebuttal/scripts/analyze_llmbench_eval12.py
python rebuttal/scripts/analyze_llmbench_eval12_all_pairs.py
python rebuttal/scripts/build_eval12_pairwise_ci_packet.py
python rebuttal/scripts/build_eval12_absolute_ci_packet.py
```

The generated outputs are written to `rebuttal/results/` by default.

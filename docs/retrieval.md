# Retrieval (label docs + BM25)

This document describes the CTI retrieval path used by TARBA.

## Overview
- Label docs live under `dataset/retrieval/label_docs/*.jsonl`.
- A BM25 index is built under `dataset/retrieval/index/`.
- Retrieval runs in one of two modes:
  - `per_type`: query only the label-type-specific index (current default).
  - `global`: query a single merged index across all label types.
- The runtime flow is:
  - Tool call (`cti_retrieve`) -> `minerva.retrieval.server` -> `minerva.retrieval.engine.RetrievalEngine`.

## Modes and config
Set the mode when starting the retrieval server:
- `--retrieval_mode per_type` (default)
- `--retrieval_mode global`

When building label docs, a global file can be emitted from config:
- `configs/retrieval/label_docs.yaml` supports `retrieval_mode: global`.
- Or pass `--global` to `minerva.retrieval.build_label_docs`.

## Exact ID extraction
For label types with a known ID pattern (ATT&CK technique/tactic, mitigation, detection, CWE, CAPEC), the engine:
1) Extracts **all** matching IDs from the query (not just the first).
2) Inserts each exact match at the front of the results (score = `inf`).
3) Fills the remaining slots with BM25 results.

If the query contains more exact IDs than `topk`, only the first `topk` matches are returned.
Label types without a regex (e.g., `threat_actor_name`) rely purely on BM25.

Global mode extracts **all** ID types (technique, tactic, mitigation, detection, CWE, CAPEC) from the query, inserts them first, then fills the rest with BM25 across the merged corpus.

## Top-k and budgets
- The retrieval tool enforces `topk_cap` from `rlvr/cti-scripts/tool_config/cti_retrieval_tool.yaml` (currently 8).
- TARBA also supplies a per-call `budget_B`; if present, `topk` is capped by `budget_B`.
- Budgeting is **per call**. Multiple tool calls (if enabled) would each get their own cap unless a separate global budget is implemented.

## Query constraints
- Queries are limited to 128 characters.
- Use short keyword phrases; avoid pasting long lists.
- When an ID is known, include it in the query to force exact-match retrieval.

## Notes
- Restart the retrieval server after code changes to the engine or label docs.
- Label docs include `doc_id`, `canonical_id`, `name`, and `text_for_retrieval`; BM25 ranks by `text_for_retrieval`.

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

Global retrieval can also be triggered per request by sending `label_type=all`.
In that case the engine loads/merges global docs on demand even if the server
is running in `per_type` mode.

## Exact ID extraction
For label types with a known ID pattern (ATT&CK technique/tactic, mitigation, detection, CWE, CAPEC), the engine:
1) Extracts **all** matching IDs from the query (not just the first).
2) Computes a binary ID match score (1.0 if the doc's canonical ID is matched, else 0.0).
3) Adds that to a normalized BM25 score in [0, 1] to rank results.

Because BM25 is normalized to [0, 1], ID-matched docs always outrank non-ID docs (scores are in [1, 2] vs [0, 1]).
Label types without a regex (e.g., `threat_actor_name`) rely purely on normalized BM25.

Global mode extracts **all** ID types (technique, tactic, mitigation, detection, CWE, CAPEC) from the query and applies the same ID-bonus + normalized BM25 ranking across the merged corpus.

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
- Tool responses return the truncated `text_for_retrieval` (bounded by `max_snippet_chars=1536`).
- ATT&CK technique docs include linked mitigation IDs + names and detection strategy text (with IDs); procedure examples are intentionally excluded.
- Mitigation docs include a "Techniques Addressed by Mitigation" section listing technique IDs + names.
- The default label-doc build config omits `detection_id` docs; detection strategies still appear inside technique docs.

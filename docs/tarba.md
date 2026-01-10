# IMPLEMENTATION TASK: TARBA (Tool-Augmented Retrieval Budget Annealing) for Minerva RLVR

This document is the implementation spec for TARBA in this repo. It mirrors the SLHC pattern:
- dataset wrapper + online controller
- trainer patch for controller updates
- tool config + multi-turn rollout
- custom reward function

Reference for SGLang multi-turn + tools (verl):
- https://verl.readthedocs.io/en/v0.5.x/sglang_multiturn/search_tool_example.html

---

## Goal

Implement Tool-Augmented Retrieval RLVR with Budget Annealing for CTI label prediction tasks:
- Replace LHC hint injection with an agentic retrieval tool over canonical label docs (no dataset examples).
- Each RL episode is multi-turn: optional tool call -> tool returns top-B docs -> model answers.
- A per-task controller (keyed by data_source) anneals retrieval access down as no-tool accuracy improves.
- Add retrieval rank reward shaping.
- Evaluate in two modes: retrieval enabled (ret-on) and retrieval disabled (ret-off).

Optional extension:
- Distill from tool-augmented (ret-on) rollouts into no-tool prompts.

---

## Assumptions and data format

Minerva RLVR uses Verl RL format (Parquet) with columns:
- prompt: list of chat messages [{role, content}, ...]
- data_source: task key (e.g., "reward_technique_id", "reward_cwe_ids")
- reward_model: {ground_truth: ...}
- extra_info: dict (we will add tarba_* metadata here)
- uid: stable per-prompt identifier (create if missing)

Training uses GRPO/RLVR with repeated rollouts per prompt uid (G usually 8).

---

## Repo integration points (use these paths)

Minerva (dataset + taxonomy sources):
- Use corpus builders from minerva/analysis/retrieval_candidates.py where possible.
- Taxonomy caches live under dataset/ (generated artifacts):
  - dataset/mitre/enterprise-attack.json
  - dataset/capec/stix-capec.json
  - dataset/cwe/cwec_v4.19.xml

Verl (training + dataset wrappers + tools):
- Dataset wrappers live under rlvr/verl/utils/dataset/
- Reward functions live under rlvr/verl/utils/reward_score/
- Tools live under rlvr/verl/tools/
- Trainer patch is in rlvr/verl/trainer/ppo/ray_trainer.py

---

## Task 0: Task specs and label types (no hardcoding)

We must map data_source -> label_type (entity type) and multi-label behavior.

Preferred source:
- Reuse minerva/analysis/retrieval_candidates.py::TASK_SPECS and ID normalization rules.
  - label_type values include: attack_technique_id, attack_tactic_id, mitigation_id,
    detection_id, cwe_id, capec_id, threat_actor_name.

Fallback (for new tasks):
- Determine multi-label by inspecting reward_model.ground_truth:
  - str -> single-label
  - list[str] -> multi-label
- Derive label_type from data_source naming conventions.

Implement a small helper (module location flexible) that yields:
TASK_SPECS[data_source] = {
  "label_type": str,
  "is_multilabel": bool,
  "label_id_regex": optional regex,
}

Notes:
- Use the same normalization as retrieval_candidates (e.g., DET-1234 -> DET1234).
- If no mapping is available, TARBA should disable retrieval for that sample (B=0).

---

## Task 1: Build canonical label docs (no dataset examples)

Create a script that emits one JSONL per label_type from taxonomy sources.
Place it under minerva/retrieval/ (or scripts/) and use the existing corpus builders.

JSONL schema (one line per label):
```json
{
  "doc_id": "attack_technique_id:T1059.003",
  "label_type": "attack_technique_id",
  "canonical_id": "T1059.003",
  "name": "Windows Command Shell",
  "aliases": ["cmd.exe", "Command Prompt"],
  "short_definition": "Adversaries may abuse cmd.exe ...",
  "metadata": {
    "tactic": ["Execution"],
    "parent": "attack_technique_id:T1059"
  },
  "text_for_retrieval": "T1059.003 Windows Command Shell (cmd.exe, Command Prompt). Tactic: Execution. Definition: ..."
}
```

Rules:
- doc_id must be globally unique: "{label_type}:{canonical_id}".
- text_for_retrieval is the indexed field. Keep it compact and clean.
- Do not include dataset examples.

Recommended outputs (generated artifacts):
- dataset/retrieval/label_docs/<label_type>.jsonl
- dataset/retrieval/label_docs/_manifest.json (counts + hashes)

Example CLI:
```bash
python -m minerva.retrieval.build_label_docs \
  --config configs/retrieval/label_docs.yaml \
  --out_dir dataset/retrieval/label_docs
```

Config example (adapt to actual sources):
```yaml
entity_types:
  attack_technique_id:
    source_paths:
      - dataset/mitre/enterprise-attack.json
    id_field: "external_id"
    name_field: "name"
    aliases_field: "aliases"
    definition_field: "description"
  cwe_id:
    source_paths:
      - dataset/cwe/cwec_v4.19.xml
    id_field: "cwe_id"
    name_field: "name"
    aliases_field: "aliases"
    definition_field: "description"
```
Note: in this repo, `build_label_docs.py` uses the retrieval_candidates corpus builders; it only uses
`entity_types.source_paths` to override default files and ignores the field mapping keys above.

---

## Task 2: Retrieval engine + index + server

Implement a minimal retrieval stack:
- Exact ID match (fast path)
- BM25
- Optional dense/hybrid after BM25 works

Reuse tokenization and ID normalization from minerva/analysis/retrieval_candidates.py.

Engine API (example):
```
retrieve(label_type: str, query: str, topk: int) -> list[Result]
Result = {doc_id, canonical_id, title, snippet, score}
```

Behavior:
- If query contains an exact ID (via label_id_regex), return it as rank 1.
- Enforce caps: topk <= TOPK_CAP, max query chars, max snippet chars.
- Defaults: topk_cap=8, max_query_chars=128, max_snippet_chars=800.
- Keep snippets short to avoid prompt bloat.

Expose an HTTP server (FastAPI or Flask):
```
POST /retrieve
{
  "label_type": "attack_technique_id",
  "query": "windows command shell cmd.exe",
  "topk": 5,
  "return_scores": true
}
```

Response:
```
{
  "results": [
    {
      "doc_id": "attack_technique_id:T1059.003",
      "canonical_id": "T1059.003",
      "title": "Windows Command Shell",
      "snippet": "T1059.003 Windows Command Shell ...",
      "score": 12.3
    }
  ]
}
```

---

## Task 3: Tool integration (SGLang multi-turn)

Use SGLang multi-turn with tools in verl. Two options:

Option A (recommended): custom CTI retrieval tool
- File: rlvr/verl/tools/cti_retrieval_tool.py
- Base class: verl.tools.base_tool.BaseTool
- Tool name: cti_retrieve
- Args: query (string), topk (int)
- Tool executes HTTP POST to /retrieve and returns formatted text.

Notes:
- label_type is injected from the dataset (tools_kwargs) and not supplied by the model.
- The rollout layer accepts a plain JSON object with `query` and `topk` and wraps it into a cti_retrieve call.
- If `topk` is missing or invalid, the rollout defaults to 8 and the tool clamps to `budget_B` and `topk_cap`.

Tool output MUST include machine-parsable doc IDs:
```
DOC_IDS: ["attack_technique_id:T1059.003", "attack_technique_id:T1059.001"]
1) attack_technique_id:T1059.003 - Windows Command Shell
2) attack_technique_id:T1059.001 - PowerShell
```

Option B: use SearchTool (verl.tools.search_tool.SearchTool)
- Encode label_type in the query or use per-type endpoints.
- Only use if Option A is too costly.

Tool config YAML (example):
```yaml
tools:
  - class_name: verl.tools.cti_retrieval_tool.CTIRetrievalTool
    config:
      type: native
      retrieval_url: http://127.0.0.1:8000/retrieve
      timeout: 30
      topk_cap: 8
      max_query_chars: 128
      max_snippet_chars: 800
    tool_schema:
      type: function
      function:
        name: cti_retrieve
        description: Retrieve CTI label docs by keyword query.
        parameters:
          type: object
          properties:
            query: {type: string}
            topk: {type: integer}
          required: [query, topk]
```

Rollout config (SGLang multi-turn):
```yaml
actor_rollout_ref:
  rollout:
    name: sglang
    multi_turn:
      enable: true
      max_assistant_turns: 3
      tool_config_path: /abs/path/to/cti_retrieval_tool.yaml
      max_tool_response_length: 256
```

Notes:
- If `TARBA_HIDE_TOOL_SCHEMA=0`, ensure your model chat template supports tool calling (see rlvr/examples/sglang_multiturn).
- data.return_raw_chat=true is required for SGLang multi-turn tools (raw_prompt is needed by the rollout engine).
- To avoid injecting tool schemas into the prompt, set `TARBA_HIDE_TOOL_SCHEMA=1` (the training script defaults to on).
- When `TARBA_HIDE_TOOL_SCHEMA=1`, the model is instructed to output JSON with `query`/`topk` only; the rollout parses it and injects the tool call.
- Use `TARBA_DEBUG_SAMPLES=N` to log the prompt, model output, tool call, and post-tool prompt for the first N samples.

---

## Task 4: TARBA controller + dataset wrapper

Create rlvr/verl/utils/dataset/minerva_tarba_retrieval_dataset.py with:
- RetrievalBudgetController
- TarbaRLHFDataset (inherits RLHFDataset)

Controller state (per task key = data_source):
- B[t]: retrieval budget (topk cap), init B_max
- p_noret[t]: probability retrieval is disabled, init p_noret_init
- ema_acc_ret[t], ema_acc_noret[t]
- steps_ret[t], steps_noret[t]

Config fields under data.tarba:
- enabled: bool
- ema_beta: 0.90
- target_acc_noret: 0.70
- tol: 0.05
- p_noret_init: 0.10
- p_step: 0.05
- B_min: 0
- default_B_max: 8
- B_max_by_task: optional dict
- B_step: 1
- min_steps_before_anneal: 50
- seed: optional

Sampling rule per example:
- allow_retrieval = rng.random() >= p_noret[t]
- if allow_retrieval: budget_B = B[t]
- else: budget_B = 0

Update rule per uid group (acc@G):
- If allow_retrieval == False:
  - update ema_acc_noret
  - if ema_acc_noret > target_acc_noret + tol and steps_noret >= min_steps_before_anneal:
    - p_noret += p_step (cap 1.0)
    - optionally B -= B_step (floor B_min)
- If allow_retrieval == True:
  - update ema_acc_ret
  - optional slow decrease in B if ema_acc_ret is very high

Expose metrics:
- tarba_ctrl/B/<task>
- tarba_ctrl/p_noret/<task>
- tarba_ctrl/ema_ret/<task>
- tarba_ctrl/ema_noret/<task>

Dataset behavior (TarbaRLHFDataset):
- data.dataloader_num_workers can be > 0; no special restriction is required.
- Derive task_key = data_source and label_type from TASK_SPECS.
- Parse ground_truth into gold_ids (list of labels).
- gold_doc_ids = [f"{label_type}:{id}"]
- If task has no label_type mapping, set allow_retrieval=false and B=0.
- Insert tool instructions into the last user message:
  - If allow_retrieval:
    - "You may request retrieval at most once."
    - "Request at most B={budget_B} docs."
    - "Use short keyword queries; do not paste long lists. Limit query to 128 characters."
    - "Output exactly one JSON object with `query` and `topk` fields only; no extra text."
    - Examples:
      - {"query":"cwe-79 xss sql inj","topk":3}
      - {"query":"credential dumping dcsync process injection","topk":5}
      - {"query":"APT28 Lazarus FIN7 spearphishing banking malware","topk":2}
  - If not allowed:
    - "Retrieval disabled for this sample. Do not call tools."
- Store metadata in extra_info:
  - tarba_allow_retrieval
  - tarba_budget_B
  - tarba_label_type
  - tarba_gold_doc_ids
  - tarba_seed
  - tarba_prompt_no_tool (optional, for distillation)
- Inject tools_kwargs for cti_retrieve with execute_kwargs {budget_B, label_type}.

Prompt-length guardrail:
- If prompt is near max length, reduce B for this sample or rely on tool output truncation.

---

## Task 5: Retrieval-aware reward

Create rlvr/verl/utils/reward_score/reward_tarba.py:

reward_tarba(data_source, solution_str, ground_truth, extra_info=None) -> float or dict

Components:
- r_ans: answer verification via reward_minerva.reward_minerva
- r_ret: retrieval reward based on rank of gold docs
- penalties: tool call penalty, illegal tool call penalty

Rank-shaped retrieval reward (single-label):
- r_ret = exp(-alpha * (rank - 1)) if retrieved, else 0
- alpha = ln(2) gives rank1=1.0, rank2=0.5, rank3=0.25

Multi-label:
- Default policy "mean": average over gold labels (missing -> 0)
- Support optional policies: "min", "all_or_nothing"

Anti-gaming:
- Gate retrieval reward by answer correctness:
  - if r_ans < ans_threshold, set r_ret = 0
- Tool call cost:
  - r_total = r_ans + lambda_ret * r_ret - penalty_tool_call
- If retrieval disabled but tool called:
  - r_total -= penalty_illegal_tool

Gold rank computation (robust):
- Preferred: parse tool call args from solution_str (<tool_call> JSON) and re-run retrieval.
- Fallback: parse DOC_IDS: [...] from tool output text in solution_str.
- If no tool call, r_ret = 0.
Note: label_type is sourced from tarba_label_type/task specs; any label_type in tool calls is ignored unless the task has no label_type.

Return dict for logging (recommended):
- score (final reward)
- r_ans, r_ret, tool_called, illegal_tool, ranks
- is_correct (boolean) for controller updates

---

## Task 6: Trainer patch (controller update)

Patch rlvr/verl/trainer/ppo/ray_trainer.py near the SLHC update section:

- After reward computation, compute per-rollout correctness:
  - Prefer reward_extra_info["is_correct"] if available.
  - Else use reward_scalar >= score_threshold.
- Group by uid (same as SLHC):
  - acc_g = mean(correct) over G rollouts
  - task_key = data_source
  - allow_retrieval = extra_info["tarba_allow_retrieval"]
- Call dataset hook if present:
  - self.train_dataset.update_tarba_controller_from_groups(...)
- Log controller metrics returned by the dataset.

Guardrails:
- If fields missing, skip update gracefully.
- No special requirement on dataloader_num_workers.

---

## Task 7: Config + training/eval

Training config (Hydra CLI style):
```bash
custom_dataset_path="$ROOT_DIR/verl/utils/dataset/minerva_tarba_retrieval_dataset.py"
reward_fn_path="$ROOT_DIR/verl/utils/reward_score/reward_tarba.py"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.custom_cls.path=$custom_dataset_path \
  data.custom_cls.name=TarbaRLHFDataset \
  data.dataloader_num_workers=16 \
  data.return_raw_chat=true \
  +data.tarba.enabled=true \
  +data.tarba.p_noret_init=0.10 \
  +data.tarba.default_B_max=8 \
  +data.tarba.B_min=0 \
  +data.tarba.B_step=1 \
  +data.tarba.ema_beta=0.90 \
  +data.tarba.target_acc_noret=0.70 \
  +data.tarba.tol=0.05 \
  +data.tarba.min_steps_before_anneal=50 \
  custom_reward_function.path=$reward_fn_path \
  custom_reward_function.name=reward_tarba \
  +custom_reward_function.reward_kwargs.lambda_ret=0.2 \
  +custom_reward_function.reward_kwargs.alpha=0.6931 \
  +custom_reward_function.reward_kwargs.ans_threshold=0.5 \
  +custom_reward_function.reward_kwargs.penalty_tool_call=0.02 \
  +custom_reward_function.reward_kwargs.penalty_illegal_tool=0.5 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/abs/path/cti_retrieval_tool.yaml
```

Environment toggles:
- `TARBA_HIDE_TOOL_SCHEMA=1` to hide tool schemas and use JSON-only tool calls (default in the training script).
- `TARBA_DEBUG_SAMPLES=N` to emit debug logs for the first N samples.

Validation modes:
- ret-on: force retrieval allowed (p_noret=0, B_eval=5)
- ret-off: force retrieval disabled (p_noret=1, B_eval=0)

Log both:
- accuracy ret-on, accuracy ret-off
- tool-call rate
- retrieval recall@B, mean rank (if available)

---

## Task 8: Optional distillation (ret-on -> ret-off)

After TARBA stabilizes, distill tool-augmented correct answers into no-tool prompts:
- Collect ret-on rollouts where r_ans >= threshold.
- Use tarba_prompt_no_tool (stored in extra_info) as the prompt.
- Run SFT on (no-tool prompt, ret-on answer) pairs.
- This can be done offline or online (see docs/slhc-sft.md for a working OVSD pattern).

---

## Operational checklist

1) Build label docs (generated artifacts):
```bash
python -m minerva.retrieval.build_label_docs --config configs/retrieval/label_docs.yaml --out_dir dataset/retrieval/label_docs
```

2) Build retrieval index (if separate step):
```bash
python -m minerva.retrieval.build_index --label_docs_dir dataset/retrieval/label_docs --out_dir dataset/retrieval/index
```

3) Start retrieval server:
```bash
python -m minerva.retrieval.server --index_dir dataset/retrieval/index --host 0.0.0.0 --port 8000
```

4) Train TARBA RLVR:
```bash
python -m verl.trainer.main_ppo --config <tarba_config>
```

5) Evaluate ret-on and ret-off (two configs or CLI overrides).

---

## Guardrails

- Enforce tool caps in code (not only prompt): max tool calls per episode = 1.
- Topk clipped by budget B and by tool config topk_cap.
- If the model omits topk, the rollout defaults to 8 before clamping.
- Cap query length to 128 characters in tool/server.
- Truncate tool output snippets to avoid context bloat.
- Retrieval docs must be canonical (no dataset examples).
- Always report both ret-on and ret-off validation.
- If a task has no label_type mapping, disable retrieval for that task.

---

# END TASK

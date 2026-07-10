# Minimal Paper Writing Edits Needed for the Four TMLR Reviews

Scope: only prose edits still needed after the new result tables already added. Do not add more result tables unless a later decision explicitly changes this.

## Required Edits

None currently.

## Changes We Should Not Make Now

- Do not add more numeric tables unless we explicitly decide to.
- Do not add a task-family result table now.
- Do not promise multi-seed LLM RLVR.
- Do not claim full compute matching unless the text says exactly what is matched.
- Do not claim no-heuristic-only ablation unless that exact result exists.
- Do not rewrite unrelated related-work sections unless needed for space or claim calibration.
- Do not make broad claims about general reasoning or general cross-domain transfer.

## Minimal Editing Sequence

All planned prose edits are resolved.

## Resolved Edits

### Reframe the Theory

Why: R1, R2, and R4 say the theorem assumes the mechanism and should not be presented as full theory for GRPO/SFT language-model training.

Files:
- `paper/latex/sections/theory.tex`
- `paper/latex/appendix/minervarl_support.tex`

Status:
- Resolved in the main theory subsection and appendix.
- Added a main-text scope sentence framing the result as a finite-sampling support effect rather than a full optimization theory of joint GRPO and SFT language-model training.
- Added an appendix paragraph stating that exposure and distillation assumptions are modeling assumptions, not derived guarantees of neural LM optimization.
- Added appendix scope text noting that the result does not analyze gradient interference, partial-credit reward dynamics, checkpoint selection, or full LM fine-tuning dynamics.

### Add Data/Code Availability and Licensing Statement

Why: R2 explicitly asks for data/code availability and licensing; R4 asks how release plans handle misuse risk.

Files:
- `paper/latex/minerva-tmlr.tex`

Status:
- Resolved before the broader-impact statement.
- Added a concise release statement covering the codebase, data-construction pipeline, evaluation tools, and derived Minerva-CTI splits for research use.
- Added a licensing/attribution sentence noting public reuse-compatible CTI sources and preserving upstream notices, licenses, and attribution.

### Expand Broader Impact

Why: all four reviewers mention dual-use or reliability risk.

Files:
- `paper/latex/minerva-tmlr.tex`

Status:
- Resolved in the broader-impact statement.
- Added intended defensive uses: CTI analysis, entity extraction, vulnerability triage, ATT&CK/CWE/CVSS mapping, mitigation recommendation, and detection-rule interpretation.
- Added explicit non-goals and dual-use risks around exploit-related diagnostics, vulnerability knowledge organization, attack-technique mapping, target prioritization, and automated threat analysis.
- Added release mitigations and reliability framing: defensive-use documentation, model/data cards, cyber-safety review, staged release if needed, and analyst-assistive rather than authoritative use.

### Soften Main Result Claims

Why: R1, R2, and R4 object to strong aggregate claims and small margins without qualification.

Files:
- `paper/latex/sections/results.tex`
- `paper/latex/sections/conclusion.tex`

Status:
- Resolved in Section 6.1 and the conclusion.
- Replaced the old Section 6.1 framing with four table-aligned paragraphs: aggregate scores, controlled baselines, per-task paired uncertainty versus GRPO, and pairwise baseline summary.
- Removed the live “consistent gains over GRPO” language and cleaned stale commented-out prose that contained older stronger claims.
- Explicitly qualifies Qwen3-4B versus GRPO as the least separated comparison and LUFFY as the closest controlled baseline.
- The abstract remains unchanged because it explicitly reports an aggregate average across four backbones and 12 CTI benchmarks, which is already appropriately scoped.

### Clarify Task Heterogeneity Without Adding More Tables

Why: R1 asks for clearer task/metric interpretation and objects to relying only on the heterogeneous average.

Files:
- `paper/latex/sections/experiments.tex`
- `paper/latex/sections/results.tex`

Status:
- Resolved through the existing evaluation-metric paragraph and revised Section 6.1.
- Section 5 already describes the heterogeneous metrics across multiple-choice, taxonomy/mapping, CVSS scoring, extraction, multi-label, and subtask-average benchmarks.
- Section 6.1 now uses per-task paired CIs and task-count summaries as the main support around Tables 2 and 3.
- No task-family numeric table or additional task grouping sentence was added; this avoids unnecessary extra structure beyond the per-task evidence.

### Limitations Scope

Why: We reviewed the current limitations text and found that the genuinely needed limitations are already covered.

Files:
- `paper/latex/sections/limitations.tex`
- `paper/latex/sections/experiments.tex`

Status:
- Resolved with no additional manuscript edit.
- The uncertainty paragraph already states that paired bootstrap intervals quantify evaluation-example uncertainty for fixed checkpoints and do not estimate training-seed variation.
- The limitations section already covers compute overhead, including the 39.3\% timing-study overhead, and filtering limitations.
- We will not add a new verifier-scope limitation because verifiable rewards are the intended RLVR setting, not a defect of the approach.
- We will not add a hard-prompt failure taxonomy unless it becomes necessary for a rebuttal response.

### Clarify What ACR Does and Does Not Prove

Why: R3 and R4 worry that answer-conditioned rationales may teach answer-driven explanation patterns rather than genuine reasoning. R2 asks for answer-only controls.

Files:
- `paper/latex/sections/method.tex`
- `paper/latex/sections/results.tex`

Status:
- Resolved in `paper/latex/sections/method.tex` and `paper/latex/sections/results.tex`.
- Method text now states that label-revealing ACR prompts are used only to generate candidate training traces, accepted traces are distilled on original prompts, and all reported evaluations use the same answer-free prompts as GRPO.
- Ablation discussion now maps GRPO to the no-ACR control, GRPO 12 rollouts to the extra-rollout control, and in-loop answer-only SFT to the matched auxiliary-distillation/direct-answer control.
- Results text avoids claiming that the ablation proves reasoning improvement; it says the full ACR path is not reducible to direct final-answer SFT with matched auxiliary updates.
- Rebuttal-only: state that we do not measure mechanistic rationale faithfulness; the response-quality evaluation measures judged usefulness/correctness rather than faithful internal reasoning.

### Clarify Evaluation Protocol

Why: R2 and R4 explicitly ask for checkpoint selection, decoding setup, answer extraction, and evaluation procedure.

Files:
- `paper/latex/sections/experiments.tex`

Status:
- Resolved in `paper/latex/sections/experiments.tex`.
- Added main single-sample decoding details: greedy decoding, `temperature=0.0`, and `max_new_tokens=2048`.
- Added fixed benchmark-prompt wording, no model-specific prompt rewriting, task-output format summary, shared parser/normalizer/scorer wording, and empty-prediction handling for unparseable outputs.
- Added that external security-SFT baselines use the same evaluation rules.
- Clarified checkpoint selection: GRPO, MinervaRL, and controlled trained baselines use the same validation criterion, average Minerva-Dev and AthenaBench-Mini performance.
- Kept/updated the uncertainty paragraph to state that bootstrap CIs are fixed-checkpoint evaluation-example uncertainty and do not estimate training-seed variation.
- Rebuttal-only: state explicitly to R2 that AthenaBench-Mini is not fully instance-disjoint from Athena-derived final columns, and that comparisons remain symmetric because trained methods use the same selection rule. We are not adding this caveat to the manuscript unless the AE asks, since the paper avoids describing those columns as pure held-out transfer.

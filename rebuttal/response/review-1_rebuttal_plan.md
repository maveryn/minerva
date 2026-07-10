# Review 1 Rebuttal Plan

Target: one reviewer-specific response. This reviewer is skeptical but interested, so the response should be evidence-first and revision-oriented. Do not sound defensive. Do not omit requested uncertainty analyses: report absolute confidence intervals for model scores and paired bootstrap intervals for model comparisons.

## Response Strategy

Main message:

> We agree the original submission leaned too heavily on a heterogeneous average and did not make the method/evaluation accessible enough. We now add uncertainty, task-family/per-task analyses, rollout-aware held-out results, and final Eval12 ablations, and we will revise presentation/theory accordingly.

Use this order:

1. Statistical uncertainty and heterogeneous average
2. Rollout-aware evaluation
3. Component-wise ablations
4. Clarity, notation, and task explanation
5. Theory reframing
6. Broader impact

## Must Include

### 1. Uncertainty / aggregation

Reviewer concern:
- `Avg.` mixes accuracy, exact match, F1, extraction F1, and normalized CVSS.
- No CIs, bootstrap intervals, significance tests, or multi-seed variation.
- Wants per-task uncertainty, task-family averages, sample-size-aware analyses.

Response:
- Acknowledge `Avg.` should be supplemented, not be the only evidence.
- Add absolute score CIs over evaluation examples for each model/task.
- Add paired bootstrap delta CIs for model comparisons.
- Include per-task denominators and task-family summaries.
- Be explicit that Qwen-4B is not clearly separated on the 12-task average.

Required tables to add:
- Absolute score CI table: per-task score CIs for all models. Use `rebuttal/results/eval12_absolute_score_ci_packet.md`.
- Paired model-comparison CI table: `MinervaRL - baseline` CIs, including the central `MinervaRL - GRPO` per-task table. Use `rebuttal/results/eval12_pairwise_ci_packet.md`.
- Task-family CI table for `MinervaRL - GRPO`.
- Per-task denominators table or note, preferably next to the task/metric table.

Numbers:
- Llama-8B: +4.2 pp [3.5, 5.1]
- Llama-3B: +6.2 pp [5.3, 7.2]
- Qwen-8B: +6.4 pp [5.6, 7.7]
- Qwen-4B: +0.4 pp [-0.8, 1.3]
- Taxonomy mapping family: +9.6, +13.4, +7.2, +3.2 pp; all CIs exclude zero.

Suggested wording:

> We agree that the heterogeneous `Avg.` should not be the sole evidence. We therefore computed two complementary uncertainty analyses from the saved row-level Eval12 outputs: absolute score CIs for every model/task and paired bootstrap delta CIs for model comparisons. The absolute CIs quantify evaluation-sample uncertainty for each reported score, while the paired deltas directly test claims such as MinervaRL versus GRPO on the same examples. We will add both tables, per-task denominators, and task-family averages. The 12-task MinervaRL-GRPO deltas are ... Thus the average gain is clearly separated for 3/4 backbones, while Qwen-4B should be described cautiously.

Training-seed wording:
- State that multiple training seeds are computationally expensive for LLM RLVR and remain a limitation.
- Do not let this sound like a refusal to do statistics: we are adding evaluation-example CIs and paired bootstrap tests, which directly address the reported benchmark uncertainty and model-comparison uncertainty.

### 2. Rollout-aware evaluation

Reviewer concern:
- Final benchmark lacks pass@k / verifier success@k / best-of-k analysis.

Response:
- Say we added held-out best-of-8 under matched sampling.
- Explain metric: row-level best verifier score/F1 across samples, then recompute Eval12 metrics.
- This directly tests support-expansion / detectability.

Numbers:

| Backbone | GRPO | MinervaRL | Delta |
| --- | ---: | ---: | ---: |
| Llama-3B | 53.2 | 60.5 | +7.2 |
| Llama-8B | 62.4 | 70.9 | +8.5 |
| Qwen-4B | 62.1 | 65.0 | +2.9 |
| Qwen-8B | 62.8 | 67.5 | +4.6 |

Settings:
- k=8
- temperature 0.7
- top-p 0.95
- max response 2048 new tokens

Suggested wording:

> To directly test this mechanism, we added held-out best-of-\(k\) evaluation with \(k=8\) under matched sampling. MinervaRL improves oracle best-of-8 Avg. over GRPO for all four backbones ...

### 3. Component-wise ablations

Reviewer concern:
- Wants no ACR, no EMA, no TextCNN, no heuristic filtering, matched additional SFT compute.

Response:
- Clarify mappings:
  - No ACR = GRPO.
  - Matched additional SFT compute = in-loop answer-only SFT control.
  - No TextCNN filter = heuristic filtering remains, learned filter removed.
  - No heuristic filtering = learned filter remains, heuristic rules removed.
  - No filtering = both disabled. If using only short response, mention only the variants in table.
- Use final Eval12 Llama-8B controls.

Numbers, 12-task Eval12 Avg:

| Model | Avg |
| --- | ---: |
| GRPO / no ACR | 56.0 |
| GRPO-12 | 53.9 |
| in-loop answer-only SFT | 57.8 |
| pure answer-only SFT | 47.2 |
| no EMA teacher | 57.2 |
| no TextCNN filter | 57.7 |
| no heuristic filtering | 57.8 |
| full MinervaRL | 60.2 |

Suggested takeaway:
- Extra rollouts do not recover MinervaRL.
- Answer-only in-loop SFT helps over GRPO, but remains below full MinervaRL.
- Removing teacher/filtering components lowers performance relative to full MinervaRL, though these are smaller effects than the ACR/distillation mechanism.

Suggested wording:

> We also added final Eval12 Llama-8B controls. Full MinervaRL reaches 60.2 Avg. versus GRPO/no-ACR 56.0. GRPO with 12 rollouts is 53.9, so extra rollout compute does not explain the gain. In-loop answer-only SFT is 57.8, improving over GRPO but remaining 2.5 pp below full MinervaRL, showing that the rationale-conditioned path is not reducible to simply injecting gold answers.

### 4. Clarity / CTI accessibility / notation

Reviewer concern:
- Intro assumes CTI background.
- Figure 1 is hard to follow.
- Section 4.1 notation and Algorithm 1 symbols unclear.
- Task/metric definitions missing before results.
- Claims too strong.

Response:
- Commit concrete changes:
  - rewrite introduction around one example: unstructured report -> CWE / ATT&CK / CVSS / mitigation / IoC
  - add task-family/metric table before results
  - simplify Figure 1 into base GRPO loop + auxiliary ACR/SFT loop
  - unify notation for verifier reward, original prompt, answer-conditioned prompt, dataset tuples
  - soften language where task-level results are mixed

Suggested wording:

> We will revise the introduction around a concrete CTI mapping example, add a compact task/metric table before Table 1, and simplify Figure 1 by separating the standard GRPO loop from the auxiliary ACR/SFT loop. We will also clean up Section 4.1 notation and tone down claims where task-level results are mixed.

### 5. Theory reframing

Reviewer concern:
- Appendix H theorem assumes the mechanism it is meant to justify.
- Does not analyze full GRPO/SFT LM dynamics.

Response:
- Agree and reframe as toy finite-sampling formalization.
- Do not defend it as full theory.

Suggested wording:

> We agree Appendix H should be framed more modestly. Its role is to formalize finite-sampling intuition for all-zero rollout groups, not to prove full LM optimization behavior. We will state the assumptions plainly, shorten/move the theorem if needed, and avoid implying it analyzes GRPO/SFT interaction, gradient interference, partial-credit rewards, or full LM dynamics.

### 6. Broader impact

Reviewer concern:
- Dual-use risk and concrete mitigations.

Response:
- Mention defensive intended use.
- Add mitigation plan: red-teaming, model cards, release controls, documentation.

Suggested wording:

> We will expand the broader-impact section with concrete mitigations: defensive-use documentation, cyber-safety red-teaming before release, model cards describing misuse risks and limitations, and caution around artifacts that could materially lower the cost of offensive automation.

## Required Manuscript Additions

Add all of the following:

1. Full absolute-score CI appendix table:
   - 4 backbone tables.
   - Rows: 12 Eval12 tasks.
   - Columns: Base, STaR, DART, GRPO, LUFFY, MinervaRL.
   - Cells: score [95% CI].

2. Full paired-comparison CI appendix table:
   - For each baseline: Base, STaR, DART, GRPO, LUFFY.
   - Rows: 12 Eval12 tasks.
   - Columns: Llama-8B, Llama-3B, Qwen-8B, Qwen-4B.
   - Cells: MinervaRL - baseline delta [95% CI].

3. Main-text compact uncertainty table:
   - Task-family CIs for MinervaRL - GRPO.
   - Average delta CIs for comparability with existing Table 1.
   - Count summary: CI-positive, CI-negative, CI-overlap-zero against each baseline.

4. Per-task sample-size/denominator table:
   - Include evaluation sizes and metric type.
   - This answers the sample-size-aware analysis request.

## What Not To Do

- Do not promise multi-seed training.
- Do not overclaim Qwen-4B average gains.
- Do not argue that the reviewer is unreasonable. Make the revised evidence do the work.

## Likely Final Shape

One paragraph each:

1. Thanks + acknowledge main issues.
2. CIs / aggregation / task families.
3. Rollout-aware best-of-8.
4. Ablation table in prose.
5. Presentation/notation/theory.
6. Safety and closing.

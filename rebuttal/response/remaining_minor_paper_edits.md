# Remaining Minor Paper Edits From Reviews

Scope: small manuscript edits or review suggestions not already fully addressed by the current revision. Major numerical additions already handled: paired CIs, rollout-aware best-of-k summaries, component ablations, data/code availability, broader impact, theory reframing, and evaluation protocol.

## Recommended Small Edits

None currently.

## Rebuttal-Only Or Optional

### 3. AthenaBench-Mini Overlap With Final Evaluation

Review source: R2/R4 ask whether AthenaBench-Mini overlaps with downstream evaluation and how this affects transfer claims.

Current status: paper now states the checkpoint-selection criterion and has a contamination/overlap paragraph for training vs evaluation. It still does not explicitly state that AthenaBench-Mini is not fully instance-disjoint from Athena-derived final columns.

Recommendation:
- Keep this rebuttal-only unless we decide the manuscript must include it. Current manuscript avoids claiming those Athena-derived columns are pure held-out transfer and uses symmetric checkpoint selection across trained methods.
- If added to the paper, make it one sentence in `Experimental Settings`, not a long caveat.

### 4. Hard-Prompt Failure Taxonomy

Review source: R3 suggested a fine-grained failure-mode analysis of hard prompts.

Current status: not added to the paper.

Recommendation:
- Rebuttal-only. In MinervaRL, a hard prompt is defined operationally during training by the verifier gate: the current policy samples no fully verified completion under the rollout budget, i.e. `max reward < 1`.
- We do not distinguish failure causes during training, and should not add an unannotated taxonomy to the paper.
- In rebuttal, state that because benchmark prompts specify structured formats and even base models usually produce parseable answers, hard prompts are more likely due to missing CTI knowledge, incorrect mapping/reasoning, partial correctness, ambiguity, or a combination of these, rather than formatting alone.

### 5. Partial-Reward Discussion

Review source: R3 suggested discussing partial rewards or partial correctness.

Current status: method already notes structured partial credit, set overlap, and dense CVSS scoring; no new paper edit is necessary.

Recommendation:
- Skip for now. If needed, answer in rebuttal by pointing to the existing reward design and noting broader partial-credit shaping is future work.

### 6. Verifier-Score Limits For Analyst-Facing Reasoning

Review source: R3/R4 ask for limitations of verifier-based evaluation, evidence grounding, and explanation correctness.

Recommendation:
- Keep this out of the manuscript unless needed in rebuttal. It is a generic RLVR evaluation-scope issue, and the current revision already includes response-quality preference evaluation plus broader-impact reliability language stating that outputs should be analyst-assistive rather than authoritative.

## Already Handled

- Figure 1 simplified into separate GRPO and ACR/distillation flows.
- Main uncertainty reporting added with paired bootstrap CIs and task-count summaries.
- Rollout-aware best-of-k evaluation added.
- Component ablations on final 12-task suite added, including in-loop answer-only SFT and GRPO 12 rollouts.
- Evaluation protocol, decoding setup, answer extraction, and shared scoring rules added.
- Data/code availability and release/licensing statement added.
- Broader impact expanded with defensive uses, dual-use risks, release mitigations, and reliability risk.
- Theory reframed as finite-sampling support intuition rather than full optimization theory.
- Main claims softened around small margins and closest baselines.
- Algorithm notation cleaned up: generic GRPO rollout indices now use `g/h`, the Minerva-CTI dataset is indexed as `\mathcal{D}=\{(x_i,y_i^\star)\}_{i=1}^{n}`, Algorithm 1 uses scalar rollout rewards `r_{i,j}^{\mathrm{base}}` and vector rewards `\mathbf{r}_i^{\mathrm{base}}`, and the filtering notation is unified as `\mathrm{Filter}`.
- Introduction now references a top-of-page public-CVE input/output figure: CVE-2024-20092 maps from an out-of-bounds-write description excerpt to CWE-787 and a CVSS v3.1 vector with 7.8/High severity.
# Remaining Minor Paper Edits From Reviews

Scope: small manuscript edits or review suggestions not already fully addressed by the current revision. Major numerical additions already handled: paired CIs, rollout-aware best-of-k summaries, component ablations, data/code availability, broader impact, theory reframing, and evaluation protocol.

## Recommended Small Edits

None currently.

## Rebuttal-Only Or Optional

### 3. AthenaBench-Mini Overlap With Final Evaluation

Review source: R2/R4 ask whether AthenaBench-Mini overlaps with downstream evaluation and how this affects transfer claims.

Current status: paper now states the checkpoint-selection criterion and has a contamination/overlap paragraph for training vs evaluation. It still does not explicitly state that AthenaBench-Mini is not fully instance-disjoint from Athena-derived final columns.

Recommendation:
- Keep this rebuttal-only unless we decide the manuscript must include it. Current manuscript avoids claiming those Athena-derived columns are pure held-out transfer and uses symmetric checkpoint selection across trained methods.
- If added to the paper, make it one sentence in `Experimental Settings`, not a long caveat.

### 4. Hard-Prompt Failure Taxonomy

Review source: R3 suggested a fine-grained failure-mode analysis of hard prompts.

Current status: not added to the paper.

Recommendation:
- Rebuttal-only. In MinervaRL, a hard prompt is defined operationally during training by the verifier gate: the current policy samples no fully verified completion under the rollout budget, i.e. `max reward < 1`.
- We do not distinguish failure causes during training, and should not add an unannotated taxonomy to the paper.
- In rebuttal, state that because benchmark prompts specify structured formats and even base models usually produce parseable answers, hard prompts are more likely due to missing CTI knowledge, incorrect mapping/reasoning, partial correctness, ambiguity, or a combination of these, rather than formatting alone.

### 5. Partial-Reward Discussion

Review source: R3 suggested discussing partial rewards or partial correctness.

Current status: method already notes structured partial credit, set overlap, and dense CVSS scoring; no new paper edit is necessary.

Recommendation:
- Skip for now. If needed, answer in rebuttal by pointing to the existing reward design and noting broader partial-credit shaping is future work.

### 6. Verifier-Score Limits For Analyst-Facing Reasoning

Review source: R3/R4 ask for limitations of verifier-based evaluation, evidence grounding, and explanation correctness.

Recommendation:
- Keep this out of the manuscript unless needed in rebuttal. It is a generic RLVR evaluation-scope issue, and the current revision already includes response-quality preference evaluation plus broader-impact reliability language stating that outputs should be analyst-assistive rather than authoritative.

## Already Handled

- Figure 1 simplified into separate GRPO and ACR/distillation flows.
- Main uncertainty reporting added with paired bootstrap CIs and task-count summaries.
- Rollout-aware best-of-k evaluation added.
- Component ablations on final 12-task suite added, including in-loop answer-only SFT and GRPO 12 rollouts.
- Evaluation protocol, decoding setup, answer extraction, and shared scoring rules added.
- Data/code availability and release/licensing statement added.
- Broader impact expanded with defensive uses, dual-use risks, release mitigations, and reliability risk.
- Theory reframed as finite-sampling support intuition rather than full optimization theory.
- Main claims softened around small margins and closest baselines.
- Algorithm notation cleaned up: generic GRPO rollout indices now use `g/h`, the Minerva-CTI dataset is indexed as `\mathcal{D}=\{(x_i,y_i^\star)\}_{i=1}^{n}`, Algorithm 1 uses scalar rollout rewards `r_{i,j}^{\mathrm{base}}` and vector rewards `\mathbf{r}_i^{\mathrm{base}}`, and the filtering notation is unified as `\mathrm{Filter}`.
- Introduction now references a top-of-page public-CVE input/output figure: CVE-2024-20092 maps from an out-of-bounds-write description excerpt to CWE-787 and a CVSS v3.1 vector with 7.8/High severity.

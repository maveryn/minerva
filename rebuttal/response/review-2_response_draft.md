# Review 2 Response Draft

This reviewer is not mainly objecting to the heterogeneous Eval12 average. They accept the broad evidence, but ask for uncertainty, checkpoint-selection clarification, missing evaluation details, release/licensing, extra ablations, theory framing, and dual-use discussion. The response should be narrower than Review 1 and should not spend space on the full 12-task CI table unless needed.

## Result blocks to include

### 1. Uncertainty over evaluation examples

Use the paired bootstrap CIs we already computed from row-level Eval12 outputs. This directly addresses the requested "confidence intervals, at least by bootstrapping evaluation instances" and the concern about small margins.

Recommended compact table:

| Baseline | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |
| --- | ---: | ---: | ---: | ---: |
| Base | +11.6 [10.8, 12.5] | +15.2 [14.5, 16.0] | +15.7 [14.5, 17.1] | +20.4 [19.0, 21.3] |
| STaR | +10.5 [9.7, 11.3] | +11.0 [10.3, 11.8] | +16.5 [15.4, 17.6] | +2.6 [1.5, 3.6] |
| DART | +6.6 [5.7, 7.5] | +4.3 [3.5, 5.1] | +10.5 [9.5, 11.8] | +8.6 [7.4, 9.7] |
| GRPO | +4.2 [3.5, 5.1] | +6.2 [5.3, 7.1] | +6.4 [5.6, 7.6] | +0.4 [-0.8, 1.3] |
| LUFFY | +3.3 [2.5, 4.1] | +2.9 [2.1, 4.0] | +0.1 [-0.6, 0.8] | +4.0 [3.0, 4.7] |

Text to include:

> We computed paired nonparametric bootstrap intervals over evaluation examples using the saved row-level outputs, with the same resampled examples used for both models and 2000 bootstrap replicates. These intervals measure evaluation-sample uncertainty, not training-seed variation. The GRPO comparison is robust for Llama-8B, Llama-3B, and Qwen-8B, while the Qwen-4B GRPO margin overlaps zero. We will therefore qualify claims based on sub-1 point margins, especially Qwen-4B vs GRPO and Qwen-8B vs LUFFY.

Also useful, but probably not a table in the final R2 reply unless we need task-level qualification:

| Baseline | Point wins | Point losses | CI positive | CI negative | CI overlaps 0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 42/48 | 6/48 | 35/48 | 2/48 | 11/48 |
| STaR | 38/48 | 10/48 | 35/48 | 6/48 | 7/48 |
| DART | 38/48 | 10/48 | 34/48 | 3/48 | 11/48 |
| GRPO | 34/48 | 14/48 | 28/48 | 7/48 | 13/48 |
| LUFFY | 32/48 | 16/48 | 20/48 | 6/48 | 22/48 |

This gives a task-level qualification without importing the entire Review 1 per-task CI table.

### 2. Checkpoint selection and AthenaBench-Mini overlap

This is a serious Review 2-specific concern. The answer should be explicit: the AthenaBench-Mini validation columns are not instance-disjoint from the corresponding Athena-derived final Eval12 columns. They are subsets of the full Eval12 Athena columns by ID/hash. This does not invalidate pairwise comparisons because all trained methods use the same checkpoint-selection criterion, but it does mean we should not present Athena-derived columns as pure held-out transfer evidence.

No-new-generation analysis we can add: remove the AthenaBench-Mini selection subset from the final Athena-derived columns and recompute MinervaRL - GRPO on the remaining examples.

Subset counts:

| Task | Full Eval12 examples | AthenaBench-Mini selected | Remaining examples |
| --- | ---: | ---: | ---: |
| CKT | 3000 | 300 | 2700 |
| RCM | 2000 | 200 | 1800 |
| VSP | 2000 | 200 | 1800 |
| ATE | 500 | 100 | 400 |
| RMS | 500 | 100 | 400 |

Non-mini subset per-task deltas:

| Task | Llama-8B | Llama-3B | Qwen-8B | Qwen-4B |
| --- | ---: | ---: | ---: | ---: |
| CKT | +2.0 [0.5, 3.7] | -0.3 [-1.7, 1.1] | +8.1 [6.5, 9.7] | -4.6 [-6.3, -3.0] |
| RCM | +2.4 [0.9, 3.9] | +8.5 [6.9, 10.1] | +4.3 [2.8, 5.8] | -1.2 [-2.4, 0.0] |
| VSP | +5.2 [4.5, 6.0] | +20.7 [18.9, 22.6] | +10.8 [9.8, 11.7] | -8.0 [-8.9, -7.1] |
| ATE | +17.2 [12.8, 22.0] | +17.0 [13.2, 20.8] | +8.5 [5.2, 12.0] | +6.2 [3.7, 9.2] |
| RMS | +11.7 [8.0, 15.4] | +13.7 [10.3, 17.0] | +11.5 [8.1, 14.5] | +0.9 [-1.2, 2.9] |

Five-task macro-average on the non-mini subset:

| Backbone | MinervaRL - GRPO | 95% CI |
| --- | ---: | ---: |
| Llama-8B | +7.7 | [6.4, 9.0] |
| Llama-3B | +11.9 | [10.8, 13.1] |
| Qwen-8B | +8.6 | [7.6, 9.7] |
| Qwen-4B | -1.3 | [-2.1, -0.5] |

How to use this:

- Good: "AthenaBench-Mini is a subset of the full Athena-derived evaluation columns; we will make this explicit."
- Good: "All trained baselines used the same selection criterion, so this does not favor MinervaRL over GRPO/STaR/DART/LUFFY through an asymmetric model-selection rule."
- Good: "To assess whether the result depends only on the selected subset, we recomputed CKT/RCM/VSP/ATE/RMS after removing Mini instances. The Llama-8B, Llama-3B, and Qwen-8B gains remain positive; Qwen-4B is worse than GRPO on this Athena-only non-mini subset, consistent with our broader decision to soften Qwen-4B claims."
- Avoid: claiming fully disjoint Athena validation/evaluation instances.
- Avoid: calling CKT a pure transfer benchmark after selection on AthenaBench-Mini.

### 3. Evaluation protocol details

No new result needed. Add explicit protocol text:

- Main Eval12 generation used the same benchmark prompts, scoring scripts, and task-specific answer extractors for every model and baseline.
- For local HuggingFace/vLLM models, the Eval12 runner defaults to deterministic decoding with `temperature=0.0`, `top_p=1.0`, and `max_new_tokens=2048`.
- Output-format differences from external security-SFT baselines are handled only through the same task-level extractors/scorers used for all systems, not baseline-specific scoring rules.
- The rollout-aware best-of-k analysis is separate and uses sampled decoding (`temperature=0.7`, `top_p=0.95`, `k=8`, `max_new_tokens=2048`).

R2 did not explicitly ask for rollout-aware results, so do not spend main R2 space on the large pass@k table unless the final reply has room.

### 4. Additional ablations requested by R2

R2 asks for in-loop answer-only distillation, compute-matched GRPO, standard SFT-then-GRPO, and more modest theory framing. We can answer most of this with the Llama-8B ablation results.

Recommended compact ablation table:

| Control / ablation | Eval12 Avg. | Full - control | CI sign summary |
| --- | ---: | ---: | --- |
| GRPO / no ACR | 56.0 | +4.2 | 8 positive, 0 negative, 4 overlap |
| GRPO-12 | 53.9 | +6.3 | 7 positive, 1 negative, 4 overlap |
| Answer-only SFT | 52.6 | +7.6 | 7 positive, 3 negative, 2 overlap |
| In-loop answer-only SFT | 57.8 | +2.4 | 6 positive, 4 negative, 2 overlap |
| No EMA teacher | 57.2 | +3.0 | 8 positive, 0 negative, 4 overlap |
| No TextCNN filter | 57.7 | +2.5 | 7 positive, 1 negative, 4 overlap |
| No filtering | 57.8 | +2.4 | 7 positive, 0 negative, 5 overlap |
| Full MinervaRL | 60.2 | -- | -- |

Interpretation:

- `GRPO / no ACR` is the direct no-ACR control.
- `GRPO-12` is the compute/sampling-budget control for extra rollout compute; it does not close the gap.
- `In-loop answer-only SFT` is the cleanest answer to "separate rationales from simply injecting gold answers": same in-loop schedule, direct answer-only SFT targets instead of ACR rationales.
- The standard SFT-then-GRPO request is only partly covered by existing results. We should either point to the existing DART-CTI SFT baseline plus DART-initialized GRPO if we decide to include that run, or state that we prioritize the more targeted in-loop answer-only and compute-matched controls in this revision. Do not imply the paper already fully contains a standard SFT-then-GRPO baseline unless we add the exact result.

### 5. Theory appendix

No new experiment needed for the rebuttal. Make the main-text framing more modest:

> We agree that the theorem should be read as a finite-sampling formalization of the support-expansion intuition, not as independent empirical evidence. We will revise the main text to state the assumptions explicitly and move stronger claims to the appendix.

Optionally tie assumptions to measured diagnostics:

- all-zero rollout group frequency from Figure 3,
- GRPO-12 not closing the gap,
- rollout-aware best-of-k improvements from the new R1 analysis.

### 6. Release, licensing, and dual-use

No numeric results needed, but the response should promise concrete revisions:

- Release code, verifier implementations, training scripts, evaluation scripts, and generated splits where redistribution is permitted.
- For upstream CTI sources with redistribution restrictions, release preprocessing scripts, metadata/manifests, and instructions to rebuild from the original sources rather than redistributing restricted raw content.
- Add dual-use discussion: CTI capability gains can help defensive triage, enrichment, and analyst workflows, but may also lower the cost of offensive misuse on exploit-generation-like tasks such as CanaryExploit.
- Clarify release controls for models/data: staged release, task-specific verifiers/data where allowed, documentation of intended defensive use, and safeguards for exploit-generation evaluations.

## Draft response skeleton

Thank you for the constructive review. We agree that the revision should make uncertainty, checkpoint selection, and evaluation protocol details explicit.

First, we added paired bootstrap uncertainty estimates over evaluation examples using the saved row-level Eval12 outputs. Against GRPO, MinervaRL is +4.2 [3.5, 5.1], +6.2 [5.3, 7.1], +6.4 [5.6, 7.6], and +0.4 [-0.8, 1.3] pp for Llama-8B, Llama-3B, Qwen-8B, and Qwen-4B respectively. The corresponding deltas are also positive with intervals above zero against Base, STaR, and DART for all four backbones. Against LUFFY, the intervals are positive for Llama-8B, Llama-3B, and Qwen-4B, but overlap zero for Qwen-8B: +0.1 [-0.6, 0.8]. We will therefore qualify sub-1 point claims rather than describe them as robust gains.

Second, we will clarify checkpoint selection. AthenaBench-Mini is not instance-disjoint from the Athena-derived final columns; it is a subset of the full CKT/RCM/VSP/ATE/RMS evaluation files. All trained baselines use the same selection criterion, so the model-comparison rule is symmetric, but these columns should not be described as pure transfer. To check dependence on the selected subset, we recomputed those five columns after removing Mini instances. The remaining-example deltas for MinervaRL - GRPO are +7.7 [6.4, 9.0], +11.9 [10.8, 13.1], +8.6 [7.6, 9.7], and -1.3 [-2.1, -0.5] pp for Llama-8B, Llama-3B, Qwen-8B, and Qwen-4B. This supports the main Llama/Qwen-8B conclusions but also motivates softening Qwen-4B claims.

Third, we will add the missing evaluation-protocol details: main Eval12 uses deterministic decoding for local models (`temperature=0.0`, `top_p=1.0`, `max_new_tokens=2048`) and applies the same task prompts, answer extractors, and scorers to all systems, including external security-SFT baselines.

Finally, we added Llama-8B controls requested by the review. Full MinervaRL averages 60.2, compared with 56.0 for GRPO/no-ACR, 53.9 for GRPO-12, 57.8 for in-loop answer-only SFT, 57.2 without the EMA teacher, 57.7 without the TextCNN filter, and 57.8 without filtering. The in-loop answer-only control directly separates ACR rationales from merely injecting gold answers; GRPO-12 controls extra rollout compute. We will also revise the theory discussion to frame the theorem as a support-expansion formalization under explicit assumptions, and add a concrete release/licensing and dual-use statement, including the risk implications of improved CanaryExploit performance.

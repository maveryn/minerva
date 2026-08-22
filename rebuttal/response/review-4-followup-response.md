# Reviewer 9rjT Follow-up Response

Thank you for the follow-up and for confirming that the revision addresses the four critical requests. We incorporated the remaining final-version clarifications into the revised manuscript.

**Uncertainty and ablation scope.** The Limitations section now states that the paired bootstrap confidence intervals quantify evaluation-example variation for fixed checkpoints and do not capture variation across independent training seeds. Revised Sec. 6.3 also states explicitly that the Table 5 component ablations use Llama-3.1-8B only.

**Answer-only control.** Revised Sec. 6.3 now brings the mixed task-level signs for Full MinervaRL versus in-loop answer-only SFT into the main discussion: 6 intervals are positive, 4 are negative, and 2 overlap zero. The accompanying text therefore states that answer-only distillation does not recover the full method on average, while the advantage of ACR distillation is not uniform across tasks.

**AthenaBench-Mini overlap.** Revised Sec. 5.2 now discloses the overlap between AthenaBench-Mini and the five corresponding final evaluation sets. Appendix F.1 contains the previously reported sensitivity analysis excluding the overlapping instances.

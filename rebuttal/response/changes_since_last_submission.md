# Changes Since Last Submission

We uploaded the latest revised manuscript and include a cumulative diff PDF against the original submission in the supplementary material. The revision incorporates the reviewer feedback as follows.

- **Clearer CTI task framing.** We revised the introduction and added a concrete CTI mapping example in Figure 1, showing how an unstructured public vulnerability description maps to structured CWE/CVSS outputs. We also clarified CTI standards and terminology for a broader ML audience.

- **Uncertainty and heterogeneous metrics.** We added paired nonparametric bootstrap 95% confidence intervals over evaluation examples in Sec. 5.2. Revised Table 2 reports per-task intervals for MinervaRL minus GRPO, and Table 3 summarizes MinervaRL against Base, STaR, DART, GRPO, and LUFFY across 48 backbone-task comparisons. Against GRPO, MinervaRL has 34/48 positive point deltas, with 28 CI-positive, 7 CI-negative, and 13 overlapping-zero intervals. Across all baselines, the CI-positive counts are 35/48 vs Base, 35/48 vs STaR, 34/48 vs DART, 28/48 vs GRPO, and 20/48 vs LUFFY. Full paired tables are in Appendix Tables 16-21.

- **Task-family analysis.** To avoid relying only on a heterogeneous average, we added a task-family count summary in Sec. 6.2/Table 4, grouping tasks into QA/selection, taxonomy mapping, vulnerability scoring, and information extraction. This identifies where the gains are strongest while keeping the per-task analysis as the main evidence.

- **Evaluation protocol and checkpoint selection.** Sec. 5.2 now specifies the main decoding setup, including greedy decoding with temperature 0.0 and a 2048-token generation limit. It also states that all systems use the same fixed benchmark prompts, parsers/normalizers, and scoring scripts, including external security-SFT baselines. We clarified that trained methods use the same checkpoint-selection rule based on Minerva-Dev and AthenaBench-Mini performance, disclosed AthenaBench-Mini's overlap with five final evaluation sets, and added a sensitivity analysis excluding the overlapping instances.

- **Rollout-aware evaluation.** We added matched-budget best-of-$k$ evaluation for GRPO and MinervaRL on the 12 CTI evaluation tasks in Sec. 7.2/Table 8, with full rollout scores and per-task rollout CIs in Appendix Tables 22-26. MinervaRL improves over GRPO on 32/48, 36/48, 38/48, and 39/48 comparisons for $k\in\{1,2,4,8\}$, respectively; at $k=8$, 31/48 paired intervals are positive, 6/48 are negative, and 11/48 overlap zero.

- **Component ablations.** We added final Llama-3.1-8B ablations in Sec. 6.3/Table 5, with full per-task scores in Appendix Table 27. The revision includes the new in-loop answer-only SFT control requested by reviewers, alongside GRPO/no-ACR, GRPO with 12 rollouts, no EMA teacher, no TextCNN filter, and no filtering controls. Full MinervaRL reaches a 60.2 average, compared with 57.8 for in-loop answer-only SFT and 56.0 for GRPO/no-ACR.

- **Training-aligned vs. not-in-training tasks.** We added Sec. 6.5 and Tables 6-7 to separate tasks directly aligned with Minerva-CTI training objectives from tasks not used as Minerva-CTI training objectives. The revised discussion narrows the claim scope: MinervaRL is strongest on verifier-aligned structured tasks, with more mixed gains on less directly aligned tasks.

- **Method, notation, and theory scope.** We clarified that answer-conditioned labels are used only to generate candidate training traces, while all evaluations use original answer-free prompts. We updated Sec. 4 notation and Figure 2 to separate the standard GRPO loop from the auxiliary ACR path. We also reframed the support analysis as a stylized conditional finite-budget detectability argument rather than a full optimization theory for joint GRPO/SFT training in language models.

- **Release and broader impact.** We added a Data and Code Availability statement and expanded the Broader Impact Statement. The revision now discusses intended defensive uses, dual-use risks, analyst-facing reliability concerns, preservation of upstream attribution/licensing, cyber-safety review, model/data cards, and staged release when needed.

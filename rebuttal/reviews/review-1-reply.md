# Reviewer 1 Follow-up

Thank you for revising the manuscript and for providing a Markdown version that makes the updates easy to trace. Empirically, this version of the paper is substantially stronger. I still have a few comments regarding the theoretical interpretation of the method and some aspects of the presentation.

## Major comments

- The theoretical definition of a successful rollout seems slightly different from the verifier used by MinervaRL. In Eqs. (11), (13), and (14), success is defined through $g(y)=a^\star$, whereas the algorithm uses $R_{\mathrm{Minerva}}(x,y,a^\star)=1$. These do not appear to be equivalent in general under the reward functions used in the paper, which also involve normalization, partial credit, set-valued scoring, alias matching, and CVSS score-based rewards. If the intention is to formalize "full verifier success," it may be cleaner to define the success event directly through $R_{\mathrm{Minerva}}$ and use this definition consistently in the subsequent analysis.

- I think the relationship between "no fully verified rollout," the ACR gate, and the absence of useful GRPO signal could be clarified. For binary rewards, an all-zero rollout group indeed provides no useful within-group reward variation. However, several Minerva tasks use partial-credit rewards. A group such as $(0,0,0.5,0,0.5,\ldots)$ contains no fully verified response but still provides non-zero relative advantages to GRPO. Relatedly, MinervaRL triggers ACR whenever $\max_j r_{i,j}<1$, whereas the reported zero-solve statistic corresponds to $\max_j r_{i,j}=0$. It would help to distinguish these cases more explicitly. A small ablation comparing the current gate with a strict zero-reward gate could also help clarify which regime benefits most from ACR.

- I am still not fully sure how much the theoretical result tells us about the full MinervaRL optimization procedure. The key assumption states that each successful ACR distillation update increases the log probability of the correct answer by at least some fixed $\Delta>0$. Given this assumption, the result that sufficiently many updates eventually cross the detectability threshold is fairly direct. The harder question seems to be whether ACR distillation actually produces such an increase when GRPO and SFT updates interact and parameters are shared across prompts. Appendix J appropriately acknowledges that the analysis does not model these interactions; I think it would help to carry this qualification into the main-text discussion and present the result explicitly as a stylized finite-sampling argument providing intuition for the method. Along similar lines, "finite-budget detectability" may be slightly more precise terminology than "support expansion," since the analysis already assumes $p_{\theta_0}(a^\star\mid x)>0$.

- The component ablations are much more informative in this version, although the role of the hard-example gate could still be isolated more directly. For example, an always-ACR or random-prompt ACR control with a matched number of additional generations and supervised updates would help distinguish the benefit of adaptive hard-example selection from the more general benefit of adding answer-conditioned supervised training. I do not think a large additional ablation study is necessary, but one such control would make the interpretation of the method cleaner.

## Minor comments on notation and presentation

- Eq. (13): I assume the symbol on the right-hand side is intended to denote an indicator function. Also, $x$ is currently an argument of $S(x,y;a^\star)$ but does not appear on the right-hand side. If $S$ is intended to represent full verifier success, perhaps the cleanest definition would be

  $$
  S(x,y;a^\star):=\mathbf{1}\!\left\{R_{\mathrm{Minerva}}(x,y,a^\star)=1\right\}.
  $$

- $\mathrm{Filter}(\cdot)$ appears in Algorithm 1 without being formally defined at that point. Since the TextCNN condition $q_{i,k}\ge\tau_q$ is included separately, it would help to clarify whether $\mathrm{Filter}$ refers only to the heuristic filtering stages.

- Algorithm 1 applies $\mathrm{Sample}(\mathcal{Q},M)$ and the subsequent SFT update every $I$ steps. What happens if no ACR candidate survives filtering and $\mathcal{Q}=\varnothing$? A short conditional for this case would make the pseudocode fully specified.

- Section 4.1 introduces the reward as $r_j$, whereas Section 4.2 uses $r_{i,j}^{\mathrm{base}}$ for what appears to be the same verifier reward with additional indices. If these refer to the same quantity, using consistent notation would make the presentation easier to follow.

- Appendix H.1: The abbreviation TAA is used but, as far as I can tell, is never introduced. I assume this refers to Threat Actor Attribution.

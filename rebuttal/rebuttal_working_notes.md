# TMLR Rebuttal Working Notes

Context read on 2026-06-29:
- Paper PDF: `rebuttal/8809_Minerva_Reinforcement_Lea (1).pdf`
- Review: `rebuttal/review.txt`
- Local rebuttal skill: `ml-codex-skills/paper-reviewer-rebuttal/SKILL.md`
- Supporting skills inspected: `ml-results-reporting`, `paper-prose-revision`, `paper-section-writing`
- Code paths inspected: dataset builder, RLVR launchers, reward functions, ACR prompt/filter/distillation trainer code, DART/STaR artifacts, paper figure/table artifacts

## Review Triage

Highest-impact reviewer concerns:
1. Evaluation reliability: no uncertainty intervals, no paired significance tests, and a heterogeneous `Avg.` that mixes accuracy, F1, extraction F1, and normalized CVSS score.
2. Rollout-aware evidence: the method is motivated by sparse finite rollout support, but final held-out evaluation does not report pass@k, verifier success@k, or oracle best-of-k metrics.
3. Component attribution: the paper has some ablations, but the reviewer wants clearer component-wise ablations on the main setting, including ACR, EMA teacher, TextCNN, heuristic filtering, and matched SFT compute.
4. Clarity and notation: introduction, Figure 1, Section 4.1 notation, task/metric definitions, and theory framing are not accessible enough for a broad TMLR audience.
5. Broader impact: the current statement acknowledges dual use but does not list concrete mitigation strategies.

Lower-impact but easy fixes:
- Add a compact task-family/metric table before Table 1.
- Tone down "consistent gains" and "outperforms" where task-level results are mixed.
- Reframe Appendix H as a toy finite-sampling formalization, not theory for full LM optimization.
- Unify notation around `r`, `R_minerva`, `x`, `x_acr`, and dataset tuples.

## Evidence Already In The Paper

Useful existing paper evidence:
- Main result: Table 1 reports 12 CTI tasks across four backbones. MinervaRL average is 52.5 vs. GRPO 48.2 and base 36.7.
- Task grouping: Table 2 splits training-aligned vs. not-in-training tasks. MinervaRL improves not-in-training averages over GRPO for all four backbones.
- Support mechanism: Figure 3 reports zero-solve fraction over training and shows MinervaRL below GRPO across all four backbones; Llama-8B also includes GRPO with 12 rollouts.
- Validation dynamics: Figure 4 shows MinervaRL reaches the best GRPO validation score and continues improving.
- Ablations: Table 4(a) includes GRPO, GRPO with 12 rollouts, answer-only SFT, full MinervaRL, no EMA teacher, no filtering, and no ML filter. This is Llama-8B validation, not full 12-task final evaluation.
- Distillation scale: Table 4(b) sweeps gamma in {0.01, 0.02, 0.05, 0.10}.
- Response quality: Appendix G reports GPT-5.2 pointwise judging, human agreement, and Claude cross-judge agreement.
- Baseline fairness: Appendix F states STaR, DART, and LUFFY use the same 32,000-example Minerva-CTI training split and same verifier where applicable.
- Safety: Broader Impact Statement acknowledges dual use but needs concrete mitigations.

## Codebase Context

Key implementation facts confirmed from code:
- GRPO launcher: `rlvr/cti-scripts/train_minerva_grpo.sh`
  - train: `rlvr/mydata/minerva_base/minerva_base_train.parquet`
  - validation: Minerva dev, Athena CTI parquets, SecEval mini
  - rollout n = 8, total steps = 500, actor LR = 1e-6
  - reward: `rlvr/verl/utils/reward_score/reward_minerva.py`
- MinervaRL/Noctua launcher: `rlvr/cti-scripts/train_minerva_noctua.sh`
  - same train/validation path and GRPO reward
  - adds ACR prompt generation, optional EMA teacher, filtering, and SFT distillation
  - default Llama-8B paper preset: `rlvr/cti-scripts/train_minerva_noctua_llama8b_lr0.05_textcnn_mlh_t0.7_p0.9_defer_ema.sh`
- ACR prompting: `minerva/acr_prompt.py`
  - appends a `GROUND_TRUTH_LABELS` block only for training trace generation
  - stores original prompt separately as `acr_orig_prompt`
- ACR reward/filter: `rlvr/verl/utils/reward_score/reward_acr.py`
  - requires verifier correctness through `reward_minerva`
  - rejects leakage phrases/regexes, short reasoning, low overlap, and ID overuse when configured
- Trainer path: `rlvr/verl/trainer/ppo/ray_trainer.py`
  - computes hard UIDs from grouped rollout rewards
  - builds ACR batches only for eligible hard prompts
  - collects distillation records with `prompt_nohint = acr_orig_prompt`
  - runs SFT distillation every configured interval, default paper preset interval = 10
- TextCNN filter artifacts:
  - training/eval scripts under `minerva-judge/classifier/`
  - paper error analysis in `paper/textcnn_filter_error_analysis/validation_error_analysis.md`
- DART artifacts:
  - `dart-artifacts/README.md` documents the 64k fixed SFT corpus and generation stages
  - raw attempt logs are intentionally not tracked
- STaR artifacts:
  - `star/RESULTS.md` summarizes completed STaR rounds and selected checkpoints

## LLMBench Evaluation Context

Evaluation artifacts are available in `/home/jovyan/work/llmbench`.

Key files inspected:
- `/home/jovyan/work/llmbench/AGENTS.md`
- `/home/jovyan/work/llmbench/athena_eval/config.yaml`
- `/home/jovyan/work/llmbench/athena_eval/evaluate.py`
- `/home/jovyan/work/llmbench/paper/eval12_reproduction.md`
- `/home/jovyan/work/llmbench/paper/build_eval12_table.py`
- `/home/jovyan/work/llmbench/paper/eval12_8b_model_results.md`
- `/home/jovyan/work/llmbench/paper/results.tex`
- `rebuttal/analyze_llmbench_eval12.py`
- `rebuttal/llmbench_eval12_stats.md`

Paper Table 1 provenance:
- The paper's `eval12` columns are backed by `runs/<model>/<task>.jsonl`, `runs/<model>/<task>-scored.jsonl`, and a summary JSON.
- `Avg.` is the arithmetic mean over the 12 shown paper columns, not the broader `overall_avg` in `evalall-summary.json`.
- `evalall-summary.json` key mapping for the paper columns:
  - CKT = `mcq3k_accuracy`
  - CyberMetric = `cybermetric_accuracy`
  - SOCEval = `threat_intel_reasoning_avg_score`
  - RCM = `rcm_accuracy`
  - VSP = `vsp_accuracy`
  - ATE = `ate_accuracy`
  - RMS = `rms_f1`
  - ElasticRule = `elastictoattack_accuracy`
  - APTNER = `aptner_f1`
  - LANCE = `prism_avg_f1`
  - AnnoCTR = `annoctr_avg`
  - AZERG = `azerg_avg`

GRPO and MinervaRL run directories with full scored outputs:
- Llama-8B:
  - GRPO: `/home/jovyan/work/llmbench/runs/xashru/minerva_grpo_llama8b_500_490`
  - MinervaRL: `/home/jovyan/work/llmbench/runs/xashru/minerva_noctua_llama8b_ema_0.05`
- Llama-3B:
  - GRPO: `/home/jovyan/work/llmbench/runs/xashru/minerva_grpo_llama3b_500`
  - MinervaRL: `/home/jovyan/work/llmbench/runs/xashru/minerva_noctua_llama3b_500`
- Qwen-8B:
  - GRPO: `/home/jovyan/work/llmbench/runs/xashru/minerva_grpo_qwen8b_base`
  - MinervaRL: `/home/jovyan/work/llmbench/runs/xashru/minerva_noctua_qwen8b_base`
- Qwen-4B:
  - GRPO: `/home/jovyan/work/llmbench/runs/xashru/minerva_grpo_qwen4b_base`
  - MinervaRL: `/home/jovyan/work/llmbench/runs/xashru/minerva_noctua_qwen4b_base`

Scored row counts are identical across those GRPO/MinervaRL runs:
- MCQ3k: 3000
- CyberMetric: 2000
- ThreatIntelReasoning/SOCEval: 588
- RCM: 2000
- VSP: 2000
- ATE: 500
- RMS: 500
- ElasticToAttack/ElasticRule: 432
- APTNER: 1505
- PRISM/LANCE subtasks: IP 89, URL 111, Domain 159, Hash 107
- AZERG subtasks: T1 25, T2 504, T3 536, T4 268
- AnnoCTR subtasks: T1 149, T2 409, T3 448, T4 224

Scoring details confirmed from `athena_eval/evaluate.py`:
- MCQ/CyberMetric/RCM/ATE/ElasticToAttack use exact-match style row scores.
- ThreatIntelReasoning/SOCEval uses per-example Jaccard `score` and reports `avg_score`.
- VSP stores per-example normalized score as `1 - MAD / 7.7`; `vsp_mad_denominator` is 7.7 in config.
- RMS stores per-example mitigation-set F1 and reports the mean F1.
- APTNER/CyNER use micro precision/recall/F1 from row-level `tp`, `fp`, and `fn`.
- PRISM/LANCE uses micro F1 for each IoC type and averages the four type F1s.
- AZERG/AnnoCTR use T1/T3/T4 exact accuracy and T2 macro-F1, then average the four subtasks.

Recomputed paper 12-column averages from `evalall-summary.json`:
- Llama-8B: GRPO 56.0, MinervaRL 60.2
- Llama-3B: GRPO 42.1, MinervaRL 48.3
- Qwen-8B: GRPO 46.9, MinervaRL 53.4
- Qwen-4B: GRPO 47.6, MinervaRL 48.0

Bootstrap results generated in `rebuttal/llmbench_eval12_stats.md`:
- Script: `rebuttal/analyze_llmbench_eval12.py`
- Resamples: 2000; seed: 8809
- The script verifies all computed point estimates against `evalall-summary.json` before writing the report.
- SOCEval detail: `ThreatIntelReasoning` averages Jaccard over parsed answers and reports parsing errors separately; the script matches that denominator.
- 12-column average paired deltas, MinervaRL - GRPO:
  - Llama-8B: +4.2 pp, 95% paired bootstrap CI [3.5, 5.1]
  - Llama-3B: +6.2 pp, 95% CI [5.3, 7.2]
  - Qwen-8B: +6.4 pp, 95% CI [5.6, 7.7]
  - Qwen-4B: +0.4 pp, 95% CI [-0.8, 1.3]
- Family-level takeaway:
  - Choice/reasoning tasks are mixed by backbone.
  - Taxonomy mapping improves for all four backbones: +9.6, +13.4, +7.2, +3.2 pp; all paired CIs exclude zero.
  - Structured extraction/scoring improves for Llama-8B, Llama-3B, and Qwen-8B; Qwen-4B is indistinguishable on this family.
- Rebuttal implication: avoid saying the average gain is statistically clear for every backbone. Stronger, defensible claim: the row-level analysis supports clear average gains for 3/4 backbones and consistent gains on taxonomy-mapping tasks across all four.

## Best Rebuttal Strategy

Use a grouped, evidence-first rebuttal. Proposed order:
1. Evaluation protocol: acknowledge the point, supplement `Avg.` with task-family means and paired bootstrap intervals, and explicitly note the Qwen-4B average is not clearly separated from GRPO.
2. Rollout-aware metrics: add matched-budget success@k/best-of-k analysis if possible. If not ready, state that Figure 3 already diagnoses training-time group-level solve coverage and commit to adding final held-out success@k.
3. Component attribution: point to existing Table 4 and add a concise new ablation table if checkpoints are available. Be transparent that Table 4 is validation-side and will be moved/expanded.
4. Clarity/theory: commit concrete manuscript edits: simpler intro example, two-color Figure 1, task/metric table, notation cleanup, Appendix H reframed as toy finite-sampling intuition.
5. Safety: add concrete mitigation text for defensive intended use, red-teaming, artifact release controls, and documentation.

## Analyses To Run If Artifacts Are Available

Priority A: statistical uncertainty for Table 1
- Inputs: row-level scored outputs in `/home/jovyan/work/llmbench/runs/<model>/`.
- Metrics:
  - per-task 95% bootstrap CI over examples
  - paired bootstrap or approximate randomization for MinervaRL vs GRPO per backbone/task
  - task-family means with bootstrap CIs
- Reporting rule: do not report a percentage without denominator and source file.

Priority A: rollout-aware final evaluation
- For exact-match/MCQ/taxonomy tasks: pass@1, pass@k, verifier success@k.
- For F1/CVSS/extraction tasks: best-of-k verifier score or thresholded success@k.
- Recommended k values: k = 1, 4, 8, matching or below the training rollout budget.
- Use matched sampling params across GRPO and MinervaRL.

Priority B: ablations on final evaluation
- Existing validation ablations:
  - GRPO
  - GRPO (12 rollouts)
  - answer-only SFT
  - MinervaRL
  - no EMA teacher
  - no filtering
  - no ML filter
- Missing or ambiguous for reviewer:
  - no ACR is essentially GRPO, but state this explicitly.
  - heuristic-only is "no ML filter"; no filtering disables both heuristic and ML.
  - matched additional SFT compute should be clarified as answer-only SFT, or add a stronger matched-SFT control if available.

## Draft Rebuttal Skeleton

Opening:
We thank the reviewer for the constructive feedback. We agree that the current draft needs clearer presentation and stronger measurement of uncertainty and rollout behavior, and we will revise the paper accordingly.

1. Evaluation uncertainty and aggregation:
We agree that the heterogeneous average should not be the only summary. We located the row-level scored outputs for all Eval12 runs and computed paired bootstrap intervals over evaluation examples. The 12-column average MinervaRL-GRPO deltas are +4.2 pp [3.5, 5.1] for Llama-8B, +6.2 pp [5.3, 7.2] for Llama-3B, +6.4 pp [5.6, 7.7] for Qwen-8B, and +0.4 pp [-0.8, 1.3] for Qwen-4B. We will add these intervals, per-task denominators, and task-family averages to the revision, and we will soften the language for Qwen-4B because its average delta is not clearly separated from GRPO.

2. Rollout-aware evaluation:
The reviewer is right that final pass@k/best-of-k evaluation directly tests the support-expansion claim. The current paper includes training-time evidence through zero-solve fraction (Figure 3) and validation dynamics (Figure 4), but we will add held-out rollout-aware metrics under matched sampling budgets. For exact-match tasks we will report pass@1/pass@k or verifier success@k; for graded tasks we will report best-of-k verifier score or thresholded success@k.

3. Component attribution:
Table 4 already includes Llama-8B validation ablations for GRPO, GRPO with 12 rollouts, answer-only SFT, no EMA teacher, no filtering, and no ML filter. We will make this discussion more explicit: GRPO is the no-ACR control, no filtering removes both heuristic and TextCNN filtering, and no ML filter leaves heuristic filtering only. If final ablation checkpoints are available, we will also add 12-task evaluation for the main ablation variants.

4. Presentation and theory:
We agree that the current writing assumes too much CTI background. We will revise the introduction around a concrete CTI mapping example, add a compact task/metric table before results, simplify Figure 1 to separate the base GRPO loop from the ACR/SFT loop, and clean up Section 4.1 notation. We will reframe Appendix H as a toy finite-sampling formalization of the support-seeding intuition, not as a proof of full LM optimization dynamics.

5. Broader impact:
We will expand the broader-impact statement with concrete mitigations: defensive-use documentation, cyber-safety red-teaming before release, careful release of derived artifacts and model checkpoints, and explicit guidance against offensive automation.

## Immediate Next Steps

1. Use `rebuttal/llmbench_eval12_stats.md` to draft the evaluation/reliability portion of the TMLR response.
2. Optionally run a matched k-generation evaluation for GRPO and MinervaRL on the most relevant exact-match tasks first: RCM, ATE, RMS, ElasticRule, CKT/CyberMetric.
3. Draft the final TMLR response in 3-6 grouped points, using cautious language for mixed or non-separated task groups.

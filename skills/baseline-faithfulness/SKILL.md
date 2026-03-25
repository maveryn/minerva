---
name: baseline-faithfulness
description: Use when implementing a comparison baseline from a source paper or source repository. Enforces faithful reproduction of the original method, explicit separation of algorithmic behavior from implementation details, and user consultation before implementing or running any functional deviation that could change comparability.
---

# Baseline Faithfulness

Use this skill when the task is to implement a baseline method in the current repo from:
- an original paper
- an original source repository
- or both

The purpose of this skill is to preserve comparison integrity.

## Core Rule

Implement the baseline as faithfully as possible to the original method.

Do not silently:
- simplify the algorithm
- "modernize" the method
- adapt it for convenience
- change the training loop
- change model initialization behavior
- change data construction rules
- change validation or checkpoint-selection rules

If a functional choice is unclear in the paper, or differs between paper and source repo, clarify the intended implementation with the user before writing the code.

If a functional change from the original is required by the new repo, dataset, hardware, or training stack, consult the user before implementing it.

## Required Workflow

1. Identify the authoritative source.
   Prefer the original paper plus the original source repo if available.

2. Extract the algorithmic contract.
   Write down the parts that define the method itself:
   - outer training loop
   - sampling/generation procedure
   - filtering/selection rule
   - data accumulation rule
   - model initialization rule
   - update rule
   - evaluation rule
   - checkpoint-selection rule

3. Separate algorithm from tooling.
   Tooling choices can change between repos.
   Algorithmic behavior should not change unless explicitly approved.

4. Resolve ambiguities before implementation.
   If the paper is underspecified, or the repo behavior is not obviously the intended algorithm, stop and clarify with the user.

5. Classify every adaptation.
   - Mechanical adaptation: does not change the method, only the local integration
   - Functional adaptation: changes behavior in a way that may affect comparability

6. Consult before implementing any functional adaptation.
   This applies even before experiments are run.

7. Implement the agreed version explicitly in code and config.
   The resulting baseline should be unambiguous from the launcher, config, and comments.

## What Counts As A Functional Adaptation

These require user consultation before implementation:
- changing what model is used in each stage
- changing whether a method trains cumulatively or per round
- changing whether a round starts from base model or prior checkpoint
- changing batch size in a way that affects fairness claims
- changing prompt truncation or response truncation policy
- changing sampling count, rollout count, or repair count
- changing the selection or verifier rule
- changing validation set composition
- changing best-checkpoint metric
- changing number of rounds or stopping rule
- changing target format, supervision format, or loss target

## Paper-Only Cases

If only a paper exists:
- extract the clearest possible algorithm from the paper
- identify underspecified parts explicitly
- ask the user to confirm the intended implementation before writing code for those parts

Do not fill in important missing algorithmic choices by intuition alone.

Examples of paper-only ambiguities that must be clarified:
- whether training is cumulative across rounds
- whether each round starts from the latest checkpoint or the base model
- which validation metric chooses the final checkpoint
- how to handle long outputs or truncation
- how to match compute fairly against another method

## Source-Repo Cases

If source code exists:
- verify whether the source repo matches the paper
- treat the source repo as the default implementation reference for procedural behavior when both paper and repo are given
- if the repo appears to differ from the paper, surface the difference explicitly
- default to source-repo behavior unless the user asks to follow the paper instead

## Implementation Requirements

When you implement the baseline in the current repo:
- create a dedicated config
- create a dedicated launcher script if the repo uses launchers
- preserve the source method's algorithmic structure
- document approved adaptations briefly in code or config comments

## Before Running Experiments

Before any experiment run, confirm that:
- the implemented loop still matches the agreed source behavior
- validation and best-checkpoint logic match the intended comparison target
- any non-comparable changes were explicitly approved

If not, stop and consult.

## Expected Output Style

When reporting status, summarize:
- source behavior
- local implementation behavior
- any remaining differences
- whether those differences are mechanical or functional

If a difference is functional and not approved, do not proceed silently.

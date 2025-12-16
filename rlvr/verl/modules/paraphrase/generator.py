# verl/modules/paraphrase/generator.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Suffix-only paraphrase generator with multiple 'rethinking' guideline variants.

This generator *only* appends a suffix to the original prompt text. The suffixes
are phrased differently from any upstream prefix prompt you might be using
(e.g., those mentioning `<think>` tags), and they consistently require placing
the final answer within \\boxed{}.

Usage (typical from trainer code):
    paraphraser = TemplateParaphraser(strategy="cycle", seed=123)
    x_primes = paraphraser.sample({"text": original_prompt}, m=2)
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union


# ---- Curated suffix variants (worded differently from the user's prefix) ----
SUFFIX_VARIANTS: Sequence[str] = (
    "Guidelines: Please think step by step, and **regularly perform self-questioning, "
    "self-verification, and self-correction** throughout your reasoning. Use connectives "
    'like “Wait, does it seem right?” or “Wait, let’s double check.” Put your final answer '
    "inside \\boxed{}.",

    "While reasoning, proceed step by step and **frequently pause to self-question, verify, "
    "and correct** any issues. You may use cues such as “Wait, does it seem right?” or "
    "“Wait, there might be a mistake.” The final answer must be written inside \\boxed{}.",

    "Work in careful steps and **continually audit your thinking with self-questioning, "
    "verification, and correction** (e.g., “Wait, does it seem right?”, “Wait, let’s double check”). "
    "Conclude by placing the final result inside \\boxed{}.",

    "Adopt a step-by-step approach and **periodically apply self-questioning, self-verification, "
    "and self-correction** to your reasoning (e.g., “Wait, does it seem right?”, "
    "“Wait, there might be a mistake”). Ensure the final answer appears inside \\boxed{}.",

    "Use a deliberate, stepwise process and **routinely ask yourself if each step is sound, verify it, "
    "and correct if needed** (e.g., “Wait, let’s double check”, “Wait, does it seem right?”). "
    "Report the final answer inside \\boxed{}.",

    "Reason in clear steps, and **at regular intervals perform self-questioning, verification, "
    "and correction** (e.g., “Wait, there might be a mistake”, “Wait, does it seem right?”). "
    "Write the final answer inside \\boxed{}.",

    "Proceed step by step with **ongoing self-questioning, self-verification, and self-correction** "
    "to validate your chain of thought (e.g., “Wait, does it seem right?”, “Wait, let’s double check”). "
    "Put the final answer in \\boxed{}.",

    "Follow a stepwise plan and **consistently check yourself via self-questioning, verification, "
    "and correction** (e.g., “Wait, there might be a mistake”, “Wait, does it seem right?”). "
    "The final numerical result must be inside \\boxed{}.",

    "Think in incremental steps and **repeatedly use self-questioning, verification, and correction** "
    "to guard against errors (e.g., “Wait, let’s double check”, “Wait, does it seem right?”). "
    "Present the final answer within \\boxed{}.",

    "Use methodical steps and **frequent self-questioning, self-verification, and self-correction** "
    "(e.g., “Wait, does it seem right?”, “Wait, there might be a mistake”) to maintain accuracy. "
    "State the final answer inside \\boxed{}.",

    "Develop the solution step by step while **regularly interrogating, verifying, and correcting** "
    "your reasoning (e.g., “Wait, let’s double check”, “Wait, does it seem right?”). "
    "Record the final answer in \\boxed{}.",

    "Keep a stepwise cadence and **habitually perform self-questioning, verification, and correction** "
    "(e.g., “Wait, there might be a mistake”, “Wait, does it seem right?”). "
    "Finish by enclosing the final answer in \\boxed{}.",
)


@dataclass
class TemplateParaphraser:
    """Suffix-only paraphraser that appends one of several guideline variants.

    Parameters
    ----------
    gamma_mix : float
        Kept for config compatibility (mixture handled in trainer). Unused here.
    variants : Sequence[str]
        Pool of suffix variants to choose from.
    strategy : {"cycle","random"}
        Selection strategy when more than one paraphrase is requested:
          - "cycle": iterate through variants in fixed order (stateful cursor)
          - "random": sample without replacement (then with replacement if needed)
    seed : Optional[int]
        RNG seed used when strategy == "random".
    """

    gamma_mix: float = 0.0
    variants: Sequence[str] = field(default_factory=lambda: SUFFIX_VARIANTS)
    strategy: str = "cycle"  # or "random"
    seed: Optional[int] = None

    # internal cursor for cycle strategy
    _cursor: int = field(default=0, init=False, repr=False)

    def _pick_indices(self, m: int) -> List[int]:
        n = len(self.variants)
        if n == 0 or m <= 0:
            return []
        if self.strategy == "random":
            rng = random.Random(self.seed)
            if m <= n:
                return rng.sample(range(n), k=m)
            # sample without replacement, then with replacement for the remainder
            idxs = list(range(n))
            idxs.extend(rng.choices(range(n), k=m - n))
            return idxs
        # "cycle" (stateful)
        start = self._cursor
        idxs = [(start + i) % n for i in range(m)]
        self._cursor = (start + m) % n
        return idxs

    def sample(self, x: Union[dict, str], m: int = 2) -> List[str]:
        """Generate up to ``m`` suffix-only paraphrases for item ``x``.

        ``x`` is expected to be a dict containing the key ``"text"``; if a string
        is passed, it is treated directly as the base prompt.
        """
        base_text = x["text"] if isinstance(x, dict) else str(x)
        idxs = self._pick_indices(m)
        # prepend two newlines so the suffix is visually separated from the question
        return [f"{base_text}\n\n{self.variants[i]}" for i in idxs]


__all__ = ["TemplateParaphraser", "SUFFIX_VARIANTS"]

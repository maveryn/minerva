# verl/trainer/prefix_guided/prefix_hint_samplers.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class FractionalStepBandit:
    """
    Thompson-sampling bandit over FRACTIONS of CoT steps.

    We discretize [min_frac, max_frac] into `frac_bins` bins and maintain a Beta posterior
    for each bin with successes ~ (reach correct answer) and failures otherwise.

    During warmup (first warmup_total_steps trainer iterations), choose a random bin.
    Afterwards, use TS to sample a success-probability from each bin's Beta posterior and
    pick the bin whose sampled probability is *closest* to target_success (e.g., 0.5).
    """

    frac_bins: int
    target_success: float
    min_frac: float
    max_frac: float
    warmup_total_steps: int

    # internal stats
    _success: Optional[list[int]] = None
    _fail: Optional[list[int]] = None
    _calls: int = 0

    def __post_init__(self):
        self.min_frac = float(self.min_frac)
        self.max_frac = float(self.max_frac)
        assert 0.0 <= self.min_frac <= self.max_frac <= 1.0
        assert self.frac_bins >= 2
        self._success = [0 for _ in range(self.frac_bins)]
        self._fail = [0 for _ in range(self.frac_bins)]

    # ---- helpers over bins -------------------------------------------------

    def _bin_bounds(self, idx: int) -> tuple[float, float]:
        w = (self.max_frac - self.min_frac) / self.frac_bins
        lo = self.min_frac + idx * w
        hi = lo + w
        return lo, hi

    def _bin_center(self, idx: int) -> float:
        lo, hi = self._bin_bounds(idx)
        return 0.5 * (lo + hi)

    def _mean_success_prob(self, idx: int) -> float:
        # Posterior mean of Beta(1 + succ, 1 + fail)
        return (1 + self._success[idx]) / (2 + self._success[idx] + self._fail[idx])

    # ---- public API --------------------------------------------------------

    def select_prefix_steps(self, *, total_steps: int, rng: random.Random, trainer_step: Optional[int] = None) -> tuple[int, float, int]:
        """
        Returns:
            k_steps: integer number of steps to reveal (>=1, <= total_steps)
            chosen_frac_center: the chosen fraction (bin center) for logging
            bin_idx: which bin was selected (for updating posterior)
    
        Warm-up semantics:
          - If `trainer_step` is provided: warm-up is active while trainer_step < warmup_total_steps.
          - Else (back-compat): warm-up is active for the first `warmup_total_steps` *calls*.
        """
        total_steps = max(1, int(total_steps))
    
        # Decide warm-up by trainer steps if provided; otherwise by internal call counter
        if trainer_step is not None:
            in_warmup = int(trainer_step) < int(self.warmup_total_steps)
        else:
            self._calls += 1
            in_warmup = self._calls <= self.warmup_total_steps
    
        if in_warmup:
            bin_idx = rng.randrange(self.frac_bins)
        else:
            # Thompson sample prob-of-success for each bin and pick the one
            # whose sampled prob is closest to the target.
            best_idx, best_score = 0, -1e9
            for i in range(self.frac_bins):
                a = 1 + self._success[i]
                b = 1 + self._fail[i]
                p = rng.betavariate(a, b)
                score = -abs(p - self.target_success)
                if score > best_score:
                    best_score = score
                    best_idx = i
            bin_idx = best_idx
    
        frac = self._bin_center(bin_idx)
        k = max(1, min(total_steps, math.ceil(frac * total_steps)))
        return k, frac, bin_idx


    def update(self, *, bin_idx: int, success: bool) -> None:
        if success:
            self._success[bin_idx] += 1
        else:
            self._fail[bin_idx] += 1

    # ---- diagnostics -------------------------------------------------------

    def current_recommended_frac(self) -> tuple[float, int, float]:
        """
        Returns (frac_center, bin_idx, p_hat) where bin_idx has posterior-mean
        success probability closest to the target. Useful to log even during warmup.
        """
        best_idx, best_gap = 0, float("inf")
        for i in range(self.frac_bins):
            p_hat = self._mean_success_prob(i)
            gap = abs(p_hat - self.target_success)
            if gap < best_gap:
                best_gap = gap
                best_idx = i
        return self._bin_center(best_idx), best_idx, self._mean_success_prob(best_idx)

    def summary(self) -> dict:
        centers = [self._bin_center(i) for i in range(self.frac_bins)]
        p_hats = [self._mean_success_prob(i) for i in range(self.frac_bins)]
        return {
            "target_success": self.target_success,
            "bin_centers": centers,
            "success": list(self._success),
            "fail": list(self._fail),
            "p_hat": p_hats,
        }

# verl/losses/distill_ce.py
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

"""Cross-entropy loss used for mandatory self-distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_shifted(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Compute next-token cross entropy.

    Parameters
    ----------
    logits: Tensor
        Model logits of shape (batch, seq, vocab).
    labels: Tensor
        Target token ids of shape (batch, seq).
    ignore_index: int
        Tokens with this label will be ignored.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


class DistillCELoss:
    """Compute mean negative log-likelihood for teacher-forced pairs."""

    def __call__(self, actor, batch_xy) -> torch.Tensor:
        model_inputs = batch_xy["model_inputs"]
        labels = batch_xy["labels"]
        logits = actor(**model_inputs).logits
        return cross_entropy_shifted(logits, labels)


__all__ = ["DistillCELoss", "cross_entropy_shifted"]

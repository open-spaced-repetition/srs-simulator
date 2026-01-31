from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiUserBehavior:
    attendance_prob: torch.Tensor
    lazy_good_bias: torch.Tensor
    max_new_per_day: torch.Tensor
    max_reviews_per_day: torch.Tensor
    max_cost_per_day: torch.Tensor
    success_weights: torch.Tensor
    first_rating_prob: torch.Tensor


@dataclass
class MultiUserCost:
    base: torch.Tensor
    penalty: torch.Tensor
    learn_costs: torch.Tensor
    review_costs: torch.Tensor

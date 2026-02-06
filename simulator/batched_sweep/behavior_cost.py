from __future__ import annotations

import math
from pathlib import Path

import torch

from simulator.button_usage import load_button_usage_config, normalize_button_usage
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost


def load_usage(
    user_ids: list[int], button_usage: Path | None
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    learn_costs = []
    review_costs = []
    first_rating_prob = []
    review_rating_prob = []
    learning_rating_prob = []
    relearning_rating_prob = []
    state_rating_costs = []
    for user_id in user_ids:
        config = (
            load_button_usage_config(button_usage, user_id)
            if button_usage is not None
            else None
        )
        usage = normalize_button_usage(config)
        learn_costs.append(usage["learn_costs"])
        review_costs.append(usage["review_costs"])
        first_rating_prob.append(usage["first_rating_prob"])
        review_rating_prob.append(usage["review_rating_prob"])
        learning_rating_prob.append(usage["learning_rating_prob"])
        relearning_rating_prob.append(usage["relearning_rating_prob"])
        state_rating_costs.append(usage["state_rating_costs"])
    return (
        torch.tensor(learn_costs, dtype=torch.float32),
        torch.tensor(review_costs, dtype=torch.float32),
        torch.tensor(first_rating_prob, dtype=torch.float32),
        torch.tensor(review_rating_prob, dtype=torch.float32),
        torch.tensor(learning_rating_prob, dtype=torch.float32),
        torch.tensor(relearning_rating_prob, dtype=torch.float32),
        torch.tensor(state_rating_costs, dtype=torch.float32),
    )


def build_behavior_cost(
    user_count: int,
    *,
    deck_size: int,
    learn_limit: int,
    review_limit: int | None,
    cost_limit_minutes: float | None,
    learn_costs: torch.Tensor,
    review_costs: torch.Tensor,
    first_rating_prob: torch.Tensor,
    review_rating_prob: torch.Tensor,
    learning_rating_prob: torch.Tensor,
    relearning_rating_prob: torch.Tensor,
    state_rating_costs: torch.Tensor,
    short_term: bool,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    max_reviews = review_limit if review_limit is not None else deck_size
    max_cost = cost_limit_minutes * 60.0 if cost_limit_minutes is not None else math.inf
    if short_term:
        # In short-term mode we use state-specific rating costs.
        # new+learning: learning (0), review: review (1), relearning: relearning (2)
        learn_costs = state_rating_costs[:, 0]
        review_costs = state_rating_costs[:, 1]
    behavior = MultiUserBehavior(
        attendance_prob=torch.full((user_count,), 1.0),
        lazy_good_bias=torch.zeros(user_count),
        max_new_per_day=torch.full((user_count,), learn_limit, dtype=torch.int64),
        max_reviews_per_day=torch.full((user_count,), max_reviews, dtype=torch.int64),
        max_cost_per_day=torch.full((user_count,), max_cost),
        success_weights=review_rating_prob,
        learning_success_weights=learning_rating_prob,
        relearning_success_weights=relearning_rating_prob,
        first_rating_prob=first_rating_prob,
    )
    cost = MultiUserCost(
        base=torch.zeros(user_count),
        penalty=torch.zeros(user_count),
        learn_costs=learn_costs,
        review_costs=review_costs,
        learning_review_costs=state_rating_costs[:, 0],
        relearning_review_costs=state_rating_costs[:, 2],
    )
    return behavior, cost

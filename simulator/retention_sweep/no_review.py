from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch

from simulator.core import Card, MemoryModel


@dataclass(frozen=True, slots=True)
class NoReviewMemorySeries:
    no_review_memorized: list[float]
    review_memory_gain: list[float]


def normalize_first_rating_prob(values: Sequence[float]) -> list[float]:
    probabilities = [float(value) for value in values]
    if len(probabilities) != 4:
        raise ValueError("first_rating_prob must contain four probabilities.")
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("first_rating_prob must contain finite non-negative values.")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("first_rating_prob must have positive total weight.")
    return [value / total for value in probabilities]


def build_no_review_retention_kernel(
    memory_model: MemoryModel,
    first_rating_prob: Sequence[float],
    days: int,
) -> list[float]:
    if days < 0:
        raise ValueError("days must be non-negative.")
    probabilities = normalize_first_rating_prob(first_rating_prob)
    cards: list[Card] = []
    for rating in range(1, 5):
        card = Card(id=rating - 1, reps=1, last_review=0.0)
        memory_model.init_card(card, rating)
        cards.append(card)

    return [
        sum(
            probability * memory_model.predict_retention(card, float(age))
            for probability, card in zip(probabilities, cards, strict=True)
        )
        for age in range(days)
    ]


@torch.inference_mode()
def build_batched_no_review_retention_kernels(
    env_ops,
    first_rating_prob: torch.Tensor,
    days: int,
    *,
    age_chunk_size: int = 64,
) -> list[list[float]]:
    if days < 0:
        raise ValueError("days must be non-negative.")
    if age_chunk_size <= 0:
        raise ValueError("age_chunk_size must be positive.")
    probabilities = first_rating_prob.to(device=env_ops.device, dtype=env_ops.dtype)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError("first_rating_prob must have shape (users, 4).")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("first_rating_prob must contain finite non-negative values.")
    totals = probabilities.sum(dim=1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("first_rating_prob rows must have positive total weight.")
    probabilities = probabilities / totals

    user_count = int(probabilities.shape[0])
    state = env_ops.init_state(user_count, 4)
    init_user = torch.arange(user_count, device=env_ops.device).repeat_interleave(4)
    init_card = torch.arange(4, device=env_ops.device).repeat(user_count)
    init_rating = init_card + 1
    env_ops.update_learn(state, init_user, init_card, init_rating)

    kernel = torch.empty(
        (user_count, days),
        device=env_ops.device,
        dtype=env_ops.dtype,
    )
    for start in range(0, days, age_chunk_size):
        stop = min(days, start + age_chunk_size)
        chunk_days = stop - start
        ages = torch.arange(
            start,
            stop,
            device=env_ops.device,
            dtype=env_ops.dtype,
        )
        user_idx = (
            torch.arange(user_count, device=env_ops.device)[None, :, None]
            .expand(chunk_days, user_count, 4)
            .reshape(-1)
        )
        card_idx = (
            torch.arange(4, device=env_ops.device)[None, None, :]
            .expand(chunk_days, user_count, 4)
            .reshape(-1)
        )
        elapsed = ages[:, None, None].expand(chunk_days, user_count, 4).reshape(-1)
        retention = env_ops.retrievability_entries(
            state,
            user_idx,
            card_idx,
            elapsed,
        ).reshape(chunk_days, user_count, 4)
        weighted = (retention * probabilities[None, :, :]).sum(dim=2)
        kernel[:, start:stop] = weighted.transpose(0, 1)
    return kernel.detach().cpu().tolist()


def calculate_no_review_memory_series(
    *,
    daily_memorized: Sequence[float],
    daily_new: Sequence[int],
    deck_size: int,
    first_rating_prob: Sequence[float],
    no_review_retention_kernel: Sequence[float],
) -> NoReviewMemorySeries:
    days = len(daily_memorized)
    if len(daily_new) != days:
        raise ValueError("daily_new and daily_memorized must have equal length.")
    if len(no_review_retention_kernel) < days:
        raise ValueError("no_review_retention_kernel is shorter than the simulation.")
    if deck_size < 0:
        raise ValueError("deck_size must be non-negative.")
    if days == 0:
        return NoReviewMemorySeries([], [])

    probabilities = normalize_first_rating_prob(first_rating_prob)
    unseen_recall_prior = sum(probabilities[1:])
    new_counts = np.asarray([max(0, int(value)) for value in daily_new], dtype=float)
    lag_kernel = np.asarray(no_review_retention_kernel[:days], dtype=float).copy()
    lag_kernel[0] = 0.0
    introduced_no_review = np.convolve(new_counts, lag_kernel, mode="full")[:days]

    learned_before_day = 0
    no_review_memorized: list[float] = []
    review_memory_gain: list[float] = []
    for day, memorized in enumerate(daily_memorized):
        unlearned = max(0, deck_size - learned_before_day)
        no_review_value = unlearned * unseen_recall_prior + float(
            introduced_no_review[day]
        )
        srs_value = unlearned * unseen_recall_prior + float(memorized)
        no_review_memorized.append(no_review_value)
        review_memory_gain.append(srs_value - no_review_value)
        learned_before_day = min(
            deck_size,
            learned_before_day + int(new_counts[day]),
        )
    return NoReviewMemorySeries(no_review_memorized, review_memory_gain)

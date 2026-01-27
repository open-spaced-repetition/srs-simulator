from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from simulator.core import CardView, Scheduler

if TYPE_CHECKING:
    import torch


class AnkiSM2Scheduler(Scheduler):
    """Anki SM-2 style scheduler without FSRS dependencies."""

    def __init__(
        self,
        *,
        graduating_interval: float = 1.0,
        easy_interval: float = 4.0,
        easy_bonus: float = 1.3,
        hard_interval_factor: float = 1.2,
        ease_start: float = 2.5,
        ease_min: float = 1.3,
        ease_max: float = 5.5,
    ) -> None:
        self.graduating_interval = float(graduating_interval)
        self.easy_interval = float(easy_interval)
        self.easy_bonus = float(easy_bonus)
        self.hard_interval_factor = float(hard_interval_factor)
        self.ease_start = float(ease_start)
        self.ease_min = float(ease_min)
        self.ease_max = float(ease_max)

    def init_card(self, card_view: CardView, rating: int, day: float):
        interval, ease = self._next_interval(
            prev_interval=0.0, rating=rating, ease=self.ease_start
        )
        return interval, {"ease": ease, "ivl": interval}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {}
        ease = float(state.get("ease", self.ease_start))
        prev_interval = float(card_view.interval or 0.0)
        interval, ease = self._next_interval(
            prev_interval=prev_interval, rating=rating, ease=ease
        )
        return interval, {"ease": ease, "ivl": interval}

    def _next_interval(
        self, *, prev_interval: float, rating: int, ease: float
    ) -> tuple[float, float]:
        rating = int(rating)
        if rating == 1:
            ease -= 0.2
        elif rating == 2:
            ease -= 0.15
        elif rating == 4:
            ease += 0.15
        ease = min(self.ease_max, max(self.ease_min, ease))

        is_new_card = prev_interval == 0.0
        if rating < 4:
            new_card_interval = self.graduating_interval
        else:
            new_card_interval = self.easy_interval

        elapsed = prev_interval
        if rating == 1:
            existing_interval = prev_interval * 0.0
        elif rating == 2:
            existing_interval = max(
                elapsed * self.hard_interval_factor,
                prev_interval * self.hard_interval_factor / 2.0,
            )
        elif rating == 4:
            existing_interval = max(elapsed * ease, prev_interval) * self.easy_bonus
        else:
            existing_interval = max(elapsed * ease, prev_interval)

        interval = new_card_interval if is_new_card else existing_interval
        interval = max(1.0, interval)
        return interval, ease


def _anki_next_interval(
    prev_interval,
    rating,
    ease,
    *,
    graduating_interval: float,
    easy_interval: float,
    easy_bonus: float,
    hard_interval_factor: float,
    ease_min: float,
    ease_max: float,
):
    new_ease = ease
    new_ease = new_ease - 0.2 * (rating == 1).to(new_ease.dtype)
    new_ease = new_ease - 0.15 * (rating == 2).to(new_ease.dtype)
    new_ease = new_ease + 0.15 * (rating == 4).to(new_ease.dtype)
    new_ease = new_ease.clamp(min=ease_min, max=ease_max)

    is_new_card = prev_interval == 0.0
    new_card_interval = prev_interval.new_full(prev_interval.shape, graduating_interval)
    new_card_interval = new_card_interval.where(rating < 4, easy_interval)

    elapsed = prev_interval
    existing_interval = prev_interval
    existing_interval = existing_interval.where(rating != 1, prev_interval * 0.0)
    hard_interval = torch.maximum(
        elapsed * hard_interval_factor, prev_interval * hard_interval_factor / 2.0
    )
    existing_interval = existing_interval.where(rating != 2, hard_interval)
    easy_interval_val = torch.maximum(elapsed * new_ease, prev_interval) * easy_bonus
    existing_interval = existing_interval.where(rating != 4, easy_interval_val)
    normal_interval = torch.maximum(elapsed * new_ease, prev_interval)
    existing_interval = existing_interval.where(
        (rating == 1) | (rating == 2) | (rating == 4),
        normal_interval,
    )

    interval = torch.where(is_new_card, new_card_interval, existing_interval)
    interval = interval.clamp(min=1.0)
    return interval, new_ease


@dataclass
class AnkiVectorizedState:
    ease: "torch.Tensor"


class AnkiSM2VectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: AnkiSM2Scheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._graduating_interval = float(scheduler.graduating_interval)
        self._easy_interval = float(scheduler.easy_interval)
        self._easy_bonus = float(scheduler.easy_bonus)
        self._hard_interval_factor = float(scheduler.hard_interval_factor)
        self._ease_start = float(scheduler.ease_start)
        self._ease_min = float(scheduler.ease_min)
        self._ease_max = float(scheduler.ease_max)

    def init_state(self, deck_size: int) -> AnkiVectorizedState:
        ease = self._torch.full(
            (deck_size,), self._ease_start, device=self.device, dtype=self.dtype
        )
        return AnkiVectorizedState(ease=ease)

    def review_priority(
        self, state: AnkiVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: AnkiVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            rating,
            state.ease[idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
        )
        state.ease[idx] = new_ease
        return intervals

    def update_learn(
        self,
        state: AnkiVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        prev_interval = self._torch.zeros(
            idx.numel(), device=self.device, dtype=self.dtype
        )
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            rating,
            state.ease[idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
        )
        state.ease[idx] = new_ease
        return intervals


__all__ = ["AnkiSM2Scheduler"]

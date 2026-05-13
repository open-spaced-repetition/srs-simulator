from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from simulator.core import CardView, Scheduler


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
        new_interval_factor: float = 0.0,
        interval_multiplier: float = 1.0,
    ) -> None:
        self.graduating_interval = float(graduating_interval)
        self.easy_interval = float(easy_interval)
        self.easy_bonus = float(easy_bonus)
        self.hard_interval_factor = float(hard_interval_factor)
        self.ease_start = float(ease_start)
        self.ease_min = float(ease_min)
        self.ease_max = float(ease_max)
        self.new_interval_factor = float(new_interval_factor)
        self.interval_multiplier = float(interval_multiplier)

    def init_card(self, card_view: CardView, rating: int, day: float):
        interval, ease = self._next_interval(
            prev_interval=0.0, elapsed=0.0, rating=rating, ease=self.ease_start
        )
        return interval, {"ease": ease, "ivl": interval}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {}
        ease = float(state.get("ease", self.ease_start))
        prev_interval = float(card_view.interval or 0.0)
        interval, ease = self._next_interval(
            prev_interval=prev_interval, elapsed=elapsed, rating=rating, ease=ease
        )
        return interval, {"ease": ease, "ivl": interval}

    def _next_interval(
        self, *, prev_interval: float, elapsed: float, rating: int, ease: float
    ) -> tuple[float, float]:
        rating = int(rating)
        ease = min(self.ease_max, max(self.ease_min, ease))
        current_interval = max(1.0, prev_interval)
        is_new_card = prev_interval == 0.0
        if rating < 4:
            new_card_interval = self.graduating_interval
        else:
            new_card_interval = self.easy_interval
        if is_new_card:
            interval = max(1.0, new_card_interval)
        elif rating == 1:
            interval = max(1.0, current_interval * self.new_interval_factor)
        else:
            interval = max(
                1.0,
                _passing_review_interval(
                    prev_interval=current_interval,
                    elapsed=max(0.0, float(elapsed)),
                    ease=ease,
                    hard_multiplier=self.hard_interval_factor,
                    easy_multiplier=self.easy_bonus,
                    rating=rating,
                )
                * self.interval_multiplier,
            )

        if rating == 1:
            ease += 0.2 * -1
        elif rating == 2:
            ease += 0.15 * -1
        elif rating == 4:
            ease += 0.15
        ease = min(self.ease_max, max(self.ease_min, ease))
        return interval, ease


def _anki_next_interval(
    prev_interval,
    elapsed,
    rating,
    ease,
    *,
    graduating_interval: float | torch.Tensor,
    easy_interval: float | torch.Tensor,
    easy_bonus: float | torch.Tensor,
    hard_interval_factor: float | torch.Tensor,
    ease_min: float | torch.Tensor,
    ease_max: float | torch.Tensor,
    new_interval_factor: float | torch.Tensor,
    interval_multiplier: float | torch.Tensor,
):
    graduating_interval = _param_like(graduating_interval, prev_interval)
    easy_interval = _param_like(easy_interval, prev_interval)
    easy_bonus = _param_like(easy_bonus, prev_interval)
    hard_interval_factor = _param_like(hard_interval_factor, prev_interval)
    ease_min = _param_like(ease_min, prev_interval)
    ease_max = _param_like(ease_max, prev_interval)
    new_interval_factor = _param_like(new_interval_factor, prev_interval)
    interval_multiplier = _param_like(interval_multiplier, prev_interval)

    new_ease = torch.minimum(torch.maximum(ease, ease_min), ease_max)
    current_interval = torch.clamp(prev_interval, min=1.0)
    is_new_card = prev_interval == 0.0
    new_card_interval = torch.where(rating < 4, graduating_interval, easy_interval)

    is_early = elapsed < current_interval
    hard_multiplier = hard_interval_factor
    easy_multiplier = easy_bonus

    hard_minimum = torch.where(
        hard_multiplier <= 1.0,
        torch.zeros_like(current_interval),
        current_interval + 1.0,
    )
    hard_non_early = _constrain_passing_interval_tensor(
        current_interval * hard_multiplier, hard_minimum
    )
    good_minimum = torch.where(
        hard_multiplier <= 1.0,
        current_interval + 1.0,
        hard_non_early + 1.0,
    )
    days_late = torch.clamp(elapsed - current_interval, min=0.0)
    good_non_early = _constrain_passing_interval_tensor(
        (current_interval + days_late / 2.0) * new_ease, good_minimum
    )
    easy_non_early = _constrain_passing_interval_tensor(
        (current_interval + days_late) * new_ease * easy_multiplier,
        good_non_early + 1.0,
    )

    half_usual = hard_multiplier / 2.0
    hard_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * hard_multiplier, current_interval * half_usual),
        torch.zeros_like(current_interval),
    )
    good_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * new_ease, current_interval),
        torch.zeros_like(current_interval),
    )
    reduced_bonus = easy_multiplier - (easy_multiplier - 1.0) / 2.0
    easy_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * new_ease, current_interval) * reduced_bonus,
        torch.zeros_like(current_interval),
    )

    hard_interval = torch.where(is_early, hard_early, hard_non_early)
    good_interval = torch.where(is_early, good_early, good_non_early)
    easy_interval_val = torch.where(is_early, easy_early, easy_non_early)

    passing_interval = torch.where(
        rating == 2,
        hard_interval,
        torch.where(rating == 4, easy_interval_val, good_interval),
    )
    review_interval = torch.where(
        rating == 1,
        current_interval * new_interval_factor,
        passing_interval * interval_multiplier,
    )
    interval = torch.where(is_new_card, new_card_interval, review_interval)
    interval = interval.clamp(min=1.0)

    new_ease = new_ease + -0.2 * (rating == 1).to(new_ease.dtype)
    new_ease = new_ease + -0.15 * (rating == 2).to(new_ease.dtype)
    new_ease = new_ease + 0.15 * (rating == 4).to(new_ease.dtype)
    new_ease = torch.minimum(torch.maximum(new_ease, ease_min), ease_max)
    return interval, new_ease


def _param_like(value: float | torch.Tensor, reference: torch.Tensor):
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.expand_as(reference)
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(
        f"scheduler parameter tensor must be scalar or shape {tuple(reference.shape)}, "
        f"got {tuple(tensor.shape)}."
    )


def _param_vector(
    value: float | torch.Tensor,
    *,
    user_count: int,
    device: torch.device,
    dtype: torch.dtype,
    field_name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(user_count).clone()
    if tensor.ndim == 1 and int(tensor.shape[0]) == user_count:
        return tensor.clone()
    raise ValueError(
        f"{field_name} must be scalar or a tensor with shape ({user_count},)."
    )


@dataclass
class AnkiBatchState:
    ease: torch.Tensor
    graduating_interval: torch.Tensor
    easy_interval: torch.Tensor
    easy_bonus: torch.Tensor
    hard_interval_factor: torch.Tensor
    ease_start: torch.Tensor
    ease_min: torch.Tensor
    ease_max: torch.Tensor
    new_interval_factor: torch.Tensor
    interval_multiplier: torch.Tensor


class AnkiSM2BatchSchedulerOps:
    def __init__(
        self,
        *,
        graduating_interval: float | torch.Tensor,
        easy_interval: float | torch.Tensor,
        easy_bonus: float | torch.Tensor,
        hard_interval_factor: float | torch.Tensor,
        ease_start: float | torch.Tensor,
        ease_min: float | torch.Tensor,
        ease_max: float | torch.Tensor,
        new_interval_factor: float | torch.Tensor = 0.0,
        interval_multiplier: float | torch.Tensor = 1.0,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self._graduating_interval = graduating_interval
        self._easy_interval = easy_interval
        self._easy_bonus = easy_bonus
        self._hard_interval_factor = hard_interval_factor
        self._ease_start = ease_start
        self._ease_min = ease_min
        self._ease_max = ease_max
        self._new_interval_factor = new_interval_factor
        self._interval_multiplier = interval_multiplier

    def init_state(self, user_count: int, deck_size: int) -> AnkiBatchState:
        ease_start = _param_vector(
            self._ease_start,
            user_count=user_count,
            device=self.device,
            dtype=self.dtype,
            field_name="ease_start",
        )
        ease = ease_start[:, None].expand(user_count, deck_size).clone()
        return AnkiBatchState(
            ease=ease,
            graduating_interval=_param_vector(
                self._graduating_interval,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="graduating_interval",
            ),
            easy_interval=_param_vector(
                self._easy_interval,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="easy_interval",
            ),
            easy_bonus=_param_vector(
                self._easy_bonus,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="easy_bonus",
            ),
            hard_interval_factor=_param_vector(
                self._hard_interval_factor,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="hard_interval_factor",
            ),
            ease_start=ease_start,
            ease_min=_param_vector(
                self._ease_min,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="ease_min",
            ),
            ease_max=_param_vector(
                self._ease_max,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="ease_max",
            ),
            new_interval_factor=_param_vector(
                self._new_interval_factor,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="new_interval_factor",
            ),
            interval_multiplier=_param_vector(
                self._interval_multiplier,
                user_count=user_count,
                device=self.device,
                dtype=self.dtype,
                field_name="interval_multiplier",
            ),
        )

    def review_priority(
        self, state: AnkiBatchState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return torch.zeros_like(elapsed, device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: AnkiBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[user_idx, card_idx],
            graduating_interval=state.graduating_interval[user_idx],
            easy_interval=state.easy_interval[user_idx],
            easy_bonus=state.easy_bonus[user_idx],
            hard_interval_factor=state.hard_interval_factor[user_idx],
            ease_min=state.ease_min[user_idx],
            ease_max=state.ease_max[user_idx],
            new_interval_factor=state.new_interval_factor[user_idx],
            interval_multiplier=state.interval_multiplier[user_idx],
        )
        state.ease[user_idx, card_idx] = new_ease
        return intervals

    def update_learn(
        self,
        state: AnkiBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        prev_interval = torch.zeros_like(rating, dtype=self.dtype)
        elapsed = torch.zeros_like(prev_interval)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[user_idx, card_idx],
            graduating_interval=state.graduating_interval[user_idx],
            easy_interval=state.easy_interval[user_idx],
            easy_bonus=state.easy_bonus[user_idx],
            hard_interval_factor=state.hard_interval_factor[user_idx],
            ease_min=state.ease_min[user_idx],
            ease_max=state.ease_max[user_idx],
            new_interval_factor=state.new_interval_factor[user_idx],
            interval_multiplier=state.interval_multiplier[user_idx],
        )
        state.ease[user_idx, card_idx] = new_ease
        return intervals


def _round_half_up(value: float) -> float:
    return float(math.floor(value + 0.5))


def _constrain_passing_interval(interval: float, minimum: float) -> float:
    rounded = _round_half_up(interval)
    return max(1.0, float(max(rounded, minimum)))


def _constrain_passing_interval_tensor(
    interval: "torch.Tensor", minimum: "torch.Tensor"
) -> "torch.Tensor":
    rounded = torch.floor(interval + 0.5)
    return torch.maximum(rounded, minimum).clamp(min=1.0)


def _passing_review_interval(
    *,
    prev_interval: float,
    elapsed: float,
    ease: float,
    hard_multiplier: float,
    easy_multiplier: float,
    rating: int,
) -> float:
    current_interval = max(1.0, prev_interval)
    elapsed = max(0.0, elapsed)
    if elapsed < current_interval:
        hard = _constrain_passing_interval(
            max(elapsed * hard_multiplier, current_interval * hard_multiplier / 2.0),
            0.0,
        )
        good = _constrain_passing_interval(
            max(elapsed * ease, current_interval),
            0.0,
        )
        reduced_bonus = easy_multiplier - (easy_multiplier - 1.0) / 2.0
        easy = _constrain_passing_interval(
            max(elapsed * ease, current_interval) * reduced_bonus,
            0.0,
        )
    else:
        days_late = max(0.0, elapsed - current_interval)
        hard_minimum = 0.0 if hard_multiplier <= 1.0 else current_interval + 1.0
        hard = _constrain_passing_interval(
            current_interval * hard_multiplier, hard_minimum
        )
        good_minimum = current_interval + 1.0 if hard_multiplier <= 1.0 else hard + 1.0
        good = _constrain_passing_interval(
            (current_interval + days_late / 2.0) * ease, good_minimum
        )
        easy = _constrain_passing_interval(
            (current_interval + days_late) * ease * easy_multiplier,
            good + 1.0,
        )
    if rating == 2:
        return hard
    if rating == 4:
        return easy
    return good


@dataclass
class AnkiBatchedState:
    ease: "torch.Tensor"


class AnkiSM2BatchedSchedulerOps:
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
        self._new_interval_factor = float(scheduler.new_interval_factor)
        self._interval_multiplier = float(scheduler.interval_multiplier)

    def init_state(self, deck_size: int) -> AnkiBatchedState:
        ease = self._torch.full(
            (deck_size,), self._ease_start, device=self.device, dtype=self.dtype
        )
        return AnkiBatchedState(ease=ease)

    def review_priority(
        self, state: AnkiBatchedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: AnkiBatchedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
            new_interval_factor=self._new_interval_factor,
            interval_multiplier=self._interval_multiplier,
        )
        state.ease[idx] = new_ease
        return intervals

    def update_learn(
        self,
        state: AnkiBatchedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        prev_interval = self._torch.zeros(
            idx.numel(), device=self.device, dtype=self.dtype
        )
        elapsed = self._torch.zeros_like(prev_interval)
        intervals, new_ease = _anki_next_interval(
            prev_interval,
            elapsed,
            rating,
            state.ease[idx],
            graduating_interval=self._graduating_interval,
            easy_interval=self._easy_interval,
            easy_bonus=self._easy_bonus,
            hard_interval_factor=self._hard_interval_factor,
            ease_min=self._ease_min,
            ease_max=self._ease_max,
            new_interval_factor=self._new_interval_factor,
            interval_multiplier=self._interval_multiplier,
        )
        state.ease[idx] = new_ease
        return intervals


__all__ = ["AnkiSM2Scheduler"]

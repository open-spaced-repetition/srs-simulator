from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simulator.core import CardView, Scheduler

if TYPE_CHECKING:
    import torch


class FixedIntervalScheduler(Scheduler):
    """Stateless fixed-interval baseline."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = float(interval)

    def init_card(self, card_view: CardView, rating: int, day: float):
        return self.interval, None

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        return self.interval, None


@dataclass
class FixedVectorizedState:
    interval: float


class FixedVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: FixedIntervalScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._interval = float(scheduler.interval)
        self._interval_days = max(1, int(round(self._interval)))

    def init_state(self, deck_size: int) -> FixedVectorizedState:
        return FixedVectorizedState(interval=self._interval)

    def review_priority(
        self, state: FixedVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: FixedVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.full(
            (idx.numel(),),
            float(self._interval_days),
            device=self.device,
            dtype=self.dtype,
        )

    def update_learn(
        self,
        state: FixedVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.full(
            (idx.numel(),),
            float(self._interval_days),
            device=self.device,
            dtype=self.dtype,
        )


@dataclass
class FixedBatchState:
    interval_days: float


class FixedBatchSchedulerOps:
    def __init__(
        self,
        *,
        interval: float,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._interval = float(interval)
        self._interval_days = max(1, int(round(self._interval)))

    def init_state(self, user_count: int, deck_size: int) -> FixedBatchState:
        return FixedBatchState(interval_days=float(self._interval_days))

    def review_priority(
        self, state: FixedBatchState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros_like(elapsed, device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: FixedBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        return self._torch.full(
            (user_idx.numel(),),
            state.interval_days,
            device=self.device,
            dtype=self.dtype,
        )

    def update_learn(
        self,
        state: FixedBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        return self._torch.full(
            (user_idx.numel(),),
            state.interval_days,
            device=self.device,
            dtype=self.dtype,
        )

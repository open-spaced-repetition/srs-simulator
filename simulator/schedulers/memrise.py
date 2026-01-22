from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simulator.core import CardView, Scheduler

if TYPE_CHECKING:
    import torch


class MemriseScheduler(Scheduler):
    """Memrise sequence scheduler without FSRS dependencies."""

    def __init__(self, sequence: list[float] | None = None) -> None:
        self.sequence = [float(x) for x in (sequence or [1, 6, 12, 48, 96, 180])]

    def init_card(self, card_view: CardView, rating: int, day: float):
        interval = self._next_interval(prev_interval=0.0, rating=rating)
        return interval, {"ivl": interval}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        prev_interval = float(card_view.interval or 0.0)
        interval = self._next_interval(prev_interval=prev_interval, rating=rating)
        return interval, {"ivl": interval}

    def _next_interval(self, *, prev_interval: float, rating: int) -> float:
        if prev_interval == 0.0 or int(rating) == 1:
            return 1.0

        closest_idx = 0
        closest_dist = None
        for idx, interval in enumerate(self.sequence):
            dist = abs(prev_interval - interval)
            if closest_dist is None or dist < closest_dist:
                closest_idx = idx
                closest_dist = dist

        next_idx = min(closest_idx + 1, len(self.sequence) - 1)
        return float(self.sequence[next_idx])


@dataclass
class MemriseVectorizedState:
    interval: float


class MemriseVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: MemriseScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._sequence = torch.tensor(scheduler.sequence, device=device, dtype=dtype)
        self._seq_len = int(self._sequence.numel())

    def init_state(self, deck_size: int) -> MemriseVectorizedState:
        return MemriseVectorizedState(interval=1.0)

    def review_priority(
        self,
        state: MemriseVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: MemriseVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        if self._seq_len == 0:
            intervals = self._torch.ones_like(prev_interval, dtype=self.dtype)
        else:
            dist = self._torch.abs(
                prev_interval.unsqueeze(1) - self._sequence.unsqueeze(0)
            )
            closest = self._torch.argmin(dist, dim=1)
            next_idx = self._torch.clamp(closest + 1, max=self._seq_len - 1)
            intervals = self._sequence[next_idx]
        fail = rating == 1
        is_new = prev_interval == 0.0
        return self._torch.where(
            is_new | fail,
            self._torch.ones_like(intervals, dtype=self.dtype),
            intervals,
        )

    def update_learn(
        self,
        state: MemriseVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._torch.ones(idx.numel(), device=self.device, dtype=self.dtype)


__all__ = ["MemriseScheduler"]

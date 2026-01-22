from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

from simulator.core import CardView, Scheduler

if TYPE_CHECKING:
    import torch


class HLRScheduler(Scheduler):
    """
    Half-life regression scheduler.
    Maintains its own right/wrong counts and schedules to hit desired retention.
    """

    def __init__(
        self, weights: Sequence[float] | None = None, desired_retention: float = 0.9
    ):
        if weights is None:
            raise ValueError("HLRScheduler requires weights from srs-benchmark.")
        self.w = [float(x) for x in weights]
        if len(self.w) != 3:
            raise ValueError("HLRScheduler expects 3 weights.")
        self.desired_retention = desired_retention

    def init_card(self, card_view: CardView, rating: int, day: float):
        right, wrong = (1, 0) if rating > 1 else (0, 1)
        state = {"right": right, "wrong": wrong}
        return self._next_interval(right, wrong), state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {"right": 0, "wrong": 0}
        right = float(state.get("right", 0))
        wrong = float(state.get("wrong", 0))
        if rating > 1:
            right += 1
        else:
            wrong += 1
        state = {"right": right, "wrong": wrong}
        return self._next_interval(right, wrong), state

    def _half_life(self, right: float, wrong: float) -> float:
        w0, w1, b = self.w
        return 2.0 ** (w0 * right + w1 * wrong + b)

    def _next_interval(self, right: float, wrong: float) -> float:
        half = self._half_life(right, wrong)
        ln_half = math.log(0.5)
        return max(1.0, half * math.log(self.desired_retention) / ln_half)


@dataclass
class HLRVectorizedState:
    right: "torch.Tensor"
    wrong: "torch.Tensor"


class HLRVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: HLRScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._w = torch.tensor(scheduler.w, device=device, dtype=dtype)
        self._log_factor = math.log(scheduler.desired_retention) / math.log(0.5)

    def init_state(self, deck_size: int) -> HLRVectorizedState:
        right = self._torch.zeros(deck_size, dtype=self.dtype, device=self.device)
        wrong = self._torch.zeros(deck_size, dtype=self.dtype, device=self.device)
        return HLRVectorizedState(right=right, wrong=wrong)

    def review_priority(
        self, state: HLRVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: HLRVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        success = (rating > 1).to(self.dtype)
        state.right[idx] = state.right[idx] + success
        state.wrong[idx] = state.wrong[idx] + (1.0 - success)
        half = self._torch.pow(
            self._torch.tensor(2.0, device=self.device, dtype=self.dtype),
            self._w[0] * state.right[idx] + self._w[1] * state.wrong[idx] + self._w[2],
        )
        return half * self._log_factor

    def update_learn(
        self,
        state: HLRVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        success = (rating > 1).to(self.dtype)
        state.right[idx] = success
        state.wrong[idx] = 1.0 - success
        half = self._torch.pow(
            self._torch.tensor(2.0, device=self.device, dtype=self.dtype),
            self._w[0] * state.right[idx] + self._w[1] * state.wrong[idx] + self._w[2],
        )
        return half * self._log_factor

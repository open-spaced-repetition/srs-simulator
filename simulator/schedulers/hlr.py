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


@dataclass
class HLRBatchState:
    right: "torch.Tensor"
    wrong: "torch.Tensor"


class HLRBatchSchedulerOps:
    def __init__(
        self,
        *,
        weights: "torch.Tensor",
        desired_retention: "float | torch.Tensor",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        if weights.ndim != 2 or int(weights.shape[1]) != 3:
            raise ValueError("HLRBatchSchedulerOps expects weights shape (users, 3).")
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._w = weights.to(device=device, dtype=dtype)
        desired = torch.as_tensor(desired_retention, device=device, dtype=dtype)
        if desired.ndim == 0:
            desired = desired.expand(weights.shape[0])
        if desired.shape != (weights.shape[0],):
            raise ValueError(
                "desired_retention must be scalar or match the user dimension."
            )
        if torch.any((desired <= 0.0) | (desired >= 1.0)):
            raise ValueError("desired_retention must be between 0 and 1.")
        self._log_factor = torch.log(desired) / math.log(0.5)
        self._two = torch.tensor(2.0, device=device, dtype=dtype)

    def init_state(self, user_count: int, deck_size: int) -> HLRBatchState:
        right = self._torch.zeros(
            (user_count, deck_size), dtype=self.dtype, device=self.device
        )
        wrong = self._torch.zeros_like(right)
        return HLRBatchState(right=right, wrong=wrong)

    def review_priority(
        self, state: HLRBatchState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros_like(elapsed, dtype=self.dtype, device=self.device)

    def update_review(
        self,
        state: HLRBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        success = (rating > 1).to(self.dtype)
        state.right[user_idx, card_idx] += success
        state.wrong[user_idx, card_idx] += 1.0 - success
        return self._next_interval(state, user_idx, card_idx)

    def update_learn(
        self,
        state: HLRBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        success = (rating > 1).to(self.dtype)
        state.right[user_idx, card_idx] = success
        state.wrong[user_idx, card_idx] = 1.0 - success
        return self._next_interval(state, user_idx, card_idx)

    def _next_interval(
        self,
        state: HLRBatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
    ) -> "torch.Tensor":
        w = self._w.index_select(0, user_idx)
        right = state.right[user_idx, card_idx]
        wrong = state.wrong[user_idx, card_idx]
        half = self._torch.pow(self._two, w[:, 0] * right + w[:, 1] * wrong + w[:, 2])
        factor = self._log_factor.index_select(0, user_idx)
        return self._torch.clamp(half * factor, min=1.0)

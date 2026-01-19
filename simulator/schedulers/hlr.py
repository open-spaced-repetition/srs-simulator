from __future__ import annotations

import math
from typing import Sequence

from simulator.core import CardView, Scheduler


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

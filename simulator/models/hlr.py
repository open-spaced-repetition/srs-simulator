from __future__ import annotations

import math
from typing import Sequence

from simulator.core import Card, MemoryModel


class HLRModel(MemoryModel):
    """
    Half-life regression memory model.

    The state is (right_cnt, wrong_cnt); half-life is 2 ** (w0 * right + w1 * wrong + bias).
    Retention follows 0.5 ** (t / half_life).
    """

    def __init__(self, weights: Sequence[float] | None):
        if weights is None:
            raise ValueError("HLRModel requires weights from srs-benchmark.")
        if len(weights) != 3:
            raise ValueError("HLRModel expects exactly 3 weights.")
        self.w = tuple(float(x) for x in weights)

    def init_card(self, card: Card, rating: int) -> None:
        card.memory_state = {
            "right": 1 if rating > 1 else 0,
            "wrong": 1 if rating == 1 else 0,
        }

    def predict_retention(self, card: Card, elapsed: float) -> float:
        right = float(card.memory_state.get("right", 0))
        wrong = float(card.memory_state.get("wrong", 0))
        half_life = self._half_life(right, wrong)
        return 0.5 ** (elapsed / max(half_life, 1e-6))

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        right = float(card.memory_state.get("right", 0))
        wrong = float(card.memory_state.get("wrong", 0))
        if rating > 1:
            right += 1
        else:
            wrong += 1
        card.memory_state = {"right": right, "wrong": wrong}

    def _half_life(self, right: float, wrong: float) -> float:
        w0, w1, b = self.w
        return 2.0 ** (w0 * right + w1 * wrong + b)

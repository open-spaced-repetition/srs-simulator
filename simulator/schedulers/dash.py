from __future__ import annotations

import math
from typing import Sequence

from simulator.core import CardView, Scheduler
from simulator.math.dash import dash_time_window_features


class DASHScheduler(Scheduler):
    """
    Scheduler that mirrors the DASH logistic retention model.

    It predicts retention for candidate intervals using the same compact
    feature vector as `DASHModel` and searches for the interval whose
    predicted retention matches `desired_retention`.
    """

    def __init__(
        self,
        *,
        weights: Sequence[float] | None = None,
        desired_retention: float = 0.85,
        min_interval: float = 1.0,
        max_interval: float = 3650.0,
        search_steps: int = 24,
    ) -> None:
        if weights is None:
            raise ValueError("DASHScheduler requires weights from srs-benchmark.")
        if len(weights) != 9:
            raise ValueError("DASHScheduler expects 9 weights.")
        self.w = tuple(float(w) for w in weights)
        self.desired_retention = float(desired_retention)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.search_steps = search_steps

    def init_card(self, card_view: CardView, rating: int, day: float):
        # Without any history we fall back to the minimum interval.
        return self.min_interval, {"ivl": self.min_interval}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        interval = self._target_interval(card_view)
        return interval, {"ivl": interval}

    def _target_interval(self, view: CardView) -> float:
        # If we have no empirical history we cannot fit the logistic model.
        if not view.history:
            return self.min_interval

        def retention(days: float) -> float:
            return self._predict_retention(view, days)

        target = self.desired_retention
        low = self.min_interval
        high = max(low, view.interval or low)

        if retention(low) <= target:
            return max(low, self.min_interval)

        while high < self.max_interval and retention(high) > target:
            high = min(self.max_interval, high * 2.0)

        for _ in range(self.search_steps):
            mid = 0.5 * (low + high)
            pred = retention(mid)
            if pred > target:
                low = mid
            else:
                high = mid
        return max(self.min_interval, min(high, self.max_interval))

    def _predict_retention(self, view: CardView, candidate_interval: float) -> float:
        feats = dash_time_window_features(view.history, candidate_interval)
        score = sum(math.log1p(f) * w for f, w in zip(feats, self.w[:8])) + self.w[8]
        return 1.0 / (1.0 + math.exp(-score))

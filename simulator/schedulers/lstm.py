from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from simulator.core import CardView, Scheduler
from simulator.models.lstm import LSTMModel


@dataclass
class LSTMSchedulerState:
    curves: dict[str, list[float]]


class LSTMScheduler(Scheduler):
    """
    Scheduler that targets desired retention using the LSTM forgetting curves.

    It rebuilds LSTM features from the review history (including the latest rating)
    and searches for the interval whose predicted retention matches desired_retention.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        user_id: int | None = None,
        benchmark_root: str | Path | None = None,
        desired_retention: float = 0.9,
        use_duration_feature: bool = False,
        default_duration_ms: float = 2500.0,
        interval_scale: float = 1.0,
        min_interval: float = 1.0,
        max_interval: float = 3650.0,
        search_steps: int = 24,
        device: str | None = None,
    ) -> None:
        if desired_retention <= 0.0 or desired_retention >= 1.0:
            raise ValueError("desired_retention must be in (0, 1).")
        self.desired_retention = float(desired_retention)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.search_steps = int(search_steps)
        if self.min_interval <= 0.0:
            raise ValueError("min_interval must be > 0.")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval.")

        self.model = LSTMModel(
            weights_path=weights_path,
            user_id=user_id,
            benchmark_root=benchmark_root,
            use_duration_feature=use_duration_feature,
            default_duration_ms=default_duration_ms,
            interval_scale=interval_scale,
            device=device,
        )

    def init_card(self, card_view: CardView, rating: int, day: float):
        curves = self._curves_with_event(card_view, rating, elapsed=0.0)
        interval = self._target_interval(curves, card_view)
        return interval, LSTMSchedulerState(curves=curves)

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        curves = self._curves_with_event(card_view, rating, elapsed)
        interval = self._target_interval(curves, card_view)
        return interval, LSTMSchedulerState(curves=curves)

    def _curves_with_event(
        self, view: CardView, rating: int, elapsed: float
    ) -> dict[str, list[float]]:
        events = [(log.elapsed, log.rating) for log in view.history]
        events.append((elapsed, int(rating)))
        return self.model.curves_from_events(events)

    def _target_interval(self, curves: dict[str, list[float]], view: CardView) -> float:
        if not curves:
            return self.min_interval

        def retention(days: float) -> float:
            return self.model.predict_retention_from_curves(curves, days)

        target = self.desired_retention
        low = self.min_interval
        high = max(low, float(view.interval or low))

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

    def review_priority(self, card_view: CardView, day: float) -> Sequence[float]:
        state = card_view.scheduler_state
        if isinstance(state, LSTMSchedulerState):
            elapsed = max(0.0, float(day) - card_view.last_review)
            r = self.model.predict_retention_from_curves(state.curves, elapsed)
            return (r, card_view.due, card_view.id)
        return super().review_priority(card_view, day)

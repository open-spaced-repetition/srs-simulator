from __future__ import annotations

from simulator.core import CardView, Scheduler


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


__all__ = ["MemriseScheduler"]

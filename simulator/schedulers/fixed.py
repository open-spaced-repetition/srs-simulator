from __future__ import annotations

from simulator.core import CardView, Scheduler


class FixedIntervalScheduler(Scheduler):
    """Stateless fixed-interval baseline."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = float(interval)

    def init_card(self, card_view: CardView, rating: int, day: float):
        return self.interval, None

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        return self.interval, None

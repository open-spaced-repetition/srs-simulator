from __future__ import annotations

from simulator.core import CardView, Scheduler


class FixedIntervalScheduler(Scheduler):
    """Baseline: multiply interval on success, reset on failure."""

    def __init__(self, start: float = 1.0, multiplier: float = 2.5):
        self.start = start
        self.multiplier = multiplier

    def init_card(self, card_view: CardView, rating: int, day: float):
        return self.start, {"ivl": self.start}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        ivl = float((card_view.scheduler_state or {}).get("ivl", self.start))
        if rating == 1:
            ivl = self.start
        else:
            ivl = ivl * self.multiplier
        return ivl, {"ivl": ivl}

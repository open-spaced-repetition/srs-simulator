from __future__ import annotations

from simulator.core import CardView, Scheduler


class AnkiSM2Scheduler(Scheduler):
    """Anki SM-2 style scheduler without FSRS dependencies."""

    def __init__(
        self,
        *,
        graduating_interval: float = 1.0,
        easy_interval: float = 4.0,
        easy_bonus: float = 1.3,
        hard_interval_factor: float = 1.2,
        ease_start: float = 2.5,
        ease_min: float = 1.3,
        ease_max: float = 5.5,
    ) -> None:
        self.graduating_interval = float(graduating_interval)
        self.easy_interval = float(easy_interval)
        self.easy_bonus = float(easy_bonus)
        self.hard_interval_factor = float(hard_interval_factor)
        self.ease_start = float(ease_start)
        self.ease_min = float(ease_min)
        self.ease_max = float(ease_max)

    def init_card(self, card_view: CardView, rating: int, day: float):
        interval, ease = self._next_interval(
            prev_interval=0.0, rating=rating, ease=self.ease_start
        )
        return interval, {"ease": ease, "ivl": interval}

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {}
        ease = float(state.get("ease", self.ease_start))
        prev_interval = float(card_view.interval or 0.0)
        interval, ease = self._next_interval(
            prev_interval=prev_interval, rating=rating, ease=ease
        )
        return interval, {"ease": ease, "ivl": interval}

    def _next_interval(
        self, *, prev_interval: float, rating: int, ease: float
    ) -> tuple[float, float]:
        rating = int(rating)
        if rating == 1:
            ease -= 0.2
        elif rating == 2:
            ease -= 0.15
        elif rating == 4:
            ease += 0.15
        ease = min(self.ease_max, max(self.ease_min, ease))

        is_new_card = prev_interval == 0.0
        if rating < 4:
            new_card_interval = self.graduating_interval
        else:
            new_card_interval = self.easy_interval

        elapsed = prev_interval
        if rating == 1:
            existing_interval = prev_interval * 0.0
        elif rating == 2:
            existing_interval = max(
                elapsed * self.hard_interval_factor,
                prev_interval * self.hard_interval_factor / 2.0,
            )
        elif rating == 4:
            existing_interval = max(elapsed * ease, prev_interval) * self.easy_bonus
        else:
            existing_interval = max(elapsed * ease, prev_interval)

        interval = new_card_interval if is_new_card else existing_interval
        interval = max(1.0, interval)
        return float(interval), ease


__all__ = ["AnkiSM2Scheduler"]

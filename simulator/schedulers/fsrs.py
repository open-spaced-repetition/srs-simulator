from __future__ import annotations

from typing import Optional, Sequence

from simulator.core import CardView, Scheduler
from simulator.math.fsrs import (
    Bounds,
    FSRS3Params,
    FSRS6Params,
    fsrs3_forgetting_curve,
    fsrs3_init_state,
    fsrs3_mean_reversion,
    fsrs3_next_interval,
    fsrs3_stability_after_failure,
    fsrs3_stability_after_success,
    fsrs6_forgetting_curve,
    fsrs6_init_state,
    fsrs6_next_d,
    fsrs6_next_interval,
    fsrs6_stability_after_failure,
    fsrs6_stability_after_success,
    fsrs6_stability_short_term,
    _clamp_d,
    _clamp_s,
)
from simulator.models.fsrs import FSRS3_INIT, FSRS6_INIT


class FSRS6Scheduler(Scheduler):
    """
    FSRS v6-style scheduler maintaining its own stability/difficulty estimates.
    """

    PRIORITY_MODES = {
        "low_retrievability",
        "high_retrievability",
        "low_difficulty",
        "high_difficulty",
    }

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        desired_retention: float = 0.9,
        bounds: Bounds = Bounds(),
        priority_mode: str = "low_retrievability",
    ):
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        self.params = FSRS6Params(tuple(weights or FSRS6_INIT), bounds)
        self.desired_retention = desired_retention
        self.priority_mode = priority_mode

    def init_card(self, card_view: CardView, rating: int, day: float):
        s, d = fsrs6_init_state(self.params, rating)
        state = {"s": s, "d": d}
        interval = fsrs6_next_interval(self.params, s, self.desired_retention)
        return interval, state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {
            "s": fsrs6_init_state(self.params, 3)[0],
            "d": fsrs6_init_state(self.params, 3)[1],
        }
        s = max(self.params.bounds.s_min, float(state["s"]))
        d = max(
            self.params.bounds.d_min, min(float(state["d"]), self.params.bounds.d_max)
        )
        r = fsrs6_forgetting_curve(self.params, elapsed, s)
        if elapsed < 1.0:
            s = fsrs6_stability_short_term(self.params, s, rating)
        else:
            if rating > 1:
                s = fsrs6_stability_after_success(self.params, s, r, d, rating)
            else:
                s = fsrs6_stability_after_failure(self.params, s, r, d)
        d = fsrs6_next_d(self.params, d, rating)
        state = {
            "s": _clamp_s(self.params.bounds, s),
            "d": _clamp_d(self.params.bounds, d),
        }
        interval = fsrs6_next_interval(self.params, state["s"], self.desired_retention)
        return interval, state

    def review_priority(self, card_view: CardView, day: float) -> Sequence[float]:
        state = card_view.scheduler_state or {}
        s = state.get("s")
        d = state.get("d")
        if (
            self.priority_mode in {"low_retrievability", "high_retrievability"}
            and s is not None
        ):
            elapsed = max(0.0, float(day) - card_view.last_review)
            r = fsrs6_forgetting_curve(self.params, elapsed, float(s))
            if self.priority_mode == "low_retrievability":
                return (r, card_view.due, card_view.id)
            return (-r, card_view.due, card_view.id)
        if self.priority_mode == "low_difficulty" and d is not None:
            return (float(d), card_view.due, card_view.id)
        if self.priority_mode == "high_difficulty" and d is not None:
            return (-float(d), card_view.due, card_view.id)
        return super().review_priority(card_view, day)


class FSRS3Scheduler(Scheduler):
    """
    FSRS v3-style scheduler using 13 parameters.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        desired_retention: float = 0.9,
        bounds: Bounds = Bounds(),
    ):
        self.params = FSRS3Params(tuple(weights or FSRS3_INIT), bounds)
        self.desired_retention = desired_retention

    def init_card(self, card_view: CardView, rating: int, day: float):
        s, d = fsrs3_init_state(self.params, rating)
        state = {"s": s, "d": d}
        interval = fsrs3_next_interval(self.params, s, self.desired_retention)
        return interval, state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {
            "s": fsrs3_init_state(self.params, 3)[0],
            "d": fsrs3_init_state(self.params, 3)[1],
        }
        s = max(self.params.bounds.s_min, float(state["s"]))
        d = max(
            self.params.bounds.d_min, min(float(state["d"]), self.params.bounds.d_max)
        )
        r = fsrs3_forgetting_curve(self.params, elapsed, s)
        if rating > 1:
            s = fsrs3_stability_after_success(self.params, s, d, r)
        else:
            s = fsrs3_stability_after_failure(self.params, s, d, r)
        d = fsrs3_mean_reversion(
            self.params.weights[2], d + self.params.weights[4] * (rating - 3)
        )
        s = _clamp_s(self.params.bounds, s)
        d = _clamp_d(self.params.bounds, d)
        state = {"s": s, "d": d}
        interval = fsrs3_next_interval(self.params, s, self.desired_retention)
        return interval, state

    def review_priority(self, card_view: CardView, day: float) -> Sequence[float]:
        state = card_view.scheduler_state or {}
        s = state.get("s")
        if s is None:
            return super().review_priority(card_view, day)
        elapsed = max(0.0, float(day) - card_view.last_review)
        r = fsrs3_forgetting_curve(self.params, elapsed, float(s))
        return (r, card_view.due, card_view.id)


# Alias for backwards compatibility
FSRSScheduler = FSRS6Scheduler

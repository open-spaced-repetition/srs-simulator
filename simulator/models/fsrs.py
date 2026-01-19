from __future__ import annotations

from typing import Sequence

from simulator.core import Card, MemoryModel
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


class FSRS6Model(MemoryModel):
    """
    FSRS v6 scalar memory model adapted from the torch reference.
    """

    def __init__(self, weights: Sequence[float] | None, bounds: Bounds = Bounds()):
        if weights is None:
            raise ValueError("FSRS6Model requires weights from srs-benchmark.")
        if len(weights) != 21:
            raise ValueError("FSRS6Model expects 21 weights.")
        self.params = FSRS6Params(tuple(float(x) for x in weights), bounds)

    def init_card(self, card: Card, rating: int) -> None:
        s, d = fsrs6_init_state(self.params, rating)
        card.memory_state = {"s": s, "d": d}
        card.metadata["fsrs_state"] = {"s": s, "d": d}

    def predict_retention(self, card: Card, elapsed: float) -> float:
        s = float(card.memory_state.get("s", self.params.bounds.s_min))
        return fsrs6_forgetting_curve(self.params, elapsed, s)

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        s = float(card.memory_state.get("s", self.params.bounds.s_min))
        d = float(card.memory_state.get("d", self.params.bounds.d_min))

        if card.reps == 0:
            s, d = fsrs6_init_state(self.params, rating)
        else:
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
        card.memory_state = state
        card.metadata["fsrs_state"] = state


class FSRS3Model(MemoryModel):
    """
    FSRS v3 scalar memory model (13 parameters) with simple forgetting curve.
    """

    def __init__(self, weights: Sequence[float] | None, bounds: Bounds = Bounds()):
        if weights is None:
            raise ValueError("FSRS3Model requires weights from srs-benchmark.")
        if len(weights) != 13:
            raise ValueError("FSRS3Model expects 13 weights.")
        self.params = FSRS3Params(tuple(float(x) for x in weights), bounds)

    def init_card(self, card: Card, rating: int) -> None:
        s, d = fsrs3_init_state(self.params, rating)
        card.memory_state = {"s": s, "d": d}
        card.metadata["fsrs_state"] = {"s": s, "d": d}

    def predict_retention(self, card: Card, elapsed: float) -> float:
        s = float(card.memory_state.get("s", self.params.bounds.s_min))
        return fsrs3_forgetting_curve(self.params, elapsed, s)

    def update_card(self, card: Card, rating: int, elapsed: float) -> None:
        s = float(card.memory_state.get("s", self.params.bounds.s_min))
        d = float(card.memory_state.get("d", self.params.bounds.d_min))

        if card.reps == 0:
            s, d = fsrs3_init_state(self.params, rating)
        else:
            r = fsrs3_forgetting_curve(self.params, elapsed, s)
            d = fsrs3_mean_reversion(
                self.params.weights[2], d + self.params.weights[4] * (rating - 3)
            )
            if rating > 1:
                s = fsrs3_stability_after_success(self.params, s, d, r)
            else:
                s = fsrs3_stability_after_failure(self.params, s, d, r)

        state = {
            "s": _clamp_s(self.params.bounds, s),
            "d": _clamp_d(self.params.bounds, d),
        }
        card.memory_state = state
        card.metadata["fsrs_state"] = state

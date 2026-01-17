from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

from simulator.core import CardView, Scheduler
from simulator.math.fsrs import (
    Bounds,
    FSRS6Params,
    _clamp_d,
    _clamp_s,
    fsrs6_forgetting_curve,
    fsrs6_init_state,
    fsrs6_next_d,
    fsrs6_next_interval,
    fsrs6_stability_after_failure,
    fsrs6_stability_after_success,
    fsrs6_stability_short_term,
)
from simulator.sspmmc_policy import SSPMMCPPolicy


class SSPMMCScheduler(Scheduler):
    """
    Scheduler that consumes precomputed SSP-MMC-FSRS policies.
    It maintains its own FSRS6 stability/difficulty estimates so it can be used
    with any environment model.
    """

    def __init__(
        self,
        *,
        policy_json: str | Path,
        fsrs_weights: Optional[Sequence[float]] = None,
        retire_interval: float = 1e9,
    ) -> None:
        self.policy = SSPMMCPPolicy.from_json(policy_json)
        weights = fsrs_weights or self.policy.metadata.get("w")
        if weights is None:
            raise ValueError(
                "FSRS weights must be provided either via policy metadata or fsrs_weights."
            )
        if len(weights) != 21:
            raise ValueError("FSRS6 weights must contain 21 parameters.")
        self.params = FSRS6Params(tuple(float(w) for w in weights), bounds=Bounds())
        self.retire_interval = float(retire_interval)

    def init_card(self, card_view: CardView, rating: int, day: float):
        s, d = fsrs6_init_state(self.params, rating)
        state = {"s": s, "d": d}
        interval = self._interval_for_state(state)
        return interval, state

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        state = card_view.scheduler_state or {}
        if not isinstance(state, dict) or "s" not in state or "d" not in state:
            s, d = fsrs6_init_state(self.params, 3)
        else:
            s = float(state["s"])
            d = float(state["d"])

        s = max(self.params.bounds.s_min, s)
        d = max(self.params.bounds.d_min, min(d, self.params.bounds.d_max))
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
        interval = self._interval_for_state(state)
        return interval, state

    def _interval_for_state(self, state: dict[str, Any]) -> float:
        stability = float(state["s"])
        difficulty = float(state["d"])
        desired_retention, graduated = self.policy.lookup(stability, difficulty)
        if graduated:
            return self.retire_interval
        interval = fsrs6_next_interval(self.params, stability, desired_retention)
        return float(max(1.0, math.floor(interval)))


__all__ = ["SSPMMCScheduler"]

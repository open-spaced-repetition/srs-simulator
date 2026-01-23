from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

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

if TYPE_CHECKING:
    import torch


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
        weights: Sequence[float] | None = None,
        desired_retention: float = 0.9,
        bounds: Bounds = Bounds(),
        priority_mode: str = "low_retrievability",
    ):
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        if weights is None:
            raise ValueError("FSRS6Scheduler requires weights from srs-benchmark.")
        if len(weights) != 21:
            raise ValueError("FSRS6Scheduler expects 21 weights.")
        self.params = FSRS6Params(tuple(float(w) for w in weights), bounds)
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
        state: dict[str, float] = card_view.scheduler_state or {}
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
        weights: Sequence[float] | None = None,
        desired_retention: float = 0.9,
        bounds: Bounds = Bounds(),
    ):
        if weights is None:
            raise ValueError("FSRS3Scheduler requires weights from srs-benchmark.")
        if len(weights) != 13:
            raise ValueError("FSRS3Scheduler expects 13 weights.")
        self.params = FSRS3Params(tuple(float(w) for w in weights), bounds)
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
        state: dict[str, float] = card_view.scheduler_state or {}
        s = state.get("s")
        if s is None:
            return super().review_priority(card_view, day)
        elapsed = max(0.0, float(day) - card_view.last_review)
        r = fsrs3_forgetting_curve(self.params, elapsed, float(s))
        return (r, card_view.due, card_view.id)


# Alias for backwards compatibility
FSRSScheduler = FSRS6Scheduler


@dataclass
class FSRSVectorizedState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class FSRS6VectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: FSRS6Scheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch
        from simulator.vectorized import math as vmath

        self._torch = torch
        self._vmath = vmath
        self.device = device
        self.dtype = dtype
        self._weights = torch.tensor(
            scheduler.params.weights, device=device, dtype=dtype
        )
        self._bounds = scheduler.params.bounds
        self._decay = -self._weights[20]
        self._factor = (
            torch.pow(torch.tensor(0.9, device=device, dtype=dtype), 1.0 / self._decay)
            - 1.0
        )
        self._retention_factor = (
            torch.pow(
                torch.tensor(scheduler.desired_retention, device=device, dtype=dtype),
                1.0 / self._decay,
            )
            - 1.0
        )
        self._mean_reversion_d = vmath.clamp(
            self._weights[4] - torch.exp(self._weights[5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        self._priority_mode = scheduler.priority_mode

    def init_state(self, deck_size: int) -> FSRSVectorizedState:
        s = self._torch.full(
            (deck_size,), self._bounds.s_min, dtype=self.dtype, device=self.device
        )
        d = self._torch.full(
            (deck_size,), self._bounds.d_min, dtype=self.dtype, device=self.device
        )
        return FSRSVectorizedState(s=s, d=d)

    def review_priority(
        self, state: FSRSVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        r_sched = self._vmath.forgetting_curve(
            self._decay,
            self._factor,
            elapsed,
            state.s[idx],
            self._bounds.s_min,
        )
        if self._priority_mode == "low_retrievability":
            return r_sched
        if self._priority_mode == "high_retrievability":
            return -r_sched
        if self._priority_mode == "low_difficulty":
            return state.d[idx]
        return -state.d[idx]

    def update_review(
        self,
        state: FSRSVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        sched_s = state.s[idx]
        sched_d = state.d[idx]
        sched_r = self._vmath.forgetting_curve(
            self._decay,
            self._factor,
            elapsed,
            sched_s,
            self._bounds.s_min,
        )
        sched_short = elapsed < 1.0
        sched_success = rating > 1

        sched_new_s = sched_s
        sched_new_s = self._torch.where(
            sched_short,
            self._vmath.stability_short_term(self._weights, sched_s, rating),
            sched_new_s,
        )
        sched_new_s = self._torch.where(
            ~sched_short & sched_success,
            self._vmath.stability_after_success(
                self._weights, sched_s, sched_r, sched_d, rating
            ),
            sched_new_s,
        )
        sched_new_s = self._torch.where(
            ~sched_short & ~sched_success,
            self._vmath.stability_after_failure(
                self._weights, sched_s, sched_r, sched_d
            ),
            sched_new_s,
        )
        sched_new_d = self._vmath.next_d(
            self._weights,
            sched_d,
            rating,
            self._mean_reversion_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[idx] = self._vmath.clamp(
            sched_new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[idx] = self._vmath.clamp(
            sched_new_d, self._bounds.d_min, self._bounds.d_max
        )
        return self._torch.clamp(
            state.s[idx] / self._factor * self._retention_factor, min=1.0
        )

    def update_learn(
        self,
        state: FSRSVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        s_init, d_init = self._vmath.init_state(
            self._weights, rating, self._bounds.d_min, self._bounds.d_max
        )
        state.s[idx] = self._vmath.clamp(s_init, self._bounds.s_min, self._bounds.s_max)
        state.d[idx] = self._vmath.clamp(d_init, self._bounds.d_min, self._bounds.d_max)
        return self._torch.clamp(
            state.s[idx] / self._factor * self._retention_factor, min=1.0
        )


class FSRS3VectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: FSRS3Scheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch
        from simulator.vectorized import math as vmath

        self._torch = torch
        self._vmath = vmath
        self.device = device
        self.dtype = dtype
        self._weights = torch.tensor(
            scheduler.params.weights, device=device, dtype=dtype
        )
        self._bounds = scheduler.params.bounds
        self._fsrs3_ln = math.log(0.9)
        self._log_desired = math.log(scheduler.desired_retention)

    def init_state(self, deck_size: int) -> FSRSVectorizedState:
        s = self._torch.full(
            (deck_size,), self._bounds.s_min, dtype=self.dtype, device=self.device
        )
        d = self._torch.full(
            (deck_size,), self._bounds.d_min, dtype=self.dtype, device=self.device
        )
        return FSRSVectorizedState(s=s, d=d)

    def review_priority(
        self, state: FSRSVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        return self._vmath.fsrs3_forgetting_curve(
            elapsed, state.s[idx], self._bounds.s_min
        )

    def update_review(
        self,
        state: FSRSVectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        sched_s = state.s[idx]
        sched_d = state.d[idx]
        sched_r = self._vmath.fsrs3_forgetting_curve(
            elapsed, sched_s, self._bounds.s_min
        )
        sched_success = rating > 1
        sched_new_s = self._torch.where(
            sched_success,
            self._vmath.fsrs3_stability_after_success(
                self._weights, sched_s, sched_d, sched_r
            ),
            self._vmath.fsrs3_stability_after_failure(
                self._weights, sched_s, sched_d, sched_r
            ),
        )
        d_update = sched_d + self._weights[4] * (rating.to(self.dtype) - 3.0)
        sched_new_d = 0.5 * self._weights[2] + 0.5 * d_update
        state.s[idx] = self._vmath.clamp(
            sched_new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[idx] = self._vmath.clamp(
            sched_new_d, self._bounds.d_min, self._bounds.d_max
        )
        return state.s[idx] * (self._log_desired / self._fsrs3_ln)

    def update_learn(
        self,
        state: FSRSVectorizedState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        s_init, d_init = self._vmath.fsrs3_init_state(
            self._weights,
            rating,
            self._bounds.s_min,
            self._bounds.s_max,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[idx] = self._vmath.clamp(s_init, self._bounds.s_min, self._bounds.s_max)
        state.d[idx] = self._vmath.clamp(d_init, self._bounds.d_min, self._bounds.d_max)
        return state.s[idx] * (self._log_desired / self._fsrs3_ln)

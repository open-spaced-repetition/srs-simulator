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
from simulator.math.fsrs_batch import (
    fsrs6_forgetting_curve as fsrs6_forgetting_curve_batch,
    fsrs6_init_state as fsrs6_init_state_batch,
    fsrs6_next_d as fsrs6_next_d_batch,
    fsrs6_stability_after_failure as fsrs6_stability_after_failure_batch,
    fsrs6_stability_after_success as fsrs6_stability_after_success_batch,
    fsrs6_stability_short_term as fsrs6_stability_short_term_batch,
)
from simulator.fsrs_defaults import resolve_fsrs6_weights

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
        weights = resolve_fsrs6_weights(weights)
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


@dataclass
class FSRS6BatchState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class FSRS6BatchSchedulerOps:
    PRIORITY_MODES = {
        "low_retrievability",
        "high_retrievability",
        "low_difficulty",
        "high_difficulty",
    }

    def __init__(
        self,
        *,
        weights: "torch.Tensor",
        desired_retention: float,
        bounds: Bounds,
        priority_mode: str,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._weights = weights.to(device=device, dtype=dtype)
        self._bounds = bounds
        self._decay = -self._weights[:, 20]
        base = torch.tensor(0.9, device=device, dtype=dtype)
        self._factor = torch.pow(base, 1.0 / self._decay) - 1.0
        self._retention_factor = (
            torch.pow(
                torch.tensor(desired_retention, device=device, dtype=dtype),
                1.0 / self._decay,
            )
            - 1.0
        )
        self._init_d = torch.clamp(
            self._weights[:, 4] - torch.exp(self._weights[:, 5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        self._priority_mode = priority_mode

    def init_state(self, user_count: int, deck_size: int) -> FSRS6BatchState:
        s = self._torch.full(
            (user_count, deck_size),
            self._bounds.s_min,
            dtype=self.dtype,
            device=self.device,
        )
        d = self._torch.full(
            (user_count, deck_size),
            self._bounds.d_min,
            dtype=self.dtype,
            device=self.device,
        )
        return FSRS6BatchState(s=s, d=d)

    def review_priority(
        self, state: FSRS6BatchState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        r_sched = fsrs6_forgetting_curve_batch(
            self._decay[:, None],
            self._factor[:, None],
            elapsed,
            state.s,
            self._bounds.s_min,
        )
        if self._priority_mode == "low_retrievability":
            return r_sched
        if self._priority_mode == "high_retrievability":
            return -r_sched
        if self._priority_mode == "low_difficulty":
            return state.d
        if self._priority_mode == "high_difficulty":
            return -state.d
        raise ValueError(f"Unknown priority_mode '{self._priority_mode}'")

    def update_review(
        self,
        state: FSRS6BatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        weights = self._weights.index_select(0, user_idx)
        decay = self._decay.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention_factor = self._retention_factor.index_select(0, user_idx)
        init_d = self._init_d.index_select(0, user_idx)
        sched_s = state.s[user_idx, card_idx]
        sched_d = state.d[user_idx, card_idx]
        sched_r = fsrs6_forgetting_curve_batch(
            decay,
            factor,
            elapsed,
            sched_s,
            self._bounds.s_min,
        )
        short_term = elapsed < 1.0
        success = rating > 1
        new_s = sched_s
        new_s = self._torch.where(
            short_term,
            fsrs6_stability_short_term_batch(weights, sched_s, rating),
            new_s,
        )
        new_s = self._torch.where(
            ~short_term & success,
            fsrs6_stability_after_success_batch(
                weights, sched_s, sched_r, sched_d, rating
            ),
            new_s,
        )
        new_s = self._torch.where(
            ~short_term & ~success,
            fsrs6_stability_after_failure_batch(weights, sched_s, sched_r, sched_d),
            new_s,
        )
        new_d = fsrs6_next_d_batch(
            weights,
            sched_d,
            rating,
            init_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[user_idx, card_idx] = self._torch.clamp(
            new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = self._torch.clamp(
            new_d, self._bounds.d_min, self._bounds.d_max
        )
        return self._torch.clamp(
            state.s[user_idx, card_idx] / factor * retention_factor, min=1.0
        )

    def update_learn(
        self,
        state: FSRS6BatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        weights = self._weights.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention_factor = self._retention_factor.index_select(0, user_idx)
        s_init, d_init = fsrs6_init_state_batch(
            weights, rating, self._bounds.d_min, self._bounds.d_max
        )
        state.s[user_idx, card_idx] = self._torch.clamp(
            s_init, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = self._torch.clamp(
            d_init, self._bounds.d_min, self._bounds.d_max
        )
        return self._torch.clamp(
            state.s[user_idx, card_idx] / factor * retention_factor, min=1.0
        )


@dataclass
class FSRS3BatchState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class FSRS3BatchSchedulerOps:
    def __init__(
        self,
        *,
        weights: "torch.Tensor",
        desired_retention: float,
        bounds: Bounds,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        if not (0.0 < float(desired_retention) < 1.0):
            raise ValueError("desired_retention must be between 0 and 1.")
        if weights.ndim != 2 or int(weights.shape[1]) != 13:
            raise ValueError(
                "FSRS3BatchSchedulerOps expects weights shape (users, 13)."
            )
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._weights = weights.to(device=device, dtype=dtype)
        self._bounds = bounds
        self._base = torch.tensor(0.9, device=device, dtype=dtype)
        self._interval_factor = math.log(float(desired_retention)) / math.log(0.9)

    def init_state(self, user_count: int, deck_size: int) -> FSRS3BatchState:
        s = self._torch.full(
            (user_count, deck_size),
            self._bounds.s_min,
            dtype=self.dtype,
            device=self.device,
        )
        d = self._torch.full(
            (user_count, deck_size),
            self._bounds.d_min,
            dtype=self.dtype,
            device=self.device,
        )
        return FSRS3BatchState(s=s, d=d)

    def review_priority(
        self, state: FSRS3BatchState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        # FSRSv3 uses retrievability (lower = more urgent); we return retrievability so
        # the engine can apply its own min/max priority conventions.
        return self._torch.pow(
            self._base, elapsed / self._torch.clamp(state.s, min=self._bounds.s_min)
        )

    def update_review(
        self,
        state: FSRS3BatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)

        w = self._weights.index_select(0, user_idx)
        rating_f = rating.to(dtype=self.dtype)

        sched_s = state.s[user_idx, card_idx]
        sched_d = state.d[user_idx, card_idx]
        sched_r = self._torch.pow(
            self._base, elapsed / self._torch.clamp(sched_s, min=self._bounds.s_min)
        )

        success = rating > 1

        inc = (
            self._torch.exp(w[:, 6])
            * (11.0 - sched_d)
            * self._torch.pow(sched_s, w[:, 7])
            * (self._torch.exp((1.0 - sched_r) * w[:, 8]) - 1.0)
        )
        s_success = sched_s * (1.0 + inc)
        s_failure = (
            w[:, 9]
            * self._torch.pow(sched_d, w[:, 10])
            * self._torch.pow(sched_s, w[:, 11])
            * self._torch.exp((1.0 - sched_r) * w[:, 12])
        )
        new_s = self._torch.where(success, s_success, s_failure)

        d_update = sched_d + w[:, 4] * (rating_f - 3.0)
        new_d = 0.5 * w[:, 2] + 0.5 * d_update

        state.s[user_idx, card_idx] = self._torch.clamp(
            new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = self._torch.clamp(
            new_d, self._bounds.d_min, self._bounds.d_max
        )
        return self._torch.clamp(
            state.s[user_idx, card_idx] * self._interval_factor, min=1.0
        )

    def update_learn(
        self,
        state: FSRS3BatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)

        w = self._weights.index_select(0, user_idx)
        rating_f = rating.to(dtype=self.dtype)

        s_init = w[:, 0] + w[:, 1] * (rating_f - 1.0)
        d_init = w[:, 2] + w[:, 3] * (rating_f - 3.0)

        state.s[user_idx, card_idx] = self._torch.clamp(
            s_init, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = self._torch.clamp(
            d_init, self._bounds.d_min, self._bounds.d_max
        )
        return self._torch.clamp(
            state.s[user_idx, card_idx] * self._interval_factor, min=1.0
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

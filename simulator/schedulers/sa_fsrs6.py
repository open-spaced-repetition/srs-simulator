from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional, Sequence

from simulator.core import CardView, Scheduler
from simulator.fsrs_defaults import resolve_fsrs6_weights
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
from simulator.math.fsrs_batch import (
    fsrs6_forgetting_curve as fsrs6_forgetting_curve_batch,
    fsrs6_init_state as fsrs6_init_state_batch,
    fsrs6_next_d as fsrs6_next_d_batch,
    fsrs6_stability_after_failure as fsrs6_stability_after_failure_batch,
    fsrs6_stability_after_success as fsrs6_stability_after_success_batch,
    fsrs6_stability_short_term as fsrs6_stability_short_term_batch,
)
from simulator.sa_fsrs6_policy import SAFSRS6Policy

if TYPE_CHECKING:
    import torch


class SAFSRS6Scheduler(Scheduler):
    """
    Simulated-annealing policy scheduler over scheduler-side FSRS-6 S/D state.
    """

    PRIORITY_MODES = {
        "low_retrievability",
        "high_retrievability",
        "low_difficulty",
        "high_difficulty",
    }

    def __init__(
        self,
        *,
        policy_json: str | Path,
        fsrs_weights: Optional[Sequence[float]] = None,
        priority_mode: str = "low_retrievability",
    ) -> None:
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        self.policy = SAFSRS6Policy.from_json(policy_json)
        weights = resolve_fsrs6_weights(fsrs_weights)
        if len(weights) != 21:
            raise ValueError("SAFSRS6Scheduler expects 21 FSRS-6 weights.")
        self.params = FSRS6Params(tuple(float(w) for w in weights), bounds=Bounds())
        self.priority_mode = priority_mode

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
        elif rating > 1:
            s = fsrs6_stability_after_success(self.params, s, r, d, rating)
        else:
            s = fsrs6_stability_after_failure(self.params, s, r, d)
        d = fsrs6_next_d(self.params, d, rating)
        state = {
            "s": _clamp_s(self.params.bounds, s),
            "d": _clamp_d(self.params.bounds, d),
        }
        return self._interval_for_state(state), state

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

    def _interval_for_state(self, state: dict[str, Any]) -> float:
        stability = float(state["s"])
        difficulty = float(state["d"])
        retention = self.policy.evaluate(stability, difficulty)
        return fsrs6_next_interval(self.params, stability, retention)


@dataclass
class SAFSRS6VectorizedState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class SAFSRS6VectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: SAFSRS6Scheduler,
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
        self._policy = scheduler.policy
        self._coefficients = torch.tensor(
            self._policy.coefficients, device=device, dtype=dtype
        )
        self._retention_min = float(self._policy.retention_min)
        self._retention_max = float(self._policy.retention_max)
        self._log_s_min = float(torch.log(torch.tensor(self._bounds.s_min)).item())
        self._log_s_span = float(
            torch.log(torch.tensor(self._bounds.s_max / self._bounds.s_min)).item()
        )
        self._d_span = self._bounds.d_max - self._bounds.d_min
        self._decay = -self._weights[20]
        self._factor = (
            torch.pow(torch.tensor(0.9, device=device, dtype=dtype), 1.0 / self._decay)
            - 1.0
        )
        self._mean_reversion_d = vmath.clamp(
            self._weights[4] - torch.exp(self._weights[5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        self._priority_mode = scheduler.priority_mode

    def init_state(self, deck_size: int) -> SAFSRS6VectorizedState:
        s = self._torch.full(
            (deck_size,), self._bounds.s_min, dtype=self.dtype, device=self.device
        )
        d = self._torch.full(
            (deck_size,), self._bounds.d_min, dtype=self.dtype, device=self.device
        )
        return SAFSRS6VectorizedState(s=s, d=d)

    def review_priority(
        self,
        state: SAFSRS6VectorizedState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
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
        state: SAFSRS6VectorizedState,
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
        short = elapsed < 1.0
        success = rating > 1
        new_s = sched_s
        new_s = self._torch.where(
            short,
            self._vmath.stability_short_term(self._weights, sched_s, rating),
            new_s,
        )
        new_s = self._torch.where(
            ~short & success,
            self._vmath.stability_after_success(
                self._weights, sched_s, sched_r, sched_d, rating
            ),
            new_s,
        )
        new_s = self._torch.where(
            ~short & ~success,
            self._vmath.stability_after_failure(
                self._weights, sched_s, sched_r, sched_d
            ),
            new_s,
        )
        new_d = self._vmath.next_d(
            self._weights,
            sched_d,
            rating,
            self._mean_reversion_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[idx] = self._vmath.clamp(new_s, self._bounds.s_min, self._bounds.s_max)
        state.d[idx] = self._vmath.clamp(new_d, self._bounds.d_min, self._bounds.d_max)
        return self._interval_for_state(state.s[idx], state.d[idx])

    def update_learn(
        self,
        state: SAFSRS6VectorizedState,
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
        return self._interval_for_state(state.s[idx], state.d[idx])

    def _retention_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor"
    ) -> "torch.Tensor":
        s_norm = (
            self._torch.log(
                self._torch.clamp(s, self._bounds.s_min, self._bounds.s_max)
            )
            - self._log_s_min
        ) / self._log_s_span
        d_norm = (
            self._torch.clamp(d, self._bounds.d_min, self._bounds.d_max)
            - self._bounds.d_min
        ) / self._d_span
        s_norm = self._torch.clamp(s_norm, 0.0, 1.0)
        d_norm = self._torch.clamp(d_norm, 0.0, 1.0)
        logit = (
            self._coefficients[0]
            + self._coefficients[1] * s_norm
            + self._coefficients[2] * d_norm
            + self._coefficients[3] * s_norm * d_norm
            + self._coefficients[4] * s_norm * s_norm
            + self._coefficients[5] * d_norm * d_norm
        )
        return self._retention_min + (
            self._retention_max - self._retention_min
        ) * self._torch.sigmoid(logit)

    def _interval_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor"
    ) -> "torch.Tensor":
        retention = self._retention_for_state(s, d)
        interval = (
            s / self._factor * (self._torch.pow(retention, 1.0 / self._decay) - 1.0)
        )
        return self._torch.clamp(interval, min=1.0)


@dataclass
class SAFSRS6BatchState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class SAFSRS6BatchSchedulerOps:
    PRIORITY_MODES = SAFSRS6Scheduler.PRIORITY_MODES

    def __init__(
        self,
        *,
        weights: "torch.Tensor",
        policy: SAFSRS6Policy,
        bounds: Bounds,
        priority_mode: str,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        import torch

        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        if weights.ndim != 2 or weights.shape[1] != 21:
            raise ValueError(
                "SAFSRS6BatchSchedulerOps expects weights shape (users, 21)."
            )
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._weights = weights.to(device=device, dtype=dtype)
        self._bounds = bounds
        self._policy = policy
        self._coefficients = torch.tensor(
            policy.coefficients, device=device, dtype=dtype
        )
        self._retention_min = float(policy.retention_min)
        self._retention_max = float(policy.retention_max)
        self._log_s_min = float(torch.log(torch.tensor(bounds.s_min)).item())
        self._log_s_span = float(
            torch.log(torch.tensor(bounds.s_max / bounds.s_min)).item()
        )
        self._d_span = bounds.d_max - bounds.d_min
        self._decay = -self._weights[:, 20]
        base = torch.tensor(0.9, device=device, dtype=dtype)
        self._factor = torch.pow(base, 1.0 / self._decay) - 1.0
        self._init_d = torch.clamp(
            self._weights[:, 4] - torch.exp(self._weights[:, 5] * 3.0) + 1.0,
            bounds.d_min,
            bounds.d_max,
        )
        self._priority_mode = priority_mode

    def init_state(self, user_count: int, deck_size: int) -> SAFSRS6BatchState:
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
        return SAFSRS6BatchState(s=s, d=d)

    def review_priority(
        self, state: SAFSRS6BatchState, elapsed: "torch.Tensor"
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
        state: SAFSRS6BatchState,
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
        short = elapsed < 1.0
        success = rating > 1
        new_s = sched_s
        new_s = self._torch.where(
            short, fsrs6_stability_short_term_batch(weights, sched_s, rating), new_s
        )
        new_s = self._torch.where(
            ~short & success,
            fsrs6_stability_after_success_batch(
                weights, sched_s, sched_r, sched_d, rating
            ),
            new_s,
        )
        new_s = self._torch.where(
            ~short & ~success,
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
        return self._interval_for_state(
            state.s[user_idx, card_idx],
            state.d[user_idx, card_idx],
            user_idx,
        )

    def update_learn(
        self,
        state: SAFSRS6BatchState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        weights = self._weights.index_select(0, user_idx)
        s_init, d_init = fsrs6_init_state_batch(
            weights, rating, self._bounds.d_min, self._bounds.d_max
        )
        state.s[user_idx, card_idx] = self._torch.clamp(
            s_init, self._bounds.s_min, self._bounds.s_max
        )
        state.d[user_idx, card_idx] = self._torch.clamp(
            d_init, self._bounds.d_min, self._bounds.d_max
        )
        return self._interval_for_state(
            state.s[user_idx, card_idx],
            state.d[user_idx, card_idx],
            user_idx,
        )

    def _retention_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor"
    ) -> "torch.Tensor":
        s_norm = (
            self._torch.log(
                self._torch.clamp(s, self._bounds.s_min, self._bounds.s_max)
            )
            - self._log_s_min
        ) / self._log_s_span
        d_norm = (
            self._torch.clamp(d, self._bounds.d_min, self._bounds.d_max)
            - self._bounds.d_min
        ) / self._d_span
        s_norm = self._torch.clamp(s_norm, 0.0, 1.0)
        d_norm = self._torch.clamp(d_norm, 0.0, 1.0)
        logit = (
            self._coefficients[0]
            + self._coefficients[1] * s_norm
            + self._coefficients[2] * d_norm
            + self._coefficients[3] * s_norm * d_norm
            + self._coefficients[4] * s_norm * s_norm
            + self._coefficients[5] * d_norm * d_norm
        )
        return self._retention_min + (
            self._retention_max - self._retention_min
        ) * self._torch.sigmoid(logit)

    def _interval_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor", user_idx: "torch.Tensor"
    ) -> "torch.Tensor":
        decay = self._decay.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention = self._retention_for_state(s, d)
        interval = s / factor * (self._torch.pow(retention, 1.0 / decay) - 1.0)
        return self._torch.clamp(interval, min=1.0)


__all__ = [
    "SAFSRS6BatchSchedulerOps",
    "SAFSRS6Scheduler",
    "SAFSRS6VectorizedSchedulerOps",
]

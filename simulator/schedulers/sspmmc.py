from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import torch


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


@dataclass
class SSPMMCVectorizedState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class SSPMMCVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: SSPMMCScheduler,
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
        self._init_d = vmath.clamp(
            self._weights[4] - torch.exp(self._weights[5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        policy = scheduler.policy
        self._policy_retention = torch.tensor(
            policy.retention_matrix, device=device, dtype=dtype
        )
        self._policy_s_grid = torch.tensor(policy.s_grid, device=device, dtype=dtype)
        self._policy_s_grid_size = int(policy.s_grid.size)
        self._policy_s_state_small_len = int(policy.s_state_small_len)
        self._policy_s_mid = float(policy.s_mid)
        self._policy_s_min = float(policy.state_space["s_min"])
        self._policy_s_max = float(policy.state_space["s_max"])
        self._policy_log_s_min = math.log(self._policy_s_min)
        self._policy_short_step = float(policy.state_space["short_step"])
        self._policy_long_step = float(policy.state_space["long_step"])
        self._policy_d_min = float(policy.state_space["d_min"])
        self._policy_d_max = float(policy.state_space["d_max"])
        self._policy_d_size = int(
            min(
                math.ceil(
                    (self._policy_d_max - self._policy_d_min)
                    / float(policy.state_space["d_eps"])
                    + 1
                ),
                int(policy.d_grid.size),
            )
        )
        self._policy_retire_interval = float(scheduler.retire_interval)
        if self._policy_s_state_small_len > 0:
            self._policy_s_last = self._policy_s_grid[
                self._policy_s_state_small_len - 1
            ]
        else:
            self._policy_s_last = torch.tensor(
                self._policy_s_min, device=device, dtype=dtype
            )

    def init_state(self, deck_size: int) -> SSPMMCVectorizedState:
        s = self._torch.full(
            (deck_size,), self._bounds.s_min, dtype=self.dtype, device=self.device
        )
        d = self._torch.full(
            (deck_size,), self._bounds.d_min, dtype=self.dtype, device=self.device
        )
        return SSPMMCVectorizedState(s=s, d=d)

    def review_priority(
        self, state: SSPMMCVectorizedState, idx: "torch.Tensor", elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        return self._torch.zeros(idx.numel(), device=self.device, dtype=self.dtype)

    def update_review(
        self,
        state: SSPMMCVectorizedState,
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
            self._init_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.s[idx] = self._vmath.clamp(
            sched_new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.d[idx] = self._vmath.clamp(
            sched_new_d, self._bounds.d_min, self._bounds.d_max
        )
        return self._interval_for_state(state.s[idx], state.d[idx])

    def update_learn(
        self,
        state: SSPMMCVectorizedState,
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

    def _interval_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor"
    ) -> "torch.Tensor":
        s_idx = self._vmath.sspmmc_s2i(
            s,
            self._policy_s_min,
            self._policy_s_mid,
            self._policy_s_state_small_len,
            self._policy_log_s_min,
            self._policy_short_step,
            self._policy_long_step,
            self._policy_s_last,
            self._policy_s_grid_size,
        )
        d_idx = self._vmath.sspmmc_d2i(
            d, self._policy_d_min, self._policy_d_max, self._policy_d_size
        )
        retention = self._policy_retention[d_idx, s_idx]
        graduated = (s_idx >= self._policy_s_grid_size - 1) | (s >= self._policy_s_max)
        retire_val = self._torch.tensor(
            self._policy_retire_interval, device=self.device, dtype=self.dtype
        )
        interval = (
            s / self._factor * (self._torch.pow(retention, 1.0 / self._decay) - 1.0)
        )
        return self._torch.where(graduated, retire_val, interval)


__all__ = ["SSPMMCScheduler"]

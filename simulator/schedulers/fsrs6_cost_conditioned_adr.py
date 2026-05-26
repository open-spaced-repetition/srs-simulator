from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from simulator.core import CardView, Scheduler
from simulator.fsrs6_cost_conditioned_adr_policy import (
    ACTION_HEAD_INTERVAL,
    ACTION_HEAD_RETENTION,
    FSRS6CostConditionedADRPolicy,
)
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

if TYPE_CHECKING:
    import torch


class FSRS6CostConditionedADRScheduler(Scheduler):
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
        goal_cost_weight: float,
        fsrs_weights: Optional[Sequence[float]] = None,
        priority_mode: str = "low_retrievability",
    ) -> None:
        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        if goal_cost_weight < 0.0:
            raise ValueError("goal_cost_weight must be >= 0.")
        self.policy = FSRS6CostConditionedADRPolicy.from_json(policy_json)
        self.goal_cost_weight = float(goal_cost_weight)
        weights = resolve_fsrs6_weights(fsrs_weights)
        if len(weights) != 21:
            raise ValueError("FSRS6CostConditionedADRScheduler expects 21 weights.")
        self.params = FSRS6Params(tuple(float(w) for w in weights), bounds=Bounds())
        self.priority_mode = priority_mode

    def init_card(self, card_view: CardView, rating: int, day: float):
        s, d = fsrs6_init_state(self.params, rating)
        state = {"s": s, "d": d}
        return self._interval_for_state(state), state

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

    def _interval_for_state(self, state: dict[str, float]) -> float:
        s = float(state["s"])
        d = float(state["d"])
        if self.policy.action_head == ACTION_HEAD_INTERVAL:
            return self.policy.evaluate_interval(
                s,
                d,
                cost_weight=self.goal_cost_weight,
            )
        retention = self.policy.evaluate_retention(
            s,
            d,
            cost_weight=self.goal_cost_weight,
        )
        return fsrs6_next_interval(self.params, s, retention)


@dataclass
class FSRS6CostConditionedADRBatchState:
    s: "torch.Tensor"
    d: "torch.Tensor"


class FSRS6CostConditionedADRBatchSchedulerOps:
    PRIORITY_MODES = FSRS6CostConditionedADRScheduler.PRIORITY_MODES

    def __init__(
        self,
        *,
        weights: "torch.Tensor",
        policy: FSRS6CostConditionedADRPolicy,
        goal_cost_weight: "torch.Tensor",
        bounds: Bounds,
        priority_mode: str,
        device: "torch.device",
        dtype: "torch.dtype",
        coefficients: "torch.Tensor | None" = None,
    ) -> None:
        import torch

        if priority_mode not in self.PRIORITY_MODES:
            raise ValueError(f"Unknown priority_mode '{priority_mode}'")
        if weights.ndim != 2 or weights.shape[1] != 21:
            raise ValueError(
                "FSRS6CostConditionedADRBatchSchedulerOps expects weights "
                "shape (lanes, 21)."
            )
        if goal_cost_weight.ndim != 1 or goal_cost_weight.shape[0] != weights.shape[0]:
            raise ValueError("goal_cost_weight must have shape (lanes,).")
        self._torch = torch
        self.device = device
        self.dtype = dtype
        self._weights = weights.to(device=device, dtype=dtype)
        self._bounds = bounds
        self._policy = policy
        self._feature_count = policy.state_feature_count
        self._action_head = policy.action_head
        if coefficients is None:
            self._coefficients = torch.tensor(
                [policy.coefficients for _ in range(weights.shape[0])],
                device=device,
                dtype=dtype,
            )
        else:
            expected_shape = (weights.shape[0], policy.parameter_count)
            if coefficients.ndim != 2 or tuple(coefficients.shape) != expected_shape:
                raise ValueError(
                    "Cost-conditioned ADR coefficients must have shape "
                    f"{expected_shape}."
                )
            self._coefficients = coefficients.to(device=device, dtype=dtype)
        self._goal_cost_weight = goal_cost_weight.to(device=device, dtype=dtype)
        self._cost_weight_min = float(policy.cost_weight_min)
        self._cost_weight_max = float(policy.cost_weight_max)
        self._retention_min = float(policy.retention_min)
        self._retention_max = float(policy.retention_max)
        self._max_interval_days = policy.max_interval_days
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

    def init_state(
        self, user_count: int, deck_size: int
    ) -> FSRS6CostConditionedADRBatchState:
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
        return FSRS6CostConditionedADRBatchState(s=s, d=d)

    def review_priority(
        self, state: FSRS6CostConditionedADRBatchState, elapsed: "torch.Tensor"
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
        state: FSRS6CostConditionedADRBatchState,
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
            short,
            fsrs6_stability_short_term_batch(weights, sched_s, rating),
            new_s,
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
        state: FSRS6CostConditionedADRBatchState,
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

    def _state_features(
        self, s: "torch.Tensor", d: "torch.Tensor"
    ) -> list["torch.Tensor"]:
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
        features = [
            self._torch.ones_like(s_norm),
            s_norm,
            d_norm,
            s_norm * d_norm,
            s_norm * s_norm,
            d_norm * d_norm,
        ]
        if self._feature_count == 8:
            features.extend(
                [
                    self._torch.clamp(s_norm - 0.5, min=0.0),
                    self._torch.clamp(d_norm - 0.5, min=0.0),
                ]
            )
        return features

    def _policy_value(
        self, s: "torch.Tensor", d: "torch.Tensor", user_idx: "torch.Tensor"
    ) -> "torch.Tensor":
        features = self._torch.stack(self._state_features(s, d), dim=1)
        coefficients = self._coefficients.index_select(0, user_idx)
        groups = coefficients.reshape(-1, 4, self._feature_count)
        base = self._torch.sum(groups[:, 0, :] * features, dim=1)
        slope_1 = self._torch.nn.functional.softplus(
            self._torch.sum(groups[:, 1, :] * features, dim=1)
        )
        slope_2 = self._torch.nn.functional.softplus(
            self._torch.sum(groups[:, 2, :] * features, dim=1)
        )
        slope_3 = self._torch.nn.functional.softplus(
            self._torch.sum(groups[:, 3, :] * features, dim=1)
        )
        weights = self._goal_cost_weight.index_select(0, user_idx)
        lo = self._torch.log1p(
            self._torch.tensor(
                self._cost_weight_min,
                device=self.device,
                dtype=self.dtype,
            )
        )
        hi = self._torch.log1p(
            self._torch.tensor(
                self._cost_weight_max,
                device=self.device,
                dtype=self.dtype,
            )
        )
        z = (
            self._torch.log1p(
                self._torch.clamp(
                    weights,
                    min=self._cost_weight_min,
                    max=self._cost_weight_max,
                )
            )
            - lo
        ) / (hi - lo)
        z = self._torch.clamp(z, 0.0, 1.0)
        cost_effect = slope_1 * self._torch.sqrt(z) + slope_2 * z + slope_3 * z * z
        if self._action_head == ACTION_HEAD_RETENTION:
            return base - cost_effect
        return base + cost_effect

    def _retention_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor", user_idx: "torch.Tensor"
    ) -> "torch.Tensor":
        logit = self._policy_value(s, d, user_idx)
        return self._retention_min + (
            self._retention_max - self._retention_min
        ) * self._torch.sigmoid(logit)

    def _interval_for_state(
        self, s: "torch.Tensor", d: "torch.Tensor", user_idx: "torch.Tensor"
    ) -> "torch.Tensor":
        if self._action_head == ACTION_HEAD_INTERVAL:
            interval = self._torch.exp(self._policy_value(s, d, user_idx))
            if self._max_interval_days is not None:
                interval = self._torch.clamp(
                    interval, max=float(self._max_interval_days)
                )
            return self._torch.clamp(interval, min=1.0)
        decay = self._decay.index_select(0, user_idx)
        factor = self._factor.index_select(0, user_idx)
        retention = self._retention_for_state(s, d, user_idx)
        interval = s / factor * (self._torch.pow(retention, 1.0 / decay) - 1.0)
        return self._torch.clamp(interval, min=1.0)


__all__ = [
    "FSRS6CostConditionedADRBatchSchedulerOps",
    "FSRS6CostConditionedADRScheduler",
]

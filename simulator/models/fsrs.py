from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

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

if TYPE_CHECKING:
    import torch


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


@dataclass
class FSRS6VectorizedEnvState:
    mem_s: "torch.Tensor"
    mem_d: "torch.Tensor"


class FSRS6VectorizedEnvOps:
    def __init__(
        self,
        environment: FSRS6Model,
        *,
        device: "torch.device",
        dtype: "torch.dtype | None",
    ) -> None:
        import torch
        from simulator.vectorized import math as vmath

        self._torch = torch
        self._vmath = vmath
        self.device = device
        self.dtype = dtype or torch.float64
        self._weights = torch.tensor(
            environment.params.weights, device=device, dtype=self.dtype
        )
        self._bounds = environment.params.bounds
        self._decay = -self._weights[20]
        self._factor = (
            torch.pow(
                torch.tensor(0.9, device=device, dtype=self.dtype), 1.0 / self._decay
            )
            - 1.0
        )
        self._init_d = vmath.clamp(
            self._weights[4] - torch.exp(self._weights[5] * 3.0) + 1.0,
            self._bounds.d_min,
            self._bounds.d_max,
        )

    def init_state(self, deck_size: int) -> FSRS6VectorizedEnvState:
        mem_s = self._torch.full(
            (deck_size,),
            self._bounds.s_min,
            dtype=self.dtype,
            device=self.device,
        )
        mem_d = self._torch.full(
            (deck_size,),
            self._bounds.d_min,
            dtype=self.dtype,
            device=self.device,
        )
        return FSRS6VectorizedEnvState(mem_s=mem_s, mem_d=mem_d)

    def retrievability(
        self,
        state: FSRS6VectorizedEnvState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
    ) -> "torch.Tensor":
        return self._vmath.forgetting_curve(
            self._decay,
            self._factor,
            elapsed,
            state.mem_s[idx],
            self._bounds.s_min,
        )

    def update_review(
        self,
        state: FSRS6VectorizedEnvState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        retrievability: "torch.Tensor",
    ) -> None:
        if idx.numel() == 0:
            return
        exec_s = state.mem_s[idx]
        exec_d = state.mem_d[idx]
        short_term = elapsed < 1.0
        success = rating > 1

        new_s = exec_s
        new_s = self._torch.where(
            short_term,
            self._vmath.stability_short_term(self._weights, exec_s, rating),
            new_s,
        )
        new_s = self._torch.where(
            ~short_term & success,
            self._vmath.stability_after_success(
                self._weights, exec_s, retrievability, exec_d, rating
            ),
            new_s,
        )
        new_s = self._torch.where(
            ~short_term & ~success,
            self._vmath.stability_after_failure(
                self._weights, exec_s, retrievability, exec_d
            ),
            new_s,
        )
        new_d = self._vmath.next_d(
            self._weights,
            exec_d,
            rating,
            self._init_d,
            self._bounds.d_min,
            self._bounds.d_max,
        )
        state.mem_s[idx] = self._vmath.clamp(
            new_s, self._bounds.s_min, self._bounds.s_max
        )
        state.mem_d[idx] = self._vmath.clamp(
            new_d, self._bounds.d_min, self._bounds.d_max
        )

    def update_learn(
        self,
        state: FSRS6VectorizedEnvState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> None:
        if idx.numel() == 0:
            return
        s_init, d_init = self._vmath.init_state(
            self._weights, rating, self._bounds.d_min, self._bounds.d_max
        )
        state.mem_s[idx] = self._vmath.clamp(
            s_init, self._bounds.s_min, self._bounds.s_max
        )
        state.mem_d[idx] = self._vmath.clamp(
            d_init, self._bounds.d_min, self._bounds.d_max
        )

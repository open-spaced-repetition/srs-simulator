from __future__ import annotations

import dataclasses
import math
from typing import Sequence, Tuple


@dataclasses.dataclass(frozen=True)
class Bounds:
    s_min: float = 0.1
    s_max: float = 365.0 * 25.0
    d_min: float = 1.0
    d_max: float = 10.0


@dataclasses.dataclass(frozen=True)
class FSRS6Params:
    weights: Tuple[float, ...]
    bounds: Bounds = Bounds()

    def __post_init__(self) -> None:
        if len(self.weights) != 21:
            raise ValueError("FSRS6Params expects 21 weights.")


@dataclasses.dataclass(frozen=True)
class FSRS3Params:
    weights: Tuple[float, ...]
    bounds: Bounds = Bounds()

    def __post_init__(self) -> None:
        if len(self.weights) != 13:
            raise ValueError("FSRS3Params expects 13 weights.")


# --------------------------- FSRS6 helpers --------------------------- #


def fsrs6_init_state(p: FSRS6Params, rating: int) -> Tuple[float, float]:
    s = p.weights[rating - 1]
    d = _clamp_d(p.bounds, p.weights[4] - math.exp(p.weights[5] * (rating - 1)) + 1.0)
    return s, d


def fsrs6_forgetting_curve(p: FSRS6Params, t: float, s: float) -> float:
    decay = -p.weights[20]
    factor = 0.9 ** (1.0 / decay) - 1.0
    return (1.0 + factor * t / max(s, p.bounds.s_min)) ** decay


def fsrs6_next_d(p: FSRS6Params, d: float, rating: int) -> float:
    delta_d = -p.weights[6] * (rating - 3.0)
    new_d = d + _linear_damping(delta_d, d)
    new_d = _mean_reversion(p.weights[7], fsrs6_init_state(p, 4)[1], new_d)
    return _clamp_d(p.bounds, new_d)


def fsrs6_stability_short_term(p: FSRS6Params, s: float, rating: int) -> float:
    sinc = math.exp(p.weights[17] * (rating - 3 + p.weights[18])) * (
        s ** (-p.weights[19])
    )
    return s * (max(1.0, sinc) if rating >= 3 else sinc)


def fsrs6_stability_after_success(
    p: FSRS6Params, s: float, r: float, d: float, rating: int
) -> float:
    hard_penalty = p.weights[15] if rating == 2 else 1.0
    easy_bonus = p.weights[16] if rating == 4 else 1.0
    inc = (
        math.exp(p.weights[8])
        * (11.0 - d)
        * (s ** (-p.weights[9]))
        * (math.exp((1.0 - r) * p.weights[10]) - 1.0)
    )
    return s * (1.0 + inc * hard_penalty * easy_bonus)


def fsrs6_stability_after_failure(
    p: FSRS6Params, s: float, r: float, d: float
) -> float:
    new_s = (
        p.weights[11]
        * (d ** (-p.weights[12]))
        * ((s + 1.0) ** p.weights[13] - 1.0)
        * math.exp((1.0 - r) * p.weights[14])
    )
    new_min = s / math.exp(p.weights[17] * p.weights[18])
    return min(new_s, new_min)


def fsrs6_next_interval(p: FSRS6Params, s: float, desired_retention: float) -> float:
    decay = -p.weights[20]
    factor = 0.9 ** (1.0 / decay) - 1.0
    return max(1.0, s / factor * (desired_retention ** (1.0 / decay) - 1.0))


# --------------------------- FSRS3 helpers --------------------------- #


def fsrs3_init_state(p: FSRS3Params, rating: int) -> Tuple[float, float]:
    s = p.weights[0] + p.weights[1] * (rating - 1)
    d = p.weights[2] + p.weights[3] * (rating - 3)
    return _clamp_s(p.bounds, s), _clamp_d(p.bounds, d)


def fsrs3_forgetting_curve(p: FSRS3Params, t: float, s: float) -> float:
    return 0.9 ** (t / max(s, p.bounds.s_min))


def fsrs3_mean_reversion(init: float, current: float) -> float:
    return 0.5 * init + 0.5 * current


def fsrs3_stability_after_success(
    p: FSRS3Params, s: float, new_d: float, r: float
) -> float:
    inc = (
        math.exp(p.weights[6])
        * (11.0 - new_d)
        * (s ** p.weights[7])
        * (math.exp((1.0 - r) * p.weights[8]) - 1.0)
    )
    return s * (1.0 + inc)


def fsrs3_stability_after_failure(
    p: FSRS3Params, s: float, new_d: float, r: float
) -> float:
    return (
        p.weights[9]
        * (new_d ** p.weights[10])
        * (s ** p.weights[11])
        * math.exp((1.0 - r) * p.weights[12])
    )


def fsrs3_next_interval(p: FSRS3Params, s: float, desired_retention: float) -> float:
    ln = math.log(0.9)
    return max(1.0, s * math.log(desired_retention) / ln)


# --------------------------- shared helpers --------------------------- #


def _linear_damping(delta_d: float, old_d: float) -> float:
    return delta_d * (10.0 - old_d) / 9.0


def _mean_reversion(weight: float, init: float, current: float) -> float:
    return weight * init + (1.0 - weight) * current


def _clamp_s(bounds: Bounds, s: float) -> float:
    return max(bounds.s_min, min(s, bounds.s_max))


def _clamp_d(bounds: Bounds, d: float) -> float:
    return max(bounds.d_min, min(d, bounds.d_max))

from __future__ import annotations

DEFAULT_FSRS6_WEIGHTS: tuple[float, ...] = (
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542,
)

DEFAULT_FSRS3_WEIGHTS: tuple[float, ...] = (
    0.9605,
    1.7234,
    4.8527,
    -1.1917,
    -1.2956,
    0.0573,
    1.7352,
    -0.1673,
    1.065,
    1.8907,
    -0.3832,
    0.5867,
    1.0721,
)


from typing import Sequence


def resolve_fsrs6_weights(weights: Sequence[float] | None) -> tuple[float, ...]:
    if weights is None:
        return DEFAULT_FSRS6_WEIGHTS
    return tuple(float(x) for x in weights)


def resolve_fsrs3_weights(weights: Sequence[float] | None) -> tuple[float, ...]:
    if weights is None:
        return DEFAULT_FSRS3_WEIGHTS
    return tuple(float(x) for x in weights)


__all__ = [
    "DEFAULT_FSRS6_WEIGHTS",
    "DEFAULT_FSRS3_WEIGHTS",
    "resolve_fsrs6_weights",
    "resolve_fsrs3_weights",
]

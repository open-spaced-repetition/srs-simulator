from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

FUZZ_RANGES: list[tuple[float, float, float]] = [
    (2.5, 7.0, 0.15),
    (7.0, 20.0, 0.1),
    (20.0, float("inf"), 0.05),
]

DEFAULT_MAX_INTERVAL = 36500


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def fuzz_delta(interval: float) -> float:
    if interval < 2.5:
        return 0.0
    delta = 1.0
    for start, end, factor in FUZZ_RANGES:
        span = max(0.0, min(interval, end) - start)
        delta += factor * span
    return delta


def fuzz_bounds(interval: float) -> tuple[int, int]:
    delta = fuzz_delta(interval)
    return (
        _round_half_up(interval - delta),
        _round_half_up(interval + delta),
    )


def constrained_fuzz_bounds(
    interval: float, minimum: int, maximum: int
) -> tuple[int, int]:
    minimum = min(minimum, maximum)
    interval = max(float(minimum), min(float(maximum), interval))
    lower, upper = fuzz_bounds(interval)
    lower = max(minimum, min(maximum, lower))
    upper = max(minimum, min(maximum, upper))
    if upper == lower and upper > 2 and upper < maximum:
        upper = lower + 1
    return lower, upper


def with_review_fuzz(
    fuzz_factor: Optional[float],
    interval: float,
    minimum: int,
    maximum: int,
) -> int:
    if fuzz_factor is None:
        return max(minimum, min(maximum, _round_half_up(interval)))
    lower, upper = constrained_fuzz_bounds(interval, minimum, maximum)
    return int(math.floor(lower + fuzz_factor * (1 + upper - lower)))


def with_learning_fuzz(rng: Callable[[], float], secs: float) -> float:
    """Apply Anki-style learning fuzz: up to +25% or +5 minutes, whichever is smaller."""
    if secs <= 0:
        return 0.0
    extra = min(secs * 0.25, 300.0)
    if extra <= 0:
        return secs
    return secs + rng() * extra


def resolve_max_interval(
    obj: object, *, default_max: int = DEFAULT_MAX_INTERVAL
) -> int:
    for attr in ("max_interval", "retire_interval", "_policy_retire_interval"):
        value = getattr(obj, attr, None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric > 0:
            return max(1, int(math.floor(numeric)))
    return max(1, int(default_max))

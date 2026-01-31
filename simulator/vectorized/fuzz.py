from __future__ import annotations

import torch

FUZZ_RANGES: list[tuple[float, float, float]] = [
    (2.5, 7.0, 0.15),
    (7.0, 20.0, 0.1),
    (20.0, float("inf"), 0.05),
]


def _round_half_up(values: torch.Tensor) -> torch.Tensor:
    return torch.floor(values + 0.5)


def _fuzz_delta(intervals: torch.Tensor) -> torch.Tensor:
    delta = torch.zeros_like(intervals)
    active = intervals >= 2.5
    if not active.any():
        return delta
    delta = torch.where(active, torch.ones_like(intervals), delta)
    for start, end, factor in FUZZ_RANGES:
        span = torch.clamp(intervals, min=start, max=end) - start
        span = torch.clamp(span, min=0.0)
        delta = delta + factor * span
    return delta


def constrained_fuzz_bounds(
    intervals: torch.Tensor, minimum: int, maximum: int
) -> tuple[torch.Tensor, torch.Tensor]:
    min_val = int(min(minimum, maximum))
    max_val = int(max(minimum, maximum))
    intervals = intervals.clamp(min=float(min_val), max=float(max_val))
    delta = _fuzz_delta(intervals)
    lower = _round_half_up(intervals - delta).to(torch.int64)
    upper = _round_half_up(intervals + delta).to(torch.int64)
    lower = lower.clamp(min=min_val, max=max_val)
    upper = upper.clamp(min=min_val, max=max_val)
    bump = (upper == lower) & (upper > 2) & (upper < max_val)
    upper = torch.where(bump, upper + 1, upper)
    return lower, upper


def with_review_fuzz(
    intervals: torch.Tensor,
    fuzz_factors: torch.Tensor,
    *,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    lower, upper = constrained_fuzz_bounds(intervals, minimum, maximum)
    span = (upper - lower + 1).to(fuzz_factors.dtype)
    fuzzed = lower.to(fuzz_factors.dtype) + torch.floor(fuzz_factors * span)
    return fuzzed.to(torch.int64)


def round_intervals(
    intervals: torch.Tensor, minimum: int, maximum: int | None = None
) -> torch.Tensor:
    rounded = _round_half_up(intervals).to(torch.int64)
    if maximum is None:
        return rounded.clamp(min=minimum)
    return rounded.clamp(min=minimum, max=maximum)

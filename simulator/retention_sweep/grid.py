from __future__ import annotations

import math


def dr_values(start: float, end: float, step: float) -> list[float]:
    """Desired-retention grid values (rounded to 2 decimals).

    Mirrors the retention sweep CLI semantics:
    - start/end/step are interpreted as 0-1 floats
    - values are rounded to 2 decimals
    - step must be large enough to change the rounded value (>= 0.01)
    """
    if step == 0:
        raise ValueError("--step must be non-zero.")
    start = round(start, 2)
    end = round(end, 2)
    values: list[float] = []
    value = start
    epsilon = abs(step) * 1e-6
    if step > 0 and start > end:
        raise ValueError(
            "--start-retention must be <= --end-retention for positive step."
        )
    if step < 0 and start < end:
        raise ValueError(
            "--start-retention must be >= --end-retention for negative step."
        )
    while (value <= end + epsilon) if step > 0 else (value >= end - epsilon):
        values.append(round(value, 2))
        next_value = round(value + step, 2)
        if next_value == value:
            raise ValueError("Retention step is too small after rounding; use >= 0.01.")
        value = next_value
    return values


def count_dr_steps(start: float, end: float, step: float) -> int:
    """Fast count of DR grid steps, matching dr_values' inclusion semantics."""
    if step == 0:
        return 0
    start = round(start, 2)
    end = round(end, 2)
    if step > 0 and start > end:
        return 0
    if step < 0 and start < end:
        return 0
    span_abs = abs(end - start)
    step_abs = abs(step)
    return int(math.floor(span_abs / step_abs + 1e-9)) + 1

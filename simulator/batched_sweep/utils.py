from __future__ import annotations

from collections.abc import Iterable

from simulator.sweep_utils import parse_cuda_devices


def chunked(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def dr_values(start: float, end: float, step: float) -> list[float]:
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


def format_id_list(values: list[int], *, max_len: int = 10) -> str:
    if len(values) <= max_len:
        return ", ".join(str(value) for value in values)
    head = ", ".join(str(value) for value in values[:max_len])
    return f"{head}, ..."

from __future__ import annotations

from collections.abc import Sequence

MIN_TARGET_RETENTION = 0.50
DEFAULT_TARGET_RETENTIONS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.93,
    0.96,
    0.98,
]


def retention_values_are_supported(values: Sequence[float]) -> bool:
    return all(MIN_TARGET_RETENTION <= value < 1.0 for value in values)


def retention_range_message(name: str) -> str:
    return f"{name} values must be within [0.5, 1)."


def validate_retention_values(values: Sequence[float], *, name: str) -> None:
    if not retention_values_are_supported(values):
        raise SystemExit(retention_range_message(name))


def validate_retention_values_for_model(
    values: Sequence[float],
    *,
    name: str,
) -> None:
    if not retention_values_are_supported(values):
        raise ValueError(retention_range_message(name))

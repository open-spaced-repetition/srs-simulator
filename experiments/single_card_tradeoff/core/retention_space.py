from __future__ import annotations

from collections.abc import Sequence

from experiments.single_card_tradeoff.core.defaults import (
    DEFAULT_TARGET_RETENTIONS,
    MIN_TARGET_RETENTION,
)

__all__ = [
    "DEFAULT_TARGET_RETENTIONS",
    "MIN_TARGET_RETENTION",
    "retention_range_message",
    "retention_values_are_supported",
    "validate_retention_values",
    "validate_retention_values_for_model",
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

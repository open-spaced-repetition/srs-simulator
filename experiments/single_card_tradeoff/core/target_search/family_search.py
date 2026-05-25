from __future__ import annotations

from collections.abc import Sequence
import math

from experiments.single_card_tradeoff.core.target_search.types import (
    ConstrainedTarget,
    EvaluatedPoint,
)

EPSILON = 1e-12


def unique_sorted(values: Sequence[float], *, digits: int = 12) -> list[float]:
    seen: set[float] = set()
    output: list[float] = []
    for value in values:
        if not math.isfinite(value):
            continue
        key = round(float(value), digits)
        if key in seen:
            continue
        seen.add(key)
        output.append(float(value))
    return sorted(output)


def missing_theta_values(
    existing: Sequence[EvaluatedPoint],
    candidates: Sequence[float],
    *,
    digits: int = 12,
) -> list[float]:
    return missing_values_from_values(
        [point.theta_value for point in existing],
        candidates,
        digits=digits,
    )


def missing_values_from_values(
    existing: Sequence[float],
    candidates: Sequence[float],
    *,
    digits: int = 12,
) -> list[float]:
    existing_values = {round(value, digits) for value in existing}
    return [
        value
        for value in unique_sorted(candidates, digits=digits)
        if round(value, digits) not in existing_values
    ]


def _point_metric(point: EvaluatedPoint, target: ConstrainedTarget) -> float:
    return point.memory if target.target_type == "memory" else point.minutes


def _straddles(left: float, right: float, target: float) -> bool:
    return (left - target) * (right - target) <= EPSILON and abs(left - right) > EPSILON


def _sample_between(
    low: float,
    high: float,
    *,
    count: int,
    theta_kind: str,
    theta_min: float,
    theta_max: float,
) -> list[float]:
    if count <= 0:
        return []
    lo = max(theta_min, min(low, high))
    hi = min(theta_max, max(low, high))
    if hi - lo <= EPSILON:
        return []
    values = [lo + (hi - lo) * idx / float(count + 1) for idx in range(1, count + 1)]
    if theta_kind == "integer":
        return unique_sorted([float(round(value)) for value in values])
    return unique_sorted(values)


def adaptive_theta_candidates(
    points: Sequence[EvaluatedPoint],
    targets: Sequence[ConstrainedTarget],
    *,
    family: str,
    theta_kind: str,
    theta_min: float,
    theta_max: float,
    candidates_per_bracket: int,
) -> list[float]:
    if theta_kind not in {"continuous", "integer"}:
        raise ValueError("theta_kind must be 'continuous' or 'integer'.")
    if candidates_per_bracket <= 0:
        return []

    candidates: list[float] = []
    for target in targets:
        if target.user_id is None:
            continue
        target_points = sorted(
            [
                point
                for point in points
                if point.user_id == target.user_id and point.family == family
            ],
            key=lambda point: point.theta_value,
        )
        if len(target_points) < 2:
            continue
        for left, right in zip(target_points, target_points[1:], strict=False):
            left_metric = _point_metric(left, target)
            right_metric = _point_metric(right, target)
            if not _straddles(left_metric, right_metric, target.value):
                continue
            candidates.extend(
                _sample_between(
                    left.theta_value,
                    right.theta_value,
                    count=candidates_per_bracket,
                    theta_kind=theta_kind,
                    theta_min=theta_min,
                    theta_max=theta_max,
                )
            )

    return missing_theta_values(points, candidates)

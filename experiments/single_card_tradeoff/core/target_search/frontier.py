from __future__ import annotations

from collections.abc import Iterable, Sequence

from experiments.single_card_tradeoff.core.target_search.types import (
    ConstrainedTarget,
    EvaluatedPoint,
    FrontierSegment,
    TargetAnswer,
)

EPSILON = 1e-12


def _dominates(left: EvaluatedPoint, right: EvaluatedPoint) -> bool:
    if left.user_id != right.user_id or left.family != right.family:
        return False
    no_worse = (
        left.memory >= right.memory - EPSILON
        and left.minutes <= right.minutes + EPSILON
    )
    strictly_better = (
        left.memory > right.memory + EPSILON or left.minutes < right.minutes - EPSILON
    )
    return no_worse and strictly_better


def empirical_frontier(points: Sequence[EvaluatedPoint]) -> list[EvaluatedPoint]:
    frontier: list[EvaluatedPoint] = []
    for candidate in points:
        if any(
            other is not candidate and _dominates(other, candidate) for other in points
        ):
            continue
        frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda point: (
            point.user_id,
            point.memory,
            point.minutes,
            point.theta_value,
        ),
    )


def points_for_target(
    points: Iterable[EvaluatedPoint],
    target: ConstrainedTarget,
    *,
    family: str,
) -> list[EvaluatedPoint]:
    return [
        point
        for point in points
        if point.family == family
        and (target.user_id is None or point.user_id == target.user_id)
    ]


def frontier_segments(frontier: Sequence[EvaluatedPoint]) -> list[FrontierSegment]:
    by_key: dict[tuple[int, str], list[EvaluatedPoint]] = {}
    for point in frontier:
        by_key.setdefault((point.user_id, point.family), []).append(point)

    segments: list[FrontierSegment] = []
    for (user_id, family), user_points in by_key.items():
        ordered = sorted(user_points, key=lambda point: (point.memory, point.minutes))
        for low, high in zip(ordered, ordered[1:], strict=False):
            memory_delta = high.memory - low.memory
            minutes_delta = high.minutes - low.minutes
            slope = (
                memory_delta / minutes_delta if abs(minutes_delta) > EPSILON else None
            )
            segments.append(
                FrontierSegment(
                    user_id=user_id,
                    family=family,
                    low=low,
                    high=high,
                    slope_lambda=slope if slope is not None and slope >= 0.0 else None,
                    certified=False,
                    certificate_gap=None,
                )
            )
    return segments


def _supported_user_frontier(
    points: Sequence[EvaluatedPoint],
) -> list[EvaluatedPoint]:
    ordered = sorted(points, key=lambda point: (point.minutes, point.memory))
    increasing_memory: list[EvaluatedPoint] = []
    max_memory = float("-inf")
    for point in ordered:
        if point.memory <= max_memory + EPSILON:
            continue
        increasing_memory.append(point)
        max_memory = point.memory

    supported: list[EvaluatedPoint] = []
    for point in increasing_memory:
        while len(supported) >= 2:
            low = supported[-2]
            mid = supported[-1]
            left_minutes_delta = mid.minutes - low.minutes
            right_minutes_delta = point.minutes - mid.minutes
            if (
                abs(left_minutes_delta) <= EPSILON
                or abs(right_minutes_delta) <= EPSILON
            ):
                supported.pop()
                continue
            left_slope = (mid.memory - low.memory) / left_minutes_delta
            right_slope = (point.memory - mid.memory) / right_minutes_delta
            if right_slope >= left_slope - EPSILON:
                supported.pop()
                continue
            break
        supported.append(point)
    return sorted(supported, key=lambda point: (point.memory, point.minutes))


def supported_frontier(points: Sequence[EvaluatedPoint]) -> list[EvaluatedPoint]:
    by_key: dict[tuple[int, str], list[EvaluatedPoint]] = {}
    for point in empirical_frontier(points):
        by_key.setdefault((point.user_id, point.family), []).append(point)

    supported: list[EvaluatedPoint] = []
    for user_points in by_key.values():
        supported.extend(_supported_user_frontier(user_points))
    return sorted(
        supported,
        key=lambda point: (
            point.user_id,
            point.memory,
            point.minutes,
            point.theta_value,
        ),
    )


def supported_frontier_segments(
    points: Sequence[EvaluatedPoint],
) -> list[FrontierSegment]:
    return frontier_segments(supported_frontier(points))


def best_feasible_point(
    points: Sequence[EvaluatedPoint],
    target: ConstrainedTarget,
    *,
    memory_margin: float = 0.0,
    time_margin: float = 0.0,
) -> EvaluatedPoint | None:
    if target.target_type == "memory":
        threshold = target.value + memory_margin
        feasible = [point for point in points if point.memory >= threshold - EPSILON]
        if not feasible:
            return None
        return min(feasible, key=lambda point: (point.minutes, -point.memory))

    threshold = target.value - time_margin
    feasible = [point for point in points if point.minutes <= threshold + EPSILON]
    if not feasible:
        return None
    return max(feasible, key=lambda point: (point.memory, -point.minutes))


def _memory_neighbors(
    frontier: Sequence[EvaluatedPoint],
    target: ConstrainedTarget,
) -> tuple[EvaluatedPoint | None, EvaluatedPoint | None]:
    below = [point for point in frontier if point.memory < target.value - EPSILON]
    above = [point for point in frontier if point.memory >= target.value - EPSILON]
    low = max(below, key=lambda point: point.memory) if below else None
    high = min(above, key=lambda point: point.memory) if above else None
    return low, high


def _time_neighbors(
    frontier: Sequence[EvaluatedPoint],
    target: ConstrainedTarget,
) -> tuple[EvaluatedPoint | None, EvaluatedPoint | None]:
    below = [point for point in frontier if point.minutes <= target.value + EPSILON]
    above = [point for point in frontier if point.minutes > target.value + EPSILON]
    low = max(below, key=lambda point: point.minutes) if below else None
    high = min(above, key=lambda point: point.minutes) if above else None
    return low, high


def _mixed_diagnostics(
    target: ConstrainedTarget,
    low: EvaluatedPoint | None,
    high: EvaluatedPoint | None,
) -> tuple[bool, float | None, float | None, float | None]:
    if low is None or high is None:
        return False, None, None, None
    if target.target_type == "memory":
        span = high.memory - low.memory
        if span <= EPSILON:
            return False, None, None, None
        p_high = (target.value - low.memory) / span
    else:
        span = high.minutes - low.minutes
        if span <= EPSILON:
            return False, None, None, None
        p_high = (target.value - low.minutes) / span
    if p_high < -EPSILON or p_high > 1.0 + EPSILON:
        return False, None, None, None
    p_high = min(1.0, max(0.0, p_high))
    mixed_memory = (1.0 - p_high) * low.memory + p_high * high.memory
    mixed_minutes = (1.0 - p_high) * low.minutes + p_high * high.minutes
    return True, p_high, mixed_memory, mixed_minutes


def answer_target(
    points: Sequence[EvaluatedPoint],
    target: ConstrainedTarget,
    *,
    family: str,
    memory_margin: float = 0.0,
    time_margin: float = 0.0,
    certified: bool = False,
) -> TargetAnswer:
    relevant = points_for_target(points, target, family=family)
    frontier = empirical_frontier(relevant)
    best = best_feasible_point(
        frontier,
        target,
        memory_margin=memory_margin,
        time_margin=time_margin,
    )
    if target.target_type == "memory":
        low, high = _memory_neighbors(frontier, target)
    else:
        low, high = _time_neighbors(frontier, target)
    mixed_available, p_high, mixed_memory, mixed_minutes = _mixed_diagnostics(
        target,
        low,
        high,
    )
    achieved_memory = None if best is None else best.memory
    achieved_minutes = None if best is None else best.minutes
    memory_slack = None if best is None else best.memory - target.value
    time_slack = None if best is None else target.value - best.minutes
    return TargetAnswer(
        target=target,
        family=family,
        feasible=best is not None,
        point=best,
        achieved_memory=achieved_memory,
        achieved_minutes=achieved_minutes,
        memory_slack=memory_slack if target.target_type == "memory" else None,
        time_slack=time_slack if target.target_type == "time" else None,
        certified=certified,
        neighbor_low=low,
        neighbor_high=high,
        mixed_available=mixed_available,
        mixed_probability_high=p_high,
        mixed_memory=mixed_memory,
        mixed_minutes=mixed_minutes,
    )


def target_answers(
    points: Sequence[EvaluatedPoint],
    targets: Sequence[ConstrainedTarget],
    *,
    family: str,
    memory_margin: float = 0.0,
    time_margin: float = 0.0,
    certified: bool = False,
) -> list[TargetAnswer]:
    return [
        answer_target(
            points,
            target,
            family=family,
            memory_margin=memory_margin,
            time_margin=time_margin,
            certified=certified,
        )
        for target in targets
    ]

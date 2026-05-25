from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import math

from experiments.single_card_tradeoff.core.target_search.family_search import (
    unique_sorted,
)
from experiments.single_card_tradeoff.core.target_search.frontier import (
    EPSILON,
    supported_frontier_segments,
)
from experiments.single_card_tradeoff.core.target_search.types import (
    ConstrainedTarget,
    EvaluatedPoint,
    FrontierSegment,
    TargetAnswer,
)


def segment_scalar_gap(
    segment: FrontierSegment,
    point: EvaluatedPoint,
) -> float | None:
    lambda_ab = segment.slope_lambda
    if lambda_ab is None or lambda_ab < 0.0:
        return None
    segment_value = segment.low.memory - lambda_ab * segment.low.minutes
    point_value = point.memory - lambda_ab * point.minutes
    return max(0.0, point_value - segment_value)


def target_relevant_segments(
    segments: Sequence[FrontierSegment],
    targets: Sequence[ConstrainedTarget],
    *,
    family: str,
) -> list[FrontierSegment]:
    relevant: list[FrontierSegment] = []
    for segment in segments:
        if segment.family != family:
            continue
        for target in targets:
            if target.user_id is not None and segment.user_id != target.user_id:
                continue
            if target.target_type == "memory":
                low = min(segment.low.memory, segment.high.memory)
                high = max(segment.low.memory, segment.high.memory)
            else:
                low = min(segment.low.minutes, segment.high.minutes)
                high = max(segment.low.minutes, segment.high.minutes)
            if low - EPSILON <= target.value <= high + EPSILON:
                relevant.append(segment)
                break
    return relevant


def oracle_refinement_candidates(
    points: Sequence[EvaluatedPoint],
    targets: Sequence[ConstrainedTarget],
    *,
    family: str,
    theta_min: float,
    theta_max: float,
    existing_theta_values: Sequence[float],
    max_candidates: int,
) -> list[float]:
    if max_candidates <= 0:
        return []
    frontier = [
        point
        for point in points
        if point.family == family
        and point.exact
        and theta_min <= point.theta_value <= theta_max
    ]
    scored: list[tuple[int, float, float]] = []
    for segment in target_relevant_segments(
        supported_frontier_segments(frontier),
        targets,
        family=family,
    ):
        lambda_ab = segment.slope_lambda
        if lambda_ab is None or not math.isfinite(lambda_ab):
            continue
        if not (theta_min <= lambda_ab <= theta_max):
            continue
        target_hits = 0
        for target in targets:
            if target.user_id is not None and target.user_id != segment.user_id:
                continue
            if target.target_type == "memory":
                low = min(segment.low.memory, segment.high.memory)
                high = max(segment.low.memory, segment.high.memory)
            else:
                low = min(segment.low.minutes, segment.high.minutes)
                high = max(segment.low.minutes, segment.high.minutes)
            if low - EPSILON <= target.value <= high + EPSILON:
                target_hits += 1
        span = abs(segment.high.memory - segment.low.memory) + abs(
            segment.high.minutes - segment.low.minutes
        )
        scored.append((target_hits, span, lambda_ab))

    existing = {round(value, 12) for value in existing_theta_values}
    missing: list[float] = []
    seen: set[float] = set()
    for _, _, value in sorted(scored, reverse=True):
        key = round(value, 12)
        if key in existing or key in seen:
            continue
        seen.add(key)
        missing.append(value)
        if len(missing) >= max_candidates:
            break
    return unique_sorted(missing)


def certify_oracle_segments(
    segments: Sequence[FrontierSegment],
    points: Sequence[EvaluatedPoint],
    *,
    family: str,
    tolerance: float,
    digits: int = 10,
) -> list[FrontierSegment]:
    solved: dict[tuple[int, str, float], EvaluatedPoint] = {}
    for point in points:
        if not point.exact or point.family != family:
            continue
        solved[(point.user_id, point.family, round(point.theta_value, digits))] = point

    certified: list[FrontierSegment] = []
    for segment in segments:
        lambda_ab = segment.slope_lambda
        if segment.family != family or lambda_ab is None:
            certified.append(segment)
            continue
        point = solved.get((segment.user_id, segment.family, round(lambda_ab, digits)))
        if point is None:
            certified.append(segment)
            continue
        gap = segment_scalar_gap(segment, point)
        certified.append(
            replace(
                segment,
                certified=gap is not None and gap <= tolerance,
                certificate_gap=gap,
            )
        )
    return certified


def target_certification_map(
    segments: Sequence[FrontierSegment],
    targets: Sequence[ConstrainedTarget],
    *,
    family: str,
) -> dict[ConstrainedTarget, bool]:
    certifications: dict[ConstrainedTarget, bool] = {}
    for target in targets:
        certifications[target] = False
        for segment in segments:
            if segment.family != family:
                continue
            if target.user_id is not None and target.user_id != segment.user_id:
                continue
            if target.target_type == "memory":
                low = min(segment.low.memory, segment.high.memory)
                high = max(segment.low.memory, segment.high.memory)
            else:
                low = min(segment.low.minutes, segment.high.minutes)
                high = max(segment.low.minutes, segment.high.minutes)
            if low - EPSILON <= target.value <= high + EPSILON:
                certifications[target] = bool(segment.certified)
                break
    return certifications


def apply_target_certifications(
    answers: Sequence[TargetAnswer],
    certifications: dict[ConstrainedTarget, bool],
) -> list[TargetAnswer]:
    return [
        replace(answer, certified=bool(certifications.get(answer.target, False)))
        for answer in answers
    ]

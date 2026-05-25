from __future__ import annotations

import math
from typing import Any

from experiments.single_card_tradeoff.core.target_search.types import (
    EvaluatedPoint,
    FrontierSegment,
    TargetAnswer,
)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def point_row(point: EvaluatedPoint) -> dict[str, Any]:
    return {
        "user_id": point.user_id,
        "family": point.family,
        "theta_name": point.theta_name,
        "theta_value": format_optional_float(point.theta_value),
        "M": format_optional_float(point.memory),
        "T": format_optional_float(point.minutes),
        "policy_ref": point.policy_ref or "",
        "cache_key": point.cache_key or "",
        "exact": point.exact,
        "eval_stage": point.eval_stage,
        "particles": "" if point.particles is None else point.particles,
        "seed": "" if point.seed is None else point.seed,
        "runtime_s": format_optional_float(point.runtime_s),
    }


def segment_row(segment: FrontierSegment) -> dict[str, Any]:
    return {
        "user_id": segment.user_id,
        "family": segment.family,
        "low_theta": format_optional_float(segment.low.theta_value),
        "low_M": format_optional_float(segment.low.memory),
        "low_T": format_optional_float(segment.low.minutes),
        "high_theta": format_optional_float(segment.high.theta_value),
        "high_M": format_optional_float(segment.high.memory),
        "high_T": format_optional_float(segment.high.minutes),
        "slope_lambda": format_optional_float(segment.slope_lambda),
        "certified": segment.certified,
        "certificate_gap": format_optional_float(segment.certificate_gap),
    }


def answer_row(answer: TargetAnswer) -> dict[str, Any]:
    point = answer.point
    low = answer.neighbor_low
    high = answer.neighbor_high
    return {
        "user_id": answer.target.user_id or "",
        "target_type": answer.target.target_type,
        "target_value": format_optional_float(answer.target.value),
        "family": answer.family,
        "feasible": answer.feasible,
        "theta_name": "" if point is None else point.theta_name,
        "theta_value": ""
        if point is None
        else format_optional_float(point.theta_value),
        "achieved_M": format_optional_float(answer.achieved_memory),
        "achieved_T": format_optional_float(answer.achieved_minutes),
        "memory_slack": format_optional_float(answer.memory_slack),
        "time_slack": format_optional_float(answer.time_slack),
        "certified": answer.certified,
        "policy_ref": ""
        if point is None or point.policy_ref is None
        else point.policy_ref,
        "cache_key": ""
        if point is None or point.cache_key is None
        else point.cache_key,
        "neighbor_low_theta": ""
        if low is None
        else format_optional_float(low.theta_value),
        "neighbor_low_M": "" if low is None else format_optional_float(low.memory),
        "neighbor_low_T": "" if low is None else format_optional_float(low.minutes),
        "neighbor_high_theta": ""
        if high is None
        else format_optional_float(high.theta_value),
        "neighbor_high_M": "" if high is None else format_optional_float(high.memory),
        "neighbor_high_T": "" if high is None else format_optional_float(high.minutes),
        "mixed_available": answer.mixed_available,
        "mixed_probability_high": format_optional_float(answer.mixed_probability_high),
        "mixed_M": format_optional_float(answer.mixed_memory),
        "mixed_T": format_optional_float(answer.mixed_minutes),
    }

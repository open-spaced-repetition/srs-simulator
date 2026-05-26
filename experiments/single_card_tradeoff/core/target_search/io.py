from __future__ import annotations

import csv
import math
from pathlib import Path
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
    return f"{value:.17g}"


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


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value
    return None


def point_from_row(row: dict[str, str]) -> EvaluatedPoint:
    memory = _first_present(row, ("M", "memory", "card_expected_retrievability"))
    minutes = _first_present(row, ("T", "minutes", "card_minutes_per_day"))
    theta_value = _first_present(row, ("theta_value", "theta", "goal_cost_weight"))
    if memory is None or minutes is None or theta_value is None:
        raise ValueError("point row must include theta_value, M, and T.")
    return EvaluatedPoint(
        user_id=int(row["user_id"]),
        family=row["family"],
        theta_name=row.get("theta_name") or "theta",
        theta_value=float(theta_value),
        memory=float(memory),
        minutes=float(minutes),
        policy_ref=row.get("policy_ref") or None,
        cache_key=row.get("cache_key") or None,
        exact=_parse_bool(row.get("exact")),
        eval_stage=row.get("eval_stage") or "loaded",
        particles=_parse_optional_int(row.get("particles")),
        seed=_parse_optional_int(row.get("seed")),
        runtime_s=_parse_optional_float(row.get("runtime_s")),
    )


def read_points_csv(path: Path) -> list[EvaluatedPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        points: list[EvaluatedPoint] = []
        for row in reader:
            points.append(point_from_row(dict(row)))
        return points


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

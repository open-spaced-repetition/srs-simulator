from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from experiments.single_card_tradeoff.core.target_search.io import (
    format_optional_float,
)
from experiments.single_card_tradeoff.core.target_search.types import TargetType


@dataclass(frozen=True, slots=True)
class TargetAnswerRecord:
    user_id: int
    target_type: TargetType
    target_value: float
    family: str
    feasible: bool
    theta_name: str | None
    theta_value: float | None
    achieved_memory: float | None
    achieved_minutes: float | None
    memory_slack: float | None
    time_slack: float | None
    certified: bool
    policy_ref: str | None
    cache_key: str | None
    mixed_available: bool
    mixed_probability_high: float | None
    mixed_memory: float | None
    mixed_minutes: float | None


@dataclass(frozen=True, slots=True)
class TargetOracleGap:
    candidate: TargetAnswerRecord
    oracle: TargetAnswerRecord | None
    target_value_delta: float | None


def _row_value(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _required_float(row: Mapping[str, Any], field: str) -> float:
    value = _row_value(row, field)
    if value is None:
        raise ValueError(f"target answer row must include {field}.")
    return float(value)


def _optional_float(row: Mapping[str, Any], field: str) -> float | None:
    value = _row_value(row, field)
    if value is None:
        return None
    return float(value)


def _required_int(row: Mapping[str, Any], field: str) -> int:
    value = _row_value(row, field)
    if value is None:
        raise ValueError(f"target answer row must include {field}.")
    return int(value)


def _optional_text(row: Mapping[str, Any], field: str) -> str | None:
    return _row_value(row, field)


def _bool_value(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def target_answer_record_from_row(row: Mapping[str, Any]) -> TargetAnswerRecord:
    target_type_raw = _row_value(row, "target_type")
    if target_type_raw not in {"memory", "time"}:
        raise ValueError("target answer row target_type must be 'memory' or 'time'.")
    target_type = cast(TargetType, target_type_raw)
    return TargetAnswerRecord(
        user_id=_required_int(row, "user_id"),
        target_type=target_type,
        target_value=_required_float(row, "target_value"),
        family=_row_value(row, "family") or "",
        feasible=_bool_value(row, "feasible"),
        theta_name=_optional_text(row, "theta_name"),
        theta_value=_optional_float(row, "theta_value"),
        achieved_memory=_optional_float(row, "achieved_M"),
        achieved_minutes=_optional_float(row, "achieved_T"),
        memory_slack=_optional_float(row, "memory_slack"),
        time_slack=_optional_float(row, "time_slack"),
        certified=_bool_value(row, "certified"),
        policy_ref=_optional_text(row, "policy_ref"),
        cache_key=_optional_text(row, "cache_key"),
        mixed_available=_bool_value(row, "mixed_available"),
        mixed_probability_high=_optional_float(row, "mixed_probability_high"),
        mixed_memory=_optional_float(row, "mixed_M"),
        mixed_minutes=_optional_float(row, "mixed_T"),
    )


def target_answer_records_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[TargetAnswerRecord]:
    return [target_answer_record_from_row(row) for row in rows]


def resolve_target_answers_path(path: Path) -> Path:
    if path.is_dir():
        return path / "target_answers.csv"
    return path


def read_target_answer_records(path: Path) -> list[TargetAnswerRecord]:
    resolved = resolve_target_answers_path(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return target_answer_records_from_rows(
            [dict(row) for row in csv.DictReader(handle)]
        )


def _constraint_violation(record: TargetAnswerRecord) -> float | None:
    if record.target_type == "memory":
        if record.achieved_memory is None:
            return None
        return max(0.0, record.target_value - record.achieved_memory)
    if record.achieved_minutes is None:
        return None
    return max(0.0, record.achieved_minutes - record.target_value)


def _deterministic_objective_gap(
    candidate: TargetAnswerRecord,
    oracle: TargetAnswerRecord | None,
) -> float | None:
    if oracle is None or not candidate.feasible or not oracle.feasible:
        return None
    if candidate.target_type == "memory":
        if candidate.achieved_minutes is None or oracle.achieved_minutes is None:
            return None
        return candidate.achieved_minutes - oracle.achieved_minutes
    if candidate.achieved_memory is None or oracle.achieved_memory is None:
        return None
    return oracle.achieved_memory - candidate.achieved_memory


def _mixed_objective_gap(
    candidate: TargetAnswerRecord,
    oracle: TargetAnswerRecord | None,
) -> float | None:
    if oracle is None or not candidate.feasible or not oracle.mixed_available:
        return None
    if candidate.target_type == "memory":
        if candidate.achieved_minutes is None or oracle.mixed_minutes is None:
            return None
        return candidate.achieved_minutes - oracle.mixed_minutes
    if candidate.achieved_memory is None or oracle.mixed_memory is None:
        return None
    return oracle.mixed_memory - candidate.achieved_memory


def _raw_delta(
    candidate_value: float | None,
    oracle_value: float | None,
) -> float | None:
    if candidate_value is None or oracle_value is None:
        return None
    return candidate_value - oracle_value


def _find_oracle_match(
    candidate: TargetAnswerRecord,
    oracle_records: Sequence[TargetAnswerRecord],
    *,
    target_tolerance: float,
) -> tuple[TargetAnswerRecord | None, float | None]:
    matches = [
        oracle
        for oracle in oracle_records
        if oracle.user_id == candidate.user_id
        and oracle.target_type == candidate.target_type
        and abs(oracle.target_value - candidate.target_value) <= target_tolerance
    ]
    if not matches:
        return None, None
    best = min(
        matches, key=lambda oracle: abs(oracle.target_value - candidate.target_value)
    )
    return best, best.target_value - candidate.target_value


def compare_target_answers_to_oracle(
    candidate_records: Sequence[TargetAnswerRecord],
    oracle_records: Sequence[TargetAnswerRecord],
    *,
    target_tolerance: float = 1e-9,
) -> list[TargetOracleGap]:
    if target_tolerance < 0.0 or not math.isfinite(target_tolerance):
        raise ValueError("target_tolerance must be finite and >= 0.")
    gaps: list[TargetOracleGap] = []
    for candidate in candidate_records:
        oracle, delta = _find_oracle_match(
            candidate,
            oracle_records,
            target_tolerance=target_tolerance,
        )
        gaps.append(
            TargetOracleGap(
                candidate=candidate, oracle=oracle, target_value_delta=delta
            )
        )
    return gaps


def oracle_gap_row(gap: TargetOracleGap) -> dict[str, Any]:
    candidate = gap.candidate
    oracle = gap.oracle
    deterministic_gap = _deterministic_objective_gap(candidate, oracle)
    mixed_gap = _mixed_objective_gap(candidate, oracle)
    candidate_violation = _constraint_violation(candidate)
    oracle_violation = None if oracle is None else _constraint_violation(oracle)
    return {
        "user_id": candidate.user_id,
        "target_type": candidate.target_type,
        "target_value": format_optional_float(candidate.target_value),
        "candidate_family": candidate.family,
        "oracle_family": "" if oracle is None else oracle.family,
        "oracle_match": oracle is not None,
        "target_value_delta": format_optional_float(gap.target_value_delta),
        "candidate_feasible": candidate.feasible,
        "oracle_feasible": "" if oracle is None else oracle.feasible,
        "feasibility_match": ""
        if oracle is None
        else candidate.feasible == oracle.feasible,
        "candidate_constraint_violation": format_optional_float(candidate_violation),
        "oracle_constraint_violation": format_optional_float(oracle_violation),
        "candidate_theta_name": candidate.theta_name or "",
        "candidate_theta_value": format_optional_float(candidate.theta_value),
        "oracle_theta_name": ""
        if oracle is None or oracle.theta_name is None
        else oracle.theta_name,
        "oracle_theta_value": ""
        if oracle is None
        else format_optional_float(oracle.theta_value),
        "candidate_M": format_optional_float(candidate.achieved_memory),
        "candidate_T": format_optional_float(candidate.achieved_minutes),
        "oracle_det_M": ""
        if oracle is None
        else format_optional_float(oracle.achieved_memory),
        "oracle_det_T": ""
        if oracle is None
        else format_optional_float(oracle.achieved_minutes),
        "delta_M_candidate_minus_oracle": ""
        if oracle is None
        else format_optional_float(
            _raw_delta(candidate.achieved_memory, oracle.achieved_memory)
        ),
        "delta_T_candidate_minus_oracle": ""
        if oracle is None
        else format_optional_float(
            _raw_delta(candidate.achieved_minutes, oracle.achieved_minutes)
        ),
        "deterministic_objective_gap": format_optional_float(deterministic_gap),
        "oracle_certified": "" if oracle is None else oracle.certified,
        "oracle_mixed_available": "" if oracle is None else oracle.mixed_available,
        "oracle_mixed_probability_high": ""
        if oracle is None
        else format_optional_float(oracle.mixed_probability_high),
        "oracle_mixed_M": ""
        if oracle is None
        else format_optional_float(oracle.mixed_memory),
        "oracle_mixed_T": ""
        if oracle is None
        else format_optional_float(oracle.mixed_minutes),
        "mixed_objective_gap": format_optional_float(mixed_gap),
        "candidate_policy_ref": candidate.policy_ref or "",
        "oracle_policy_ref": ""
        if oracle is None or oracle.policy_ref is None
        else oracle.policy_ref,
        "oracle_cache_key": ""
        if oracle is None or oracle.cache_key is None
        else oracle.cache_key,
    }


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / float(len(finite))


def summarize_oracle_gaps(gaps: Sequence[TargetOracleGap]) -> dict[str, Any]:
    rows = [oracle_gap_row(gap) for gap in gaps]
    matched = [gap for gap in gaps if gap.oracle is not None]
    feasibility_mismatches = [
        gap
        for gap in matched
        if gap.oracle is not None and gap.candidate.feasible != gap.oracle.feasible
    ]
    memory_gaps = [
        _deterministic_objective_gap(gap.candidate, gap.oracle)
        for gap in matched
        if gap.candidate.target_type == "memory"
    ]
    time_gaps = [
        _deterministic_objective_gap(gap.candidate, gap.oracle)
        for gap in matched
        if gap.candidate.target_type == "time"
    ]
    memory_mixed_gaps = [
        _mixed_objective_gap(gap.candidate, gap.oracle)
        for gap in matched
        if gap.candidate.target_type == "memory"
    ]
    time_mixed_gaps = [
        _mixed_objective_gap(gap.candidate, gap.oracle)
        for gap in matched
        if gap.candidate.target_type == "time"
    ]
    violations = [_constraint_violation(gap.candidate) for gap in gaps]
    return {
        "candidate_count": len(gaps),
        "oracle_matched_count": len(matched),
        "oracle_missing_count": len(gaps) - len(matched),
        "candidate_feasible_count": sum(1 for gap in gaps if gap.candidate.feasible),
        "oracle_feasible_count": sum(
            1 for gap in matched if gap.oracle is not None and gap.oracle.feasible
        ),
        "feasibility_mismatch_count": len(feasibility_mismatches),
        "mean_memory_target_extra_T_vs_oracle": _mean_optional(memory_gaps),
        "mean_time_target_memory_loss_vs_oracle": _mean_optional(time_gaps),
        "mean_memory_target_extra_T_vs_oracle_mixed": _mean_optional(memory_mixed_gaps),
        "mean_time_target_memory_loss_vs_oracle_mixed": _mean_optional(time_mixed_gaps),
        "max_candidate_constraint_violation": max(
            (value for value in violations if value is not None),
            default=None,
        ),
        "columns": list(rows[0].keys()) if rows else [],
    }

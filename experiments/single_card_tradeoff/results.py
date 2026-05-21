from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.single_card_tradeoff.types import MemoryTargetRegretAucSummary

RESULT_FIELDNAMES = [
    "user_id",
    "environment",
    "scheduler",
    "scheduler_spec",
    "desired_retention",
    "fixed_interval",
    "goal_cost_weight",
    "seed",
    "days",
    "particles",
    "deck_scale",
    "card_expected_retrievability",
    "card_minutes_per_day",
    "card_reviews_per_day",
    "card_total_reviews",
    "card_total_lapses",
    "card_total_cost_seconds",
    "card_final_projected_retrievability",
    "observed_retention",
    "deck_expected_memorized",
    "deck_minutes_per_day",
    "deck_reviews_per_day",
    "total_reviews",
    "total_lapses",
    "total_cost_seconds",
    "runtime_s",
    "fuzz",
    "review_markov_transition",
    "engine",
    "fsrs6_adr_policy",
    "fsrs6_adr_baseline_desired_retention",
    "fsrs6_adr_lambda_value",
    "fsrs6_adr_policy_index",
    "fsrs6_adr_policy_title",
    "fsrs6_adr_feature_version",
    "fsrs6_adr_point_label",
]

REGRET_AUC_FIELDNAMES = [
    "user_id",
    "environment",
    "review_markov_transition",
    "baseline_scheduler",
    "scheduler",
    "baseline_point_count",
    "scheduler_point_count",
    "baseline_frontier_count",
    "scheduler_frontier_count",
    "target_count",
    "covered_target_count",
    "total_span",
    "covered_span",
    "span_coverage_percent",
    "same_target_time_saved_auc",
    "baseline_time_auc",
    "relative_same_target_time_saved_auc_percent",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in rows:
        candidate_mem = float(candidate["deck_expected_memorized"])
        candidate_minutes = float(candidate["deck_minutes_per_day"])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_mem = float(other["deck_expected_memorized"])
            other_minutes = float(other["deck_minutes_per_day"])
            no_worse = other_mem >= candidate_mem and other_minutes <= candidate_minutes
            strictly_better = (
                other_mem > candidate_mem or other_minutes < candidate_minutes
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            float(row["deck_expected_memorized"]),
            float(row["deck_minutes_per_day"]),
        ),
    )


def row_review_markov_transition(row: dict[str, Any]) -> bool | None:
    value = row.get("review_markov_transition")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def row_user_id(row: dict[str, Any]) -> int:
    value = row.get("user_id", 1)
    if value is None or value == "":
        return 1
    return int(value)


def regret_auc_group_key(row: dict[str, Any]) -> tuple[int, str, bool | None, str]:
    scheduler = str(row["scheduler"])
    scheduler_label = "fixed" if scheduler == "fixed" else str(row["scheduler_spec"])
    return (
        row_user_id(row),
        str(row["environment"]),
        row_review_markov_transition(row),
        scheduler_label,
    )


def frontier_memory_time_points(
    rows: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    min_time_by_memory: dict[float, float] = {}
    for row in pareto_frontier(rows):
        memory = float(row["deck_expected_memorized"])
        minutes = float(row["deck_minutes_per_day"])
        if not (math.isfinite(memory) and math.isfinite(minutes)):
            continue
        previous = min_time_by_memory.get(memory)
        if previous is None or minutes < previous:
            min_time_by_memory[memory] = minutes
    return sorted(min_time_by_memory.items())


def common_interval(
    left_min: float,
    left_max: float,
    right_min: float,
    right_max: float,
) -> tuple[float, float] | None:
    start = max(left_min, right_min)
    end = min(left_max, right_max)
    if end <= start:
        return None
    return start, end


def values_in_interval(
    values: Sequence[float], start: float, end: float
) -> list[float]:
    return sorted(
        {
            value
            for value in values
            if (start < value < end)
            or math.isclose(value, start)
            or math.isclose(value, end)
        }
    )


def integration_grid(
    baseline_values: Sequence[float],
    target_values: Sequence[float],
    start: float,
    end: float,
) -> list[float]:
    return sorted(
        {
            start,
            end,
            *values_in_interval(baseline_values, start, end),
            *values_in_interval(target_values, start, end),
        }
    )


def interpolated_time_for_memory_target(
    points: Sequence[tuple[float, float]],
    target: float,
) -> float | None:
    if not points:
        return None
    if target < points[0][0] and not math.isclose(target, points[0][0]):
        return None
    if math.isclose(target, points[0][0]):
        return points[0][1]
    if target > points[-1][0] and not math.isclose(target, points[-1][0]):
        return None
    if math.isclose(target, points[-1][0]):
        return points[-1][1]

    for (left_memory, left_minutes), (
        right_memory,
        right_minutes,
    ) in zip(points[:-1], points[1:]):
        if not (left_memory <= target <= right_memory):
            continue
        if math.isclose(left_memory, right_memory):
            return min(left_minutes, right_minutes)
        ratio = (target - left_memory) / (right_memory - left_memory)
        return left_minutes + ratio * (right_minutes - left_minutes)
    return None


def memory_target_regret_auc_summary(
    *,
    user_id: int,
    environment: str,
    review_markov_transition: bool | None,
    baseline_scheduler: str,
    baseline_rows: list[dict[str, Any]],
    scheduler: str,
    scheduler_rows: list[dict[str, Any]],
) -> MemoryTargetRegretAucSummary | None:
    baseline_frontier = frontier_memory_time_points(baseline_rows)
    scheduler_frontier = frontier_memory_time_points(scheduler_rows)
    if not baseline_frontier:
        return None

    baseline_targets = [memory for memory, _ in baseline_frontier]
    total_span = (
        max(baseline_targets) - min(baseline_targets)
        if len(baseline_targets) > 1
        else 0.0
    )
    covered_span = 0.0
    time_saved_area = 0.0
    baseline_time_area = 0.0
    covered_target_count = 0

    if scheduler_frontier:
        interval = common_interval(
            baseline_frontier[0][0],
            baseline_frontier[-1][0],
            scheduler_frontier[0][0],
            scheduler_frontier[-1][0],
        )
        if interval is not None:
            start, end = interval
            covered_target_count = len(values_in_interval(baseline_targets, start, end))
            scheduler_targets = [memory for memory, _ in scheduler_frontier]
            targets = integration_grid(baseline_targets, scheduler_targets, start, end)
            for left_target, right_target in zip(targets[:-1], targets[1:]):
                width = right_target - left_target
                if width <= 0.0:
                    continue
                left_baseline_time = interpolated_time_for_memory_target(
                    baseline_frontier,
                    left_target,
                )
                right_baseline_time = interpolated_time_for_memory_target(
                    baseline_frontier,
                    right_target,
                )
                left_scheduler_time = interpolated_time_for_memory_target(
                    scheduler_frontier,
                    left_target,
                )
                right_scheduler_time = interpolated_time_for_memory_target(
                    scheduler_frontier,
                    right_target,
                )
                if (
                    left_baseline_time is None
                    or right_baseline_time is None
                    or left_scheduler_time is None
                    or right_scheduler_time is None
                ):
                    continue
                left_saved = left_baseline_time - left_scheduler_time
                right_saved = right_baseline_time - right_scheduler_time
                time_saved_area += width * ((left_saved + right_saved) / 2.0)
                baseline_time_area += width * (
                    (left_baseline_time + right_baseline_time) / 2.0
                )
                covered_span += width

    return MemoryTargetRegretAucSummary(
        user_id=user_id,
        environment=environment,
        review_markov_transition=review_markov_transition,
        baseline_scheduler=baseline_scheduler,
        scheduler=scheduler,
        baseline_point_count=len(baseline_rows),
        scheduler_point_count=len(scheduler_rows),
        baseline_frontier_count=len(baseline_frontier),
        scheduler_frontier_count=len(scheduler_frontier),
        target_count=len(baseline_targets),
        covered_target_count=covered_target_count,
        total_span=total_span,
        covered_span=covered_span,
        same_target_time_saved_auc=(time_saved_area / covered_span)
        if covered_span
        else None,
        baseline_time_auc=(baseline_time_area / covered_span) if covered_span else None,
    )


def finite_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def summary_to_regret_auc_row(
    summary: MemoryTargetRegretAucSummary,
) -> dict[str, Any]:
    span_coverage_percent = (
        (summary.covered_span / summary.total_span) * 100.0
        if summary.total_span
        else 0.0
    )
    relative_same_target_time_saved_auc_percent = (
        (summary.same_target_time_saved_auc / summary.baseline_time_auc) * 100.0
        if summary.same_target_time_saved_auc is not None
        and summary.baseline_time_auc is not None
        and summary.baseline_time_auc
        else None
    )
    return {
        "user_id": summary.user_id,
        "environment": summary.environment,
        "review_markov_transition": summary.review_markov_transition,
        "baseline_scheduler": summary.baseline_scheduler,
        "scheduler": summary.scheduler,
        "baseline_point_count": summary.baseline_point_count,
        "scheduler_point_count": summary.scheduler_point_count,
        "baseline_frontier_count": summary.baseline_frontier_count,
        "scheduler_frontier_count": summary.scheduler_frontier_count,
        "target_count": summary.target_count,
        "covered_target_count": summary.covered_target_count,
        "total_span": finite_float(summary.total_span),
        "covered_span": finite_float(summary.covered_span),
        "span_coverage_percent": finite_float(span_coverage_percent),
        "same_target_time_saved_auc": finite_float(summary.same_target_time_saved_auc),
        "baseline_time_auc": finite_float(summary.baseline_time_auc),
        "relative_same_target_time_saved_auc_percent": finite_float(
            relative_same_target_time_saved_auc_percent
        ),
    }


def build_regret_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, bool | None, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = regret_auc_group_key(row)
        groups.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    group_modes = sorted(
        {
            (user_id, environment, markov)
            for user_id, environment, markov, _scheduler in groups
        },
        key=lambda item: (item[0], item[1], str(item[2])),
    )
    for user_id, environment, review_markov_transition in group_modes:
        scheduler_labels = sorted(
            scheduler
            for group_user_id, group_env, group_markov, scheduler in groups
            if group_user_id == user_id
            and group_env == environment
            and group_markov == review_markov_transition
        )
        for baseline_scheduler in scheduler_labels:
            baseline_rows = groups[
                (user_id, environment, review_markov_transition, baseline_scheduler)
            ]
            for scheduler in scheduler_labels:
                summary = memory_target_regret_auc_summary(
                    user_id=user_id,
                    environment=environment,
                    review_markov_transition=review_markov_transition,
                    baseline_scheduler=baseline_scheduler,
                    baseline_rows=baseline_rows,
                    scheduler=scheduler,
                    scheduler_rows=groups[
                        (user_id, environment, review_markov_transition, scheduler)
                    ],
                )
                if summary is not None:
                    output_rows.append(summary_to_regret_auc_row(summary))
    return output_rows


def write_regret_auc_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGRET_AUC_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

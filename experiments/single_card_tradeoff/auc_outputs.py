from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SINGLE_SCHEDULER_AUC_FIELDS = [
    "environment",
    "span_coverage_percent",
    "same_target_time_saved_auc",
    "relative_same_target_time_saved_auc_percent",
    "covered_target_count",
    "target_count",
]

MULTI_SCHEDULER_AUC_FIELDS = [
    "environment",
    "scheduler",
    "span_coverage_percent",
    "same_target_time_saved_auc",
    "relative_same_target_time_saved_auc_percent",
    "covered_target_count",
    "target_count",
]

DETAILED_AUC_FIELDS = [
    "environment",
    "baseline_scheduler",
    "scheduler",
    "span_coverage_percent",
    "same_target_time_saved_auc",
    "baseline_time_auc",
    "relative_same_target_time_saved_auc_percent",
    "covered_target_count",
    "target_count",
]

MEAN_AUC_FIELDS = [
    "scheduler",
    "user_count",
    "mean_span_coverage_percent",
    "mean_same_target_time_saved_auc",
    "mean_relative_same_target_time_saved_auc_percent",
    "covered_target_count_sum",
    "target_count_sum",
]

DETAILED_MEAN_AUC_FIELDS = [
    "baseline_scheduler",
    *MEAN_AUC_FIELDS,
]


def select_auc_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str],
    schedulers: Sequence[str],
) -> list[Mapping[str, Any]]:
    baseline_set = set(baselines)
    scheduler_set = set(schedulers)
    return [
        row
        for row in rows
        if str(row["baseline_scheduler"]) in baseline_set
        and str(row["scheduler"]) in scheduler_set
    ]


def write_auc_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str],
    schedulers: Sequence[str],
    fieldnames: Sequence[str],
) -> None:
    selected = select_auc_rows(rows, baselines=baselines, schedulers=schedulers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in selected:
            writer.writerow({field: _metric_value(row, field) for field in fieldnames})


def build_mean_auc_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str],
    schedulers: Sequence[str],
    include_baseline_scheduler: bool,
) -> list[dict[str, Any]]:
    mean_rows: list[dict[str, Any]] = []
    for baseline in baselines:
        for scheduler in schedulers:
            selected = [
                row
                for row in rows
                if str(row["baseline_scheduler"]) == baseline
                and str(row["scheduler"]) == scheduler
            ]
            if not selected:
                continue
            mean_row: dict[str, Any] = {
                "scheduler": scheduler,
                "user_count": len(selected),
                "mean_span_coverage_percent": _mean(selected, "span_coverage_percent"),
                "mean_same_target_time_saved_auc": _mean(
                    selected, "same_target_time_saved_auc"
                ),
                "mean_relative_same_target_time_saved_auc_percent": _mean(
                    selected, "relative_same_target_time_saved_auc_percent"
                ),
                "covered_target_count_sum": sum(
                    int(row["covered_target_count"]) for row in selected
                ),
                "target_count_sum": sum(int(row["target_count"]) for row in selected),
            }
            if include_baseline_scheduler:
                mean_row = {"baseline_scheduler": baseline, **mean_row}
            mean_rows.append(mean_row)
    return mean_rows


def write_mean_auc_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str],
    schedulers: Sequence[str],
    include_baseline_scheduler: bool,
) -> list[dict[str, Any]]:
    mean_rows = build_mean_auc_rows(
        rows,
        baselines=baselines,
        schedulers=schedulers,
        include_baseline_scheduler=include_baseline_scheduler,
    )
    fieldnames = (
        DETAILED_MEAN_AUC_FIELDS if include_baseline_scheduler else MEAN_AUC_FIELDS
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in mean_rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return mean_rows


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := _metric_value(row, key)) is not None and str(value) != ""
    ]
    if not values:
        return None
    return sum(values) / float(len(values))


def _metric_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    legacy_map: dict[str, tuple[str, float]] = {
        "same_target_time_saved_auc": ("time_regret_auc", -1.0),
        "relative_same_target_time_saved_auc_percent": (
            "relative_regret_auc_percent",
            -1.0,
        ),
        "mean_same_target_time_saved_auc": ("mean_time_regret_auc", -1.0),
        "mean_relative_same_target_time_saved_auc_percent": (
            "mean_relative_regret_auc_percent",
            -1.0,
        ),
    }
    legacy = legacy_map.get(key)
    if legacy is None:
        return None
    legacy_key, sign = legacy
    value = row.get(legacy_key)
    if value is None or value == "":
        return value
    return sign * float(value)

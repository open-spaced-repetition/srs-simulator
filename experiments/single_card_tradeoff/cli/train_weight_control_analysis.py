from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import format_float


FSRS6_SCHEDULER = "fsrs6"
EXACT_SCHEDULER = "fsrs6_oracle_stationary_finite"
ADR_SCHEDULER = "fsrs6_adr"
PER_USER_DISTILL_SCHEDULER = "fsrs6_oracle_stationary_finite_distill_per_user"
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/train_weight_1_4_control_markov_off"
)
DEFAULT_FORMAL_EXACT_DIR = DEFAULT_OUT_DIR / "exact_value_formal"
DEFAULT_DENSE_LOWW_EXACT_DIR = DEFAULT_OUT_DIR / "exact_value_user2_dense_loww"
DEFAULT_TREATMENTS = (
    "sparse_baseline=artifacts/single_card_tradeoff/"
    "stationary_finite_distill_train_weights_sparse_first8_markov_off",
    "add_1_4=artifacts/single_card_tradeoff/"
    "stationary_finite_distill_train_weights_add_1_4_first8_markov_off",
)
DEFAULT_SEGMENT = (9400.0, 9750.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the stationary finite distill train-weight 1/4 control "
            "experiment into gate, pairwise, dense-low-weight, and segment outputs."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--treatment",
        action="append",
        default=None,
        metavar="LABEL=DIR",
        help=(
            "Training artifact root for one treatment. May be repeated. Defaults "
            "to sparse_baseline and add_1_4 roots."
        ),
    )
    parser.add_argument("--baseline-label", default="sparse_baseline")
    parser.add_argument("--candidate-label", default="add_1_4")
    parser.add_argument(
        "--formal-exact-dir", type=Path, default=DEFAULT_FORMAL_EXACT_DIR
    )
    parser.add_argument(
        "--dense-loww-exact-dir",
        type=Path,
        default=DEFAULT_DENSE_LOWW_EXACT_DIR,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--segment",
        default=",".join(format_float(value) for value in DEFAULT_SEGMENT),
        help="Two comma-separated deck-memory bounds for user-2 segment AUC.",
    )
    return parser.parse_args()


def _parse_treatment(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Treatment must be LABEL=DIR, got '{raw}'.")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit(f"Treatment must be LABEL=DIR, got '{raw}'.")
    return label, Path(path)


def _parse_segment(raw: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise SystemExit("--segment must contain exactly two comma-separated values.")
    start, end = values
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        raise SystemExit("--segment must be finite and increasing.")
    return start, end


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    fields = list(fieldnames) if fieldnames is not None else _fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields or ["label"]


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return payload


def _optional_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json_object(path)


def _float_value(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return float(raw)


def _int_value(row: Mapping[str, Any], key: str) -> int | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    return int(float(raw))


def _user_id_from_row(row: Mapping[str, Any]) -> int:
    raw = row.get("user_id")
    if raw not in {None, ""}:
        return int(float(str(raw)))
    environment = str(row.get("environment", ""))
    if "_user_" in environment:
        return int(environment.rsplit("_user_", 1)[1])
    if environment.startswith("fsrs6_user_"):
        return int(environment.rsplit("_", 1)[1])
    raise ValueError(f"Cannot infer user id from row: {row}")


def _distill_scheduler_spec(label: str) -> str:
    if label == "distill_476":
        return PER_USER_DISTILL_SCHEDULER
    suffix = label.removeprefix("distill_")
    return f"{PER_USER_DISTILL_SCHEDULER}_{suffix}"


def _display_scheduler(label: str, scheduler: str) -> str:
    if scheduler == FSRS6_SCHEDULER:
        return "fsrs6"
    if scheduler == EXACT_SCHEDULER:
        return "exact"
    if scheduler == ADR_SCHEDULER:
        return "adr"
    return label


def _format_csv_floats(values: Sequence[float] | None) -> str | None:
    if values is None:
        return None
    return ",".join(format_float(float(value)) for value in values)


def _load_training_cost_weights(root: Path) -> list[float] | None:
    config = _optional_json_object(root / "run_config.json")
    raw = config.get("training_cost_weights")
    if isinstance(raw, list):
        return [float(value) for value in raw]
    checkpoint_paths = sorted(root.glob("user_*_policy.pt"))
    if not checkpoint_paths:
        return None
    try:
        import torch

        checkpoint = torch.load(
            checkpoint_paths[0],
            map_location="cpu",
            weights_only=False,
        )
    except (ImportError, OSError, RuntimeError):
        return None
    raw = checkpoint.get("cost_weights")
    if not isinstance(raw, Sequence):
        return None
    return [float(value) for value in raw]


def _load_train_rows(label: str, root: Path) -> list[dict[str, Any]]:
    summary_path = root / "summary.csv"
    train_path = root / "train_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing treatment summary: {summary_path}")
    summary_rows = _read_csv(summary_path)
    train_rows = _read_csv(train_path) if train_path.exists() else []
    train_by_user = {
        _user_id_from_row(row): row for row in train_rows if row.get("user_id")
    }
    weights = _load_training_cost_weights(root)
    config = _optional_json_object(root / "run_config.json")
    performance = _optional_json_object(root / "performance_summary.json")
    gpu = _optional_json_object(root / "gpu_monitor" / "summary.json")
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        user_id = _user_id_from_row(row)
        train_row = train_by_user.get(user_id, {})
        out.append(
            {
                "treatment": label,
                "user_id": user_id,
                "train_root": str(root),
                "training_cost_weights": _format_csv_floats(weights),
                "train_weight_count": len(weights) if weights is not None else None,
                "params_per_user": _int_value(train_row, "params_per_user"),
                "epochs": _int_value(train_row, "epochs") or config.get("epochs"),
                "steps_per_epoch": _int_value(train_row, "steps_per_epoch")
                or config.get("steps_per_epoch"),
                "table_samples_per_weight": _int_value(
                    train_row,
                    "table_samples_per_weight",
                )
                or config.get("table_samples_per_weight"),
                "final_ce_loss": _float_value(train_row, "final_ce_loss"),
                "final_teacher_action_agreement": _float_value(
                    train_row,
                    "final_teacher_action_agreement",
                ),
                "eval_teacher_action_agreement": _float_value(
                    train_row,
                    "eval_teacher_action_agreement",
                ),
                "formal_vs_fsrs6_same_target_time_saved_auc": _float_value(
                    row,
                    "same_target_time_saved_auc",
                ),
                "formal_vs_fsrs6_relative_time_saved_percent": _float_value(
                    row,
                    "relative_same_target_time_saved_auc_percent",
                ),
                "formal_vs_fsrs6_span_coverage_percent": _float_value(
                    row,
                    "span_coverage_percent",
                ),
                "formal_vs_fsrs6_covered_target_count": _int_value(
                    row,
                    "covered_target_count",
                ),
                "formal_vs_fsrs6_target_count": _int_value(row, "target_count"),
                "gpu_monitor_shared_memory_spill_detected": gpu.get(
                    "shared_memory_spill_detected",
                    performance.get("gpu_monitor_shared_memory_spill_detected"),
                ),
                "gpu_monitor_shared_memory_peak_single_adapter_bytes": gpu.get(
                    "shared_memory_peak_single_adapter_bytes",
                    performance.get(
                        "gpu_monitor_shared_memory_peak_single_adapter_bytes"
                    ),
                ),
                "gpu_monitor_nvidia_smi_peak_memory_used_mib": gpu.get(
                    "nvidia_smi_peak_memory_used_mib",
                    performance.get("gpu_monitor_nvidia_smi_peak_memory_used_mib"),
                ),
            }
        )
    return sorted(out, key=lambda item: (int(item["user_id"]), str(item["treatment"])))


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row.get(key)) is not None and value != ""
    ]
    if not values:
        return None
    return sum(values) / float(len(values))


def _minimum(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row.get(key)) is not None and value != ""
    ]
    return min(values) if values else None


def _row_for_user(
    rows: Sequence[Mapping[str, Any]], label: str, user_id: int
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("treatment") == label and int(row.get("user_id", -1)) == user_id:
            return row
    return None


def _regret_rows_for_pair(
    regret_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_scheduler: str,
    scheduler: str,
    user_id: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in regret_rows:
        if str(row.get("baseline_scheduler")) != baseline_scheduler:
            continue
        if str(row.get("scheduler")) != scheduler:
            continue
        if user_id is not None and _user_id_from_row(row) != user_id:
            continue
        rows.append(dict(row))
    return sorted(rows, key=_user_id_from_row)


def _annotate_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    pair: str,
    source: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": source,
                "pair": pair,
                "treatment": treatment,
                "user_id": _user_id_from_row(row),
                "baseline_scheduler": row.get("baseline_scheduler"),
                "scheduler": row.get("scheduler"),
                "span_coverage_percent": row.get("span_coverage_percent"),
                "same_target_time_saved_auc": row.get("same_target_time_saved_auc"),
                "baseline_time_auc": row.get("baseline_time_auc"),
                "relative_same_target_time_saved_auc_percent": row.get(
                    "relative_same_target_time_saved_auc_percent"
                ),
                "covered_target_count": row.get("covered_target_count"),
                "target_count": row.get("target_count"),
                "baseline_frontier_count": row.get("baseline_frontier_count"),
                "scheduler_frontier_count": row.get("scheduler_frontier_count"),
            }
        )
    return out


def _lookup_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_scheduler: str,
    scheduler: str,
    user_id: int,
    key: str,
) -> float | None:
    for row in rows:
        if (
            str(row.get("baseline_scheduler")) == baseline_scheduler
            and str(row.get("scheduler")) == scheduler
            and _user_id_from_row(row) == user_id
        ):
            return _float_value(row, key)
    return None


def _plot_sort_key(row: Mapping[str, Any]) -> tuple[float, float]:
    goal = row.get("goal_cost_weight")
    if goal is not None and goal != "":
        return 1.0, float(goal)
    policy_index = row.get("fsrs6_adr_policy_index")
    if policy_index is not None and policy_index != "":
        return 2.0, float(policy_index)
    baseline_dr = row.get("fsrs6_adr_baseline_desired_retention")
    if baseline_dr is not None and baseline_dr != "":
        return 2.0, float(baseline_dr)
    desired = row.get("desired_retention")
    if desired is not None and desired != "":
        return 0.0, float(desired)
    return 3.0, 0.0


def _pareto_frontier(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    frontier: list[Mapping[str, Any]] = []
    for candidate in rows:
        candidate_mem = _float_value(candidate, "deck_expected_memorized")
        candidate_minutes = _float_value(candidate, "deck_minutes_per_day")
        if candidate_mem is None or candidate_minutes is None:
            continue
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_mem = _float_value(other, "deck_expected_memorized")
            other_minutes = _float_value(other, "deck_minutes_per_day")
            if other_mem is None or other_minutes is None:
                continue
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
            _plot_sort_key(row),
        ),
    )


def _frontier_points(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    min_time_by_memory: dict[float, float] = {}
    for row in _pareto_frontier(rows):
        memory = _float_value(row, "deck_expected_memorized")
        minutes = _float_value(row, "deck_minutes_per_day")
        if memory is None or minutes is None:
            continue
        if not (math.isfinite(memory) and math.isfinite(minutes)):
            continue
        previous = min_time_by_memory.get(memory)
        if previous is None or minutes < previous:
            min_time_by_memory[memory] = minutes
    return sorted(min_time_by_memory.items())


def _values_in_interval(
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


def _interpolated_time(
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
        if not left_memory <= target <= right_memory:
            continue
        if math.isclose(left_memory, right_memory):
            return min(left_minutes, right_minutes)
        ratio = (target - left_memory) / (right_memory - left_memory)
        return left_minutes + ratio * (right_minutes - left_minutes)
    return None


def _segment_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    user_id: int,
    baseline_scheduler: str,
    scheduler: str,
    segment_start: float,
    segment_end: float,
) -> dict[str, Any]:
    baseline_rows = [
        row
        for row in rows
        if _user_id_from_row(row) == user_id
        and str(row.get("scheduler_spec")) == baseline_scheduler
    ]
    scheduler_rows = [
        row
        for row in rows
        if _user_id_from_row(row) == user_id
        and str(row.get("scheduler_spec")) == scheduler
    ]
    baseline_frontier = _frontier_points(baseline_rows)
    scheduler_frontier = _frontier_points(scheduler_rows)
    covered_span = 0.0
    time_saved_area = 0.0
    baseline_time_area = 0.0
    target_count = 0
    if baseline_frontier and scheduler_frontier:
        start = max(segment_start, baseline_frontier[0][0], scheduler_frontier[0][0])
        end = min(segment_end, baseline_frontier[-1][0], scheduler_frontier[-1][0])
        if end > start:
            baseline_targets = [memory for memory, _ in baseline_frontier]
            scheduler_targets = [memory for memory, _ in scheduler_frontier]
            targets = sorted(
                {
                    start,
                    end,
                    *_values_in_interval(baseline_targets, start, end),
                    *_values_in_interval(scheduler_targets, start, end),
                }
            )
            target_count = len(_values_in_interval(baseline_targets, start, end))
            for left, right in zip(targets[:-1], targets[1:]):
                width = right - left
                if width <= 0.0:
                    continue
                left_baseline = _interpolated_time(baseline_frontier, left)
                right_baseline = _interpolated_time(baseline_frontier, right)
                left_scheduler = _interpolated_time(scheduler_frontier, left)
                right_scheduler = _interpolated_time(scheduler_frontier, right)
                if (
                    left_baseline is None
                    or right_baseline is None
                    or left_scheduler is None
                    or right_scheduler is None
                ):
                    continue
                time_saved_area += width * (
                    (left_baseline - left_scheduler + right_baseline - right_scheduler)
                    / 2.0
                )
                baseline_time_area += width * ((left_baseline + right_baseline) / 2.0)
                covered_span += width
    same_target = time_saved_area / covered_span if covered_span else None
    baseline_time = baseline_time_area / covered_span if covered_span else None
    relative = (
        (same_target / baseline_time) * 100.0
        if same_target is not None and baseline_time
        else None
    )
    return {
        "user_id": user_id,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "baseline_scheduler": baseline_scheduler,
        "scheduler": scheduler,
        "covered_span": covered_span,
        "segment_coverage_percent": (
            covered_span / (segment_end - segment_start) * 100.0
        ),
        "covered_baseline_frontier_target_count": target_count,
        "same_target_time_saved_auc": same_target,
        "baseline_time_auc": baseline_time,
        "relative_same_target_time_saved_auc_percent": relative,
    }


def _write_frontier_plot(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    user_id: int,
    scheduler_labels: Mapping[str, str],
    segment: tuple[float, float] | None,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for scheduler, label in scheduler_labels.items():
        selected = [
            row
            for row in rows
            if _user_id_from_row(row) == user_id
            and str(row.get("scheduler_spec")) == scheduler
        ]
        if not selected:
            continue
        frontier = _frontier_points(selected)
        if not frontier:
            continue
        ax.plot(
            [memory for memory, _ in frontier],
            [minutes for _, minutes in frontier],
            marker="o",
            linewidth=1.4,
            markersize=4.5,
            label=label,
        )
    if segment is not None:
        ax.axvspan(segment[0], segment[1], color="0.7", alpha=0.18, linewidth=0)
    ax.set_xlabel("Expected memorized cards per day (deck scaled)")
    ax.set_ylabel("Study minutes per day (deck scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bool_label(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value in {None, ""}:
        return ""
    return str(value)


def _build_summary_rows(
    *,
    labels: Sequence[str],
    by_user_rows: Sequence[Mapping[str, Any]],
    formal_regret_rows: Sequence[Mapping[str, Any]],
    dense_regret_rows: Sequence[Mapping[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    rows_by_label = {
        label: [row for row in by_user_rows if row.get("treatment") == label]
        for label in labels
    }
    baseline_rows = rows_by_label.get(baseline_label, [])
    baseline_user2 = _row_for_user(by_user_rows, baseline_label, 2)
    baseline_spec = _distill_scheduler_spec(baseline_label)
    baseline_dense_user2 = _lookup_metric(
        dense_regret_rows,
        baseline_scheduler=FSRS6_SCHEDULER,
        scheduler=baseline_spec,
        user_id=2,
        key="same_target_time_saved_auc",
    )
    baseline_direct_exact = _mean(
        _annotate_pair_rows(
            _regret_rows_for_pair(
                formal_regret_rows,
                baseline_scheduler=EXACT_SCHEDULER,
                scheduler=baseline_spec,
            ),
            treatment=baseline_label,
            pair="treatment_vs_exact",
            source="formal_exact_value",
        ),
        "same_target_time_saved_auc",
    )

    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        treatment_rows = rows_by_label[label]
        spec = _distill_scheduler_spec(label)
        user2 = _row_for_user(by_user_rows, label, 2)
        dense_user2 = _lookup_metric(
            dense_regret_rows,
            baseline_scheduler=FSRS6_SCHEDULER,
            scheduler=spec,
            user_id=2,
            key="same_target_time_saved_auc",
        )
        direct_exact_rows = _annotate_pair_rows(
            _regret_rows_for_pair(
                formal_regret_rows,
                baseline_scheduler=EXACT_SCHEDULER,
                scheduler=spec,
            ),
            treatment=label,
            pair="treatment_vs_exact",
            source="formal_exact_value",
        )
        direct_adr_rows = _annotate_pair_rows(
            _regret_rows_for_pair(
                formal_regret_rows,
                baseline_scheduler=ADR_SCHEDULER,
                scheduler=spec,
            ),
            treatment=label,
            pair="treatment_vs_adr",
            source="formal_exact_value",
        )
        mean_direct_exact = _mean(direct_exact_rows, "same_target_time_saved_auc")
        mean_formal_time = _mean(
            treatment_rows,
            "formal_vs_fsrs6_same_target_time_saved_auc",
        )
        mean_formal_relative = _mean(
            treatment_rows,
            "formal_vs_fsrs6_relative_time_saved_percent",
        )
        mean_formal_coverage = _mean(
            treatment_rows,
            "formal_vs_fsrs6_span_coverage_percent",
        )
        min_formal_coverage = _minimum(
            treatment_rows,
            "formal_vs_fsrs6_span_coverage_percent",
        )
        max_other_user_relative_loss = None
        if label != baseline_label and baseline_rows:
            losses: list[float] = []
            for row in treatment_rows:
                user_id = int(row["user_id"])
                if user_id == 2:
                    continue
                base = _row_for_user(by_user_rows, baseline_label, user_id)
                if base is None:
                    continue
                base_relative = base.get("formal_vs_fsrs6_relative_time_saved_percent")
                candidate_relative = row.get(
                    "formal_vs_fsrs6_relative_time_saved_percent"
                )
                if base_relative is None or candidate_relative is None:
                    continue
                losses.append(float(candidate_relative) - float(base_relative))
            max_other_user_relative_loss = min(losses) if losses else None

        formal_user2 = (
            user2.get("formal_vs_fsrs6_same_target_time_saved_auc")
            if user2 is not None
            else None
        )
        baseline_formal_user2 = (
            baseline_user2.get("formal_vs_fsrs6_same_target_time_saved_auc")
            if baseline_user2 is not None
            else None
        )
        formal_user2_delta = (
            float(formal_user2) - float(baseline_formal_user2)
            if formal_user2 is not None and baseline_formal_user2 is not None
            else None
        )
        dense_user2_delta = (
            float(dense_user2) - float(baseline_dense_user2)
            if dense_user2 is not None and baseline_dense_user2 is not None
            else None
        )
        direct_exact_delta = (
            float(mean_direct_exact) - float(baseline_direct_exact)
            if mean_direct_exact is not None and baseline_direct_exact is not None
            else None
        )
        baseline_mean_coverage = _mean(
            baseline_rows,
            "formal_vs_fsrs6_span_coverage_percent",
        )
        coverage_delta = (
            float(mean_formal_coverage) - float(baseline_mean_coverage)
            if mean_formal_coverage is not None and baseline_mean_coverage is not None
            else None
        )
        spill_values = {
            row.get("gpu_monitor_shared_memory_spill_detected")
            for row in treatment_rows
        }
        gpu_no_spill = spill_values == {False} or spill_values == {"False"}
        gate_user2_formal = (
            formal_user2 is not None and float(formal_user2) >= 0.0
        ) or (formal_user2_delta is not None and formal_user2_delta >= 3.0)
        gate_dense_user2 = (
            dense_user2 is not None
            and dense_user2_delta is not None
            and float(dense_user2) >= 1.0
            and dense_user2_delta > 0.0
        )
        gate_direct_exact = direct_exact_delta is not None and direct_exact_delta > 0.0
        gate_mean_coverage = coverage_delta is not None and coverage_delta > -1.0
        gate_min_coverage = (
            min_formal_coverage is not None and min_formal_coverage >= 90.0
        )
        gate_other_users = (
            max_other_user_relative_loss is not None
            and max_other_user_relative_loss >= -2.0
        )
        is_candidate = label != baseline_label
        gates = [
            gate_user2_formal,
            gate_dense_user2,
            gate_direct_exact,
            gate_mean_coverage,
            gate_min_coverage,
            gate_other_users,
            gpu_no_spill,
        ]
        summary_rows.append(
            {
                "treatment": label,
                "scheduler_spec": spec,
                "user_count": len(treatment_rows),
                "training_cost_weights": treatment_rows[0].get("training_cost_weights")
                if treatment_rows
                else None,
                "mean_formal_vs_fsrs6_same_target_time_saved_auc": mean_formal_time,
                "mean_formal_vs_fsrs6_relative_time_saved_percent": (
                    mean_formal_relative
                ),
                "mean_formal_vs_fsrs6_span_coverage_percent": mean_formal_coverage,
                "min_formal_vs_fsrs6_span_coverage_percent": min_formal_coverage,
                "user2_formal_vs_fsrs6_same_target_time_saved_auc": formal_user2,
                "user2_formal_vs_fsrs6_delta_vs_baseline": formal_user2_delta,
                "user2_dense_loww_vs_fsrs6_same_target_time_saved_auc": dense_user2,
                "user2_dense_loww_vs_fsrs6_delta_vs_baseline": dense_user2_delta,
                "mean_direct_vs_exact_same_target_time_saved_auc": (mean_direct_exact),
                "mean_direct_vs_exact_delta_vs_baseline": direct_exact_delta,
                "mean_direct_vs_adr_same_target_time_saved_auc": _mean(
                    direct_adr_rows,
                    "same_target_time_saved_auc",
                ),
                "mean_coverage_delta_vs_baseline_pp": coverage_delta,
                "worst_non_user2_relative_time_saved_delta_vs_baseline_pp": (
                    max_other_user_relative_loss
                ),
                "gpu_monitor_shared_memory_spill_detected_values": ",".join(
                    sorted(_bool_label(value) for value in spill_values)
                ),
                "gate_user2_formal": gate_user2_formal if is_candidate else None,
                "gate_user2_dense_loww": gate_dense_user2 if is_candidate else None,
                "gate_direct_vs_exact": gate_direct_exact if is_candidate else None,
                "gate_mean_coverage": gate_mean_coverage if is_candidate else None,
                "gate_min_coverage": gate_min_coverage if is_candidate else None,
                "gate_other_users": gate_other_users if is_candidate else None,
                "gate_gpu_no_spill": gpu_no_spill if is_candidate else None,
                "passes_all_gates": all(gates) if is_candidate else None,
            }
        )
    return summary_rows


def main() -> int:
    args = parse_args()
    treatment_specs = [
        _parse_treatment(raw) for raw in (args.treatment or DEFAULT_TREATMENTS)
    ]
    labels = [label for label, _root in treatment_specs]
    if args.baseline_label not in labels:
        raise SystemExit(f"Missing baseline treatment: {args.baseline_label}")
    if args.candidate_label not in labels:
        raise SystemExit(f"Missing candidate treatment: {args.candidate_label}")
    segment = _parse_segment(args.segment)

    by_user_rows: list[dict[str, Any]] = []
    for label, root in treatment_specs:
        by_user_rows.extend(_load_train_rows(label, root))

    formal_regret_rows = _read_csv(args.formal_exact_dir / "regret_auc.csv")
    formal_result_rows = _read_csv(args.formal_exact_dir / "results.csv")
    dense_regret_rows = _read_csv(args.dense_loww_exact_dir / "regret_auc.csv")
    dense_result_rows = _read_csv(args.dense_loww_exact_dir / "results.csv")

    direct_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    for label in labels:
        spec = _distill_scheduler_spec(label)
        direct_rows.extend(
            _annotate_pair_rows(
                _regret_rows_for_pair(
                    formal_regret_rows,
                    baseline_scheduler=EXACT_SCHEDULER,
                    scheduler=spec,
                ),
                treatment=label,
                pair="treatment_vs_exact",
                source="formal_exact_value",
            )
        )
        direct_rows.extend(
            _annotate_pair_rows(
                _regret_rows_for_pair(
                    formal_regret_rows,
                    baseline_scheduler=ADR_SCHEDULER,
                    scheduler=spec,
                ),
                treatment=label,
                pair="treatment_vs_adr",
                source="formal_exact_value",
            )
        )
        for baseline in (FSRS6_SCHEDULER, EXACT_SCHEDULER, ADR_SCHEDULER):
            dense_rows.extend(
                _annotate_pair_rows(
                    _regret_rows_for_pair(
                        dense_regret_rows,
                        baseline_scheduler=baseline,
                        scheduler=spec,
                        user_id=2,
                    ),
                    treatment=label,
                    pair=f"treatment_vs_{_display_scheduler(label, baseline)}",
                    source="dense_loww_exact_value",
                )
            )

    for scheduler in (EXACT_SCHEDULER, ADR_SCHEDULER):
        dense_rows.extend(
            _annotate_pair_rows(
                _regret_rows_for_pair(
                    dense_regret_rows,
                    baseline_scheduler=FSRS6_SCHEDULER,
                    scheduler=scheduler,
                    user_id=2,
                ),
                treatment=_display_scheduler("", scheduler),
                pair=f"{_display_scheduler('', scheduler)}_vs_fsrs6",
                source="dense_loww_exact_value",
            )
        )

    segment_rows: list[dict[str, Any]] = []
    segment_schedulers = {
        FSRS6_SCHEDULER: "fsrs6",
        EXACT_SCHEDULER: "exact",
        ADR_SCHEDULER: "adr",
        **{_distill_scheduler_spec(label): label for label in labels},
    }
    for baseline in (FSRS6_SCHEDULER, EXACT_SCHEDULER, ADR_SCHEDULER):
        for scheduler in segment_schedulers:
            if scheduler == baseline:
                continue
            row = _segment_auc(
                dense_result_rows,
                user_id=2,
                baseline_scheduler=baseline,
                scheduler=scheduler,
                segment_start=segment[0],
                segment_end=segment[1],
            )
            row["baseline_label"] = segment_schedulers.get(baseline, baseline)
            row["scheduler_label"] = segment_schedulers.get(scheduler, scheduler)
            segment_rows.append(row)

    summary_rows = _build_summary_rows(
        labels=labels,
        by_user_rows=by_user_rows,
        formal_regret_rows=formal_regret_rows,
        dense_regret_rows=dense_regret_rows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "train_weight_by_user.csv", by_user_rows)
    _write_csv(args.out_dir / "train_weight_summary.csv", summary_rows)
    _write_csv(args.out_dir / "train_weight_direct_vs_exact.csv", direct_rows)
    _write_csv(args.out_dir / "train_weight_user2_dense_loww.csv", dense_rows)
    _write_csv(args.out_dir / "train_weight_user2_segment_auc.csv", segment_rows)

    plot_labels = {
        FSRS6_SCHEDULER: "FSRS6",
        EXACT_SCHEDULER: "Exact",
        _distill_scheduler_spec(args.baseline_label): "Sparse baseline",
        _distill_scheduler_spec(args.candidate_label): "Add 1,4",
        ADR_SCHEDULER: "ADR",
    }
    _write_frontier_plot(
        args.out_dir / "train_weight_user2_frontier_formal.png",
        rows=formal_result_rows,
        user_id=2,
        scheduler_labels=plot_labels,
        segment=None,
        title="User 2 Formal Sparse-Grid Frontiers",
    )
    _write_frontier_plot(
        args.out_dir / "train_weight_user2_frontier_dense_loww.png",
        rows=dense_result_rows,
        user_id=2,
        scheduler_labels=plot_labels,
        segment=segment,
        title="User 2 Dense Low-Weight Frontiers",
    )
    print(f"Wrote train-weight control outputs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

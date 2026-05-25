from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.single_card_tradeoff.core.target_search.comparison import (  # noqa: E402
    TargetAnswerRecord,
    compare_target_answers_to_oracle,
    oracle_gap_row,
    read_target_answer_records,
    resolve_target_answers_path,
    target_answer_record_row,
)
from experiments.single_card_tradeoff.core.target_search.io import (  # noqa: E402
    format_optional_float,
)

EPSILON = 1e-12
DEFAULT_TARGET_MEMORIES = "0.70,0.75,0.80,0.85,0.90,0.93,0.96"
DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/target_memory_scheduler_comparison_first8/"
    "comparison"
)
THETA_COLUMNS = (
    "desired_retention",
    "fixed_interval",
    "goal_cost_weight",
    "fsrs6_adr_lambda_value",
    "fsrs6_adr_policy_index",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate fixed target-memory scheduler answers and convert "
            "tradeoff frontier CSVs into target-answer rows."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--target-answer",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Target-answer CSV, or directory containing target_answers.csv. "
            "May be passed multiple times. NAME becomes the scheduler label."
        ),
    )
    parser.add_argument(
        "--oracle",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Oracle target-answer CSV, or directory containing target_answers.csv. "
            "The oracle is included in the combined outputs and used for gaps."
        ),
    )
    parser.add_argument(
        "--tradeoff-result",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Tradeoff combined_results.csv to convert into target-answer rows. "
            "NAME becomes the scheduler label. By default rows are filtered "
            "where scheduler or scheduler_spec equals NAME."
        ),
    )
    parser.add_argument(
        "--tradeoff-scheduler",
        action="append",
        default=[],
        metavar="NAME=SCHEDULER",
        help=(
            "Override the scheduler/scheduler_spec value used to filter a "
            "--tradeoff-result entry."
        ),
    )
    parser.add_argument(
        "--theta-column",
        action="append",
        default=[],
        metavar="NAME=COLUMN",
        help="Override theta column used when converting a tradeoff result.",
    )
    parser.add_argument(
        "--target-memories",
        default=DEFAULT_TARGET_MEMORIES,
        help="Comma-separated memory targets used for tradeoff-result conversion.",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for matching target values to the oracle.",
    )
    parser.add_argument(
        "--require-complete-target-grid",
        action="store_true",
        help="Fail if any scheduler/user is missing a requested target memory.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _parse_name_path(raw: str, *, option_name: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"{option_name} entries must have NAME=PATH form.")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"{option_name} NAME must not be empty.")
    resolved = Path(path.strip())
    if not resolved.exists():
        raise SystemExit(f"{option_name} path does not exist: {resolved}")
    return name, resolved


def _parse_name_value(raw: str, *, option_name: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"{option_name} entries must have NAME=VALUE form.")
    name, value = raw.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise SystemExit(f"{option_name} NAME and VALUE must not be empty.")
    return name, value


def _parse_csv_floats(raw: str, *, option_name: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid {option_name} value: {token!r}") from exc
        if not math.isfinite(value):
            raise SystemExit(f"{option_name} values must be finite.")
        values.append(value)
    if not values:
        raise SystemExit(f"{option_name} must contain at least one value.")
    return values


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_float(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _tradeoff_theta(
    row: Mapping[str, str],
    *,
    override_column: str | None,
) -> tuple[str | None, float | None]:
    columns = (override_column,) if override_column is not None else THETA_COLUMNS
    for column in columns:
        if column is None:
            continue
        value = _maybe_float(row.get(column))
        if value is not None:
            return column, value
    return None, None


def _tradeoff_row_matches(
    row: Mapping[str, str],
    *,
    label: str,
    scheduler_filter: str | None,
) -> bool:
    scheduler = scheduler_filter or label
    return row.get("scheduler") == scheduler or row.get("scheduler_spec") == scheduler


def _target_record_with_label(
    record: TargetAnswerRecord,
    *,
    label: str,
) -> TargetAnswerRecord:
    return replace(record, family=label)


def read_labeled_target_answers(label: str, path: Path) -> list[TargetAnswerRecord]:
    records = read_target_answer_records(resolve_target_answers_path(path))
    return [_target_record_with_label(record, label=label) for record in records]


def convert_tradeoff_results(
    *,
    label: str,
    path: Path,
    target_memories: Sequence[float],
    scheduler_filter: str | None,
    theta_column: str | None,
) -> list[TargetAnswerRecord]:
    rows = [
        row
        for row in _read_csv_rows(path)
        if _tradeoff_row_matches(row, label=label, scheduler_filter=scheduler_filter)
    ]
    if not rows:
        scheduler = scheduler_filter or label
        raise SystemExit(f"No rows in {path} matched scheduler {scheduler!r}.")

    rows_by_user: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        user_id_raw = row.get("user_id")
        memory = _maybe_float(row.get("card_expected_retrievability"))
        minutes = _maybe_float(row.get("card_minutes_per_day"))
        if user_id_raw is None or memory is None or minutes is None:
            continue
        rows_by_user[int(user_id_raw)].append(row)

    records: list[TargetAnswerRecord] = []
    for user_id, user_rows in sorted(rows_by_user.items()):
        for target in target_memories:
            feasible_rows = [
                row
                for row in user_rows
                if float(row["card_expected_retrievability"]) >= target - EPSILON
            ]
            if feasible_rows:
                selected = min(
                    feasible_rows,
                    key=lambda row: (
                        float(row["card_minutes_per_day"]),
                        -float(row["card_expected_retrievability"]),
                    ),
                )
                feasible = True
            else:
                selected = max(
                    user_rows,
                    key=lambda row: (
                        float(row["card_expected_retrievability"]),
                        -float(row["card_minutes_per_day"]),
                    ),
                )
                feasible = False

            achieved_memory = float(selected["card_expected_retrievability"])
            achieved_minutes = float(selected["card_minutes_per_day"])
            theta_name, theta_value = _tradeoff_theta(
                selected,
                override_column=theta_column,
            )
            records.append(
                TargetAnswerRecord(
                    user_id=user_id,
                    target_type="memory",
                    target_value=float(target),
                    family=label,
                    feasible=feasible,
                    theta_name=theta_name,
                    theta_value=theta_value,
                    achieved_memory=achieved_memory,
                    achieved_minutes=achieved_minutes,
                    memory_slack=achieved_memory - float(target),
                    time_slack=None,
                    certified=False,
                    policy_ref=str(path),
                    cache_key=None,
                    mixed_available=False,
                    mixed_probability_high=None,
                    mixed_memory=None,
                    mixed_minutes=None,
                )
            )
    return records


def _record_key(record: TargetAnswerRecord) -> tuple[int, str, float]:
    return (record.user_id, record.target_type, round(record.target_value, 12))


def _oracle_by_key(
    records: Sequence[TargetAnswerRecord],
) -> dict[tuple[int, str, float], TargetAnswerRecord]:
    return {_record_key(record): record for record in records}


def _extra_vs_oracle(
    record: TargetAnswerRecord,
    oracle: TargetAnswerRecord | None,
) -> float | None:
    if oracle is None or not record.feasible or not oracle.feasible:
        return None
    if record.target_type == "memory":
        if record.achieved_minutes is None or oracle.achieved_minutes is None:
            return None
        return record.achieved_minutes - oracle.achieved_minutes
    if record.achieved_memory is None or oracle.achieved_memory is None:
        return None
    return oracle.achieved_memory - record.achieved_memory


def _combined_rows(
    records: Sequence[TargetAnswerRecord],
    *,
    oracle_records: Sequence[TargetAnswerRecord],
) -> list[dict[str, Any]]:
    oracle_lookup = _oracle_by_key(oracle_records)
    rows: list[dict[str, Any]] = []
    for record in records:
        oracle = oracle_lookup.get(_record_key(record))
        row = target_answer_record_row(record)
        row["extra_objective_vs_oracle"] = format_optional_float(
            _extra_vs_oracle(record, oracle)
        )
        rows.append(row)
    return rows


def _matrix_rows(
    records: Sequence[TargetAnswerRecord],
    *,
    oracle_records: Sequence[TargetAnswerRecord],
) -> list[dict[str, Any]]:
    families = sorted({record.family for record in records})
    by_target: dict[tuple[int, str, float], dict[str, TargetAnswerRecord]] = (
        defaultdict(dict)
    )
    for record in records:
        by_target[_record_key(record)][record.family] = record
    oracle_lookup = _oracle_by_key(oracle_records)
    rows: list[dict[str, Any]] = []
    for (user_id, target_type, target_value), family_records in sorted(
        by_target.items()
    ):
        row: dict[str, Any] = {
            "user_id": user_id,
            "target_type": target_type,
            "target_value": format_optional_float(target_value),
        }
        oracle = oracle_lookup.get((user_id, target_type, target_value))
        for family in families:
            record = family_records.get(family)
            prefix = family
            if record is None:
                row[f"{prefix}_feasible"] = ""
                row[f"{prefix}_theta"] = ""
                row[f"{prefix}_M"] = ""
                row[f"{prefix}_T"] = ""
                row[f"{prefix}_memory_slack"] = ""
                row[f"{prefix}_extra_T_vs_oracle"] = ""
                continue
            row[f"{prefix}_feasible"] = record.feasible
            row[f"{prefix}_theta"] = format_optional_float(record.theta_value)
            row[f"{prefix}_M"] = format_optional_float(record.achieved_memory)
            row[f"{prefix}_T"] = format_optional_float(record.achieved_minutes)
            row[f"{prefix}_memory_slack"] = format_optional_float(record.memory_slack)
            row[f"{prefix}_extra_T_vs_oracle"] = format_optional_float(
                _extra_vs_oracle(record, oracle)
            )
        rows.append(row)
    return rows


def _finite(values: Sequence[float | None]) -> list[float]:
    return [value for value in values if value is not None and math.isfinite(value)]


def _mean(values: Sequence[float | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return sum(finite) / float(len(finite))


def _median(values: Sequence[float | None]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    return statistics.median(finite)


def _p90(values: Sequence[float | None]) -> float | None:
    finite = sorted(_finite(values))
    if not finite:
        return None
    index = min(len(finite) - 1, math.ceil(0.9 * len(finite)) - 1)
    return finite[index]


def _scheduler_summary_rows(
    records: Sequence[TargetAnswerRecord],
    *,
    oracle_records: Sequence[TargetAnswerRecord],
) -> list[dict[str, Any]]:
    oracle_lookup = _oracle_by_key(oracle_records)
    by_family: dict[str, list[TargetAnswerRecord]] = defaultdict(list)
    for record in records:
        by_family[record.family].append(record)
    rows: list[dict[str, Any]] = []
    for family, family_records in sorted(by_family.items()):
        extras = [
            _extra_vs_oracle(record, oracle_lookup.get(_record_key(record)))
            for record in family_records
        ]
        feasible = [record for record in family_records if record.feasible]
        finite_extras = _finite(extras)
        rows.append(
            {
                "scheduler": family,
                "target_count": len(family_records),
                "feasible_count": len(feasible),
                "coverage": format_optional_float(
                    len(feasible) / float(len(family_records))
                    if family_records
                    else None
                ),
                "mean_achieved_T": format_optional_float(
                    _mean([record.achieved_minutes for record in feasible])
                ),
                "mean_memory_slack": format_optional_float(
                    _mean([record.memory_slack for record in feasible])
                ),
                "mean_extra_T_vs_oracle": format_optional_float(_mean(extras)),
                "median_extra_T_vs_oracle": format_optional_float(_median(extras)),
                "p90_extra_T_vs_oracle": format_optional_float(_p90(extras)),
                "worst_positive_extra_T_vs_oracle": format_optional_float(
                    max((value for value in finite_extras if value > 0.0), default=None)
                ),
                "negative_gap_count": sum(
                    1 for value in finite_extras if value < -EPSILON
                ),
            }
        )
    return rows


def _target_summary_rows(
    records: Sequence[TargetAnswerRecord],
    *,
    oracle_label: str | None,
    oracle_records: Sequence[TargetAnswerRecord],
) -> list[dict[str, Any]]:
    by_target: dict[tuple[str, float], list[TargetAnswerRecord]] = defaultdict(list)
    for record in records:
        by_target[(record.target_type, round(record.target_value, 12))].append(record)
    oracle_by_target: dict[tuple[str, float], list[TargetAnswerRecord]] = defaultdict(
        list
    )
    for record in oracle_records:
        oracle_by_target[(record.target_type, round(record.target_value, 12))].append(
            record
        )
    rows: list[dict[str, Any]] = []
    for (target_type, target_value), target_records in sorted(by_target.items()):
        non_oracle = [
            record
            for record in target_records
            if oracle_label is None or record.family != oracle_label
        ]
        feasible = [record for record in non_oracle if record.feasible]
        by_family: dict[str, list[TargetAnswerRecord]] = defaultdict(list)
        for record in feasible:
            by_family[record.family].append(record)
        family_mean_t = {
            family: _mean([record.achieved_minutes for record in values])
            for family, values in by_family.items()
        }
        best_family = None
        best_t = None
        for family, mean_t in family_mean_t.items():
            if mean_t is None:
                continue
            if best_t is None or mean_t < best_t:
                best_family = family
                best_t = mean_t
        oracle_t = _mean(
            [
                record.achieved_minutes
                for record in oracle_by_target[(target_type, target_value)]
            ]
        )
        extra = (
            best_t - oracle_t if best_t is not None and oracle_t is not None else None
        )
        rows.append(
            {
                "target_type": target_type,
                "target_value": format_optional_float(target_value),
                "best_non_oracle_scheduler": best_family or "",
                "best_non_oracle_T": format_optional_float(best_t),
                "oracle_T": format_optional_float(oracle_t),
                "best_non_oracle_extra_T": format_optional_float(extra),
                "feasible_scheduler_count": len(by_family),
                "mean_memory_slack": format_optional_float(
                    _mean([record.memory_slack for record in feasible])
                ),
            }
        )
    return rows


def _user_summary_rows(
    records: Sequence[TargetAnswerRecord],
    *,
    oracle_label: str | None,
    oracle_records: Sequence[TargetAnswerRecord],
) -> list[dict[str, Any]]:
    oracle_lookup = _oracle_by_key(oracle_records)
    by_user: dict[int, list[TargetAnswerRecord]] = defaultdict(list)
    for record in records:
        if oracle_label is not None and record.family == oracle_label:
            continue
        by_user[record.user_id].append(record)
    rows: list[dict[str, Any]] = []
    for user_id, user_records in sorted(by_user.items()):
        by_family: dict[str, list[TargetAnswerRecord]] = defaultdict(list)
        for record in user_records:
            by_family[record.family].append(record)
        family_mean_extra = {
            family: _mean(
                [
                    _extra_vs_oracle(record, oracle_lookup.get(_record_key(record)))
                    for record in family_records
                ]
            )
            for family, family_records in by_family.items()
        }
        best_family = None
        best_extra = None
        for family, mean_extra in family_mean_extra.items():
            if mean_extra is None:
                continue
            if best_extra is None or mean_extra < best_extra:
                best_family = family
                best_extra = mean_extra
        largest_record = None
        largest_extra = None
        for record in user_records:
            extra = _extra_vs_oracle(record, oracle_lookup.get(_record_key(record)))
            if extra is None:
                continue
            if largest_extra is None or extra > largest_extra:
                largest_extra = extra
                largest_record = record
        rows.append(
            {
                "user_id": user_id,
                "best_non_oracle_scheduler": best_family or "",
                "mean_extra_T_vs_oracle": format_optional_float(best_extra),
                "hardest_target": ""
                if largest_record is None
                else format_optional_float(largest_record.target_value),
                "largest_extra_T_vs_oracle": format_optional_float(largest_extra),
                "infeasible_count": sum(
                    1 for record in user_records if not record.feasible
                ),
            }
        )
    return rows


def _validate_complete_grid(
    records: Sequence[TargetAnswerRecord],
    *,
    target_memories: Sequence[float],
) -> None:
    users = sorted({record.user_id for record in records})
    families = sorted({record.family for record in records})
    expected = {round(value, 12) for value in target_memories}
    by_family_user: dict[tuple[str, int], set[float]] = defaultdict(set)
    for record in records:
        if record.target_type == "memory":
            by_family_user[(record.family, record.user_id)].add(
                round(record.target_value, 12)
            )
    missing: list[str] = []
    for family in families:
        for user_id in users:
            actual = by_family_user[(family, user_id)]
            diff = sorted(expected - actual)
            if diff:
                missing.append(
                    f"{family}/user_{user_id}:"
                    + ",".join(format_optional_float(value) for value in diff)
                )
    if missing:
        preview = "; ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f"; ... {len(missing) - 8} more"
        raise SystemExit(f"Missing target grid rows: {preview}{suffix}")


def _plot_outputs(
    *,
    records: Sequence[TargetAnswerRecord],
    oracle_records: Sequence[TargetAnswerRecord],
    oracle_label: str | None,
    out_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    oracle_lookup = _oracle_by_key(oracle_records)
    families = sorted({record.family for record in records})

    mean_t_rows: dict[str, list[tuple[float, float]]] = {}
    extra_rows: dict[str, list[tuple[float, float]]] = {}
    slack_rows: dict[str, list[float]] = {}
    feasible_counts: dict[tuple[str, float], int] = {}
    for family in families:
        family_records = [record for record in records if record.family == family]
        targets = sorted({record.target_value for record in family_records})
        mean_t_rows[family] = []
        extra_rows[family] = []
        slack_rows[family] = []
        for target in targets:
            target_records = [
                record
                for record in family_records
                if abs(record.target_value - target) <= EPSILON
            ]
            mean_t = _mean(
                [
                    record.achieved_minutes
                    for record in target_records
                    if record.feasible
                ]
            )
            extra = _mean(
                [
                    _extra_vs_oracle(record, oracle_lookup.get(_record_key(record)))
                    for record in target_records
                ]
            )
            feasible_counts[(family, target)] = sum(
                1 for record in target_records if record.feasible
            )
            if mean_t is not None:
                mean_t_rows[family].append((target, mean_t))
            if extra is not None:
                extra_rows[family].append((target, extra))
            slack_rows[family].extend(
                [
                    record.memory_slack
                    for record in target_records
                    if record.feasible and record.memory_slack is not None
                ]
            )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for family, values in mean_t_rows.items():
        if not values:
            continue
        ax.plot(
            [item[0] for item in values],
            [item[1] for item in values],
            marker="o",
            label=family,
        )
    ax.set_xlabel("Target memory M0")
    ax.set_ylabel("Mean achieved T")
    ax.set_title("Mean target time by scheduler")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = plot_dir / "mean_T_vs_target.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if oracle_label is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for family, values in extra_rows.items():
            if family == oracle_label or not values:
                continue
            ax.plot(
                [item[0] for item in values],
                [item[1] for item in values],
                marker="o",
                label=family,
            )
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_xlabel("Target memory M0")
        ax.set_ylabel("Mean extra T vs oracle")
        ax.set_title("Mean target-time gap vs oracle")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "mean_extra_T_vs_oracle.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    targets = sorted({record.target_value for record in records})
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(families))))
    heatmap = [
        [feasible_counts.get((family, target), 0) for target in targets]
        for family in families
    ]
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(targets)), [format_optional_float(value) for value in targets]
    )
    ax.set_yticks(range(len(families)), families)
    ax.set_xlabel("Target memory M0")
    ax.set_title("Feasible user count")
    fig.colorbar(image, ax=ax, label="users")
    fig.tight_layout()
    path = plot_dir / "feasibility_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    box_values = [slack_rows[family] for family in families if slack_rows[family]]
    box_labels = [family for family in families if slack_rows[family]]
    if box_values:
        ax.boxplot(box_values, tick_labels=box_labels, vert=True)
        ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylabel("Memory slack")
    ax.set_title("Memory slack distribution by scheduler")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "memory_slack_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_tolerance < 0.0 or not math.isfinite(args.target_tolerance):
        raise SystemExit("--target-tolerance must be finite and >= 0.")
    target_memories = _parse_csv_floats(
        args.target_memories,
        option_name="--target-memories",
    )
    scheduler_overrides = dict(
        _parse_name_value(raw, option_name="--tradeoff-scheduler")
        for raw in args.tradeoff_scheduler
    )
    theta_overrides = dict(
        _parse_name_value(raw, option_name="--theta-column")
        for raw in args.theta_column
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records: list[TargetAnswerRecord] = []
    source_paths: dict[str, str] = {}

    oracle_label = None
    oracle_records: list[TargetAnswerRecord] = []
    if args.oracle is not None:
        oracle_label, oracle_path = _parse_name_path(
            args.oracle, option_name="--oracle"
        )
        oracle_records = read_labeled_target_answers(oracle_label, oracle_path)
        records.extend(oracle_records)
        source_paths[oracle_label] = str(resolve_target_answers_path(oracle_path))

    for raw in args.target_answer:
        label, path = _parse_name_path(raw, option_name="--target-answer")
        if label == oracle_label:
            raise SystemExit(f"Duplicate target-answer label: {label}")
        records.extend(read_labeled_target_answers(label, path))
        source_paths[label] = str(resolve_target_answers_path(path))

    for raw in args.tradeoff_result:
        label, path = _parse_name_path(raw, option_name="--tradeoff-result")
        if label == oracle_label:
            raise SystemExit(f"Duplicate tradeoff-result label: {label}")
        records.extend(
            convert_tradeoff_results(
                label=label,
                path=path,
                target_memories=target_memories,
                scheduler_filter=scheduler_overrides.get(label),
                theta_column=theta_overrides.get(label),
            )
        )
        source_paths[label] = str(path)

    if not records:
        raise SystemExit(
            "Provide at least one --oracle, --target-answer, or --tradeoff-result."
        )
    if args.require_complete_target_grid:
        _validate_complete_grid(records, target_memories=target_memories)

    combined_path = args.out_dir / "combined_target_answers.csv"
    matrix_path = args.out_dir / "scheduler_target_matrix.csv"
    gaps_path = args.out_dir / "scheduler_oracle_gaps.csv"
    scheduler_summary_path = args.out_dir / "scheduler_summary.csv"
    user_summary_path = args.out_dir / "user_summary.csv"
    target_summary_path = args.out_dir / "target_summary.csv"
    metadata_path = args.out_dir / "metadata.json"

    _write_csv(
        combined_path,
        _combined_rows(records, oracle_records=oracle_records),
    )
    _write_csv(
        matrix_path,
        _matrix_rows(records, oracle_records=oracle_records),
    )
    gap_rows: list[dict[str, Any]] = []
    if oracle_records:
        candidates = [
            record
            for record in records
            if oracle_label is None or record.family != oracle_label
        ]
        gap_rows = [
            oracle_gap_row(gap)
            for gap in compare_target_answers_to_oracle(
                candidates,
                oracle_records,
                target_tolerance=args.target_tolerance,
            )
        ]
    _write_csv(gaps_path, gap_rows)
    _write_csv(
        scheduler_summary_path,
        _scheduler_summary_rows(records, oracle_records=oracle_records),
    )
    _write_csv(
        user_summary_path,
        _user_summary_rows(
            records,
            oracle_label=oracle_label,
            oracle_records=oracle_records,
        ),
    )
    _write_csv(
        target_summary_path,
        _target_summary_rows(
            records,
            oracle_label=oracle_label,
            oracle_records=oracle_records,
        ),
    )

    plot_paths: list[str] = []
    if not args.no_plots:
        plot_paths = _plot_outputs(
            records=records,
            oracle_records=oracle_records,
            oracle_label=oracle_label,
            out_dir=args.out_dir,
        )

    metadata = {
        "target_memories": target_memories,
        "target_tolerance": args.target_tolerance,
        "oracle_label": oracle_label,
        "scheduler_labels": sorted({record.family for record in records}),
        "record_count": len(records),
        "source_paths": source_paths,
        "outputs": {
            "combined_target_answers": str(combined_path),
            "scheduler_target_matrix": str(matrix_path),
            "scheduler_oracle_gaps": str(gaps_path),
            "scheduler_summary": str(scheduler_summary_path),
            "user_summary": str(user_summary_path),
            "target_summary": str(target_summary_path),
            "plots": plot_paths,
            "metadata": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote combined target answers: {combined_path}")
    print(f"Wrote scheduler target matrix: {matrix_path}")
    print(f"Wrote oracle gaps: {gaps_path}")
    print(f"Wrote scheduler summary: {scheduler_summary_path}")
    print(f"Wrote user summary: {user_summary_path}")
    print(f"Wrote target summary: {target_summary_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")
    print(f"Wrote metadata: {metadata_path}")
    return metadata


def main_from_args(args: argparse.Namespace) -> None:
    run_compare(args)


def main() -> None:
    main_from_args(parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.fsrs6_cost_conditioned_adr_policy import (  # noqa: E402
    FSRS6CostConditionedADRPolicy,
)


DEFAULT_ANALYSIS_ROOT = Path("artifacts/rl_scheduler/cost_adr_horizon_days_users_1_8")
DEFAULT_TRAIN365_ROOT = Path(
    "artifacts/rl_scheduler/fsrs6_cost_adr_horizon_days_users_1_8/"
    "fsrs6_cost_adr_rethead_365d_users_1_8_pop16_gen20_v1"
)
DEFAULT_TRAIN1825_ROOT = Path(
    "artifacts/rl_scheduler/fsrs6_cost_adr_rethead_structure_ablation_users_1_8/"
    "fsrs6_cost_adr_rethead_ablate_drop_sqrt_z_xd2_users_1_8_pop16_gen20_v1"
)
DEFAULT_OUT_DIR = DEFAULT_ANALYSIS_ROOT / "horizon_analysis"
DEFAULT_COST_WEIGHTS = (
    0.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    48.0,
    64.0,
    96.0,
    128.0,
    192.0,
    256.0,
    384.0,
    512.0,
    1024.0,
)


@dataclass(frozen=True, slots=True)
class CellSummary:
    cell: str
    train_days: int
    eval_days: int
    baseline_hv_sum: float
    scheduler_hv_sum: float
    hv_delta_sum: float
    hv_delta_baseline_ratio_percent: float
    relative_time_saved_auc_percent: float
    time_saved_auc_mean: float
    target_span_coverage_percent: float
    relative_memory_lift_auc_percent: float
    memory_lift_auc_mean: float
    budget_span_coverage_percent: float
    scheduler_frontier_points: int
    cost_adr_policy_point_avg_time: float
    cost_adr_policy_point_avg_memorized: float
    best_efficiency_winner_users: tuple[int, ...]
    per_user_hv_delta: dict[int, float]


@dataclass(frozen=True, slots=True)
class PolicySet:
    label: str
    train_days: int
    policies: dict[int, FSRS6CostConditionedADRPolicy]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Cost-ADR 365-day vs 1825-day train/eval performance "
            "and policy distribution differences."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--train365-run-root", type=Path, default=DEFAULT_TRAIN365_ROOT)
    parser.add_argument(
        "--train1825-run-root", type=Path, default=DEFAULT_TRAIN1825_ROOT
    )
    parser.add_argument("--users", default="1-8")
    parser.add_argument(
        "--cost-weights",
        default=",".join(format_float(value) for value in DEFAULT_COST_WEIGHTS),
    )
    parser.add_argument("--s-points", type=int, default=41)
    parser.add_argument("--d-points", type=int, default=41)
    parser.add_argument("--monotonicity-tolerance", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def parse_users(raw: str) -> tuple[int, ...]:
    value = raw.strip()
    if "-" in value and "," not in value:
        lo_raw, hi_raw = value.split("-", maxsplit=1)
        lo = int(lo_raw)
        hi = int(hi_raw)
        if hi < lo:
            raise SystemExit("--users range must be ascending.")
        return tuple(range(lo, hi + 1))
    users = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not users:
        raise SystemExit("--users must not be empty.")
    if len(set(users)) != len(users):
        raise SystemExit("--users must not contain duplicates.")
    return users


def parse_cost_weights(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise SystemExit("--cost-weights must not be empty.")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise SystemExit("--cost-weights must be finite and >= 0.")
    if len(set(values)) != len(values):
        raise SystemExit("--cost-weights must not contain duplicates.")
    return values


def format_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return raw


def required_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(f"Expected numeric field {key!r}.")
    return float(value)


def required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"Expected integer field {key!r}.")
    return value


def scheduler_entry(
    entries: Sequence[Mapping[str, Any]], scheduler: str
) -> Mapping[str, Any]:
    for entry in entries:
        if entry.get("scheduler") == scheduler:
            return entry
    raise SystemExit(f"Missing scheduler={scheduler!r} entry.")


def load_cell_summary(analysis_root: Path, cell: str) -> CellSummary:
    train_raw, eval_raw = cell.split("_")
    train_days = int(train_raw.removeprefix("train"))
    eval_days = int(eval_raw.removeprefix("eval"))
    path = analysis_root / cell / "analysis_summary.json"
    raw = load_json_object(path)
    env_summary = raw["environments"]["fsrs6"]
    hv = scheduler_entry(env_summary["primary_hypervolume_summary"], "fsrs6_cost_adr")
    time_saved = scheduler_entry(
        env_summary["same_target_time_saved_auc"], "fsrs6_cost_adr"
    )
    memory_lift = scheduler_entry(
        env_summary["same_budget_memory_lift_auc"], "fsrs6_cost_adr"
    )
    policy_points = scheduler_entry(
        env_summary["policy_point_diagnostics"], "fsrs6_cost_adr"
    )
    winners = scheduler_entry(env_summary["best_efficiency_winners"], "fsrs6_cost_adr")
    per_user_hv = {
        int(entry["user_id"]): required_float(entry, "hv_delta")
        for entry in env_summary["per_user_hypervolume"]["fsrs6_cost_adr"]
    }
    return CellSummary(
        cell=cell,
        train_days=train_days,
        eval_days=eval_days,
        baseline_hv_sum=required_float(hv, "baseline_hv_sum"),
        scheduler_hv_sum=required_float(hv, "scheduler_hv_sum"),
        hv_delta_sum=required_float(hv, "hv_delta_sum"),
        hv_delta_baseline_ratio_percent=required_float(
            hv, "hv_delta_baseline_ratio_percent"
        ),
        relative_time_saved_auc_percent=required_float(
            time_saved, "relative_same_target_time_saved_auc_percent"
        ),
        time_saved_auc_mean=required_float(
            time_saved, "same_target_time_saved_auc_mean"
        ),
        target_span_coverage_percent=required_float(
            time_saved, "span_coverage_percent"
        ),
        relative_memory_lift_auc_percent=required_float(
            memory_lift, "relative_same_budget_memory_lift_auc_percent"
        ),
        memory_lift_auc_mean=required_float(
            memory_lift, "same_budget_memory_lift_auc_mean"
        ),
        budget_span_coverage_percent=required_float(
            memory_lift, "span_coverage_percent"
        ),
        scheduler_frontier_points=required_int(hv, "scheduler_frontier_points"),
        cost_adr_policy_point_avg_time=required_float(
            policy_points, "policy_point_avg_time"
        ),
        cost_adr_policy_point_avg_memorized=required_float(
            policy_points, "policy_point_avg_memorized"
        ),
        best_efficiency_winner_users=tuple(int(user) for user in winners["users"]),
        per_user_hv_delta=per_user_hv,
    )


def policy_output_root(run_root: Path) -> Path:
    return run_root / "train-overfit" / "train_outputs"


def load_policy_set(
    *,
    label: str,
    train_days: int,
    run_root: Path,
    users: Sequence[int],
) -> PolicySet:
    root = policy_output_root(run_root)
    policies: dict[int, FSRS6CostConditionedADRPolicy] = {}
    for user_id in users:
        path = root / f"user_{user_id}" / "policy.json"
        if not path.exists():
            raise SystemExit(f"Missing policy: {path}")
        policies[int(user_id)] = FSRS6CostConditionedADRPolicy.from_json(path)
    return PolicySet(label=label, train_days=train_days, policies=policies)


def linspace(start: float, end: float, count: int) -> list[float]:
    if count < 2:
        raise SystemExit("Grid point count must be >= 2.")
    step = (end - start) / (count - 1)
    return [start + step * index for index in range(count)]


def logspace(start: float, end: float, count: int) -> list[float]:
    if start <= 0.0:
        raise SystemExit("Stability lower bound must be > 0.")
    return [
        math.exp(value) for value in linspace(math.log(start), math.log(end), count)
    ]


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    frac = position - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def describe(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "q05": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "q95": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "q05": quantile(values, 0.05),
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "q95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def bin_label(position: float) -> str:
    if position < 1.0 / 3.0:
        return "low"
    if position < 2.0 / 3.0:
        return "mid"
    return "high"


def classify_monotone(values: Sequence[float], *, tolerance: float) -> str:
    diffs = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    increasing = any(diff > tolerance for diff in diffs)
    decreasing = any(diff < -tolerance for diff in diffs)
    if increasing and decreasing:
        return "mixed"
    if increasing:
        return "increasing"
    if decreasing:
        return "decreasing"
    return "flat"


def policy_retention_distribution(
    policy_set: PolicySet,
    *,
    cost_weights: Sequence[float],
    s_points: int,
    d_points: int,
    monotonicity_tolerance: float,
) -> dict[str, Any]:
    values_by_weight: dict[str, list[float]] = {
        format_float(weight): [] for weight in cost_weights
    }
    values_by_s_bin: dict[str, list[float]] = {
        key: [] for key in ("low", "mid", "high")
    }
    values_by_d_bin: dict[str, list[float]] = {
        key: [] for key in ("low", "mid", "high")
    }
    all_values: list[float] = []
    monotonicity = {"w": Counter(), "S": Counter(), "D": Counter()}
    w_violation_steps = 0

    for policy in policy_set.policies.values():
        bounds = policy.bounds
        s_values = logspace(bounds.s_min, bounds.s_max, s_points)
        d_values = linspace(bounds.d_min, bounds.d_max, d_points)
        for s_index, stability in enumerate(s_values):
            s_bin = bin_label(s_index / (len(s_values) - 1))
            for d_index, difficulty in enumerate(d_values):
                d_bin = bin_label(d_index / (len(d_values) - 1))
                w_line = []
                for weight in cost_weights:
                    retention = policy.evaluate_retention(
                        stability,
                        difficulty,
                        cost_weight=weight,
                    )
                    w_line.append(retention)
                    values_by_weight[format_float(weight)].append(retention)
                    values_by_s_bin[s_bin].append(retention)
                    values_by_d_bin[d_bin].append(retention)
                    all_values.append(retention)
                monotonicity["w"][
                    classify_monotone(
                        w_line,
                        tolerance=monotonicity_tolerance,
                    )
                ] += 1
                for index in range(len(w_line) - 1):
                    if w_line[index + 1] - w_line[index] > monotonicity_tolerance:
                        w_violation_steps += 1
        for difficulty in d_values:
            for weight in cost_weights:
                values = [
                    policy.evaluate_retention(stability, difficulty, cost_weight=weight)
                    for stability in s_values
                ]
                monotonicity["S"][
                    classify_monotone(
                        values,
                        tolerance=monotonicity_tolerance,
                    )
                ] += 1
        for stability in s_values:
            for weight in cost_weights:
                values = [
                    policy.evaluate_retention(stability, difficulty, cost_weight=weight)
                    for difficulty in d_values
                ]
                monotonicity["D"][
                    classify_monotone(
                        values,
                        tolerance=monotonicity_tolerance,
                    )
                ] += 1

    return {
        "overall": describe(all_values),
        "by_cost_weight": {
            weight: describe(values) for weight, values in values_by_weight.items()
        },
        "by_stability_bin": {
            label: describe(values) for label, values in values_by_s_bin.items()
        },
        "by_difficulty_bin": {
            label: describe(values) for label, values in values_by_d_bin.items()
        },
        "monotonicity": {axis: dict(counter) for axis, counter in monotonicity.items()},
        "w_nonincreasing_violation_steps": w_violation_steps,
    }


def coefficient_difference(
    left: PolicySet,
    right: PolicySet,
    users: Sequence[int],
) -> dict[str, Any]:
    per_user: list[dict[str, Any]] = []
    all_abs_deltas: list[float] = []
    for user_id in users:
        left_coefficients = left.policies[user_id].coefficients
        right_coefficients = right.policies[user_id].coefficients
        if len(left_coefficients) != len(right_coefficients):
            raise SystemExit(f"Policy parameter count mismatch for user {user_id}.")
        deltas = [
            float(left_value) - float(right_value)
            for left_value, right_value in zip(
                left_coefficients,
                right_coefficients,
                strict=True,
            )
        ]
        abs_deltas = [abs(value) for value in deltas]
        all_abs_deltas.extend(abs_deltas)
        per_user.append(
            {
                "user_id": user_id,
                "l2_delta": math.sqrt(sum(value * value for value in deltas)),
                "mean_abs_delta": statistics.fmean(abs_deltas),
                "max_abs_delta": max(abs_deltas),
            }
        )
    return {
        "left_minus_right": f"{left.label} - {right.label}",
        "per_user": per_user,
        "overall_abs_delta": describe(all_abs_deltas),
    }


def distribution_delta(
    left_distribution: Mapping[str, Any],
    right_distribution: Mapping[str, Any],
) -> dict[str, Any]:
    by_weight: dict[str, dict[str, float]] = {}
    for weight, left_stats in left_distribution["by_cost_weight"].items():
        right_stats = right_distribution["by_cost_weight"][weight]
        by_weight[weight] = {
            "mean_delta": left_stats["mean"] - right_stats["mean"],
            "median_delta": left_stats["median"] - right_stats["median"],
            "q05_delta": left_stats["q05"] - right_stats["q05"],
            "q95_delta": left_stats["q95"] - right_stats["q95"],
        }
    return {
        "left_minus_right": "train365 - train1825",
        "overall_mean_delta": (
            left_distribution["overall"]["mean"] - right_distribution["overall"]["mean"]
        ),
        "overall_median_delta": (
            left_distribution["overall"]["median"]
            - right_distribution["overall"]["median"]
        ),
        "by_cost_weight": by_weight,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def percent(value: float) -> str:
    return f"{value:.3f}%"


def number(value: float) -> str:
    return f"{value:,.2f}"


def write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    cells = summary["performance_matrix"]
    rows = []
    for cell in cells:
        rows.append(
            [
                str(cell["train_days"]),
                str(cell["eval_days"]),
                number(cell["hv_delta_sum"]),
                percent(cell["hv_delta_baseline_ratio_percent"]),
                percent(cell["relative_time_saved_auc_percent"]),
                percent(cell["target_span_coverage_percent"]),
                percent(cell["relative_memory_lift_auc_percent"]),
                percent(cell["budget_span_coverage_percent"]),
            ]
        )
    lines = [
        "# FSRS6 Cost-ADR Horizon Analysis",
        "",
        markdown_table(
            [
                "train days",
                "eval days",
                "HV delta",
                "HV / baseline",
                "relative time-save AUC",
                "target coverage",
                "relative memory-lift AUC",
                "budget coverage",
            ],
            rows,
        ),
        "",
        "## Retention Distribution",
        "",
    ]
    for label, distribution in summary["policy_distributions"].items():
        overall = distribution["overall"]
        lines.append(
            f"- {label}: mean={overall['mean']:.4f}, "
            f"median={overall['median']:.4f}, q05={overall['q05']:.4f}, "
            f"q95={overall['q95']:.4f}"
        )
    lines.extend(["", "## Retention Delta By Cost Weight", ""])
    delta_rows = []
    for weight, stats in summary["retention_delta"]["by_cost_weight"].items():
        delta_rows.append(
            [
                weight,
                f"{stats['mean_delta']:+.4f}",
                f"{stats['median_delta']:+.4f}",
                f"{stats['q05_delta']:+.4f}",
                f"{stats['q95_delta']:+.4f}",
            ]
        )
    lines.append(
        markdown_table(
            ["cost weight", "mean delta", "median delta", "q05 delta", "q95 delta"],
            delta_rows,
        )
    )
    lines.extend(["", "## Monotonicity", ""])
    mono_rows = []
    for label, distribution in summary["policy_distributions"].items():
        for axis, counts in distribution["monotonicity"].items():
            total = sum(counts.values())
            mixed = counts.get("mixed", 0)
            mono_rows.append(
                [
                    label,
                    axis,
                    str(counts),
                    percent(100.0 * mixed / total if total else 0.0),
                    str(distribution["w_nonincreasing_violation_steps"])
                    if axis == "w"
                    else "",
                ]
            )
    lines.append(
        markdown_table(
            ["policy", "axis", "counts", "mixed share", "w violation steps"],
            mono_rows,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    users = parse_users(args.users)
    cost_weights = parse_cost_weights(args.cost_weights)
    if args.s_points < 2 or args.d_points < 2:
        raise SystemExit("--s-points and --d-points must be >= 2.")
    if args.monotonicity_tolerance < 0.0:
        raise SystemExit("--monotonicity-tolerance must be >= 0.")

    cells = [
        load_cell_summary(args.analysis_root, cell)
        for cell in (
            "train365_eval365",
            "train365_eval1825",
            "train1825_eval365",
            "train1825_eval1825",
        )
    ]
    train365 = load_policy_set(
        label="train365",
        train_days=365,
        run_root=args.train365_run_root,
        users=users,
    )
    train1825 = load_policy_set(
        label="train1825",
        train_days=1825,
        run_root=args.train1825_run_root,
        users=users,
    )
    distribution365 = policy_retention_distribution(
        train365,
        cost_weights=cost_weights,
        s_points=args.s_points,
        d_points=args.d_points,
        monotonicity_tolerance=args.monotonicity_tolerance,
    )
    distribution1825 = policy_retention_distribution(
        train1825,
        cost_weights=cost_weights,
        s_points=args.s_points,
        d_points=args.d_points,
        monotonicity_tolerance=args.monotonicity_tolerance,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "users": list(users),
        "cost_weights": list(cost_weights),
        "grid": {
            "s_points": args.s_points,
            "d_points": args.d_points,
            "monotonicity_tolerance": args.monotonicity_tolerance,
        },
        "sources": {
            "analysis_root": str(args.analysis_root),
            "train365_run_root": str(args.train365_run_root),
            "train1825_run_root": str(args.train1825_run_root),
        },
        "performance_matrix": [
            {
                "cell": cell.cell,
                "train_days": cell.train_days,
                "eval_days": cell.eval_days,
                "baseline_hv_sum": cell.baseline_hv_sum,
                "scheduler_hv_sum": cell.scheduler_hv_sum,
                "hv_delta_sum": cell.hv_delta_sum,
                "hv_delta_baseline_ratio_percent": (
                    cell.hv_delta_baseline_ratio_percent
                ),
                "relative_time_saved_auc_percent": (
                    cell.relative_time_saved_auc_percent
                ),
                "time_saved_auc_mean": cell.time_saved_auc_mean,
                "target_span_coverage_percent": cell.target_span_coverage_percent,
                "relative_memory_lift_auc_percent": (
                    cell.relative_memory_lift_auc_percent
                ),
                "memory_lift_auc_mean": cell.memory_lift_auc_mean,
                "budget_span_coverage_percent": cell.budget_span_coverage_percent,
                "scheduler_frontier_points": cell.scheduler_frontier_points,
                "cost_adr_policy_point_avg_time": (cell.cost_adr_policy_point_avg_time),
                "cost_adr_policy_point_avg_memorized": (
                    cell.cost_adr_policy_point_avg_memorized
                ),
                "best_efficiency_winner_users": list(cell.best_efficiency_winner_users),
                "per_user_hv_delta": cell.per_user_hv_delta,
            }
            for cell in cells
        ],
        "policy_distributions": {
            "train365": distribution365,
            "train1825": distribution1825,
        },
        "retention_delta": distribution_delta(distribution365, distribution1825),
        "coefficient_delta": coefficient_difference(train365, train1825, users),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "horizon_analysis_summary.json"
    markdown_path = args.out_dir / "horizon_analysis_summary.md"
    write_json(json_path, summary)
    write_markdown(markdown_path, summary)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

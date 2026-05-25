from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


DEFAULT_RESULTS = Path(
    "artifacts/single_card_tradeoff/"
    "oracle_interval_grid_stationary_first8_eval_weights_add_025_05_markov_off/"
    "combined_results.csv"
)
DEFAULT_OUT_DIR = Path("artifacts/single_card_tradeoff/fsrs6_implied_cost_weight")
DEFAULT_SCHEDULER = "fsrs6"
EPSILON = 1e-12


@dataclass(frozen=True)
class PolicyPoint:
    group_key: tuple[str, ...]
    user_id: str
    environment: str
    review_markov_transition: str
    seed: str
    days: str
    particles: str
    scheduler: str
    desired_retention: float
    memory: float
    minutes: float
    source_row_index: int


@dataclass(frozen=True)
class ImpliedLambdaResult:
    point: PolicyPoint
    supported: bool
    dominated: bool
    lambda_min: float
    lambda_max: float
    lambda_representative: float
    best_fit_lambda: float
    best_fit_regret: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer rollout-level scalarization weights for FSRS6 desired-retention "
            "points within a single scheduler frontier."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Single-card tradeoff results.csv or combined_results.csv.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER,
        help="Scheduler name to analyze. First supported use is fsrs6.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(row: Mapping[str, str], key: str, *, row_index: int) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        raise ValueError(f"Missing {key!r} at row {row_index}.")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} at row {row_index}: {raw!r}.")
    return value


def _group_key(row: Mapping[str, str], scheduler: str) -> tuple[str, ...]:
    return (
        row.get("user_id") or "1",
        row.get("environment") or "",
        row.get("review_markov_transition") or "",
        row.get("seed") or "",
        row.get("days") or "",
        row.get("particles") or "",
        scheduler,
    )


def load_scheduler_points(
    rows: Sequence[Mapping[str, str]],
    *,
    scheduler: str,
) -> list[PolicyPoint]:
    points: list[PolicyPoint] = []
    for row_index, row in enumerate(rows, start=2):
        if row.get("scheduler") != scheduler:
            continue
        desired_raw = row.get("desired_retention")
        if desired_raw is None or desired_raw == "":
            continue
        desired_retention = float(desired_raw)
        if not math.isfinite(desired_retention):
            raise ValueError(
                f"Non-finite desired_retention at row {row_index}: {desired_raw!r}."
            )
        memory = _float(row, "card_expected_retrievability", row_index=row_index)
        minutes = _float(row, "card_minutes_per_day", row_index=row_index)
        if minutes < 0.0:
            raise ValueError(
                f"card_minutes_per_day must be non-negative at row {row_index}."
            )
        group_key = _group_key(row, scheduler)
        points.append(
            PolicyPoint(
                group_key=group_key,
                user_id=group_key[0],
                environment=group_key[1],
                review_markov_transition=group_key[2],
                seed=group_key[3],
                days=group_key[4],
                particles=group_key[5],
                scheduler=scheduler,
                desired_retention=desired_retention,
                memory=memory,
                minutes=minutes,
                source_row_index=row_index,
            )
        )
    return points


def _dominates(left: PolicyPoint, right: PolicyPoint) -> bool:
    no_worse = (
        left.memory >= right.memory - EPSILON
        and left.minutes <= right.minutes + EPSILON
    )
    strictly_better = (
        left.memory > right.memory + EPSILON or left.minutes < right.minutes - EPSILON
    )
    return no_worse and strictly_better


def _best_fit_lambda(
    point: PolicyPoint,
    candidates: Sequence[PolicyPoint],
) -> tuple[float, float]:
    lambdas = {0.0}
    for left in candidates:
        for right in candidates:
            delta_time = left.minutes - right.minutes
            if abs(delta_time) <= EPSILON:
                continue
            crossing = (left.memory - right.memory) / delta_time
            if crossing >= 0.0 and math.isfinite(crossing):
                lambdas.add(crossing)

    best_lambda = 0.0
    best_regret = math.inf
    for value in sorted(lambdas):
        point_score = point.memory - value * point.minutes
        best_score = max(
            candidate.memory - value * candidate.minutes for candidate in candidates
        )
        regret = max(0.0, best_score - point_score)
        if regret < best_regret - EPSILON:
            best_regret = regret
            best_lambda = value
    return best_lambda, best_regret


def _representative_lambda(lambda_min: float, lambda_max: float) -> float:
    if math.isinf(lambda_max):
        return lambda_min
    if lambda_min > 0.0 and lambda_max > 0.0:
        return math.sqrt(lambda_min * lambda_max)
    return max(0.0, lambda_max / 2.0)


def implied_lambda_for_point(
    point: PolicyPoint,
    candidates: Sequence[PolicyPoint],
) -> ImpliedLambdaResult:
    lower = 0.0
    upper = math.inf
    feasible = True

    for other in candidates:
        if other is point:
            continue
        memory_delta = point.memory - other.memory
        minutes_delta = point.minutes - other.minutes
        if abs(minutes_delta) <= EPSILON:
            if memory_delta < -EPSILON:
                feasible = False
                break
            continue
        bound = memory_delta / minutes_delta
        if minutes_delta > 0.0:
            upper = min(upper, bound)
        else:
            lower = max(lower, bound)

    lower = max(0.0, lower)
    supported = feasible and upper >= -EPSILON and lower <= upper + EPSILON
    if supported:
        upper = max(upper, lower)
        representative = _representative_lambda(lower, upper)
        best_lambda = representative
        best_regret = 0.0
    else:
        representative = math.nan
        best_lambda, best_regret = _best_fit_lambda(point, candidates)

    return ImpliedLambdaResult(
        point=point,
        supported=supported,
        dominated=any(
            _dominates(other, point) for other in candidates if other is not point
        ),
        lambda_min=lower if supported else math.nan,
        lambda_max=upper if supported else math.nan,
        lambda_representative=representative,
        best_fit_lambda=best_lambda,
        best_fit_regret=best_regret,
    )


def infer_implied_lambdas(points: Sequence[PolicyPoint]) -> list[ImpliedLambdaResult]:
    by_group: dict[tuple[str, ...], list[PolicyPoint]] = defaultdict(list)
    for point in points:
        by_group[point.group_key].append(point)

    results: list[ImpliedLambdaResult] = []
    for group_points in by_group.values():
        seen_retentions: set[float] = set()
        for point in group_points:
            if point.desired_retention in seen_retentions:
                raise ValueError(
                    "Duplicate desired_retention within inverse-lambda group: "
                    f"group={point.group_key} retention={point.desired_retention:g}"
                )
            seen_retentions.add(point.desired_retention)
        ordered = sorted(group_points, key=lambda point: point.desired_retention)
        for point in ordered:
            results.append(implied_lambda_for_point(point, ordered))
    return sorted(
        results,
        key=lambda result: (
            result.point.user_id,
            result.point.environment,
            result.point.review_markov_transition,
            result.point.seed,
            result.point.desired_retention,
        ),
    )


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.12g}"


def point_result_rows(results: Sequence[ImpliedLambdaResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        point = result.point
        rows.append(
            {
                "user_id": point.user_id,
                "environment": point.environment,
                "review_markov_transition": point.review_markov_transition,
                "seed": point.seed,
                "days": point.days,
                "particles": point.particles,
                "scheduler": point.scheduler,
                "desired_retention": _format_float(point.desired_retention),
                "card_expected_retrievability": _format_float(point.memory),
                "card_minutes_per_day": _format_float(point.minutes),
                "supported": result.supported,
                "dominated": result.dominated,
                "lambda_min": _format_float(result.lambda_min),
                "lambda_max": _format_float(result.lambda_max),
                "lambda_representative": _format_float(result.lambda_representative),
                "best_fit_lambda": _format_float(result.best_fit_lambda),
                "best_fit_regret": _format_float(result.best_fit_regret),
                "source_row_index": point.source_row_index,
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else math.nan


def _median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def retention_summary_rows(
    results: Sequence[ImpliedLambdaResult],
) -> list[dict[str, Any]]:
    by_retention: dict[float, list[ImpliedLambdaResult]] = defaultdict(list)
    for result in results:
        by_retention[result.point.desired_retention].append(result)

    rows: list[dict[str, Any]] = []
    for retention, retention_results in sorted(by_retention.items()):
        supported = [result for result in retention_results if result.supported]
        representative = [result.lambda_representative for result in supported]
        best_fit = [result.best_fit_lambda for result in retention_results]
        regret = [result.best_fit_regret for result in retention_results]
        rows.append(
            {
                "desired_retention": _format_float(retention),
                "point_count": len(retention_results),
                "supported_count": len(supported),
                "dominated_count": sum(
                    1 for result in retention_results if result.dominated
                ),
                "mean_lambda_representative": _format_float(_mean(representative)),
                "median_lambda_representative": _format_float(_median(representative)),
                "mean_best_fit_lambda": _format_float(_mean(best_fit)),
                "median_best_fit_lambda": _format_float(_median(best_fit)),
                "mean_best_fit_regret": _format_float(_mean(regret)),
                "median_best_fit_regret": _format_float(_median(regret)),
            }
        )
    return rows


def _finite_positive(values: Iterable[float]) -> list[float]:
    return sorted(value for value in values if math.isfinite(value) and value > 0.0)


def write_plot(path: Path, results: Sequence[ImpliedLambdaResult]) -> None:
    import matplotlib.pyplot as plt

    supported = [result for result in results if result.supported]
    unsupported = [result for result in results if not result.supported]
    positive_values = _finite_positive(
        [
            *(result.lambda_min for result in supported),
            *(result.lambda_max for result in supported),
            *(result.lambda_representative for result in supported),
            *(result.best_fit_lambda for result in unsupported),
        ]
    )
    floor = positive_values[0] / 10.0 if positive_values else 1e-6
    ceiling = positive_values[-1] * 10.0 if positive_values else 1.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for result in supported:
        x = result.point.desired_retention
        y = max(result.lambda_representative, floor)
        ymin = max(result.lambda_min, floor)
        ymax = (
            ceiling if math.isinf(result.lambda_max) else max(result.lambda_max, floor)
        )
        ax.vlines(x, ymin, ymax, color="#2563eb", alpha=0.35, linewidth=2.0)
        ax.scatter(x, y, color="#2563eb", s=28, zorder=3)
    for result in unsupported:
        ax.scatter(
            result.point.desired_retention,
            max(result.best_fit_lambda, floor),
            facecolors="none",
            edgecolors="#dc2626" if result.dominated else "#f97316",
            s=38,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor, top=ceiling)
    ax.set_xlabel("Desired retention")
    ax.set_ylabel("Implied cost weight lambda")
    ax.set_title("FSRS6 desired-retention implied cost weights")
    ax.grid(True, which="both", alpha=0.25)
    ax.text(
        0.01,
        0.01,
        "Filled: supported interval; hollow: best-fit only",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.scheduler != DEFAULT_SCHEDULER:
        raise SystemExit(
            "First-phase implied cost-weight analysis supports only fsrs6."
        )
    rows = _read_csv(args.results)
    points = load_scheduler_points(rows, scheduler=args.scheduler)
    if not points:
        raise SystemExit(
            f"No {args.scheduler} desired-retention rows found in {args.results}."
        )

    results = infer_implied_lambdas(points)
    point_path = args.out_dir / "implied_lambda_by_point.csv"
    summary_path = args.out_dir / "implied_lambda_by_retention.csv"
    _write_csv(point_path, point_result_rows(results))
    _write_csv(summary_path, retention_summary_rows(results))
    print(f"Wrote point CSV: {point_path}")
    print(f"Wrote retention summary CSV: {summary_path}")
    if not args.no_plot:
        plot_path = args.out_dir / "fsrs6_implied_lambda.png"
        write_plot(plot_path, results)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()

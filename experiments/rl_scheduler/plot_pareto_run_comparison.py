from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True, slots=True)
class RunSeries:
    label: str
    results_dir: Path


@dataclass(frozen=True, slots=True)
class Point:
    user_id: int
    x: float
    y: float
    title: str
    run_id: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-user Pareto comparisons from build-pareto result JSON files. "
            "Each --series value is LABEL=RUN_ROOT or LABEL=BUILD_PARETO_OUTPUTS."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help=(
            "Series to plot, formatted as LABEL=PATH. PATH may be a formal run "
            "root or a build-pareto/build_pareto_outputs directory."
        ),
    )
    parser.add_argument(
        "--env",
        default="lstm",
        help="Environment to plot from the result JSON files. Default: lstm.",
    )
    parser.add_argument(
        "--scheduler",
        default="fsrs6_adr",
        help="Scheduler to compare across series. Default: fsrs6_adr.",
    )
    parser.add_argument(
        "--users",
        default=None,
        help="Comma-separated user ids. Default: discover all users in series.",
    )
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--include-baseline",
        dest="include_baseline",
        action="store_true",
        default=True,
        help=(
            "Plot --baseline-scheduler from the first series as a gray reference. "
            "This is the default."
        ),
    )
    baseline_group.add_argument(
        "--no-baseline",
        dest="include_baseline",
        action="store_false",
        help="Do not plot the baseline scheduler reference.",
    )
    parser.add_argument(
        "--baseline-scheduler",
        default="fsrs6",
        help="Baseline scheduler used with --include-baseline. Default: fsrs6.",
    )
    parser.add_argument(
        "--x-field",
        default="memorized_average",
        help="Numeric x-axis field. Default: memorized_average.",
    )
    parser.add_argument(
        "--y-field",
        default="time_average",
        help="Numeric y-axis field. Default: time_average.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments")
        / "rl_scheduler"
        / "plots"
        / "pareto_run_comparison",
        help="Directory for output PNG files.",
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Optional plot title prefix. Default describes env and scheduler.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution. Default: 160.",
    )
    return parser.parse_args(argv)


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise SystemExit("--users must contain at least one user id when provided.")
    return values


def _parse_series(raw_values: Sequence[str]) -> list[RunSeries]:
    series: list[RunSeries] = []
    for raw in raw_values:
        if "=" not in raw:
            raise SystemExit(f"--series must be formatted LABEL=PATH, got {raw!r}.")
        label, path_text = raw.split("=", 1)
        label = label.strip()
        path = Path(path_text.strip())
        if not label:
            raise SystemExit(f"--series label must be non-empty: {raw!r}.")
        if not path_text.strip():
            raise SystemExit(f"--series path must be non-empty: {raw!r}.")
        series.append(RunSeries(label=label, results_dir=_resolve_results_dir(path)))
    return series


def _resolve_results_dir(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if (resolved / "build-pareto" / "build_pareto_outputs").is_dir():
        return resolved / "build-pareto" / "build_pareto_outputs"
    if resolved.is_dir() and resolved.name == "build_pareto_outputs":
        return resolved
    if resolved.is_dir() and any(
        resolved.glob("simulation_results_retention_sweep_user_*.json")
    ):
        return resolved
    raise SystemExit(
        "Could not resolve build-pareto outputs from path: "
        f"{path}. Expected a run root or build_pareto_outputs directory."
    )


def _discover_users(series: Sequence[RunSeries]) -> tuple[int, ...]:
    users: set[int] = set()
    for item in series:
        for path in item.results_dir.glob(
            "simulation_results_retention_sweep_user_*.json"
        ):
            match = re.fullmatch(
                r"simulation_results_retention_sweep_user_(\d+)\.json",
                path.name,
            )
            if match:
                users.add(int(match.group(1)))
    if not users:
        raise SystemExit(
            "No simulation_results_retention_sweep_user_*.json files found."
        )
    return tuple(sorted(users))


def _read_user_records(results_dir: Path, user_id: int) -> list[dict[str, Any]]:
    path = results_dir / f"simulation_results_retention_sweep_user_{user_id}.json"
    if not path.exists():
        raise SystemExit(f"Missing user result JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise SystemExit(f"{path} must contain a JSON array.")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"{path} record {index} must be an object.")
        records.append(item)
    return records


def _number_field(record: dict[str, Any], field: str, source: str) -> float:
    raw = record.get(field)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise SystemExit(f"{source} field {field!r} must be numeric.")
    value = float(raw)
    if not math.isfinite(value):
        raise SystemExit(f"{source} field {field!r} must be finite.")
    return value


def _points_for(
    records: Iterable[dict[str, Any]],
    *,
    user_id: int,
    environment: str,
    scheduler: str,
    x_field: str,
    y_field: str,
) -> list[Point]:
    points: list[Point] = []
    for index, record in enumerate(records):
        if record.get("environment") != environment:
            continue
        if record.get("scheduler") != scheduler:
            continue
        source = f"user {user_id} record {index}"
        title = record.get("title")
        run_id = record.get("run_id")
        points.append(
            Point(
                user_id=user_id,
                x=_number_field(record, x_field, source),
                y=_number_field(record, y_field, source),
                title=title if isinstance(title, str) else "",
                run_id=run_id if isinstance(run_id, str) else None,
            )
        )
    return points


def _plot_ordered_points(points: Sequence[Point]) -> list[Point]:
    return sorted(points, key=lambda point: (point.x, point.y, point.title))


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return token.strip("_") or "series"


def _axis_label(field: str) -> str:
    labels = {
        "time_average": "Minutes of studying per day (average)\n(lower=better)",
        "memorized_average": "Memorized cards (average, all days)\n(higher=better)",
        "avg_accum_memorized_per_hour": "Average accumulated memorized per hour",
        "memorized_per_minute": "Memorized per minute",
        "reviews_average": "Average reviews per day",
    }
    return labels.get(field, field)


def _plot_user(
    *,
    user_id: int,
    series_points: Sequence[tuple[RunSeries, list[Point]]],
    baseline_points: list[Point],
    environment: str,
    scheduler: str,
    baseline_scheduler: str,
    x_field: str,
    y_field: str,
    title_prefix: str | None,
    out_dir: Path,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = [
        point for _series, points in series_points for point in points
    ] + baseline_points
    if not all_points:
        raise SystemExit(f"No points found for user {user_id}.")

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 9))
    if baseline_points:
        ordered_points = _plot_ordered_points(baseline_points)
        ax.scatter(
            [point.x for point in baseline_points],
            [point.y for point in baseline_points],
            s=28,
            marker="o",
            facecolors="none",
            edgecolors="#777777",
            linewidths=1.0,
            alpha=0.55,
            label=f"{baseline_scheduler} points",
        )
        ax.plot(
            [point.x for point in ordered_points],
            [point.y for point in ordered_points],
            color="#777777",
            linewidth=1.7,
            linestyle="--",
            alpha=0.85,
            label=f"{baseline_scheduler} Pareto",
        )

    for index, (series, points) in enumerate(series_points):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        ordered_points = _plot_ordered_points(points)
        ax.scatter(
            [point.x for point in points],
            [point.y for point in points],
            s=44,
            marker=marker,
            color=color,
            alpha=0.64,
            label=f"{series.label} points",
        )
        ax.plot(
            [point.x for point in ordered_points],
            [point.y for point in ordered_points],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            label=f"{series.label} Pareto",
        )

    title = title_prefix or f"{environment} {scheduler} Pareto comparison"
    ax.set_title(f"{title} - user {user_id}", fontsize=22)
    ax.set_xlabel(_axis_label(x_field), fontsize=18, color="black")
    ax.set_ylabel(_axis_label(y_field), fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, linestyle="--")
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=16)

    _apply_axis_limits(
        ax=ax,
        points=all_points,
        x_field=x_field,
        y_field=y_field,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        out_dir
        / f"user_{user_id}_{_safe_token(environment)}_{_safe_token(scheduler)}_pareto_comparison.png"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _apply_axis_limits(
    *,
    ax: Any,
    points: Sequence[Point],
    x_field: str,
    y_field: str,
) -> None:
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    if x_field == "memorized_average" and y_field == "time_average":
        min_x = min(x_values)
        max_x = max(x_values)
        max_y = max(y_values)
        x_min = 200 * math.floor(min_x / 200) if min_x else 0
        x_max = 200 * math.ceil(max_x / 200) if max_x else 1
        if x_min == x_max:
            x_max = x_min + 200
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([0, max_y * 1.03 if max_y else 1])
        return
    _pad_axis(ax.set_xlim, min(x_values), max(x_values))
    _pad_axis(ax.set_ylim, min(y_values), max(y_values))


def _pad_axis(setter: Any, lower: float, upper: float) -> None:
    if math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-9):
        pad = 1.0 if math.isclose(lower, 0.0, abs_tol=1e-9) else abs(lower) * 0.05
    else:
        pad = (upper - lower) * 0.06
    setter(lower - pad, upper + pad)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    series = _parse_series(args.series)
    users = _parse_csv_ints(args.users) or _discover_users(series)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir

    output_paths: list[Path] = []
    for user_id in users:
        series_points: list[tuple[RunSeries, list[Point]]] = []
        for item in series:
            records = _read_user_records(item.results_dir, user_id)
            points = _points_for(
                records,
                user_id=user_id,
                environment=args.env,
                scheduler=args.scheduler,
                x_field=args.x_field,
                y_field=args.y_field,
            )
            if not points:
                raise SystemExit(
                    f"No {args.scheduler!r} records for user {user_id}, "
                    f"env {args.env!r}, series {item.label!r}."
                )
            series_points.append((item, points))

        baseline_points: list[Point] = []
        if args.include_baseline:
            baseline_records = _read_user_records(series[0].results_dir, user_id)
            baseline_points = _points_for(
                baseline_records,
                user_id=user_id,
                environment=args.env,
                scheduler=args.baseline_scheduler,
                x_field=args.x_field,
                y_field=args.y_field,
            )
            if not baseline_points:
                raise SystemExit(
                    f"No baseline {args.baseline_scheduler!r} records for user "
                    f"{user_id}, env {args.env!r}."
                )

        output_paths.append(
            _plot_user(
                user_id=user_id,
                series_points=series_points,
                baseline_points=baseline_points,
                environment=args.env,
                scheduler=args.scheduler,
                baseline_scheduler=args.baseline_scheduler,
                x_field=args.x_field,
                y_field=args.y_field,
                title_prefix=args.title_prefix,
                out_dir=out_dir,
                dpi=args.dpi,
            )
        )

    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

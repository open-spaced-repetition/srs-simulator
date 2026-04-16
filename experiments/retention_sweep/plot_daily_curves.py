# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.cli_utils import LogFilenameFilter, parse_csv
from simulator.scheduler_spec import format_float


@dataclass(slots=True)
class DailyCurve:
    desired_retention: float
    csv_path: Path
    days: list[int]
    new_per_day: list[float]
    reviews_per_day: list[float]
    memorized: list[float]
    observed_retention: list[float]


def _load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                data = payload.get("data")
                if isinstance(data, dict):
                    return data
                break
    return {}


def _normalize_user_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_desired_retentions(value: str) -> list[float]:
    retentions: list[float] = []
    for item in parse_csv(value):
        try:
            retention = round(float(item), 2)
        except ValueError as exc:
            raise SystemExit(f"Invalid desired retention '{item}'.") from exc
        if not 0.0 < retention < 1.0:
            raise SystemExit(f"Desired retention must be between 0 and 1, got {item}.")
        if retention not in retentions:
            retentions.append(retention)
    if not retentions:
        raise SystemExit("At least one desired retention is required.")
    return retentions


def _iter_log_paths(log_root: Path, user_id: int) -> list[Path]:
    user_dir = log_root / f"user_{user_id}"
    if user_dir.is_dir():
        return sorted(user_dir.glob("*.jsonl"))
    return sorted(log_root.glob("*.jsonl"))


def _meta_matches(
    meta: dict[str, Any],
    *,
    user_id: int,
    env: str | None,
    sched: str | None,
    engine: str,
    short_term: str,
    short_term_source: str,
) -> bool:
    meta_user_id = _normalize_user_id(meta.get("user_id"))
    if meta_user_id != user_id:
        return False
    if env is not None and meta.get("environment") != env:
        return False
    if sched is not None and meta.get("scheduler") != sched:
        return False
    if engine != "any" and meta.get("engine") != engine:
        return False

    source_value = meta.get("short_term_source")
    if short_term == "on" and source_value in {None, "", "off"}:
        return False
    if short_term == "off" and source_value not in {None, "", "off"}:
        return False
    if short_term_source != "any" and source_value != short_term_source:
        return False
    return True


def _load_daily_curve(csv_path: Path, desired_retention: float) -> DailyCurve:
    days: list[int] = []
    new_per_day: list[float] = []
    reviews_per_day: list[float] = []
    memorized: list[float] = []
    observed_retention: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            days.append(int(float(row["day"])))
            new_per_day.append(float(row["new"]))
            reviews_per_day.append(float(row["reviews"]))
            memorized.append(float(row["memorized"]))
            raw_retention = row.get("retention", "")
            if not raw_retention or raw_retention.lower() == "nan":
                observed_retention.append(math.nan)
            else:
                observed_retention.append(float(raw_retention))

    return DailyCurve(
        desired_retention=desired_retention,
        csv_path=csv_path,
        days=days,
        new_per_day=new_per_day,
        reviews_per_day=reviews_per_day,
        memorized=memorized,
        observed_retention=observed_retention,
    )


def _select_curves(
    *,
    log_root: Path,
    user_id: int,
    env: str | None,
    sched: str | None,
    engine: str,
    short_term: str,
    short_term_source: str,
    desired_retentions: list[float],
) -> tuple[list[DailyCurve], int]:
    min_retention = min(desired_retentions)
    max_retention = max(desired_retentions)
    effective_short_term = (
        "on" if short_term == "any" and short_term_source != "any" else short_term
    )
    log_filter = LogFilenameFilter(
        envs=[env] if env else [],
        scheds=[sched] if sched else [],
        engine=engine,
        short_term=effective_short_term,
        short_term_source=short_term_source,
        start_retention=min_retention,
        end_retention=max_retention,
    )

    desired_retention_set = set(desired_retentions)
    latest_by_retention: dict[float, tuple[int, Path]] = {}
    duplicates_skipped = 0

    for path in _iter_log_paths(log_root, user_id):
        if not log_filter.matches(path.name):
            continue
        meta = _load_meta(path)
        if not _meta_matches(
            meta,
            user_id=user_id,
            env=env,
            sched=sched,
            engine=engine,
            short_term=effective_short_term,
            short_term_source=short_term_source,
        ):
            continue
        raw_retention = meta.get("desired_retention")
        if raw_retention is None:
            continue
        try:
            desired_retention = round(float(raw_retention), 2)
        except (TypeError, ValueError):
            continue
        if desired_retention not in desired_retention_set:
            continue
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            continue

        current = latest_by_retention.get(desired_retention)
        sort_key = path.stat().st_mtime_ns
        if current is None or sort_key > current[0]:
            if current is not None:
                duplicates_skipped += 1
            latest_by_retention[desired_retention] = (sort_key, csv_path)
        else:
            duplicates_skipped += 1

    missing = [
        retention
        for retention in desired_retentions
        if retention not in latest_by_retention
    ]
    if missing:
        missing_text = ", ".join(format_float(value) for value in missing)
        raise SystemExit(
            f"No matching daily CSV found for user {user_id} at desired retention(s): "
            f"{missing_text}"
        )

    curves = [
        _load_daily_curve(latest_by_retention[retention][1], retention)
        for retention in desired_retentions
    ]
    return curves, duplicates_skipped


def _default_output_path(
    *,
    plot_dir: Path,
    user_id: int,
    env: str | None,
    sched: str | None,
    engine: str,
    short_term: str,
    short_term_source: str,
    desired_retentions: list[float],
    start_day: int,
) -> Path:
    parts = [f"user_{user_id}", "daily-curves"]
    if env is not None:
        parts.append(f"env={env}")
    if sched is not None:
        parts.append(f"sched={sched}")
    if engine != "any":
        parts.append(f"engine={engine}")
    if short_term != "any":
        parts.append(f"st={short_term}")
    if short_term_source != "any":
        parts.append(f"stsrc={short_term_source}")
    parts.append("dr=" + "_".join(format_float(value) for value in desired_retentions))
    if start_day > 0:
        parts.append(f"day-start={start_day}")
    return plot_dir / ("_".join(parts) + ".png")


def _centered_rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)

    prefix_sum = [0.0]
    prefix_count = [0]
    for value in values:
        if math.isnan(value):
            prefix_sum.append(prefix_sum[-1])
            prefix_count.append(prefix_count[-1])
        else:
            prefix_sum.append(prefix_sum[-1] + value)
            prefix_count.append(prefix_count[-1] + 1)

    half_left = (window - 1) // 2
    half_right = window // 2
    smoothed: list[float] = []
    last_index = len(values) - 1
    for index in range(len(values)):
        start = max(0, index - half_left)
        end = min(last_index, index + half_right)
        total = prefix_sum[end + 1] - prefix_sum[start]
        count = prefix_count[end + 1] - prefix_count[start]
        smoothed.append(math.nan if count == 0 else total / count)
    return smoothed


def _plot_curves(
    curves: list[DailyCurve],
    *,
    user_id: int,
    env: str | None,
    sched: str | None,
    engine: str,
    short_term_source: str,
    start_day: int,
    out_path: Path,
    review_scale: str,
    retention_smoothing_window: int,
    show_plot: bool,
) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10").colors

    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        start_index = 0
        while start_index < len(curve.days) and curve.days[start_index] < start_day:
            start_index += 1
        days = curve.days[start_index:]
        smoothed_retention = _centered_rolling_mean(
            curve.observed_retention, retention_smoothing_window
        )

        label = f"DR={format_float(curve.desired_retention)}"
        axes[0].step(
            days,
            curve.new_per_day[start_index:],
            where="mid",
            color=color,
            linewidth=1.6,
            label=label,
        )
        axes[1].plot(
            days,
            curve.reviews_per_day[start_index:],
            color=color,
            linewidth=1.4,
        )
        axes[2].plot(
            days,
            curve.memorized[start_index:],
            color=color,
            linewidth=1.7,
        )
        axes[3].axhline(
            curve.desired_retention,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
        axes[3].plot(
            days,
            smoothed_retention[start_index:],
            color=color,
            linewidth=1.2,
            zorder=2,
        )

    axes[0].set_ylabel("New / day")
    axes[1].set_ylabel("Reviews / day")
    axes[2].set_ylabel("Memorized")
    axes[3].set_ylabel("Observed retention")
    axes[3].set_xlabel("Day")
    axes[3].set_ylim(0.0, 1.0)

    if review_scale == "symlog":
        axes[1].set_yscale("symlog", linthresh=10.0)
    elif review_scale == "log":
        axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=0.0)

    title_parts = [f"User {user_id} daily curves"]
    if env is not None:
        title_parts.append(f"env={env}")
    if sched is not None:
        title_parts.append(f"sched={sched}")
    if engine != "any":
        title_parts.append(f"engine={engine}")
    if short_term_source != "any":
        title_parts.append(f"short-term-source={short_term_source}")
    title_parts.append(
        "DRs=" + ", ".join(format_float(curve.desired_retention) for curve in curves)
    )
    title_parts.append(f"day>={start_day}")
    axes[0].set_title(", ".join(title_parts))
    axes[0].legend(ncol=min(4, len(curves)))
    axes[3].text(
        0.99,
        0.96,
        (
            "Dashed lines = target DR\n"
            f"Solid lines = {retention_smoothing_window}d centered mean"
        ),
        transform=axes[3].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    for axis in axes:
        axis.grid(alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-day new/reviews/memorized/observed-retention curves for one user "
            "at multiple desired-retention values."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs") / "retention_sweep",
        help="Root directory containing retention_sweep logs.",
    )
    parser.add_argument("--user-id", type=int, required=True, help="User id to plot.")
    parser.add_argument("--env", type=str, default=None, help="Filter by environment.")
    parser.add_argument("--sched", type=str, default=None, help="Filter by scheduler.")
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized", "batched", "any"],
        default="any",
        help="Filter by engine.",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="any",
        help="Filter by short-term setting.",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched", "any"],
        default="any",
        help="Filter by short-term source.",
    )
    parser.add_argument(
        "--desired-retentions",
        required=True,
        help="Comma-separated desired-retention values, e.g. 0.55,0.68,0.86,0.96.",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=0,
        help="Plot from this day onward.",
    )
    parser.add_argument(
        "--review-scale",
        choices=["linear", "symlog", "log"],
        default="symlog",
        help="Y-axis scale for reviews/day.",
    )
    parser.add_argument(
        "--retention-smoothing-window",
        type=int,
        default=30,
        help="Centered rolling window for observed retention (1 disables smoothing).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window.",
    )
    args = parser.parse_args()
    if args.retention_smoothing_window < 1:
        raise SystemExit("--retention-smoothing-window must be >= 1.")

    desired_retentions = _parse_desired_retentions(args.desired_retentions)
    curves, duplicates_skipped = _select_curves(
        log_root=args.log_dir,
        user_id=args.user_id,
        env=args.env,
        sched=args.sched,
        engine=args.engine,
        short_term=args.short_term,
        short_term_source=args.short_term_source,
        desired_retentions=desired_retentions,
    )
    out_path = args.out or _default_output_path(
        plot_dir=Path("experiments") / "retention_sweep" / "plots",
        user_id=args.user_id,
        env=args.env,
        sched=args.sched,
        engine=args.engine,
        short_term=args.short_term,
        short_term_source=args.short_term_source,
        desired_retentions=desired_retentions,
        start_day=args.start_day,
    )
    _plot_curves(
        curves,
        user_id=args.user_id,
        env=args.env,
        sched=args.sched,
        engine=args.engine,
        short_term_source=args.short_term_source,
        start_day=args.start_day,
        out_path=out_path,
        review_scale=args.review_scale,
        retention_smoothing_window=args.retention_smoothing_window,
        show_plot=not args.no_show,
    )

    if duplicates_skipped:
        print(
            f"Note: skipped {duplicates_skipped} duplicate log(s) after keeping the "
            "newest match per desired retention."
        )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate retention_sweep logs across users and write summary JSON.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Root directory containing retention_sweep logs (default logs/retention_sweep).",
    )
    parser.add_argument(
        "--env",
        dest="env",
        default="lstm",
        help="Comma-separated environments to include.",
    )
    parser.add_argument(
        "--sched",
        dest="sched",
        default="fsrs6,anki_sm2,memrise",
        help="Comma-separated schedulers to include.",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=0.50,
        help="Minimum desired retention to include.",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=0.98,
        help="Maximum desired retention to include.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Where to write aggregate results JSON.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting.",
    )
    parser.add_argument(
        "--equiv-report",
        nargs="?",
        const="fsrs6",
        default="off",
        choices=["off", "fsrs6", "fsrs3", "fsrs6_default"],
        help=(
            "Print equivalence report; optionally set target DR scheduler "
            "(fsrs6/fsrs3/fsrs6_default). `--equiv-report` defaults to fsrs6."
        ),
    )
    parser.add_argument(
        "--equiv-report-path",
        type=Path,
        default=None,
        help="Optional path to write equivalence summary JSON.",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized", "batched", "any"],
        default="any",
        help="Filter logs by simulation engine (default: any).",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="any",
        help="Filter logs by short-term flag.",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched", "any"],
        default="any",
        help="Filter logs by short-term source (steps/sched).",
    )
    parser.add_argument(
        "--fsrs3-vs-fsrs6-boxplot",
        action="store_true",
        help=(
            "Plot FSRSv3/FSRS-6 efficiency ratio at equivalent memorized-average, "
            "binned by equivalent FSRS-6 DR."
        ),
    )
    parser.add_argument(
        "--fsrs6-default-vs-fsrs6-boxplot",
        action="store_true",
        help=(
            "Plot FSRS-6 default/FSRS-6 efficiency ratio at equivalent "
            "memorized-average, binned by equivalent FSRS-6 DR."
        ),
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _load_meta_totals(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = None
    totals = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload.get("data")
            elif payload.get("type") == "totals":
                totals = payload.get("data")
            if meta is not None and totals is not None:
                break
    if meta is None or totals is None:
        raise ValueError(f"Missing meta or totals in {path}")
    return meta, totals


def _iter_log_paths(log_root: Path) -> Iterable[Path]:
    if not log_root.exists():
        return []
    user_dirs = sorted(path for path in log_root.iterdir() if path.is_dir())
    if user_dirs:
        for user_dir in user_dirs:
            for path in sorted(user_dir.glob("*.jsonl")):
                yield path
    else:
        for path in sorted(log_root.glob("*.jsonl")):
            yield path


def _format_scheduler_title(scheduler: str) -> str:
    labels = {
        "anki_sm2": "Anki-SM-2",
        "fsrs3": "FSRSv3",
        "fsrs6": "FSRS-6",
        "fsrs6_default": "FSRS-6 (default)",
        "memrise": "Memrise",
    }
    return labels.get(scheduler, scheduler)


def _normalize_user_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_user_id_from_path(path: Path) -> Optional[int]:
    parent = path.parent.name
    if parent.startswith("user_"):
        try:
            return int(parent.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    match = re.search(r"_user=(\d+)_", path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _percentile(values: List[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty list.")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * p
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_vals[lower])
    weight = pos - lower
    return (
        float(sorted_vals[lower]) * (1.0 - weight) + float(sorted_vals[upper]) * weight
    )


def _filter_outliers_iqr(values: List[float]) -> List[float]:
    if len(values) < 4:
        return values
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return values
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    filtered = [value for value in values if low <= value <= high]
    return filtered or values


def _mean_without_outliers(values: List[float]) -> float:
    return mean(_filter_outliers_iqr(values))


def _infer_engine(meta: Dict[str, Any], path: Path) -> str | None:
    value = meta.get("engine")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "batch":
            lowered = "batched"
        if lowered in {"event", "vectorized", "batched"}:
            return lowered
    match = re.search(r"_engine=([^_]+)_", path.name)
    if match:
        lowered = match.group(1).strip().lower()
        if lowered == "batch":
            lowered = "batched"
        if lowered in {"event", "vectorized", "batched"}:
            return lowered
    return None


def _format_title(base: str, user_ids: List[int]) -> str:
    if not user_ids:
        return base
    if len(user_ids) == 1:
        return f"{base} (user {user_ids[0]})"
    user_ids_sorted = sorted(user_ids)
    if user_ids_sorted[-1] - user_ids_sorted[0] + 1 == len(user_ids_sorted):
        return f"{base} (users {user_ids_sorted[0]}-{user_ids_sorted[-1]}, n={len(user_ids_sorted)})"
    return f"{base} (n={len(user_ids_sorted)})"


def main() -> None:
    args = parse_args()

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    envs = _parse_csv(args.env)
    schedulers = [item for item in _parse_csv(args.sched) if item != "sspmmc"]
    if not schedulers:
        schedulers = ["fsrs6"]
    try:
        scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dr_schedulers: List[str] = []
    fixed_intervals: List[float] = []
    include_all_fixed = False
    for name, interval, raw in scheduler_specs:
        if name == "fixed":
            if interval is None or raw == "fixed":
                include_all_fixed = True
            else:
                fixed_intervals.append(interval)
            continue
        if name not in dr_schedulers:
            dr_schedulers.append(name)
    if include_all_fixed:
        fixed_intervals = []
    run_dr = bool(dr_schedulers)
    run_fixed = include_all_fixed or bool(fixed_intervals)
    if not run_dr and not run_fixed:
        raise SystemExit("No schedulers specified. Use --sched to select plots.")

    results_path = args.results_path or (
        log_root / "simulation_results_retention_sweep_user_averages.json"
    )
    plot_dir = args.plot_dir or (
        REPO_ROOT / "experiments" / "retention_sweep" / "plots" / "user_averages"
    )

    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]] = {}
    duplicate_count = 0

    for path in _iter_log_paths(log_root):
        try:
            meta, totals = _load_meta_totals(path)
        except ValueError:
            continue

        engine = _infer_engine(meta, path)
        if args.engine != "any":
            if engine != args.engine:
                continue

        environment = meta.get("environment")
        scheduler = meta.get("scheduler")
        if not isinstance(environment, str):
            continue
        if envs and environment not in envs:
            continue
        if not isinstance(scheduler, str):
            continue
        if scheduler == "sspmmc":
            continue
        if scheduler not in {name for name, _, _ in scheduler_specs}:
            continue

        short_term_value = _normalize_bool(meta.get("short_term"))
        if args.short_term != "any":
            if args.short_term == "on":
                if short_term_value is not True:
                    continue
            elif short_term_value is True:
                continue
        if args.short_term_source != "any":
            short_term_source = meta.get("short_term_source")
            if short_term_source != args.short_term_source:
                continue
            if short_term_value is not True:
                continue

        desired = meta.get("desired_retention")
        fixed_interval = None
        if scheduler == "fixed":
            raw_interval = meta.get("fixed_interval")
            fixed_interval = normalize_fixed_interval(
                float(raw_interval) if raw_interval is not None else None
            )
            if run_fixed and fixed_intervals:
                if not any(
                    math.isclose(fixed_interval, value, rel_tol=0.0, abs_tol=1e-6)
                    for value in fixed_intervals
                ):
                    continue
        if scheduler_uses_desired_retention(scheduler):
            if desired is None:
                continue
            try:
                desired = round(float(desired), 2)
            except (TypeError, ValueError):
                continue
            if desired < args.start_retention or desired > args.end_retention:
                continue
        else:
            desired = None

        time_average = totals.get("time_average")
        if time_average is None:
            continue
        time_average = float(time_average)
        if time_average <= 0:
            continue

        memorized_average = float(totals.get("memorized_average", 0.0))
        memorized_per_minute = memorized_average / time_average
        user_id = _normalize_user_id(meta.get("user_id"))
        if user_id is None:
            user_id = _infer_user_id_from_path(path)
        user_key = user_id if user_id is not None else f"unknown::{path}"
        mtime = path.stat().st_mtime

        key = (environment, scheduler, desired, fixed_interval)
        group = groups.setdefault(
            key,
            {
                "environment": environment,
                "scheduler": scheduler,
                "desired_retention": desired,
                "fixed_interval": fixed_interval,
                "users": {},
            },
        )
        existing = group["users"].get(user_key)
        if existing is not None:
            duplicate_count += 1
            if mtime <= existing["mtime"]:
                continue
        group["users"][user_key] = {
            "metrics": {
                "memorized_average": memorized_average,
                "memorized_per_minute": memorized_per_minute,
            },
            "mtime": mtime,
        }

    common_user_ids: Optional[set[int]] = None
    groups_with_users = 0
    for group in groups.values():
        group_user_ids = {
            user_key for user_key in group["users"].keys() if isinstance(user_key, int)
        }
        if not group_user_ids:
            continue
        if common_user_ids is None:
            common_user_ids = set(group_user_ids)
        else:
            common_user_ids &= group_user_ids
        groups_with_users += 1

    if common_user_ids is None:
        raise SystemExit("No user ids found across logs; cannot compute intersections.")
    if not common_user_ids:
        raise SystemExit(
            "No overlapping user ids across configs; intersection is empty."
        )

    results: List[Dict[str, Any]] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2] or -1,
            item[0][3] or -1,
        ),
    ):
        users = [
            payload["metrics"]
            for user_key, payload in group["users"].items()
            if isinstance(user_key, int) and user_key in common_user_ids
        ]
        if not users:
            continue
        scheduler = group["scheduler"]
        desired = group["desired_retention"]
        fixed_interval = group["fixed_interval"]
        if scheduler == "fixed":
            title = f"Ivl={format_float(fixed_interval)}"
        elif scheduler_uses_desired_retention(scheduler):
            title = f"DR={format_float(float(desired) * 100)}%"
        else:
            title = _format_scheduler_title(scheduler)
        memorized_average_values = [item["memorized_average"] for item in users]
        memorized_per_minute_values = [item["memorized_per_minute"] for item in users]
        memorized_average_mean = _mean_without_outliers(memorized_average_values)
        memorized_per_minute_mean = _mean_without_outliers(memorized_per_minute_values)
        memorized_average_median = median(memorized_average_values)
        memorized_per_minute_median = median(memorized_per_minute_values)
        results.append(
            {
                "environment": group["environment"],
                "scheduler": group["scheduler"],
                "desired_retention": group["desired_retention"],
                "fixed_interval": group["fixed_interval"],
                "user_count": len(common_user_ids),
                "memorized_average": memorized_average_mean,
                "memorized_per_minute": memorized_per_minute_mean,
                "memorized_average_mean": memorized_average_mean,
                "memorized_per_minute_mean": memorized_per_minute_mean,
                "memorized_average_median": memorized_average_median,
                "memorized_per_minute_median": memorized_per_minute_median,
                "title": title,
            }
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if duplicate_count:
        print(f"Note: skipped {duplicate_count} duplicate logs (kept newest per user).")
    if groups_with_users:
        print(
            "Note: using intersection of users across configs: "
            f"{len(common_user_ids)} users."
        )

    wants_equiv = args.equiv_report != "off" or args.equiv_report_path is not None
    needs_equiv = wants_equiv or not args.no_plot
    equiv_target = args.equiv_report if args.equiv_report != "off" else "fsrs6"
    equivalent_distributions: List[Dict[str, Any]] = []
    if needs_equiv:
        equivalent_distributions = _compute_equivalent_dr_distributions(
            groups,
            envs,
            common_user_ids,
            target=equiv_target,
            baselines=["anki_sm2", "memrise"],
        )

    if not args.no_plot:
        plot_dir.mkdir(parents=True, exist_ok=True)
        _setup_plot_style()
        st_suffix = f"_st={args.short_term}"
        sts_suffix = f"_sts={args.short_term_source}"
        engine_suffix = "" if args.engine == "any" else f"_engine={args.engine}"
        for entry in equivalent_distributions:
            env_label = entry["environment"]
            env_suffix = f"_env={env_label}"
            distribution_title = _format_title(
                (
                    f"{_format_scheduler_title(entry['baseline'])} vs "
                    f"{_format_scheduler_title(entry.get('target', 'fsrs6'))} "
                    f"equiv distributions (env={env_label}, short-term={args.short_term})"
                ),
                sorted(entry["user_ids"]),
            )
            baseline_suffix = entry["baseline"]
            target_suffix = entry.get("target", "fsrs6")
            if target_suffix == "fsrs6":
                distribution_path = (
                    plot_dir
                    / f"retention_sweep_equivalent_fsrs6_distributions_{baseline_suffix}{env_suffix}{st_suffix}{engine_suffix}.png"
                )
            elif target_suffix == "fsrs6_default":
                distribution_path = (
                    plot_dir
                    / f"retention_sweep_equivalent_fsrs6_default_distributions_{baseline_suffix}{env_suffix}{st_suffix}{engine_suffix}.png"
                )
            else:
                distribution_path = (
                    plot_dir
                    / f"retention_sweep_equivalent_{target_suffix}_distributions_{baseline_suffix}{env_suffix}{st_suffix}{engine_suffix}.png"
                )
            _plot_equivalent_distributions(entry, distribution_path, distribution_title)
            print(f"Saved plot to {distribution_path}")

        if args.fsrs3_vs_fsrs6_boxplot:
            ratio_entries = _compute_fsrs3_vs_fsrs6_ratio_by_fsrs6_dr(
                groups,
                envs,
                common_user_ids,
            )
            for entry in ratio_entries:
                env_label = entry["environment"]
                env_suffix = f"_env={env_label}"
                title = _format_title(
                    (
                        "FSRSv3 / FSRS-6 ratio at equivalent memorized-average "
                        f"(env={env_label}, short-term={args.short_term})"
                    ),
                    sorted(entry["user_ids"]),
                )
                out_path = (
                    plot_dir
                    / f"retention_sweep_fsrs3_over_fsrs6_ratio_by_fsrs6_dr{env_suffix}{st_suffix}{sts_suffix}{engine_suffix}.png"
                )
                _plot_fsrs3_vs_fsrs6_ratio_boxplot(
                    entry, output_path=out_path, title=title
                )
                print(f"Saved plot to {out_path}")

        if args.fsrs6_default_vs_fsrs6_boxplot:
            ratio_entries = _compute_fsrs6_default_vs_fsrs6_ratio_by_fsrs6_dr(
                groups,
                envs,
                common_user_ids,
            )
            for entry in ratio_entries:
                env_label = entry["environment"]
                env_suffix = f"_env={env_label}"
                title = _format_title(
                    (
                        "FSRS-6 default / FSRS-6 ratio at equivalent memorized-average "
                        f"(env={env_label}, short-term={args.short_term})"
                    ),
                    sorted(entry["user_ids"]),
                )
                out_path = (
                    plot_dir
                    / f"retention_sweep_fsrs6_default_over_fsrs6_ratio_by_fsrs6_dr{env_suffix}{st_suffix}{sts_suffix}{engine_suffix}.png"
                )
                _plot_fsrs6_default_vs_fsrs6_ratio_boxplot(
                    entry, output_path=out_path, title=title
                )
                print(f"Saved plot to {out_path}")

    if wants_equiv:
        summaries = _summarize_equivalent_distributions(equivalent_distributions)
        _print_equiv_report(summaries)
        if args.equiv_report_path is not None:
            args.equiv_report_path.parent.mkdir(parents=True, exist_ok=True)
            with args.equiv_report_path.open("w", encoding="utf-8") as fh:
                json.dump(summaries, fh, indent=2)
            print(f"Wrote {args.equiv_report_path}")

    if args.no_plot:
        print(f"Wrote {results_path}")
        return
    print(f"Wrote {results_path}")


def _setup_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")


def _interpolate_equivalent_point(
    candidates: List[Tuple[float, Dict[str, float]]],
    *,
    target_x: float,
    allow_extrapolation: bool = True,
) -> Tuple[float, float] | None:
    """Interpolate (dr, y) at target memorized-average x.

    candidates is a list of (desired_retention, metrics) for a single user and
    scheduler where metrics has keys: memorized_average (x) and memorized_per_minute (y).
    """

    if len(candidates) < 2:
        return None
    points = [
        {
            "dr": float(dr),
            "x": float(payload["memorized_average"]),
            "y": float(payload["memorized_per_minute"]),
        }
        for dr, payload in candidates
    ]
    points.sort(key=lambda item: item["x"])
    lower = None
    upper = None
    for point in points:
        if point["x"] <= target_x:
            lower = point
        if point["x"] >= target_x and upper is None:
            upper = point
    if lower is None or upper is None:
        if not allow_extrapolation:
            return None
        points.sort(key=lambda item: abs(item["x"] - target_x))
        lower = points[0]
        upper = points[1]
    x1 = lower["x"]
    x2 = upper["x"]
    if math.isclose(x1, x2):
        dr_equiv = (lower["dr"] + upper["dr"]) / 2.0
        y_equiv = (lower["y"] + upper["y"]) / 2.0
        return dr_equiv, y_equiv
    t = (target_x - x1) / (x2 - x1)
    t = max(0.0, min(1.0, t))
    dr_equiv = lower["dr"] + t * (upper["dr"] - lower["dr"])
    y_equiv = lower["y"] + t * (upper["y"] - lower["y"])
    return dr_equiv, y_equiv


def _compute_equivalent_fsrs6_distributions(
    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]],
    envs: List[str],
    common_user_ids: set[int],
    baselines: List[str],
) -> List[Dict[str, Any]]:
    # Backwards-compatible wrapper: original behavior is fsrs6 equivalence.
    return _compute_equivalent_dr_distributions(
        groups,
        envs,
        common_user_ids,
        target="fsrs6",
        baselines=baselines,
    )


def _compute_equivalent_dr_distributions(
    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]],
    envs: List[str],
    common_user_ids: set[int],
    *,
    target: str,
    baselines: List[str],
) -> List[Dict[str, Any]]:
    distributions: List[Dict[str, Any]] = []
    for env in envs:
        target_users: Dict[int, List[Tuple[float, Dict[str, float]]]] = {}

        for (group_env, scheduler, desired, _), group in groups.items():
            if group_env != env:
                continue
            if scheduler == target and desired is not None:
                for user_key, payload in group["users"].items():
                    if not isinstance(user_key, int):
                        continue
                    target_users.setdefault(user_key, []).append(
                        (float(desired), payload["metrics"])
                    )

        for baseline in baselines:
            baseline_users: Dict[int, Dict[str, float]] = {}
            for (group_env, scheduler, _, _), group in groups.items():
                if group_env != env or scheduler != baseline:
                    continue
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        baseline_users[user_key] = payload["metrics"]

            eligible_users = set(baseline_users) & set(target_users) & common_user_ids
            if not eligible_users:
                continue

            baseline_per_minute: List[float] = []
            target_per_minute: List[float] = []
            target_dr_equiv: List[float] = []
            used_users: set[int] = set()
            for user_id in sorted(eligible_users):
                baseline_metrics = baseline_users[user_id]
                candidates = target_users[user_id]
                target_value = float(baseline_metrics["memorized_average"])
                interp = _interpolate_equivalent_point(
                    candidates, target_x=target_value
                )
                if interp is None:
                    continue
                dr_equiv, y_equiv = interp

                baseline_per_minute.append(baseline_metrics["memorized_per_minute"])
                target_per_minute.append(y_equiv)
                target_dr_equiv.append(dr_equiv)
                used_users.add(user_id)

            if not used_users:
                continue

            distributions.append(
                {
                    "environment": env,
                    "baseline": baseline,
                    "baseline_per_minute": baseline_per_minute,
                    "target": target,
                    "target_per_minute": target_per_minute,
                    "target_dr_equiv": target_dr_equiv,
                    "user_ids": used_users,
                }
            )

    return distributions


def _compute_fsrs3_vs_fsrs6_ratio_by_fsrs6_dr(
    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]],
    envs: List[str],
    common_user_ids: set[int],
) -> List[Dict[str, Any]]:
    """Compute per-user ratio at equivalent memorized-average, binned by FSRS-6 DR.

    For each user and each FSRS-6 desired retention point:
    - take FSRS-6 (dr6, x6=memorized_average, y6=memorized_per_minute)
    - interpolate FSRSv3 to match x6 => y3_equiv
      - interpolation only (no extrapolation): if x6 is outside FSRSv3's x-range,
        skip that (user, dr6) point.
    - ratio = y3_equiv / y6
    - group ratio by dr6 (x axis is FSRS-6 DR)
    """

    entries: List[Dict[str, Any]] = []
    for env in envs:
        fsrs6_users: Dict[int, Dict[float, Dict[str, float]]] = {}
        fsrs3_users: Dict[int, Dict[float, Dict[str, float]]] = {}

        for (group_env, scheduler, desired, _), group in groups.items():
            if group_env != env or desired is None:
                continue
            dr = round(float(desired), 2)
            if scheduler == "fsrs6":
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        fsrs6_users.setdefault(user_key, {})[dr] = payload["metrics"]
            elif scheduler == "fsrs3":
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        fsrs3_users.setdefault(user_key, {})[dr] = payload["metrics"]

        if not fsrs6_users or not fsrs3_users:
            continue

        eligible_users = set(fsrs6_users) & set(fsrs3_users) & common_user_ids
        if not eligible_users:
            continue

        fsrs6_dr: List[float] = []
        ratio_values: List[float] = []
        used_users: set[int] = set()
        for user_id in sorted(eligible_users):
            fsrs6_points = fsrs6_users[user_id]
            fsrs3_points = fsrs3_users[user_id]
            if len(fsrs3_points) < 2:
                continue

            fsrs3_curve = [(dr, metrics) for dr, metrics in fsrs3_points.items()]
            for dr6 in sorted(fsrs6_points):
                metrics6 = fsrs6_points[dr6]
                x6 = float(metrics6["memorized_average"])
                y6 = float(metrics6["memorized_per_minute"])
                if y6 <= 0 or not math.isfinite(y6):
                    continue
                interp = _interpolate_equivalent_point(
                    fsrs3_curve,
                    target_x=x6,
                    allow_extrapolation=False,
                )
                if interp is None:
                    continue
                _dr3, y3 = interp
                ratio = y3 / y6
                if not math.isfinite(ratio):
                    continue
                fsrs6_dr.append(float(dr6))
                ratio_values.append(ratio)
                used_users.add(user_id)

        if not used_users:
            continue

        entries.append(
            {
                "environment": env,
                "fsrs6_dr_equiv": fsrs6_dr,
                "ratio_fsrs3_over_fsrs6": ratio_values,
                "user_ids": used_users,
            }
        )

    return entries


def _compute_fsrs6_default_vs_fsrs6_ratio_by_fsrs6_dr(
    groups: Dict[Tuple[str, str, Optional[float], Optional[float]], Dict[str, Any]],
    envs: List[str],
    common_user_ids: set[int],
) -> List[Dict[str, Any]]:
    """Compute per-user ratio at equivalent memorized-average, binned by FSRS-6 DR.

    For each user and each FSRS-6 desired retention point:
    - take FSRS-6 (dr6, x6=memorized_average, y6=memorized_per_minute)
    - interpolate FSRS-6 default to match x6 => y_default_equiv
      - interpolation only (no extrapolation): if x6 is outside FSRS-6 default's
        x-range, skip that (user, dr6) point.
    - ratio = y_default_equiv / y6
    - group ratio by dr6 (x axis is FSRS-6 DR)
    """

    entries: List[Dict[str, Any]] = []
    for env in envs:
        fsrs6_users: Dict[int, Dict[float, Dict[str, float]]] = {}
        fsrs6_default_users: Dict[int, Dict[float, Dict[str, float]]] = {}

        for (group_env, scheduler, desired, _), group in groups.items():
            if group_env != env or desired is None:
                continue
            dr = round(float(desired), 2)
            if scheduler == "fsrs6":
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        fsrs6_users.setdefault(user_key, {})[dr] = payload["metrics"]
            elif scheduler == "fsrs6_default":
                for user_key, payload in group["users"].items():
                    if isinstance(user_key, int):
                        fsrs6_default_users.setdefault(user_key, {})[dr] = payload[
                            "metrics"
                        ]

        if not fsrs6_users or not fsrs6_default_users:
            continue

        eligible_users = set(fsrs6_users) & set(fsrs6_default_users) & common_user_ids
        if not eligible_users:
            continue

        fsrs6_dr: List[float] = []
        ratio_values: List[float] = []
        used_users: set[int] = set()
        for user_id in sorted(eligible_users):
            fsrs6_points = fsrs6_users[user_id]
            fsrs6_default_points = fsrs6_default_users[user_id]
            if len(fsrs6_default_points) < 2:
                continue

            default_curve = [
                (dr, metrics) for dr, metrics in fsrs6_default_points.items()
            ]
            for dr6 in sorted(fsrs6_points):
                metrics6 = fsrs6_points[dr6]
                x6 = float(metrics6["memorized_average"])
                y6 = float(metrics6["memorized_per_minute"])
                if y6 <= 0 or not math.isfinite(y6):
                    continue
                interp = _interpolate_equivalent_point(
                    default_curve,
                    target_x=x6,
                    allow_extrapolation=False,
                )
                if interp is None:
                    continue
                _dr_default, y_default = interp
                ratio = y_default / y6
                if not math.isfinite(ratio):
                    continue
                fsrs6_dr.append(float(dr6))
                ratio_values.append(ratio)
                used_users.add(user_id)

        if not used_users:
            continue

        entries.append(
            {
                "environment": env,
                "fsrs6_dr_equiv": fsrs6_dr,
                "ratio_fsrs6_default_over_fsrs6": ratio_values,
                "user_ids": used_users,
            }
        )

    return entries


def _plot_fsrs3_vs_fsrs6_ratio_boxplot(
    entry: Dict[str, Any],
    *,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    dr_values: List[float] = [float(x) for x in entry["fsrs6_dr_equiv"]]
    ratios: List[float] = [float(x) for x in entry["ratio_fsrs3_over_fsrs6"]]

    # Bin by rounded DR so x-axis remains readable and stable.
    binned: Dict[float, List[float]] = {}
    for dr, ratio in zip(dr_values, ratios):
        dr_bin = round(dr, 2)
        binned.setdefault(dr_bin, []).append(ratio)

    dr_bins = sorted(binned)
    data = [binned[value] for value in dr_bins]
    positions = list(range(1, len(dr_bins) + 1))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#2ca02c", "edgecolor": "black", "alpha": 0.5},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value * 100:.0f}%" for value in dr_bins], rotation=45)
    ax.set_xlabel("FSRS-6 DR (%)")
    ax.set_ylabel("FSRSv3 equiv / FSRS-6 equiv (cards/min)")
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _plot_fsrs6_default_vs_fsrs6_ratio_boxplot(
    entry: Dict[str, Any],
    *,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    dr_values: List[float] = [float(x) for x in entry["fsrs6_dr_equiv"]]
    ratios: List[float] = [float(x) for x in entry["ratio_fsrs6_default_over_fsrs6"]]

    binned: Dict[float, List[float]] = {}
    for dr, ratio in zip(dr_values, ratios):
        dr_bin = round(dr, 2)
        binned.setdefault(dr_bin, []).append(ratio)

    dr_bins = sorted(binned)
    data = [binned[value] for value in dr_bins]
    positions = list(range(1, len(dr_bins) + 1))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#2ca02c", "edgecolor": "black", "alpha": 0.5},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value * 100:.0f}%" for value in dr_bins], rotation=45)
    ax.set_xlabel("FSRS-6 DR (%)")
    ax.set_ylabel("FSRS-6 default equiv / FSRS-6 equiv (cards/min)")
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _summarize_equivalent_distributions(
    distributions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for entry in distributions:
        baseline_per_minute = entry.get("baseline_per_minute", [])
        target_per_minute = entry.get(
            "target_per_minute", entry.get("fsrs_per_minute", [])
        )
        dr_equiv = entry.get("target_dr_equiv", entry.get("fsrs_dr_equiv", []))
        if not baseline_per_minute or not target_per_minute:
            continue
        diffs = [
            target_value - baseline_value
            for baseline_value, target_value in zip(
                baseline_per_minute, target_per_minute
            )
        ]
        ratios = [
            target_value / baseline_value
            for baseline_value, target_value in zip(
                baseline_per_minute, target_per_minute
            )
            if baseline_value > 0
        ]
        if not ratios:
            continue
        pos = sum(1 for value in diffs if value > 0)
        neg = sum(1 for value in diffs if value < 0)
        zero = len(diffs) - pos - neg
        try:
            q1, _, q3 = statistics.quantiles(ratios, n=4, method="inclusive")
        except Exception:
            ratios_sorted = sorted(ratios)
            q1 = ratios_sorted[0]
            q3 = ratios_sorted[-1]

        summaries.append(
            {
                "environment": entry.get("environment"),
                "baseline": entry.get("baseline"),
                "target": entry.get("target", "fsrs6"),
                "user_count": len(entry.get("user_ids", [])),
                "diff_mean": _mean_without_outliers(diffs),
                "diff_median": median(diffs),
                "ratio_mean": _mean_without_outliers(ratios),
                "ratio_median": median(ratios),
                "ratio_q25": q1,
                "ratio_q75": q3,
                "pos_pct": pos / len(diffs),
                "neg_pct": neg / len(diffs),
                "zero_pct": zero / len(diffs),
                "dr_equiv_mean": _mean_without_outliers(dr_equiv) if dr_equiv else None,
                "dr_equiv_median": median(dr_equiv) if dr_equiv else None,
            }
        )
    return summaries


def _print_equiv_report(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        print("No equivalence summaries available.")
        return
    title_target = _format_scheduler_title(str(summaries[0].get("target", "fsrs6")))
    print(f"{title_target} equivalence summary (matched memorized average)")
    for summary in summaries:
        env = summary.get("environment")
        baseline = _format_scheduler_title(str(summary.get("baseline")))
        target = _format_scheduler_title(str(summary.get("target", "fsrs6")))
        user_count = summary.get("user_count")
        pos_pct = summary.get("pos_pct")
        neg_pct = summary.get("neg_pct")
        ratio_mean = summary.get("ratio_mean")
        ratio_median = summary.get("ratio_median")
        ratio_q25 = summary.get("ratio_q25")
        ratio_q75 = summary.get("ratio_q75")
        dr_mean = summary.get("dr_equiv_mean")
        dr_median = summary.get("dr_equiv_median")
        print(
            f"- {target} vs {baseline} (env={env}): n={user_count}, "
            f"superiority={pos_pct:.1%}, "
            f"mean ratio={ratio_mean:.3f}, "
            f"median ratio={ratio_median:.3f} "
            f"(IQR {ratio_q25:.3f}-{ratio_q75:.3f}), "
            f"mean DR={dr_mean:.3f}, "
            f"median DR={dr_median:.3f}"
        )


def _plot_equivalent_distributions(
    entry: Dict[str, Any],
    output_path: Path,
    title_base: str,
) -> None:
    import matplotlib.pyplot as plt

    baseline_per_minute: List[float] = [float(x) for x in entry["baseline_per_minute"]]
    target_per_minute: List[float] = [
        float(x)
        for x in (entry.get("target_per_minute", entry.get("fsrs_per_minute")) or [])
    ]
    target_dr_equiv: List[float] = [
        float(x)
        for x in (entry.get("target_dr_equiv", entry.get("fsrs_dr_equiv")) or [])
    ]
    baseline_label = _format_scheduler_title(entry["baseline"])
    target_label = _format_scheduler_title(entry.get("target", "fsrs6"))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax_middle = axes[0, 0]
    ax_left = axes[0, 1]
    ax_right = axes[1, 0]
    ax_box = axes[1, 1]
    ax_ratio_hist = axes[2, 0]
    ax_ratio_box = axes[2, 1]

    boxplot_left = ax_left.boxplot(
        [baseline_per_minute, target_per_minute],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#d9d9d9", "edgecolor": "black"},
        medianprops={"color": "black"},
    )
    for patch, color in zip(boxplot_left["boxes"], ["#1f77b4", "#2ca02c"]):
        patch.set_facecolor(color)
    ax_left.set_xticks([1, 2])
    ax_left.set_xticklabels([baseline_label, f"{target_label} equiv"])
    ax_left.set_ylabel("Memorized cards/min (average)")
    ax_left.grid(True, axis="y", ls="--", alpha=0.6)

    def _annotate_box_stats(
        ax: Any,
        values_list: List[List[float]],
        x_positions: List[float],
    ) -> Tuple[List[str], List[float], List[float]]:
        labels: List[str] = []
        means: List[float] = []
        medians: List[float] = []
        for x_pos, values in zip(x_positions, values_list):
            if not values:
                continue
            mean_value = _mean_without_outliers(values)
            median_value = median(values)
            means.append(mean_value)
            medians.append(median_value)
            labels.append(f"μ={mean_value:.2f}  med={median_value:.2f}")
            ax.scatter(
                [x_pos],
                [mean_value],
                color="black",
                marker="D",
                s=28,
                zorder=4,
            )
            ax.scatter(
                [x_pos],
                [median_value],
                color="white",
                edgecolor="black",
                marker="o",
                s=26,
                zorder=4,
            )
        return labels, means, medians

    left_labels, _, _ = _annotate_box_stats(
        ax_left,
        [baseline_per_minute, target_per_minute],
        [1, 2],
    )
    if left_labels:
        ax_left.legend(
            left_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )

    dr_percent = [value * 100 for value in target_dr_equiv]
    dr_bins = min(15, max(5, len(dr_percent) // 3))
    ax_middle.hist(dr_percent, bins=dr_bins, color="#1f77b4", alpha=0.8)
    ax_middle.set_xlabel(f"Equivalent {target_label} DR (%)")
    ax_middle.set_ylabel("User count")
    ax_middle.grid(True, axis="y", ls="--", alpha=0.6)

    diff_values = [
        target_value - baseline_value
        for baseline_value, target_value in zip(baseline_per_minute, target_per_minute)
    ]
    if diff_values:
        diff_bins = min(15, max(5, len(diff_values) // 3))
        ax_right.hist(diff_values, bins=diff_bins, color="#ff7f0e", alpha=0.8)
    ax_right.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_right.set_xlabel(f"{target_label} equiv - {baseline_label} (cards/min)")
    ax_right.set_ylabel("User count")
    ax_right.grid(True, axis="y", ls="--", alpha=0.6)

    if diff_values:
        ax_box.boxplot(
            diff_values,
            vert=True,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#ff7f0e", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        ax_box.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_box.set_xticks([1])
        ax_box.set_xticklabels(["Diff"])
        ax_box.set_ylabel(f"{target_label} equiv - {baseline_label} (cards/min)")
        ax_box.grid(True, axis="y", ls="--", alpha=0.6)
        diff_positive = sum(value > 0 for value in diff_values)
        diff_ratio = diff_positive / len(diff_values)
        ax_box.text(
            0.98,
            0.98,
            f"diff>0: {diff_ratio:.1%}",
            transform=ax_box.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        diff_labels, _, _ = _annotate_box_stats(ax_box, [diff_values], [1])
        if diff_labels:
            ax_box.legend(
                diff_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
    else:
        ax_box.axis("off")

    ratio_values = [
        target_value / baseline_value
        for baseline_value, target_value in zip(baseline_per_minute, target_per_minute)
        if baseline_value > 0
    ]
    if ratio_values:
        ratio_bins = min(15, max(5, len(ratio_values) // 3))
        ax_ratio_hist.hist(ratio_values, bins=ratio_bins, color="#2ca02c", alpha=0.8)
    ax_ratio_hist.axvline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_ratio_hist.set_xlabel(f"{target_label} equiv / {baseline_label} (cards/min)")
    ax_ratio_hist.set_ylabel("User count")
    ax_ratio_hist.grid(True, axis="y", ls="--", alpha=0.6)

    if ratio_values:
        ax_ratio_box.boxplot(
            ratio_values,
            vert=True,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#2ca02c", "edgecolor": "black"},
            medianprops={"color": "black"},
        )
        ax_ratio_box.axhline(
            1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7
        )
        ax_ratio_box.set_xticks([1])
        ax_ratio_box.set_xticklabels(["Ratio"])
        ax_ratio_box.set_ylabel(f"{target_label} equiv / {baseline_label} (cards/min)")
        ax_ratio_box.grid(True, axis="y", ls="--", alpha=0.6)
        ratio_above = sum(value > 1 for value in ratio_values)
        ratio_ratio = ratio_above / len(ratio_values)
        ax_ratio_box.text(
            0.98,
            0.98,
            f"ratio>1: {ratio_ratio:.1%}",
            transform=ax_ratio_box.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ratio_labels, _, _ = _annotate_box_stats(ax_ratio_box, [ratio_values], [1])
        if ratio_labels:
            ax_ratio_box.legend(
                ratio_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
    else:
        ax_ratio_box.axis("off")

    fig.suptitle(title_base, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()

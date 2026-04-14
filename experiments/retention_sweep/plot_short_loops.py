from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.cli_utils import LogFilenameFilter
from simulator.scheduler_spec import format_float, scheduler_uses_desired_retention


def _load_meta(path: Path) -> dict:
    meta: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload.get("data", {}) or {}
                break
    return meta


def _load_short_loops_metric(csv_path: Path, metric: str) -> float | None:
    if not csv_path.exists():
        return None

    total_loops = 0
    total_days = 0
    active_loops = 0
    active_days = 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get("short_loops", "")
            if raw is None or raw == "":
                value = 0
            else:
                value = int(float(raw))
            total_loops += value
            total_days += 1
            if value > 0:
                active_loops += value
                active_days += 1

    if metric == "avg":
        if total_days == 0:
            return None
        return total_loops / total_days
    if active_days == 0:
        return None
    return active_loops / active_days


def _iter_log_paths(
    log_root: Path, *, match_fn: Optional[Callable[[str], bool]] = None
) -> Iterable[Path]:
    if not log_root.exists():
        return []
    user_dirs = sorted(
        path
        for path in log_root.iterdir()
        if path.is_dir() and path.name.startswith("user_")
    )
    if user_dirs:
        for user_dir in user_dirs:
            for path in sorted(user_dir.glob("*.jsonl")):
                if match_fn is not None and not match_fn(path.name):
                    continue
                yield path
        return
    for path in sorted(log_root.glob("*.jsonl")):
        if match_fn is not None and not match_fn(path.name):
            continue
        yield path


def _normalize_user_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _filter_log_meta(
    meta: dict,
    *,
    env: str | None,
    sched: str | None,
    engine: str | None,
    short_term_source: str | None,
    desired_retention: float | None,
) -> bool:
    if env and meta.get("environment") != env:
        return False
    if sched and meta.get("scheduler") != sched:
        return False
    if engine and meta.get("engine") != engine:
        return False
    if short_term_source and meta.get("short_term_source") != short_term_source:
        return False
    if desired_retention is not None:
        raw_retention = meta.get("desired_retention")
        if raw_retention is None:
            return False
        try:
            actual_retention = round(float(raw_retention), 2)
        except (TypeError, ValueError):
            return False
        if actual_retention != desired_retention:
            return False
    return True


def _collect_candidate_csvs(
    log_root: Path,
    *,
    env: str | None,
    sched: str | None,
    engine: str | None,
    short_term_source: str | None,
    desired_retention: float | None,
) -> tuple[dict[int, list[Path]], int, int]:
    retention_map = None
    if (
        desired_retention is not None
        and sched is not None
        and scheduler_uses_desired_retention(sched)
    ):
        retention_map = {sched: desired_retention}
    log_filter = LogFilenameFilter(
        envs=[env] if env else [],
        scheds=[sched] if sched else [],
        engine=engine or "any",
        short_term="any",
        short_term_source=short_term_source or "any",
        start_retention=desired_retention,
        end_retention=desired_retention,
        retention_values_by_scheduler=retention_map,
    )
    candidates: dict[int, list[Path]] = {}
    extra_matches = 0

    for path in _iter_log_paths(log_root, match_fn=log_filter.matches):
        meta = _load_meta(path)
        if not _filter_log_meta(
            meta,
            env=env,
            sched=sched,
            engine=engine,
            short_term_source=short_term_source,
            desired_retention=desired_retention,
        ):
            continue
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        user_id = _normalize_user_id(meta.get("user_id"))
        paths = candidates.setdefault(user_id, [])
        if paths:
            extra_matches += 1
        paths.append(csv_path)

    users_with_multiple_matches = sum(
        1 for paths in candidates.values() if len(paths) > 1
    )
    return candidates, extra_matches, users_with_multiple_matches


def _collect_values(
    log_root: Path,
    *,
    env: str | None,
    sched: str | None,
    engine: str | None,
    short_term_source: str | None,
    desired_retention: float | None,
    metric: str,
) -> tuple[list[float], list[int], int, int]:
    items: list[tuple[int, float]] = []
    candidates, extra_matches, users_with_multiple_matches = _collect_candidate_csvs(
        log_root,
        env=env,
        sched=sched,
        engine=engine,
        short_term_source=short_term_source,
        desired_retention=desired_retention,
    )

    for user_id, csv_paths in candidates.items():
        for csv_path in csv_paths:
            value = _load_short_loops_metric(csv_path, metric)
            if value is None:
                continue
            items.append((user_id, value))
            break

    items.sort(key=lambda pair: pair[0])
    user_ids = [item[0] for item in items]
    values = [item[1] for item in items]
    return values, user_ids, extra_matches, users_with_multiple_matches


def _sample_ticks(values: list[int], max_ticks: int) -> list[int]:
    unique_values = sorted(set(values))
    if len(unique_values) <= max_ticks:
        return unique_values
    tick_count = max(2, max_ticks)
    last_index = len(unique_values) - 1
    sampled = []
    for idx in range(tick_count):
        position = round(idx * last_index / (tick_count - 1))
        sampled.append(unique_values[position])
    return sorted(set(sampled))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot per-user short-term loop distribution.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs") / "retention_sweep",
        help="Root directory containing retention_sweep logs.",
    )
    parser.add_argument("--env", type=str, default=None, help="Filter by environment.")
    parser.add_argument("--sched", type=str, default=None, help="Filter by scheduler.")
    parser.add_argument("--engine", type=str, default=None, help="Filter by engine.")
    parser.add_argument(
        "--short-term-source",
        type=str,
        default=None,
        help="Filter by short-term source (steps/sched).",
    )
    parser.add_argument(
        "--metric",
        choices=["avg", "avg-active"],
        default="avg",
        help="Metric per user: avg loops/day or avg on active days.",
    )
    parser.add_argument(
        "--desired-retention",
        type=float,
        default=None,
        help="Optional exact desired-retention filter (rounded to 2 decimals).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins.",
    )
    parser.add_argument(
        "--max-user-ticks",
        type=int,
        default=20,
        help="Maximum number of x-axis user-id ticks on the lower subplot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window.",
    )
    args = parser.parse_args()
    desired_retention = (
        None
        if args.desired_retention is None
        else round(float(args.desired_retention), 2)
    )

    values, user_ids, extra_matches, users_with_multiple_matches = _collect_values(
        args.log_dir,
        env=args.env,
        sched=args.sched,
        engine=args.engine,
        short_term_source=args.short_term_source,
        desired_retention=desired_retention,
        metric=args.metric,
    )
    if not values:
        raise SystemExit("No matching short_loops data found.")
    if extra_matches:
        print(
            "Note: matched multiple logs for some users; "
            f"kept the first matching CSV per user and skipped {extra_matches} extra matches "
            f"across {users_with_multiple_matches} users."
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    axes[0].hist(values, bins=args.bins, color="tab:blue", alpha=0.8)
    sorted_values = sorted(values)
    n_values = len(sorted_values)
    median = (
        sorted_values[n_values // 2]
        if n_values % 2
        else (sorted_values[n_values // 2 - 1] + sorted_values[n_values // 2]) / 2.0
    )
    mean = sum(sorted_values) / n_values if n_values else 0.0
    q1 = sorted_values[max(int(0.25 * (n_values - 1)), 0)]
    q3 = sorted_values[max(int(0.75 * (n_values - 1)), 0)]
    axes[0].axvline(mean, color="tab:orange", linestyle="--", label=f"Mean={mean:.2f}")
    axes[0].axvline(
        median, color="tab:green", linestyle="--", label=f"Median={median:.2f}"
    )
    axes[0].axvline(q1, color="tab:red", linestyle=":", label=f"Q1={q1:.2f}")
    axes[0].axvline(q3, color="tab:red", linestyle=":", label=f"Q3={q3:.2f}")
    axes[0].set_xlabel(
        "Short loops/day" if args.metric == "avg" else "Short loops/active day"
    )
    axes[0].set_ylabel("Users")
    title = ["Short loops distribution"]
    if args.env:
        title.append(f"env={args.env}")
    if args.sched:
        title.append(f"sched={args.sched}")
    if args.engine:
        title.append(f"engine={args.engine}")
    if args.short_term_source:
        title.append(f"st={args.short_term_source}")
    if desired_retention is not None:
        title.append(f"ret={format_float(desired_retention)}")
    axes[0].set_title(" ".join(title))
    axes[0].legend()

    if extra_matches:
        axes[0].text(
            0.99,
            0.95,
            (
                f"extra matches skipped: {extra_matches}\n"
                f"users with >1 match: {users_with_multiple_matches}"
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    axes[1].bar(user_ids, values, width=0.9, color="tab:blue", alpha=0.8)
    axes[1].set_xlabel("User id")
    axes[1].set_ylabel(
        "Short loops/day" if args.metric == "avg" else "Short loops/active day"
    )
    axes[1].set_title("Short loops by user")
    if user_ids:
        axes[1].set_xticks(_sample_ticks(user_ids, args.max_user_ticks))
        axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

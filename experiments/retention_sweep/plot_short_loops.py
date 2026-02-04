from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_log(path: Path) -> tuple[dict, dict]:
    meta = {}
    daily = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "meta":
                meta = payload.get("data", {}) or {}
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            loops: list[int] = []
            for row in reader:
                raw = row.get("short_loops", "")
                if raw is None or raw == "":
                    loops.append(0)
                else:
                    loops.append(int(float(raw)))
            if loops:
                daily["short_loops"] = loops
    return meta, daily


def _matches(value: str | None, expected: str | None) -> bool:
    if expected is None:
        return True
    if value is None:
        return False
    return value == expected


def _collect_values(
    log_root: Path,
    *,
    env: str | None,
    sched: str | None,
    engine: str | None,
    short_term_source: str | None,
    metric: str,
) -> tuple[list[float], list[int], int]:
    items: list[tuple[int, float]] = []
    seen_users: set[int] = set()
    duplicates = 0

    for path in sorted(log_root.rglob("log_*.jsonl")):
        meta, daily = _load_log(path)
        if env and meta.get("environment") != env:
            continue
        if sched and meta.get("scheduler") != sched:
            continue
        if engine and meta.get("engine") != engine:
            continue
        if short_term_source and meta.get("short_term_source") != short_term_source:
            continue
        loops = daily.get("short_loops")
        if not loops:
            continue
        user_id = int(meta.get("user_id", 0))
        if user_id in seen_users:
            duplicates += 1
            continue
        seen_users.add(user_id)
        if metric == "avg":
            value = sum(loops) / len(loops) if loops else 0.0
        else:
            active = [val for val in loops if val > 0]
            value = sum(active) / len(active) if active else 0.0
        items.append((user_id, value))

    items.sort(key=lambda pair: pair[0])
    user_ids = [item[0] for item in items]
    values = [item[1] for item in items]
    return values, user_ids, duplicates


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
        "--bins",
        type=int,
        default=30,
        help="Histogram bins.",
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

    values, user_ids, duplicates = _collect_values(
        args.log_dir,
        env=args.env,
        sched=args.sched,
        engine=args.engine,
        short_term_source=args.short_term_source,
        metric=args.metric,
    )
    if not values:
        raise SystemExit("No matching short_loops data found.")

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
    axes[0].set_title(" ".join(title))
    axes[0].legend()

    if duplicates:
        axes[0].text(
            0.99,
            0.95,
            f"duplicates skipped: {duplicates}",
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
        axes[1].set_xticks(sorted(set(user_ids)))

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

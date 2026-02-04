from __future__ import annotations

import argparse
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
            elif payload.get("type") == "daily":
                daily = payload.get("data", {}) or {}
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
    values: list[float] = []
    user_ids: list[int] = []
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
        values.append(value)
        user_ids.append(user_id)

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

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import (
    format_float,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report DR distribution for each user's max memorized_per_minute.",
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
        default="fsrs6",
        help="Environment to include.",
    )
    parser.add_argument(
        "--sched",
        default="fsrs6",
        help="Scheduler to include.",
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
        "--out-csv",
        type=Path,
        default=None,
        help="Write per-user max DRs to CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a histogram plot to the plots/user_averages directory.",
    )
    return parser.parse_args()


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


def _load_meta_totals(path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
    user_dirs = sorted(
        path
        for path in log_root.iterdir()
        if path.is_dir() and path.name.startswith("user_")
    )
    if user_dirs:
        for user_dir in user_dirs:
            for path in sorted(user_dir.glob("*.jsonl")):
                yield path
    else:
        for path in sorted(log_root.glob("*.jsonl")):
            yield path


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


def main() -> None:
    args = parse_args()

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    try:
        scheduler_name, _, _ = parse_scheduler_spec(args.sched)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not scheduler_uses_desired_retention(scheduler_name):
        raise SystemExit(f"{scheduler_name} does not use desired retention.")

    per_user_best: dict[int, tuple[float, float]] = {}

    for path in _iter_log_paths(log_root):
        try:
            meta, totals = _load_meta_totals(path)
        except ValueError:
            continue

        if meta.get("environment") != args.env:
            continue
        if meta.get("scheduler") != scheduler_name:
            continue

        engine = _infer_engine(meta, path)
        if args.engine != "any" and engine != args.engine:
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
        if desired is None:
            continue
        try:
            desired_value = round(float(desired), 2)
        except (TypeError, ValueError):
            continue
        if desired_value < args.start_retention or desired_value > args.end_retention:
            continue

        time_average = totals.get("time_average")
        memorized_average = totals.get("memorized_average")
        if time_average is None or memorized_average is None:
            continue
        try:
            time_value = float(time_average)
            mem_value = float(memorized_average)
        except (TypeError, ValueError):
            continue
        if time_value <= 0.0:
            continue

        mem_per_minute = mem_value / time_value
        user_id = _normalize_user_id(meta.get("user_id"))
        if user_id is None:
            user_id = _infer_user_id_from_path(path)
        if user_id is None:
            continue

        existing = per_user_best.get(user_id)
        if existing is None or mem_per_minute > existing[0]:
            per_user_best[user_id] = (mem_per_minute, desired_value)

    if not per_user_best:
        raise SystemExit("No matching logs found.")

    dr_counts = Counter(value[1] for value in per_user_best.values())
    total_users = len(per_user_best)
    print(
        f"Users={total_users}, sched={scheduler_name}, env={args.env}, "
        f"short-term={args.short_term}, engine={args.engine}"
    )
    for dr in sorted(dr_counts):
        count = dr_counts[dr]
        pct = count / total_users * 100.0
        print(f"DR={format_float(dr * 100)}%: {count} users ({pct:.1f}%)")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", encoding="utf-8") as fh:
            fh.write("user_id,desired_retention,memorized_per_minute\n")
            for user_id in sorted(per_user_best):
                mem_per_minute, dr = per_user_best[user_id]
                fh.write(f"{user_id},{dr:.2f},{mem_per_minute:.6f}\n")

    if args.plot:
        import matplotlib.pyplot as plt

        plot_dir = (
            REPO_ROOT / "experiments" / "retention_sweep" / "plots" / "user_averages"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        st_suffix = f"_st={args.short_term}"
        engine_suffix = "" if args.engine == "any" else f"_engine={args.engine}"
        out_path = (
            plot_dir
            / f"retention_sweep_max_dr_distribution_env={args.env}_sched={scheduler_name}{st_suffix}{engine_suffix}.png"
        )
        xs = [value[1] for value in per_user_best.values()]
        plt.style.use("ggplot")
        plt.figure(figsize=(10, 5))
        plt.hist(xs, bins=sorted(dr_counts))
        plt.title(
            f"Max memorized_per_minute DR distribution (env={args.env}, sched={scheduler_name})"
        )
        plt.xlabel("Desired retention")
        plt.ylabel("Users")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from simulator.scheduler_spec import (
    format_float,
    normalize_fixed_interval,
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-user retention_sweep results as JSONL.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Root directory containing retention_sweep logs (default logs/retention_sweep).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write per-config JSONL results.",
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
        "--no-config",
        action="store_true",
        help="Omit config details from each record.",
    )
    parser.add_argument(
        "--mean-retention",
        action="store_true",
        help="Compute mean retention from daily CSV (slow).",
    )
    return parser.parse_args()


def _parse_csv(value: Optional[str]) -> list[str]:
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
    else:
        for path in sorted(log_root.glob("*.jsonl")):
            if match_fn is not None and not match_fn(path.name):
                continue
            yield path


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


def _mean_retention_from_csv(csv_path: Path) -> Optional[float]:
    if not csv_path.exists():
        return None
    retention_idx = None
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for idx, name in enumerate(header):
            if name == "retention":
                retention_idx = idx
                break
        if retention_idx is None:
            return None
        for line in fh:
            cols = line.strip().split(",")
            if retention_idx >= len(cols):
                continue
            raw = cols[retention_idx]
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isnan(value):
                continue
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _make_output_name(
    *,
    env: str,
    sched: str,
    engine: str | None,
    short_term: Optional[bool],
    short_term_source: Optional[str],
    desired_retention: Optional[float],
    fixed_interval: Optional[float],
    fuzz: bool,
) -> str:
    parts = [f"env={env}", f"sched={sched}"]
    if engine:
        parts.append(f"engine={engine}")
    if short_term is not None:
        parts.append(f"st={'on' if short_term else 'off'}")
        if short_term and short_term_source:
            parts.append(f"sts={short_term_source}")
    if scheduler_uses_desired_retention(sched) and desired_retention is not None:
        parts.append(f"ret={format_float(desired_retention)}")
    if sched == "fixed" and fixed_interval is not None:
        parts.append(f"ivl={format_float(fixed_interval)}")
    if fuzz:
        parts.append("fuzz=1")
    return "results_" + "_".join(parts) + ".jsonl"


def _matches_filename(
    name: str,
    *,
    envs: list[str],
    scheds: list[str],
    engine: str,
    short_term: str,
    short_term_source: str,
    start_retention: float,
    end_retention: float,
) -> bool:
    if envs and not any(f"env={env}" in name for env in envs):
        return False
    if scheds and not any(f"sched={sched}" in name for sched in scheds):
        return False
    if engine != "any" and f"engine={engine}" not in name:
        return False
    if short_term == "on":
        if "_st=" not in name:
            return False
        if short_term_source != "any" and f"st={short_term_source}" not in name:
            return False
    elif short_term == "off":
        if "_st=" in name and "st=off" not in name:
            return False
    if "ret=" in name:
        match = re.search(r"ret=([0-9.]+)", name)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                return False
            if value < start_retention or value > end_retention:
                return False
    return True


def main() -> None:
    args = parse_args()

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    out_dir = args.out_dir or (log_root / "user_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = _parse_csv(args.env)
    schedulers = _parse_csv(args.sched)
    if not schedulers:
        raise SystemExit("No schedulers specified.")
    try:
        scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    scheduler_names = {name for name, _, _ in scheduler_specs}

    latest_records: dict[str, dict[int, tuple[float, Dict[str, Any]]]] = {}
    duplicate_count = 0

    log_paths = list(
        _iter_log_paths(
            log_root,
            match_fn=lambda name: _matches_filename(
                name,
                envs=envs,
                scheds=list(scheduler_names),
                engine=args.engine,
                short_term=args.short_term,
                short_term_source=args.short_term_source,
                start_retention=args.start_retention,
                end_retention=args.end_retention,
            ),
        )
    )
    for path in tqdm(log_paths, unit="log", desc="Exporting"):
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
        if scheduler not in scheduler_names:
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
        else:
            short_term_source = meta.get("short_term_source")

        desired = meta.get("desired_retention")
        fixed_interval = None
        if scheduler == "fixed":
            raw_interval = meta.get("fixed_interval")
            fixed_interval = normalize_fixed_interval(
                float(raw_interval) if raw_interval is not None else None
            )
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

        user_id = _normalize_user_id(meta.get("user_id"))
        if user_id is None:
            user_id = _infer_user_id_from_path(path)
        if user_id is None:
            continue

        output_name = _make_output_name(
            env=environment,
            sched=scheduler,
            engine=None if args.engine == "any" else engine,
            short_term=short_term_value,
            short_term_source=short_term_source,
            desired_retention=desired,
            fixed_interval=fixed_interval,
            fuzz=bool(meta.get("fuzz")),
        )
        mtime = path.stat().st_mtime

        time_average = totals.get("time_average")
        total_reviews = totals.get("total_reviews")
        total_cost = totals.get("total_cost")
        memorized_average = totals.get("memorized_average")
        avg_accum_mem_per_hour = totals.get("avg_accum_memorized_per_hour")
        mean_daily_reviews = totals.get("mean_daily_reviews")

        metrics = {
            "memorized_average": memorized_average,
            "mean_daily_reviews": mean_daily_reviews,
            "time_average": time_average,
            "total_cost": total_cost,
            "avg_accum_memorized_per_hour": avg_accum_mem_per_hour,
            "total_reviews": total_reviews,
        }
        mem_per_minute = None
        if memorized_average is not None and time_average is not None:
            try:
                mem_value = float(memorized_average)
                time_value = float(time_average)
            except (TypeError, ValueError):
                mem_value = None
                time_value = None
            if mem_value is not None and time_value is not None and time_value > 0.0:
                mem_per_minute = mem_value / time_value
        metrics["memorized_per_minute"] = mem_per_minute

        if args.mean_retention:
            csv_mean_retention = _mean_retention_from_csv(path.with_suffix(".csv"))
            if csv_mean_retention is not None:
                metrics["mean_retention"] = csv_mean_retention

        record: Dict[str, Any] = {"user": user_id, "metrics": metrics}
        if not args.no_config:
            record["config"] = {
                "environment": environment,
                "scheduler": scheduler,
                "engine": engine,
                "short_term": short_term_value,
                "short_term_source": short_term_source,
                "desired_retention": desired,
                "fixed_interval": fixed_interval,
                "priority": meta.get("priority"),
                "scheduler_priority": meta.get("scheduler_priority"),
                "days": meta.get("days"),
                "deck_size": meta.get("deck_size"),
                "learn_limit": meta.get("learn_limit"),
                "review_limit": meta.get("review_limit"),
                "cost_limit_minutes": meta.get("cost_limit_minutes"),
                "seed": meta.get("seed"),
                "fuzz": bool(meta.get("fuzz")),
            }

        output_records = latest_records.setdefault(output_name, {})
        existing = output_records.get(user_id)
        if existing is not None:
            duplicate_count += 1
            if mtime <= existing[0]:
                continue
        output_records[user_id] = (mtime, record)

    for output_name, output_records in latest_records.items():
        output_path = out_dir / output_name
        with output_path.open("w", encoding="utf-8") as writer:
            for user_id in sorted(output_records):
                record = output_records[user_id][1]
                writer.write(json.dumps(record) + "\n")

    if duplicate_count:
        print(f"Note: skipped {duplicate_count} duplicate user records.")
    print(f"Wrote {len(latest_records)} result files to {out_dir}")


if __name__ == "__main__":
    main()

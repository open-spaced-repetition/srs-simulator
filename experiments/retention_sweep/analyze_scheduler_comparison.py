from __future__ import annotations

# ruff: noqa: E402

import argparse
import io
import json
import re
import statistics
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import ExperimentConfig
from experiments.retention_sweep.cli_utils import has_flag
from experiments.rl_scheduler.train_fsrs6_adr_direct_portfolio import (
    ObjectivePoint,
    hypervolume_2d,
    non_dominated_indices,
    reference_point,
)


DEFAULT_ENVS = ("fsrs6", "lstm")
DEFAULT_SCHEDULERS = ("fsrs6", "fsrs6_adr_direct", "fsrs6_adr_delta", "fsrs6_adp")
DEFAULT_METRIC = "avg_accum_memorized_per_hour"
USER_FILE_RE = re.compile(r"simulation_results_retention_sweep_user_(\d+)\.json$")
DR_PERCENT_RE = re.compile(r"\bDR=(\d+(?:\.\d+)?)%")
DR_TOKEN_RE = re.compile(r"(?:^|[_\W])dr[_=-]([01]?(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class SweepRow:
    environment: str
    scheduler: str
    user_id: int
    desired_retention: float | None
    memorized_average: float
    time_average: float
    reviews_average: float
    efficiency: float
    path: Path
    mtime_ns: int
    series_identity: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Compare FSRS-6-family schedulers on retention_sweep user logs. "
            "Defaults match the first-8-user fsrs6/lstm batch comparison."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Unified rl_scheduler TOML config. Direct CLI flags override config values.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Formal experiment run root, used only for recorded command context.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional Markdown file to write the report to.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/retention_sweep"),
        help="Root directory containing retention_sweep result JSON files.",
    )
    parser.add_argument(
        "--env",
        default=",".join(DEFAULT_ENVS),
        help="Comma-separated environments to include.",
    )
    parser.add_argument(
        "--sched",
        default=",".join(DEFAULT_SCHEDULERS),
        help="Comma-separated schedulers to include.",
    )
    parser.add_argument(
        "--comparisons",
        default=(
            "fsrs6_adr_direct:fsrs6,fsrs6_adr_delta:fsrs6,"
            "fsrs6_adr_delta:fsrs6_adr_direct,fsrs6_adp:fsrs6"
        ),
        help="Comma-separated pairwise comparisons as left:right.",
    )
    parser.add_argument(
        "--start-user",
        type=int,
        default=1,
        help="First user ID to include.",
    )
    parser.add_argument(
        "--end-user",
        type=int,
        default=8,
        help="Last user ID to include.",
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
        default="batched",
        help="Filter logs by simulation engine.",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="off",
        help="Filter logs by short-term flag.",
    )
    parser.add_argument(
        "--fuzz",
        choices=["on", "off", "any"],
        default="off",
        help="Filter logs by fuzz flag.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Efficiency metric field to compare.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help=(
            "Keep all matching records. By default the newest file wins for each "
            "(env, user, scheduler, DR) key so stale result directories do not "
            "affect the comparison."
        ),
    )
    args = parser.parse_args(argv)
    if args.config is not None:
        args = _merge_config_args(cli_args=args, config_path=args.config, argv=argv)
    return args


def _resolve_repo_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (REPO_ROOT / expanded).resolve()


def _merge_config_args(
    *,
    cli_args: argparse.Namespace,
    config_path: Path,
    argv: list[str],
) -> argparse.Namespace:
    config = ExperimentConfig.from_toml(config_path)
    analyze = config.analyze_pareto
    build = config.build_pareto
    sweep = config.sweep_batched
    if analyze.log_dir is not None and not has_flag(argv, "--log-dir"):
        cli_args.log_dir = _resolve_repo_path(analyze.log_dir)
    if not has_flag(argv, "--env"):
        envs = analyze.envs or build.envs or sweep.envs or DEFAULT_ENVS
        cli_args.env = ",".join(envs)
    if not has_flag(argv, "--sched"):
        schedulers = (
            analyze.schedulers
            or build.schedulers
            or (
                (config.baseline.scheduler, *sweep.schedulers)
                if sweep.schedulers
                else DEFAULT_SCHEDULERS
            )
        )
        cli_args.sched = ",".join(dict.fromkeys(schedulers))
    if not (has_flag(argv, "--start-user") or has_flag(argv, "--end-user")):
        cli_args.start_user = min(config.users.train)
        cli_args.end_user = max(config.users.train)
    if not has_flag(argv, "--start-retention"):
        cli_args.start_retention = analyze.start_retention
    if not has_flag(argv, "--end-retention"):
        cli_args.end_retention = analyze.end_retention
    if not has_flag(argv, "--engine"):
        cli_args.engine = analyze.engine
    if not has_flag(argv, "--short-term"):
        cli_args.short_term = analyze.short_term
    if not has_flag(argv, "--fuzz"):
        cli_args.fuzz = analyze.fuzz
    if not has_flag(argv, "--metric"):
        cli_args.metric = analyze.metric
    if analyze.no_dedupe and not has_flag(argv, "--no-dedupe"):
        cli_args.no_dedupe = True
    if analyze.comparisons and not has_flag(argv, "--comparisons"):
        cli_args.comparisons = ",".join(analyze.comparisons)
    return cli_args


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_comparisons(value: str) -> tuple[tuple[str, str], ...]:
    comparisons: list[tuple[str, str]] = []
    for item in parse_csv(value):
        if ":" not in item:
            raise ValueError(f"Invalid comparison {item!r}; expected left:right.")
        left, right = item.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid comparison {item!r}; expected left:right.")
        comparisons.append((left, right))
    return tuple(comparisons)


def load_result_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if not isinstance(loaded, list):
            raise ValueError(f"{path} does not contain a JSON list")
        return [item for item in loaded if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            items.append(loaded)
    return items


def parse_desired_retention(item: dict[str, Any]) -> float | None:
    for key in (
        "desired_retention",
        "fsrs6_adr_direct_baseline_desired_retention",
        "fsrs6_adp_baseline_desired_retention",
        "retention",
    ):
        value = item.get(key)
        if value is not None:
            return float(value)

    title = str(item.get("title", ""))
    match = DR_PERCENT_RE.search(title)
    if match:
        return float(match.group(1)) / 100.0
    match = DR_TOKEN_RE.search(title)
    if match:
        return float(match.group(1))
    return None


def bool_filter_matches(value: Any, wanted: str) -> bool:
    if wanted == "any":
        return True
    return bool(value) is (wanted == "on")


def iter_candidate_paths(log_dir: Path, start_user: int, end_user: int) -> list[Path]:
    paths: list[Path] = []
    for path in log_dir.rglob("simulation_results_retention_sweep_user_*.json"):
        match = USER_FILE_RE.match(path.name)
        if not match:
            continue
        user_id = int(match.group(1))
        if start_user <= user_id <= end_user:
            paths.append(path)
    return sorted(paths)


def row_from_item(
    item: dict[str, Any],
    *,
    path: Path,
    mtime_ns: int,
    metric: str,
) -> SweepRow | None:
    desired_retention = parse_desired_retention(item)
    if desired_retention is None and item.get("scheduler") != "fsrs6_adr_direct":
        return None
    return SweepRow(
        environment=str(item["environment"]),
        scheduler=str(item["scheduler"]),
        user_id=int(item["user_id"]),
        desired_retention=desired_retention,
        memorized_average=float(item["memorized_average"]),
        time_average=float(item["time_average"]),
        reviews_average=float(item["reviews_average"]),
        efficiency=float(item[metric]),
        path=path,
        mtime_ns=mtime_ns,
        series_identity=_row_series_identity(item),
    )


def _row_series_identity(item: dict[str, Any]) -> str | None:
    if item.get("scheduler") == "fsrs6_adr_direct":
        policy = item.get("fsrs6_adr_direct_policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    title = item.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return None


def load_rows(args: argparse.Namespace) -> tuple[list[SweepRow], int]:
    envs = set(parse_csv(args.env))
    schedulers = set(parse_csv(args.sched))
    raw_rows: list[SweepRow] = []
    for path in iter_candidate_paths(args.log_dir, args.start_user, args.end_user):
        mtime_ns = path.stat().st_mtime_ns
        try:
            items = load_result_items(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        for item in items:
            if item.get("environment") not in envs:
                continue
            if item.get("scheduler") not in schedulers:
                continue
            user_id = int(item.get("user_id", -1))
            if not (args.start_user <= user_id <= args.end_user):
                continue
            if args.engine != "any" and item.get("engine") != args.engine:
                continue
            if not bool_filter_matches(item.get("short_term"), args.short_term):
                continue
            if not bool_filter_matches(item.get("fuzz"), args.fuzz):
                continue
            try:
                row = row_from_item(
                    item,
                    path=path,
                    mtime_ns=mtime_ns,
                    metric=args.metric,
                )
            except (KeyError, TypeError, ValueError) as exc:
                print(f"warning: skipping row in {path}: {exc}", file=sys.stderr)
                continue
            if row is not None and (
                row.desired_retention is None
                or args.start_retention <= row.desired_retention <= args.end_retention
            ):
                raw_rows.append(row)

    if args.no_dedupe:
        return raw_rows, len(raw_rows)

    latest: dict[tuple[str, int, str, object], SweepRow] = {}
    for row in raw_rows:
        dr_key: object
        if row.desired_retention is None:
            dr_key = row.series_identity or str(row.path)
        else:
            dr_key = round(row.desired_retention * 10000)
        key = (
            row.environment,
            row.user_id,
            row.scheduler,
            dr_key,
        )
        previous = latest.get(key)
        if previous is None or (row.mtime_ns, str(row.path)) > (
            previous.mtime_ns,
            str(previous.path),
        ):
            latest[key] = row
    return list(latest.values()), len(raw_rows)


def average(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def average_optional(values: list[float | None]) -> float:
    return average([value for value in values if value is not None])


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_int(value: float) -> str:
    return f"{value:.0f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def grouped(rows: list[SweepRow]) -> dict[tuple[str, str], list[SweepRow]]:
    groups: dict[tuple[str, str], list[SweepRow]] = {}
    for row in rows:
        groups.setdefault((row.environment, row.scheduler), []).append(row)
    return groups


def coverage_table(
    rows: list[SweepRow], envs: tuple[str, ...], schedulers: tuple[str, ...]
) -> str:
    groups = grouped(rows)
    output_rows: list[list[str]] = []
    for env in envs:
        for scheduler in schedulers:
            group_rows = groups.get((env, scheduler), [])
            users = sorted({row.user_id for row in group_rows})
            drs = sorted(
                {
                    round(row.desired_retention, 4)
                    for row in group_rows
                    if row.desired_retention is not None
                }
            )
            output_rows.append(
                [
                    env,
                    scheduler,
                    str(len(group_rows)),
                    f"{users[0]}-{users[-1]}" if users else "-",
                    str(len(drs)),
                ]
            )
    return markdown_table(["env", "scheduler", "records", "users", "DRs"], output_rows)


def aggregate_table(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
) -> str:
    output_rows: list[list[str]] = []
    for scheduler in schedulers:
        selected = [
            row for row in rows if row.environment == env and row.scheduler == scheduler
        ]
        output_rows.append(
            [
                scheduler,
                fmt_float(average([row.memorized_average for row in selected]), 1),
                fmt_float(average([row.time_average for row in selected]), 2),
                fmt_float(average([row.efficiency for row in selected]), 2),
                fmt_float(average([row.reviews_average for row in selected]), 2),
            ]
        )
    return markdown_table(
        ["scheduler", "avg memorized", "avg time", "avg efficiency", "avg reviews"],
        output_rows,
    )


def pairwise_rows(
    rows: list[SweepRow],
    env: str,
    comparisons: tuple[tuple[str, str], ...],
) -> list[list[str]]:
    env_rows = [row for row in rows if row.environment == env]
    by_key = {
        (row.scheduler, row.user_id, round(row.desired_retention * 10000)): row
        for row in env_rows
        if row.desired_retention is not None
    }
    output_rows: list[list[str]] = []
    for left, right in comparisons:
        deltas_mem: list[float] = []
        deltas_time: list[float] = []
        deltas_eff: list[float] = []
        mem_wins = 0
        time_wins = 0
        eff_wins = 0
        pair_count = 0
        for row in env_rows:
            if row.scheduler != left:
                continue
            if row.desired_retention is None:
                continue
            key = (right, row.user_id, round(row.desired_retention * 10000))
            other = by_key.get(key)
            if other is None:
                continue
            pair_count += 1
            delta_mem = row.memorized_average - other.memorized_average
            delta_time = row.time_average - other.time_average
            delta_eff = row.efficiency - other.efficiency
            deltas_mem.append(delta_mem)
            deltas_time.append(delta_time)
            deltas_eff.append(delta_eff)
            mem_wins += delta_mem > 0
            time_wins += delta_time < 0
            eff_wins += delta_eff > 0
        output_rows.append(
            [
                f"{left} - {right}",
                str(pair_count),
                fmt_float(average(deltas_mem), 1),
                fmt_float(average(deltas_time), 2),
                fmt_float(average(deltas_eff), 2),
                f"{eff_wins}/{pair_count}",
                f"{time_wins}/{pair_count}",
                f"{mem_wins}/{pair_count}",
            ]
        )
    return output_rows


def dominance_rows(
    rows: list[SweepRow],
    env: str,
    comparisons: tuple[tuple[str, str], ...],
) -> list[list[str]]:
    env_rows = [row for row in rows if row.environment == env]
    by_key = {
        (row.scheduler, row.user_id, round(row.desired_retention * 10000)): row
        for row in env_rows
        if row.desired_retention is not None
    }
    output_rows: list[list[str]] = []
    for left, right in comparisons:
        pair_count = 0
        left_dominates = 0
        right_dominates = 0
        left_more_mem_more_time = 0
        left_less_mem_less_time = 0
        equal = 0
        for row in env_rows:
            if row.scheduler != left:
                continue
            if row.desired_retention is None:
                continue
            key = (right, row.user_id, round(row.desired_retention * 10000))
            other = by_key.get(key)
            if other is None:
                continue
            pair_count += 1
            left_no_worse = (
                row.memorized_average >= other.memorized_average
                and row.time_average <= other.time_average
            )
            left_strictly_better = (
                row.memorized_average > other.memorized_average
                or row.time_average < other.time_average
            )
            right_no_worse = (
                other.memorized_average >= row.memorized_average
                and other.time_average <= row.time_average
            )
            right_strictly_better = (
                other.memorized_average > row.memorized_average
                or other.time_average < row.time_average
            )
            if left_no_worse and left_strictly_better:
                left_dominates += 1
            elif right_no_worse and right_strictly_better:
                right_dominates += 1
            elif (
                row.memorized_average == other.memorized_average
                and row.time_average == other.time_average
            ):
                equal += 1
            elif (
                row.memorized_average > other.memorized_average
                and row.time_average > other.time_average
            ):
                left_more_mem_more_time += 1
            elif (
                row.memorized_average < other.memorized_average
                and row.time_average < other.time_average
            ):
                left_less_mem_less_time += 1
            else:
                equal += 1

        output_rows.append(
            [
                f"{left} - {right}",
                str(pair_count),
                f"{left_dominates}/{pair_count}",
                f"{right_dominates}/{pair_count}",
                f"{left_more_mem_more_time}/{pair_count}",
                f"{left_less_mem_less_time}/{pair_count}",
                f"{equal}/{pair_count}",
            ]
        )
    return output_rows


def best_by_user(
    rows: list[SweepRow],
    *,
    env: str,
    scheduler: str,
    mode: str,
) -> dict[int, SweepRow]:
    selected = [
        row for row in rows if row.environment == env and row.scheduler == scheduler
    ]
    best: dict[int, SweepRow] = {}
    for row in selected:
        previous = best.get(row.user_id)
        if previous is None:
            best[row.user_id] = row
            continue
        if mode == "efficiency":
            row_key = (row.efficiency, row.memorized_average, -row.time_average)
            previous_key = (
                previous.efficiency,
                previous.memorized_average,
                -previous.time_average,
            )
        elif mode == "memory":
            row_key = (row.memorized_average, -row.time_average, row.efficiency)
            previous_key = (
                previous.memorized_average,
                -previous.time_average,
                previous.efficiency,
            )
        else:
            raise ValueError(f"Unknown best mode: {mode}")
        if row_key > previous_key:
            best[row.user_id] = row
    return best


def best_summary_table(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    mode: str,
) -> str:
    output_rows: list[list[str]] = []
    for scheduler in schedulers:
        best = best_by_user(rows, env=env, scheduler=scheduler, mode=mode)
        values = list(best.values())
        output_rows.append(
            [
                scheduler,
                fmt_float(average([row.efficiency for row in values]), 2),
                fmt_float(average([row.memorized_average for row in values]), 1),
                fmt_float(average([row.time_average for row in values]), 2),
                fmt_float(average([row.reviews_average for row in values]), 2),
                fmt_float(
                    average_optional([row.desired_retention for row in values]), 3
                ),
            ]
        )
    return markdown_table(
        [
            "scheduler",
            "avg efficiency",
            "avg memorized",
            "avg time",
            "avg reviews",
            "avg DR",
        ],
        output_rows,
    )


def winner_counts(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
) -> dict[str, list[int]]:
    per_scheduler = {
        scheduler: best_by_user(rows, env=env, scheduler=scheduler, mode="efficiency")
        for scheduler in schedulers
    }
    users = sorted(
        {
            user_id
            for scheduler_best in per_scheduler.values()
            for user_id in scheduler_best
        }
    )
    winners: dict[str, list[int]] = {scheduler: [] for scheduler in schedulers}
    for user_id in users:
        candidates = [
            per_scheduler[scheduler][user_id]
            for scheduler in schedulers
            if user_id in per_scheduler[scheduler]
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: (row.efficiency, row.memorized_average, -row.time_average),
        )
        winners[best.scheduler].append(user_id)
    return winners


def pareto_frontier(rows: list[SweepRow]) -> list[SweepRow]:
    frontier: list[SweepRow] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = (
                other.memorized_average >= candidate.memorized_average
                and other.time_average <= candidate.time_average
            )
            strictly_better = (
                other.memorized_average > candidate.memorized_average
                or other.time_average < candidate.time_average
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def hypervolume_summary_table(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    portfolio_scheduler: str = "fsrs6_adr_direct",
) -> str:
    output_rows: list[list[str]] = []
    user_ids = sorted({row.user_id for row in rows if row.environment == env})
    for user_id in user_ids:
        baseline = [
            row
            for row in rows
            if row.environment == env
            and row.user_id == user_id
            and row.scheduler == baseline_scheduler
        ]
        portfolio = [
            row
            for row in rows
            if row.environment == env
            and row.user_id == user_id
            and row.scheduler == portfolio_scheduler
        ]
        if not baseline or not portfolio:
            continue
        baseline_points = [
            ObjectivePoint(row.memorized_average, -row.time_average) for row in baseline
        ]
        portfolio_points = [
            ObjectivePoint(row.memorized_average, -row.time_average)
            for row in portfolio
        ]
        reference = reference_point(baseline_points, margin_fraction=0.05)
        baseline_hv = hypervolume_2d(baseline_points, reference=reference)
        combined_points = [*baseline_points, *portfolio_points]
        portfolio_hv = hypervolume_2d(combined_points, reference=reference)
        frontier_indices = non_dominated_indices(combined_points)
        frontier_child_count = sum(
            1 for index in frontier_indices if index >= len(baseline_points)
        )
        output_rows.append(
            [
                str(user_id),
                fmt_float(baseline_hv, 2),
                fmt_float(portfolio_hv, 2),
                fmt_float(portfolio_hv - baseline_hv, 2),
                str(frontier_child_count),
            ]
        )
    if not output_rows:
        return "No baseline + portfolio rows available."
    return markdown_table(
        ["user", "baseline HV", "portfolio HV", "HV delta", "frontier children"],
        output_rows,
    )


def pareto_counts(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
) -> tuple[int, dict[str, int], dict[str, list[int]]]:
    total = 0
    counts = {scheduler: 0 for scheduler in schedulers}
    users = {scheduler: set[int]() for scheduler in schedulers}
    env_users = sorted({row.user_id for row in rows if row.environment == env})
    for user_id in env_users:
        frontier = pareto_frontier(
            [row for row in rows if row.environment == env and row.user_id == user_id]
        )
        total += len(frontier)
        for row in frontier:
            if row.scheduler in counts:
                counts[row.scheduler] += 1
                users[row.scheduler].add(row.user_id)
    return (
        total,
        counts,
        {
            scheduler: sorted(scheduler_users)
            for scheduler, scheduler_users in users.items()
        },
    )


def print_env_report(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
) -> None:
    print(f"\n## {env} environment\n")
    print("### All DR grid points\n")
    print(aggregate_table(rows, env, schedulers))

    print("\n### Same-user same-DR deltas\n")
    print(
        markdown_table(
            [
                "comparison",
                "pairs",
                "delta memorized",
                "delta time",
                "delta efficiency",
                "eff wins",
                "time wins",
                "mem wins",
            ],
            pairwise_rows(rows, env, comparisons),
        )
    )

    print("\n### Same-user same-DR dominance\n")
    print(
        markdown_table(
            [
                "comparison",
                "pairs",
                "left dominates",
                "right dominates",
                "left mem+ time+",
                "left mem- time-",
                "equal",
            ],
            dominance_rows(rows, env, comparisons),
        )
    )

    print("\n### Best efficiency point per user\n")
    print(best_summary_table(rows, env, schedulers, mode="efficiency"))
    winners = winner_counts(rows, env, schedulers)
    winner_text = ", ".join(
        f"{scheduler}: {len(user_ids)}/{sum(len(ids) for ids in winners.values())}"
        f" users {user_ids}"
        for scheduler, user_ids in winners.items()
        if user_ids
    )
    print(f"\nBest-efficiency winners: {winner_text}")

    print("\n### Max-memory point per user\n")
    print(best_summary_table(rows, env, schedulers, mode="memory"))

    total, counts, users = pareto_counts(rows, env, schedulers)
    pareto_rows = [
        [scheduler, str(counts[scheduler]), str(users[scheduler])]
        for scheduler in schedulers
    ]
    print(f"\n### Pareto frontier by user\n\nTotal frontier points: {total}\n")
    print(markdown_table(["scheduler", "frontier points", "users"], pareto_rows))

    if "fsrs6" in schedulers and "fsrs6_adr_direct" in schedulers:
        print("\n### Hypervolume vs FSRS6 baseline\n")
        print(hypervolume_summary_table(rows, env))


def render_report(args: argparse.Namespace) -> str:
    envs = parse_csv(args.env)
    schedulers = parse_csv(args.sched)
    comparisons = parse_comparisons(args.comparisons)
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    rows, candidate_count = load_rows(args)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("# FSRS-6 Scheduler Comparison\n")
        print(
            "Filters: "
            f"env={','.join(envs)}, "
            f"scheduler={','.join(schedulers)}, "
            f"users={args.start_user}-{args.end_user}, "
            f"retention={args.start_retention:.2f}-{args.end_retention:.2f}, "
            f"engine={args.engine}, short_term={args.short_term}, fuzz={args.fuzz}"
        )
        print(
            f"Loaded {len(rows)} records"
            + (
                f" after latest-file dedupe from {candidate_count} matching records."
                if not args.no_dedupe
                else "."
            )
        )

        print("\n## Coverage\n")
        print(coverage_table(rows, envs, schedulers))

        for env in envs:
            print_env_report(rows, env, schedulers, comparisons)
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    report = render_report(args)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

# ruff: noqa: E402

import argparse
import io
import json
import math
import re
import statistics
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.experiment_infra import ExperimentConfig
from simulator.experiment_infra.baseline_dr_selection import (
    BaselineDRManifest,
    load_baseline_dr_manifest,
)
from experiments.retention_sweep.cli_utils import has_flag
from experiments.rl_scheduler.train_fsrs6_adr_portfolio import (
    ObjectivePoint,
    hypervolume_2d,
    non_dominated_indices,
    reference_point,
)
from simulator.scheduler_catalog import PolicySource, schedulers_for_policy_source


DEFAULT_ENVS = ("fsrs6", "lstm")
DEFAULT_SCHEDULERS = (
    "fsrs6",
    "fsrs6_adr",
    "fsrs6_adr_time",
    "fsrs6_ap",
    "anki_sm2_ap",
)
DEFAULT_METRIC = "avg_accum_memorized_per_hour"
USER_FILE_RE = re.compile(r"simulation_results_retention_sweep_user_(\d+)\.json$")
DR_PERCENT_RE = re.compile(r"\bDR=(\d+(?:\.\d+)?)%")
DR_TOKEN_RE = re.compile(r"(?:^|[_\W])dr[_=-]([01]?(?:\.\d+)?)", re.IGNORECASE)
ADR_POLICY_SCHEDULERS = schedulers_for_policy_source(PolicySource.FSRS6_ADR)
ORACLE_DISTILL_SCHEDULER = "fsrs6_oracle_stationary_finite_distill"


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


@dataclass(frozen=True)
class HypervolumeUserSummary:
    user_id: int
    baseline_hv: float
    target_hv: float
    hv_delta: float
    target_frontier_count: int


@dataclass(frozen=True)
class BudgetMemoryGainAucUserSummary:
    user_id: int
    budget_count: int
    covered_budget_count: int
    total_span: float
    covered_span: float
    memory_gain_auc: float | None
    baseline_memory_auc: float | None


@dataclass(frozen=True)
class MemoryTargetRegretAucUserSummary:
    user_id: int
    target_count: int
    covered_target_count: int
    total_span: float
    covered_span: float
    time_regret_auc: float | None
    baseline_time_auc: float | None


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
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON file to write the machine-readable analysis summary to.",
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
        default="fsrs6_adr:fsrs6,fsrs6_ap:fsrs6",
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
        choices=["event", "batched", "any"],
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
    parser.add_argument(
        "--baseline-dr-manifest",
        type=Path,
        default=None,
        help=(
            "JSON manifest of per-user FSRS6 baseline desired-retention values. "
            "Rows with a non-null DR outside the user's manifest are ignored."
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
    if config.baseline_dr_selection.manifest is not None and not has_flag(
        argv, "--baseline-dr-manifest"
    ):
        cli_args.baseline_dr_manifest = _resolve_repo_path(
            config.baseline_dr_selection.manifest
        )
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
        "fsrs6_adr_baseline_desired_retention",
        "fsrs6_ap_baseline_desired_retention",
        "anki_sm2_ap_baseline_desired_retention",
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
    if desired_retention is None and item.get("scheduler") not in {
        *ADR_POLICY_SCHEDULERS,
        "fsrs6_ap",
        "anki_sm2_ap",
        ORACLE_DISTILL_SCHEDULER,
    }:
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
    if item.get("scheduler") in ADR_POLICY_SCHEDULERS:
        policy = item.get("fsrs6_adr_policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    if item.get("scheduler") == "fsrs6_ap":
        policy = item.get("fsrs6_ap_policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    if item.get("scheduler") == "anki_sm2_ap":
        policy = item.get("anki_sm2_ap_policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    if item.get("scheduler") == ORACLE_DISTILL_SCHEDULER:
        policy = item.get("fsrs6_oracle_stationary_finite_distill_policy")
        if isinstance(policy, str) and policy.strip():
            return policy
    title = item.get("title")
    if isinstance(title, str) and title.strip():
        return title
    return None


def load_rows(args: argparse.Namespace) -> tuple[list[SweepRow], int]:
    envs = set(parse_csv(args.env))
    schedulers = set(parse_csv(args.sched))
    baseline_dr_manifest: BaselineDRManifest | None = (
        load_baseline_dr_manifest(args.baseline_dr_manifest)
        if args.baseline_dr_manifest is not None
        else None
    )
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
            engine_value = item.get("engine")
            if engine_value not in {"event", "batched"}:
                continue
            if args.engine != "any" and engine_value != args.engine:
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
                if (
                    baseline_dr_manifest is not None
                    and row.desired_retention is not None
                    and row.scheduler
                    in {
                        "fsrs6",
                        *ADR_POLICY_SCHEDULERS,
                        "fsrs6_ap",
                        "anki_sm2_ap",
                    }
                    and not baseline_dr_manifest.contains_value(
                        row.user_id,
                        row.desired_retention,
                    )
                ):
                    continue
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


def fmt_percent(value: float) -> str:
    return f"{value:.3f}%"


def fmt_quantile(value: float) -> str:
    return f"{value:.2f}"


def finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def fmt_float_optional(value: float | None, digits: int = 2) -> str:
    if value is None:
        return f"{float('nan'):.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_int_optional(value: float | None) -> str:
    if value is None:
        return f"{float('nan'):.0f}"
    return f"{value:.0f}"


def fmt_percent_optional(value: float | None) -> str:
    if value is None:
        return f"{float('nan'):.3f}%"
    return f"{value:.3f}%"


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def five_number_summary(
    values: list[float],
) -> tuple[float, float, float, float, float]:
    return (
        min(values) if values else float("nan"),
        nearest_rank(values, 0.25),
        nearest_rank(values, 0.50),
        nearest_rank(values, 0.75),
        max(values) if values else float("nan"),
    )


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
        [
            "scheduler",
            "policy-point avg memorized",
            "policy-point avg time",
            "policy-point avg efficiency",
            "policy-point avg reviews",
        ],
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


def hypervolume_user_summaries(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    target_scheduler: str = "fsrs6_adr",
) -> list[HypervolumeUserSummary]:
    summaries: list[HypervolumeUserSummary] = []
    user_ids = sorted({row.user_id for row in rows if row.environment == env})
    for user_id in user_ids:
        baseline = [
            row
            for row in rows
            if row.environment == env
            and row.user_id == user_id
            and row.scheduler == baseline_scheduler
        ]
        target = [
            row
            for row in rows
            if row.environment == env
            and row.user_id == user_id
            and row.scheduler == target_scheduler
        ]
        if not baseline or not target:
            continue
        baseline_points = [
            ObjectivePoint(row.memorized_average, -row.time_average) for row in baseline
        ]
        target_points = [
            ObjectivePoint(row.memorized_average, -row.time_average) for row in target
        ]
        reference = reference_point(baseline_points, margin_fraction=0.05)
        baseline_hv = hypervolume_2d(baseline_points, reference=reference)
        target_hv = hypervolume_2d(target_points, reference=reference)
        target_frontier_count = len(non_dominated_indices(target_points))
        summaries.append(
            HypervolumeUserSummary(
                user_id=user_id,
                baseline_hv=baseline_hv,
                target_hv=target_hv,
                hv_delta=target_hv - baseline_hv,
                target_frontier_count=target_frontier_count,
            )
        )
    return summaries


def primary_hypervolume_summary_table(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    target_schedulers: tuple[str, ...],
) -> str:
    output_rows: list[list[str]] = []
    for target_scheduler in target_schedulers:
        summaries = hypervolume_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=target_scheduler,
        )
        if not summaries:
            continue
        baseline_hv_sum = sum(summary.baseline_hv for summary in summaries)
        target_hv_sum = sum(summary.target_hv for summary in summaries)
        hv_delta_sum = sum(summary.hv_delta for summary in summaries)
        hv_delta_ratio = (
            (hv_delta_sum / baseline_hv_sum) * 100.0
            if baseline_hv_sum
            else float("nan")
        )
        hv_min, hv_p25, hv_median, hv_p75, hv_max = five_number_summary(
            [summary.hv_delta for summary in summaries]
        )
        output_rows.append(
            [
                target_scheduler,
                fmt_float(baseline_hv_sum, 2),
                fmt_float(target_hv_sum, 2),
                fmt_float(hv_delta_sum, 2),
                fmt_percent(hv_delta_ratio),
                fmt_float(hv_min, 2),
                fmt_float(hv_p25, 2),
                fmt_float(hv_median, 2),
                fmt_float(hv_p75, 2),
                fmt_float(hv_max, 2),
                str(sum(summary.target_frontier_count for summary in summaries)),
                str(len(summaries)),
            ]
        )
    if not output_rows:
        return f"No {baseline_scheduler} comparison rows available."
    return markdown_table(
        [
            "scheduler",
            "baseline HV sum",
            "scheduler HV sum",
            "HV delta sum",
            "HV delta / baseline HV",
            "HV delta min",
            "HV delta p25",
            "HV delta median",
            "HV delta p75",
            "HV delta max",
            "scheduler frontier points",
            "users",
        ],
        output_rows,
    )


def hypervolume_summary_table(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    target_scheduler: str = "fsrs6_adr",
) -> str:
    summaries = hypervolume_user_summaries(
        rows,
        env,
        baseline_scheduler=baseline_scheduler,
        target_scheduler=target_scheduler,
    )
    output_rows: list[list[str]] = []
    for summary in summaries:
        output_rows.append(
            [
                str(summary.user_id),
                fmt_float(summary.baseline_hv, 2),
                fmt_float(summary.target_hv, 2),
                fmt_float(summary.hv_delta, 2),
                str(summary.target_frontier_count),
            ]
        )
    if not output_rows:
        return f"No {baseline_scheduler} + {target_scheduler} rows available."
    return markdown_table(
        [
            "user",
            "baseline HV",
            f"{target_scheduler} HV",
            "HV delta",
            "scheduler frontier points",
        ],
        output_rows,
    )


def rows_for(
    rows: list[SweepRow],
    *,
    env: str,
    scheduler: str,
    user_id: int,
) -> list[SweepRow]:
    return [
        row
        for row in rows
        if row.environment == env
        and row.scheduler == scheduler
        and row.user_id == user_id
    ]


def sorted_frontier(rows: list[SweepRow], *, sort_key: str) -> list[SweepRow]:
    frontier = pareto_frontier(rows)
    if sort_key == "time":
        return sorted(
            frontier,
            key=lambda row: (row.time_average, row.memorized_average, row.scheduler),
        )
    if sort_key == "memory":
        return sorted(
            frontier,
            key=lambda row: (row.memorized_average, -row.time_average, row.scheduler),
        )
    raise ValueError(f"Unknown frontier sort key: {sort_key}")


def best_memorized_under_budget(rows: list[SweepRow], budget: float) -> float | None:
    feasible = [row.memorized_average for row in rows if row.time_average <= budget]
    return max(feasible) if feasible else None


def min_time_for_memory_target(rows: list[SweepRow], target: float) -> float | None:
    feasible = [row.time_average for row in rows if row.memorized_average >= target]
    return min(feasible) if feasible else None


def interpolated_memorized_under_budget(
    rows: list[SweepRow], budget: float
) -> float | None:
    frontier = sorted_frontier(rows, sort_key="time")
    if not frontier:
        return None
    if budget < frontier[0].time_average:
        return None
    if budget > frontier[-1].time_average:
        return None
    if math.isclose(budget, frontier[-1].time_average):
        return frontier[-1].memorized_average

    for left, right in zip(frontier[:-1], frontier[1:]):
        left_time = left.time_average
        right_time = right.time_average
        if not (left_time <= budget <= right_time):
            continue
        if math.isclose(left_time, right_time):
            return max(left.memorized_average, right.memorized_average)
        ratio = (budget - left_time) / (right_time - left_time)
        return left.memorized_average + ratio * (
            right.memorized_average - left.memorized_average
        )
    return None


def interpolated_min_time_for_memory_target(
    rows: list[SweepRow], target: float
) -> float | None:
    frontier = sorted_frontier(rows, sort_key="memory")
    if not frontier:
        return None
    if target < frontier[0].memorized_average:
        return None
    if math.isclose(target, frontier[0].memorized_average):
        return frontier[0].time_average
    if target > frontier[-1].memorized_average:
        return None

    for left, right in zip(frontier[:-1], frontier[1:]):
        left_memory = left.memorized_average
        right_memory = right.memorized_average
        if not (left_memory <= target <= right_memory):
            continue
        if math.isclose(left_memory, right_memory):
            return min(left.time_average, right.time_average)
        ratio = (target - left_memory) / (right_memory - left_memory)
        return left.time_average + ratio * (right.time_average - left.time_average)
    return None


def common_interval(
    left_min: float,
    left_max: float,
    right_min: float,
    right_max: float,
) -> tuple[float, float] | None:
    start = max(left_min, right_min)
    end = min(left_max, right_max)
    if end <= start:
        return None
    return start, end


def values_in_interval(values: list[float], start: float, end: float) -> list[float]:
    return sorted(
        {
            value
            for value in values
            if (start < value < end)
            or math.isclose(value, start)
            or math.isclose(value, end)
        }
    )


def integration_grid(
    baseline_values: list[float],
    target_values: list[float],
    start: float,
    end: float,
) -> list[float]:
    return sorted(
        {
            start,
            end,
            *values_in_interval(baseline_values, start, end),
            *values_in_interval(target_values, start, end),
        }
    )


def budget_memory_gain_auc_user_summaries(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    target_scheduler: str,
) -> list[BudgetMemoryGainAucUserSummary]:
    summaries: list[BudgetMemoryGainAucUserSummary] = []
    user_ids = sorted({row.user_id for row in rows if row.environment == env})
    for user_id in user_ids:
        baseline_rows = rows_for(
            rows,
            env=env,
            scheduler=baseline_scheduler,
            user_id=user_id,
        )
        target_rows = rows_for(
            rows,
            env=env,
            scheduler=target_scheduler,
            user_id=user_id,
        )
        baseline_frontier = sorted_frontier(baseline_rows, sort_key="time")
        target_frontier = sorted_frontier(target_rows, sort_key="time")
        if not baseline_frontier:
            continue

        baseline_budgets = sorted({row.time_average for row in baseline_frontier})
        total_span = (
            max(baseline_budgets) - min(baseline_budgets)
            if len(baseline_budgets) > 1
            else 0.0
        )
        covered_span = 0.0
        memory_gain_area = 0.0
        baseline_memory_area = 0.0
        covered_budget_count = 0
        if target_frontier:
            interval = common_interval(
                baseline_frontier[0].time_average,
                baseline_frontier[-1].time_average,
                target_frontier[0].time_average,
                target_frontier[-1].time_average,
            )
            if interval is not None:
                start, end = interval
                covered_budget_count = len(
                    values_in_interval(baseline_budgets, start, end)
                )
                budgets = integration_grid(
                    baseline_budgets,
                    sorted({row.time_average for row in target_frontier}),
                    start,
                    end,
                )
                for left_budget, right_budget in zip(budgets[:-1], budgets[1:]):
                    width = right_budget - left_budget
                    if width <= 0.0:
                        continue
                    left_baseline_memory = interpolated_memorized_under_budget(
                        baseline_rows, left_budget
                    )
                    right_baseline_memory = interpolated_memorized_under_budget(
                        baseline_rows, right_budget
                    )
                    left_target_memory = interpolated_memorized_under_budget(
                        target_rows, left_budget
                    )
                    right_target_memory = interpolated_memorized_under_budget(
                        target_rows, right_budget
                    )
                    if (
                        left_baseline_memory is None
                        or right_baseline_memory is None
                        or left_target_memory is None
                        or right_target_memory is None
                    ):
                        continue
                    left_gain = left_target_memory - left_baseline_memory
                    right_gain = right_target_memory - right_baseline_memory
                    memory_gain_area += width * ((left_gain + right_gain) / 2.0)
                    baseline_memory_area += width * (
                        (left_baseline_memory + right_baseline_memory) / 2.0
                    )
                    covered_span += width

        summaries.append(
            BudgetMemoryGainAucUserSummary(
                user_id=user_id,
                budget_count=len(baseline_budgets),
                covered_budget_count=covered_budget_count,
                total_span=total_span,
                covered_span=covered_span,
                memory_gain_auc=(memory_gain_area / covered_span)
                if covered_span
                else None,
                baseline_memory_auc=(baseline_memory_area / covered_span)
                if covered_span
                else None,
            )
        )
    return summaries


def _relative_gain_auc_percent_values(
    summaries: list[BudgetMemoryGainAucUserSummary],
) -> list[float]:
    values: list[float] = []
    for summary in summaries:
        if summary.memory_gain_auc is None or summary.baseline_memory_auc is None:
            continue
        if math.isclose(summary.baseline_memory_auc, 0.0):
            continue
        values.append((summary.memory_gain_auc / summary.baseline_memory_auc) * 100.0)
    return values


def budget_memory_gain_auc_table(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    baseline_scheduler: str = "fsrs6",
) -> str:
    output_rows: list[list[str]] = []
    for scheduler in schedulers:
        summaries = budget_memory_gain_auc_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=scheduler,
        )
        if not summaries:
            continue
        budget_count = sum(summary.budget_count for summary in summaries)
        covered_budget_count = sum(
            summary.covered_budget_count for summary in summaries
        )
        total_span = sum(summary.total_span for summary in summaries)
        covered_span = sum(summary.covered_span for summary in summaries)
        memory_gain_aucs = [
            summary.memory_gain_auc
            for summary in summaries
            if summary.memory_gain_auc is not None
        ]
        baseline_memory_aucs = [
            summary.baseline_memory_auc
            for summary in summaries
            if summary.baseline_memory_auc is not None
        ]
        relative_gain_auc_values = _relative_gain_auc_percent_values(summaries)
        span_coverage = (covered_span / total_span) * 100.0 if total_span else 0.0
        relative_gain_auc = (
            average(relative_gain_auc_values)
            if relative_gain_auc_values
            else float("nan")
        )
        output_rows.append(
            [
                scheduler,
                f"{len(memory_gain_aucs)}/{len(summaries)}",
                f"{covered_budget_count}/{budget_count}",
                fmt_percent(span_coverage),
                fmt_float(average(memory_gain_aucs), 1),
                fmt_percent(relative_gain_auc),
            ]
        )
    if not output_rows:
        return f"No {baseline_scheduler} baseline time budgets available."
    return markdown_table(
        [
            "scheduler",
            "AUC users",
            "budget coverage",
            "span coverage",
            f"memory gain AUC vs {baseline_scheduler}",
            f"relative gain AUC vs {baseline_scheduler}",
        ],
        output_rows,
    )


def memory_target_regret_auc_user_summaries(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str = "fsrs6",
    target_scheduler: str,
) -> list[MemoryTargetRegretAucUserSummary]:
    summaries: list[MemoryTargetRegretAucUserSummary] = []
    user_ids = sorted({row.user_id for row in rows if row.environment == env})
    for user_id in user_ids:
        baseline_rows = rows_for(
            rows,
            env=env,
            scheduler=baseline_scheduler,
            user_id=user_id,
        )
        target_rows = rows_for(
            rows,
            env=env,
            scheduler=target_scheduler,
            user_id=user_id,
        )
        baseline_frontier = sorted_frontier(baseline_rows, sort_key="memory")
        target_frontier = sorted_frontier(target_rows, sort_key="memory")
        if not baseline_frontier:
            continue

        baseline_targets = sorted({row.memorized_average for row in baseline_frontier})
        total_span = (
            max(baseline_targets) - min(baseline_targets)
            if len(baseline_targets) > 1
            else 0.0
        )
        covered_span = 0.0
        time_regret_area = 0.0
        baseline_time_area = 0.0
        covered_target_count = 0
        if target_frontier:
            interval = common_interval(
                baseline_frontier[0].memorized_average,
                baseline_frontier[-1].memorized_average,
                target_frontier[0].memorized_average,
                target_frontier[-1].memorized_average,
            )
            if interval is not None:
                start, end = interval
                covered_target_count = len(
                    values_in_interval(baseline_targets, start, end)
                )
                targets = integration_grid(
                    baseline_targets,
                    sorted({row.memorized_average for row in target_frontier}),
                    start,
                    end,
                )
                for left_target, right_target in zip(targets[:-1], targets[1:]):
                    width = right_target - left_target
                    if width <= 0.0:
                        continue
                    left_baseline_time = interpolated_min_time_for_memory_target(
                        baseline_rows, left_target
                    )
                    right_baseline_time = interpolated_min_time_for_memory_target(
                        baseline_rows, right_target
                    )
                    left_target_time = interpolated_min_time_for_memory_target(
                        target_rows, left_target
                    )
                    right_target_time = interpolated_min_time_for_memory_target(
                        target_rows, right_target
                    )
                    if (
                        left_baseline_time is None
                        or right_baseline_time is None
                        or left_target_time is None
                        or right_target_time is None
                    ):
                        continue
                    left_regret = left_target_time - left_baseline_time
                    right_regret = right_target_time - right_baseline_time
                    time_regret_area += width * ((left_regret + right_regret) / 2.0)
                    baseline_time_area += width * (
                        (left_baseline_time + right_baseline_time) / 2.0
                    )
                    covered_span += width

        summaries.append(
            MemoryTargetRegretAucUserSummary(
                user_id=user_id,
                target_count=len(baseline_targets),
                covered_target_count=covered_target_count,
                total_span=total_span,
                covered_span=covered_span,
                time_regret_auc=(time_regret_area / covered_span)
                if covered_span
                else None,
                baseline_time_auc=(baseline_time_area / covered_span)
                if covered_span
                else None,
            )
        )
    return summaries


def _relative_regret_auc_percent_values(
    summaries: list[MemoryTargetRegretAucUserSummary],
) -> list[float]:
    values: list[float] = []
    for summary in summaries:
        if summary.time_regret_auc is None or summary.baseline_time_auc is None:
            continue
        if math.isclose(summary.baseline_time_auc, 0.0):
            continue
        values.append((summary.time_regret_auc / summary.baseline_time_auc) * 100.0)
    return values


def memory_target_regret_auc_table(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    baseline_scheduler: str = "fsrs6",
) -> str:
    output_rows: list[list[str]] = []
    for scheduler in schedulers:
        summaries = memory_target_regret_auc_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=scheduler,
        )
        if not summaries:
            continue
        target_count = sum(summary.target_count for summary in summaries)
        covered_target_count = sum(
            summary.covered_target_count for summary in summaries
        )
        total_span = sum(summary.total_span for summary in summaries)
        covered_span = sum(summary.covered_span for summary in summaries)
        time_regret_aucs = [
            summary.time_regret_auc
            for summary in summaries
            if summary.time_regret_auc is not None
        ]
        baseline_time_aucs = [
            summary.baseline_time_auc
            for summary in summaries
            if summary.baseline_time_auc is not None
        ]
        relative_regret_auc_values = _relative_regret_auc_percent_values(summaries)
        span_coverage = (covered_span / total_span) * 100.0 if total_span else 0.0
        relative_regret_auc = (
            average(relative_regret_auc_values)
            if relative_regret_auc_values
            else float("nan")
        )
        output_rows.append(
            [
                scheduler,
                f"{len(time_regret_aucs)}/{len(summaries)}",
                f"{covered_target_count}/{target_count}",
                fmt_percent(span_coverage),
                fmt_float(average(time_regret_aucs), 2),
                fmt_percent(relative_regret_auc),
            ]
        )
    if not output_rows:
        return f"No {baseline_scheduler} baseline memory targets available."
    return markdown_table(
        [
            "scheduler",
            "AUC users",
            "target coverage",
            "span coverage",
            f"time regret AUC vs {baseline_scheduler}",
            f"relative regret AUC vs {baseline_scheduler}",
        ],
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


def build_coverage_summary(
    rows: list[SweepRow],
    envs: tuple[str, ...],
    schedulers: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups = grouped(rows)
    output_rows: list[dict[str, Any]] = []
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
                {
                    "environment": env,
                    "scheduler": scheduler,
                    "records": len(group_rows),
                    "users": users,
                    "user_range": f"{users[0]}-{users[-1]}" if users else None,
                    "desired_retention_count": len(drs),
                    "desired_retention_values": drs,
                }
            )
    return output_rows


def build_policy_point_diagnostics(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for scheduler in schedulers:
        selected = [
            row for row in rows if row.environment == env and row.scheduler == scheduler
        ]
        output_rows.append(
            {
                "scheduler": scheduler,
                "policy_point_avg_memorized": finite_float(
                    average([row.memorized_average for row in selected])
                ),
                "policy_point_avg_time": finite_float(
                    average([row.time_average for row in selected])
                ),
                "policy_point_avg_efficiency": finite_float(
                    average([row.efficiency for row in selected])
                ),
                "policy_point_avg_reviews": finite_float(
                    average([row.reviews_average for row in selected])
                ),
                "record_count": len(selected),
            }
        )
    return output_rows


def build_pairwise_delta_summary(
    rows: list[SweepRow],
    env: str,
    comparisons: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    env_rows = [row for row in rows if row.environment == env]
    by_key = {
        (row.scheduler, row.user_id, round(row.desired_retention * 10000)): row
        for row in env_rows
        if row.desired_retention is not None
    }
    output_rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        deltas_mem: list[float] = []
        deltas_time: list[float] = []
        deltas_eff: list[float] = []
        mem_wins = 0
        time_wins = 0
        eff_wins = 0
        pair_count = 0
        for row in env_rows:
            if row.scheduler != left or row.desired_retention is None:
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
            {
                "left_scheduler": left,
                "right_scheduler": right,
                "comparison": f"{left} - {right}",
                "pairs": pair_count,
                "delta_memorized_mean": finite_float(average(deltas_mem)),
                "delta_time_mean": finite_float(average(deltas_time)),
                "delta_efficiency_mean": finite_float(average(deltas_eff)),
                "efficiency_wins": eff_wins,
                "time_wins": time_wins,
                "memory_wins": mem_wins,
            }
        )
    return output_rows


def build_dominance_summary(
    rows: list[SweepRow],
    env: str,
    comparisons: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    env_rows = [row for row in rows if row.environment == env]
    by_key = {
        (row.scheduler, row.user_id, round(row.desired_retention * 10000)): row
        for row in env_rows
        if row.desired_retention is not None
    }
    output_rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        pair_count = 0
        left_dominates = 0
        right_dominates = 0
        left_more_mem_more_time = 0
        left_less_mem_less_time = 0
        equal = 0
        for row in env_rows:
            if row.scheduler != left or row.desired_retention is None:
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
            {
                "left_scheduler": left,
                "right_scheduler": right,
                "comparison": f"{left} - {right}",
                "pairs": pair_count,
                "left_dominates": left_dominates,
                "right_dominates": right_dominates,
                "left_more_memory_more_time": left_more_mem_more_time,
                "left_less_memory_less_time": left_less_mem_less_time,
                "equal": equal,
            }
        )
    return output_rows


def build_best_point_summary(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for scheduler in schedulers:
        best = best_by_user(rows, env=env, scheduler=scheduler, mode=mode)
        values = list(best.values())
        output_rows.append(
            {
                "scheduler": scheduler,
                "avg_efficiency": finite_float(
                    average([row.efficiency for row in values])
                ),
                "avg_memorized": finite_float(
                    average([row.memorized_average for row in values])
                ),
                "avg_time": finite_float(average([row.time_average for row in values])),
                "avg_reviews": finite_float(
                    average([row.reviews_average for row in values])
                ),
                "avg_desired_retention": finite_float(
                    average_optional([row.desired_retention for row in values])
                ),
                "users": sorted(best),
            }
        )
    return output_rows


def build_primary_hypervolume_summary(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str,
    target_schedulers: tuple[str, ...],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for target_scheduler in target_schedulers:
        summaries = hypervolume_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=target_scheduler,
        )
        if not summaries:
            continue
        baseline_hv_sum = sum(summary.baseline_hv for summary in summaries)
        target_hv_sum = sum(summary.target_hv for summary in summaries)
        hv_delta_sum = sum(summary.hv_delta for summary in summaries)
        hv_delta_ratio = (
            (hv_delta_sum / baseline_hv_sum) * 100.0
            if baseline_hv_sum
            else float("nan")
        )
        hv_min, hv_p25, hv_median, hv_p75, hv_max = five_number_summary(
            [summary.hv_delta for summary in summaries]
        )
        output_rows.append(
            {
                "scheduler": target_scheduler,
                "baseline_scheduler": baseline_scheduler,
                "baseline_hv_sum": finite_float(baseline_hv_sum),
                "scheduler_hv_sum": finite_float(target_hv_sum),
                "hv_delta_sum": finite_float(hv_delta_sum),
                "hv_delta_baseline_ratio_percent": finite_float(hv_delta_ratio),
                "hv_delta_five_number": {
                    "min": finite_float(hv_min),
                    "p25": finite_float(hv_p25),
                    "median": finite_float(hv_median),
                    "p75": finite_float(hv_p75),
                    "max": finite_float(hv_max),
                },
                "scheduler_frontier_points": sum(
                    summary.target_frontier_count for summary in summaries
                ),
                "users": len(summaries),
            }
        )
    return output_rows


def build_budget_memory_gain_auc_summary(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    baseline_scheduler: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for scheduler in schedulers:
        summaries = budget_memory_gain_auc_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=scheduler,
        )
        if not summaries:
            continue
        budget_count = sum(summary.budget_count for summary in summaries)
        covered_budget_count = sum(
            summary.covered_budget_count for summary in summaries
        )
        total_span = sum(summary.total_span for summary in summaries)
        covered_span = sum(summary.covered_span for summary in summaries)
        memory_gain_aucs = [
            summary.memory_gain_auc
            for summary in summaries
            if summary.memory_gain_auc is not None
        ]
        baseline_memory_aucs = [
            summary.baseline_memory_auc
            for summary in summaries
            if summary.baseline_memory_auc is not None
        ]
        mean_memory_gain_auc = (
            average(memory_gain_aucs) if memory_gain_aucs else float("nan")
        )
        mean_baseline_memory_auc = (
            average(baseline_memory_aucs) if baseline_memory_aucs else float("nan")
        )
        relative_gain_auc_values = _relative_gain_auc_percent_values(summaries)
        span_coverage = (covered_span / total_span) * 100.0 if total_span else 0.0
        relative_gain_auc = (
            average(relative_gain_auc_values)
            if relative_gain_auc_values
            else float("nan")
        )
        output_rows.append(
            {
                "scheduler": scheduler,
                "baseline_scheduler": baseline_scheduler,
                "auc_users": len(memory_gain_aucs),
                "users": len(summaries),
                "budget_count": budget_count,
                "covered_budget_count": covered_budget_count,
                "total_span": finite_float(total_span),
                "covered_span": finite_float(covered_span),
                "span_coverage_percent": finite_float(span_coverage),
                "memory_gain_auc_mean": finite_float(mean_memory_gain_auc),
                "baseline_memory_auc_mean": finite_float(mean_baseline_memory_auc),
                "relative_gain_auc_percent": finite_float(relative_gain_auc),
            }
        )
    return output_rows


def build_memory_target_regret_auc_summary(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    *,
    baseline_scheduler: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for scheduler in schedulers:
        summaries = memory_target_regret_auc_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=scheduler,
        )
        if not summaries:
            continue
        target_count = sum(summary.target_count for summary in summaries)
        covered_target_count = sum(
            summary.covered_target_count for summary in summaries
        )
        total_span = sum(summary.total_span for summary in summaries)
        covered_span = sum(summary.covered_span for summary in summaries)
        time_regret_aucs = [
            summary.time_regret_auc
            for summary in summaries
            if summary.time_regret_auc is not None
        ]
        baseline_time_aucs = [
            summary.baseline_time_auc
            for summary in summaries
            if summary.baseline_time_auc is not None
        ]
        relative_regret_auc_values = _relative_regret_auc_percent_values(summaries)
        mean_time_regret_auc = (
            average(time_regret_aucs) if time_regret_aucs else float("nan")
        )
        mean_baseline_time_auc = (
            average(baseline_time_aucs) if baseline_time_aucs else float("nan")
        )
        span_coverage = (covered_span / total_span) * 100.0 if total_span else 0.0
        relative_regret_auc = (
            average(relative_regret_auc_values)
            if relative_regret_auc_values
            else float("nan")
        )
        output_rows.append(
            {
                "scheduler": scheduler,
                "baseline_scheduler": baseline_scheduler,
                "auc_users": len(time_regret_aucs),
                "users": len(summaries),
                "target_count": target_count,
                "covered_target_count": covered_target_count,
                "total_span": finite_float(total_span),
                "covered_span": finite_float(covered_span),
                "span_coverage_percent": finite_float(span_coverage),
                "time_regret_auc_mean": finite_float(mean_time_regret_auc),
                "baseline_time_auc_mean": finite_float(mean_baseline_time_auc),
                "relative_regret_auc_percent": finite_float(relative_regret_auc),
            }
        )
    return output_rows


def build_per_user_hypervolume_summary(
    rows: list[SweepRow],
    env: str,
    *,
    baseline_scheduler: str,
    target_scheduler: str,
) -> list[dict[str, Any]]:
    return [
        {
            "user_id": summary.user_id,
            "baseline_scheduler": baseline_scheduler,
            "target_scheduler": target_scheduler,
            "baseline_hv": finite_float(summary.baseline_hv),
            "target_hv": finite_float(summary.target_hv),
            "hv_delta": finite_float(summary.hv_delta),
            "target_frontier_count": summary.target_frontier_count,
        }
        for summary in hypervolume_user_summaries(
            rows,
            env,
            baseline_scheduler=baseline_scheduler,
            target_scheduler=target_scheduler,
        )
    ]


def build_pareto_frontier_summary(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
) -> dict[str, Any]:
    total, counts, users = pareto_counts(rows, env, schedulers)
    return {
        "total_frontier_points": total,
        "by_scheduler": [
            {
                "scheduler": scheduler,
                "frontier_points": counts[scheduler],
                "users": users[scheduler],
            }
            for scheduler in schedulers
        ],
    }


def build_analysis_summary(args: argparse.Namespace) -> dict[str, Any]:
    envs = parse_csv(args.env)
    schedulers = parse_csv(args.sched)
    comparisons = parse_comparisons(args.comparisons)
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    rows, candidate_count = load_rows(args)
    environments: dict[str, Any] = {}
    for env in envs:
        env_summary: dict[str, Any] = {}
        if "fsrs6" in schedulers:
            target_schedulers = tuple(
                scheduler for scheduler in schedulers if scheduler != "fsrs6"
            )
            env_summary["primary_hypervolume_summary"] = (
                build_primary_hypervolume_summary(
                    rows,
                    env,
                    baseline_scheduler="fsrs6",
                    target_schedulers=target_schedulers,
                )
            )
            env_summary["budget_memory_gain_auc"] = (
                build_budget_memory_gain_auc_summary(
                    rows,
                    env,
                    schedulers,
                    baseline_scheduler="fsrs6",
                )
            )
            env_summary["memory_target_regret_auc"] = (
                build_memory_target_regret_auc_summary(
                    rows,
                    env,
                    schedulers,
                    baseline_scheduler="fsrs6",
                )
            )
            env_summary["per_user_hypervolume"] = {
                target_scheduler: build_per_user_hypervolume_summary(
                    rows,
                    env,
                    baseline_scheduler="fsrs6",
                    target_scheduler=target_scheduler,
                )
                for target_scheduler in target_schedulers
            }
        else:
            env_summary["primary_hypervolume_summary"] = []
            env_summary["budget_memory_gain_auc"] = []
            env_summary["memory_target_regret_auc"] = []
            env_summary["per_user_hypervolume"] = {}

        env_summary["policy_point_diagnostics"] = build_policy_point_diagnostics(
            rows, env, schedulers
        )
        env_summary["same_user_same_dr_deltas"] = build_pairwise_delta_summary(
            rows, env, comparisons
        )
        env_summary["same_user_same_dr_dominance"] = build_dominance_summary(
            rows, env, comparisons
        )
        env_summary["best_efficiency_points"] = build_best_point_summary(
            rows, env, schedulers, mode="efficiency"
        )
        env_summary["best_efficiency_winners"] = [
            {"scheduler": scheduler, "users": user_ids}
            for scheduler, user_ids in winner_counts(rows, env, schedulers).items()
        ]
        env_summary["max_memory_points"] = build_best_point_summary(
            rows, env, schedulers, mode="memory"
        )
        env_summary["pareto_frontier"] = build_pareto_frontier_summary(
            rows, env, schedulers
        )
        environments[env] = env_summary

    return {
        "type": "scheduler-comparison-analysis",
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(args.run_root) if args.run_root is not None else None,
        "analysis_summary_path": str(args.summary_path)
        if args.summary_path is not None
        else None,
        "filters": {
            "envs": list(envs),
            "schedulers": list(schedulers),
            "comparisons": [
                {"left": left, "right": right} for left, right in comparisons
            ],
            "users": {"start": args.start_user, "end": args.end_user},
            "retention": {
                "start": args.start_retention,
                "end": args.end_retention,
            },
            "engine": args.engine,
            "short_term": args.short_term,
            "fuzz": args.fuzz,
            "metric": args.metric,
            "dedupe": not args.no_dedupe,
            "baseline_dr_manifest": str(args.baseline_dr_manifest)
            if args.baseline_dr_manifest is not None
            else None,
        },
        "source": {
            "log_dir": str(args.log_dir),
            "output_path": str(args.output_path)
            if args.output_path is not None
            else None,
        },
        "coverage": {
            "loaded_records": len(rows),
            "candidate_records": candidate_count,
            "after_latest_file_dedupe": not args.no_dedupe,
            "by_env_scheduler": build_coverage_summary(rows, envs, schedulers),
        },
        "environments": environments,
    }


def render_coverage_summary(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["env", "scheduler", "records", "users", "DRs"],
        [
            [
                str(row["environment"]),
                str(row["scheduler"]),
                str(row["records"]),
                str(row["user_range"] or "-"),
                str(row["desired_retention_count"]),
            ]
            for row in rows
        ],
    )


def render_policy_point_diagnostics(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "scheduler",
            "policy-point avg memorized",
            "policy-point avg time",
            "policy-point avg efficiency",
            "policy-point avg reviews",
        ],
        [
            [
                str(row["scheduler"]),
                fmt_float_optional(row["policy_point_avg_memorized"], 1),
                fmt_float_optional(row["policy_point_avg_time"], 2),
                fmt_float_optional(row["policy_point_avg_efficiency"], 2),
                fmt_float_optional(row["policy_point_avg_reviews"], 2),
            ]
            for row in rows
        ],
    )


def render_pairwise_delta_summary(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
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
        [
            [
                str(row["comparison"]),
                str(row["pairs"]),
                fmt_float_optional(row["delta_memorized_mean"], 1),
                fmt_float_optional(row["delta_time_mean"], 2),
                fmt_float_optional(row["delta_efficiency_mean"], 2),
                f"{row['efficiency_wins']}/{row['pairs']}",
                f"{row['time_wins']}/{row['pairs']}",
                f"{row['memory_wins']}/{row['pairs']}",
            ]
            for row in rows
        ],
    )


def render_dominance_summary(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "comparison",
            "pairs",
            "left dominates",
            "right dominates",
            "left mem+ time+",
            "left mem- time-",
            "equal",
        ],
        [
            [
                str(row["comparison"]),
                str(row["pairs"]),
                f"{row['left_dominates']}/{row['pairs']}",
                f"{row['right_dominates']}/{row['pairs']}",
                f"{row['left_more_memory_more_time']}/{row['pairs']}",
                f"{row['left_less_memory_less_time']}/{row['pairs']}",
                f"{row['equal']}/{row['pairs']}",
            ]
            for row in rows
        ],
    )


def render_best_point_summary(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "scheduler",
            "avg efficiency",
            "avg memorized",
            "avg time",
            "avg reviews",
            "avg DR",
        ],
        [
            [
                str(row["scheduler"]),
                fmt_float_optional(row["avg_efficiency"], 2),
                fmt_float_optional(row["avg_memorized"], 1),
                fmt_float_optional(row["avg_time"], 2),
                fmt_float_optional(row["avg_reviews"], 2),
                fmt_float_optional(row["avg_desired_retention"], 3),
            ]
            for row in rows
        ],
    )


def render_primary_hypervolume_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_scheduler: str = "fsrs6",
) -> str:
    if not rows:
        return f"No {baseline_scheduler} comparison rows available."
    return markdown_table(
        [
            "scheduler",
            "baseline HV sum",
            "scheduler HV sum",
            "HV delta sum",
            "HV delta / baseline HV",
            "HV delta min",
            "HV delta p25",
            "HV delta median",
            "HV delta p75",
            "HV delta max",
            "scheduler frontier points",
            "users",
        ],
        [
            [
                str(row["scheduler"]),
                fmt_float_optional(row["baseline_hv_sum"], 2),
                fmt_float_optional(row["scheduler_hv_sum"], 2),
                fmt_float_optional(row["hv_delta_sum"], 2),
                fmt_percent_optional(row["hv_delta_baseline_ratio_percent"]),
                fmt_float_optional(row["hv_delta_five_number"]["min"], 2),
                fmt_float_optional(row["hv_delta_five_number"]["p25"], 2),
                fmt_float_optional(row["hv_delta_five_number"]["median"], 2),
                fmt_float_optional(row["hv_delta_five_number"]["p75"], 2),
                fmt_float_optional(row["hv_delta_five_number"]["max"], 2),
                str(row["scheduler_frontier_points"]),
                str(row["users"]),
            ]
            for row in rows
        ],
    )


def render_budget_memory_gain_auc_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_scheduler: str = "fsrs6",
) -> str:
    if not rows:
        return f"No {baseline_scheduler} baseline time budgets available."
    return markdown_table(
        [
            "scheduler",
            "AUC users",
            "budget coverage",
            "span coverage",
            f"memory gain AUC vs {baseline_scheduler}",
            f"relative gain AUC vs {baseline_scheduler}",
        ],
        [
            [
                str(row["scheduler"]),
                f"{row['auc_users']}/{row['users']}",
                f"{row['covered_budget_count']}/{row['budget_count']}",
                fmt_percent_optional(row["span_coverage_percent"]),
                fmt_float_optional(row["memory_gain_auc_mean"], 1),
                fmt_percent_optional(row["relative_gain_auc_percent"]),
            ]
            for row in rows
        ],
    )


def render_memory_target_regret_auc_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_scheduler: str = "fsrs6",
) -> str:
    if not rows:
        return f"No {baseline_scheduler} baseline memory targets available."
    return markdown_table(
        [
            "scheduler",
            "AUC users",
            "target coverage",
            "span coverage",
            f"time regret AUC vs {baseline_scheduler}",
            f"relative regret AUC vs {baseline_scheduler}",
        ],
        [
            [
                str(row["scheduler"]),
                f"{row['auc_users']}/{row['users']}",
                f"{row['covered_target_count']}/{row['target_count']}",
                fmt_percent_optional(row["span_coverage_percent"]),
                fmt_float_optional(row["time_regret_auc_mean"], 2),
                fmt_percent_optional(row["relative_regret_auc_percent"]),
            ]
            for row in rows
        ],
    )


def render_per_user_hypervolume_summary(
    rows: list[dict[str, Any]],
    *,
    baseline_scheduler: str = "fsrs6",
    target_scheduler: str,
) -> str:
    if not rows:
        return f"No {baseline_scheduler} + {target_scheduler} rows available."
    return markdown_table(
        [
            "user",
            "baseline HV",
            f"{target_scheduler} HV",
            "HV delta",
            "scheduler frontier points",
        ],
        [
            [
                str(row["user_id"]),
                fmt_float_optional(row["baseline_hv"], 2),
                fmt_float_optional(row["target_hv"], 2),
                fmt_float_optional(row["hv_delta"], 2),
                str(row["target_frontier_count"]),
            ]
            for row in rows
        ],
    )


def render_pareto_frontier_summary(summary: dict[str, Any]) -> tuple[str, str]:
    rows = [
        [
            str(row["scheduler"]),
            str(row["frontier_points"]),
            str(row["users"]),
        ]
        for row in summary["by_scheduler"]
    ]
    return (
        str(summary["total_frontier_points"]),
        markdown_table(["scheduler", "frontier points", "users"], rows),
    )


def render_env_summary(
    summary: dict[str, Any],
    env: str,
    schedulers: tuple[str, ...],
) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print(f"\n## {env} environment\n")
        if "fsrs6" in schedulers:
            target_schedulers = tuple(
                scheduler for scheduler in schedulers if scheduler != "fsrs6"
            )
            if target_schedulers:
                print("### Primary hypervolume summary vs FSRS6 baseline\n")
                print(
                    "Scheduler HV is computed from the scheduler's own Pareto "
                    "frontier with an FSRS6-baseline reference point; it is not "
                    "computed from the combined FSRS6 + scheduler frontier.\n"
                )
                print(
                    render_primary_hypervolume_summary(
                        summary["primary_hypervolume_summary"],
                        baseline_scheduler="fsrs6",
                    )
                )

                print("\n### Budget-memory gain AUC vs FSRS6 baseline\n")
                print(
                    "Gain AUC integrates memorized-card gain over the common "
                    "covered time-budget interval between the FSRS6 baseline "
                    "frontier and each scheduler frontier, using linear "
                    "interpolation only. Positive values mean the scheduler "
                    "remembers more cards at the same budget. Relative gain "
                    "is the simple average of each user's memory gain AUC divided "
                    "by that user's covered baseline memory AUC.\n"
                )
                print(
                    render_budget_memory_gain_auc_summary(
                        summary["budget_memory_gain_auc"],
                        baseline_scheduler="fsrs6",
                    )
                )

                print("\n### Memory-target regret AUC vs FSRS6 baseline\n")
                print(
                    "Regret AUC integrates time regret over the common covered "
                    "memory-target interval between the FSRS6 baseline frontier "
                    "and each scheduler frontier, using linear interpolation only. "
                    "Negative values mean the scheduler reaches the same "
                    "memorized-card targets faster. Relative regret is the simple "
                    "average of each user's time regret AUC divided by that user's "
                    "covered baseline time AUC.\n"
                )
                print(
                    render_memory_target_regret_auc_summary(
                        summary["memory_target_regret_auc"],
                        baseline_scheduler="fsrs6",
                    )
                )

        print("\n### Policy-point diagnostics\n")
        print(
            "Unweighted policy-point averages describe where the sampled policies lie; "
            "they are not the primary Pareto metrics.\n"
        )
        print(render_policy_point_diagnostics(summary["policy_point_diagnostics"]))

        print("\n### Same-user same-DR deltas\n")
        print(render_pairwise_delta_summary(summary["same_user_same_dr_deltas"]))

        print("\n### Same-user same-DR dominance\n")
        print(render_dominance_summary(summary["same_user_same_dr_dominance"]))

        print("\n### Diagnostic best-efficiency point per user\n")
        print(render_best_point_summary(summary["best_efficiency_points"]))
        winners = summary["best_efficiency_winners"]
        denominator = sum(len(row["users"]) for row in winners)
        winner_text = ", ".join(
            f"{row['scheduler']}: {len(row['users'])}/{denominator}"
            f" users {row['users']}"
            for row in winners
            if row["users"]
        )
        print(f"\nBest-efficiency winners: {winner_text}")

        print("\n### Diagnostic max-memory point per user\n")
        print(render_best_point_summary(summary["max_memory_points"]))

        total, table = render_pareto_frontier_summary(summary["pareto_frontier"])
        print(f"\n### Pareto frontier by user\n\nTotal frontier points: {total}\n")
        print(table)

        if "fsrs6" in schedulers:
            target_schedulers = tuple(
                scheduler for scheduler in schedulers if scheduler != "fsrs6"
            )
            for target_scheduler in target_schedulers:
                title = "### Per-user hypervolume vs FSRS6 baseline"
                if len(target_schedulers) > 1:
                    title = f"{title}: {target_scheduler}"
                print(f"\n{title}\n")
                print(
                    render_per_user_hypervolume_summary(
                        summary["per_user_hypervolume"].get(target_scheduler, []),
                        baseline_scheduler="fsrs6",
                        target_scheduler=target_scheduler,
                    )
                )
    return buffer.getvalue()


def render_summary_report(summary: dict[str, Any]) -> str:
    filters = summary["filters"]
    envs = tuple(str(env) for env in filters["envs"])
    schedulers = tuple(str(scheduler) for scheduler in filters["schedulers"])
    coverage = summary["coverage"]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print("# FSRS-6 Scheduler Comparison\n")
        print(
            "Filters: "
            f"env={','.join(envs)}, "
            f"scheduler={','.join(schedulers)}, "
            f"users={filters['users']['start']}-{filters['users']['end']}, "
            f"retention={filters['retention']['start']:.2f}-"
            f"{filters['retention']['end']:.2f}, "
            f"engine={filters['engine']}, short_term={filters['short_term']}, "
            f"fuzz={filters['fuzz']}, "
            "budget_gain_auc=common_covered_frontier_interval, "
            "memory_target_regret_auc=common_covered_frontier_interval"
        )
        loaded = coverage["loaded_records"]
        candidate_count = coverage["candidate_records"]
        print(
            f"Loaded {loaded} records"
            + (
                f" after latest-file dedupe from {candidate_count} matching records."
                if coverage["after_latest_file_dedupe"]
                else "."
            )
        )

        print("\n## Coverage\n")
        print(render_coverage_summary(coverage["by_env_scheduler"]))

        for env in envs:
            print(
                render_env_summary(
                    summary["environments"][env],
                    env,
                    schedulers,
                ),
                end="",
            )
    return buffer.getvalue()


def print_env_report(
    rows: list[SweepRow],
    env: str,
    schedulers: tuple[str, ...],
    comparisons: tuple[tuple[str, str], ...],
) -> None:
    print(f"\n## {env} environment\n")
    if "fsrs6" in schedulers:
        target_schedulers = tuple(
            scheduler for scheduler in schedulers if scheduler != "fsrs6"
        )
        if target_schedulers:
            print("### Primary hypervolume summary vs FSRS6 baseline\n")
            print(
                "Scheduler HV is computed from the scheduler's own Pareto "
                "frontier with an FSRS6-baseline reference point; it is not "
                "computed from the combined FSRS6 + scheduler frontier.\n"
            )
            print(
                primary_hypervolume_summary_table(
                    rows,
                    env,
                    baseline_scheduler="fsrs6",
                    target_schedulers=target_schedulers,
                )
            )

            print("\n### Budget-memory gain AUC vs FSRS6 baseline\n")
            print(
                "Gain AUC integrates memorized-card gain over the common covered "
                "time-budget interval between the FSRS6 baseline frontier and each "
                "scheduler frontier, using linear interpolation only. Positive "
                "values mean the scheduler remembers more cards at the same "
                "budget. Relative gain is the simple average of each user's memory "
                "gain AUC divided by that user's covered baseline memory AUC.\n"
            )
            print(
                budget_memory_gain_auc_table(
                    rows,
                    env,
                    schedulers,
                    baseline_scheduler="fsrs6",
                )
            )

            print("\n### Memory-target regret AUC vs FSRS6 baseline\n")
            print(
                "Regret AUC integrates time regret over the common covered "
                "memory-target interval between the FSRS6 baseline frontier and "
                "each scheduler frontier, using linear interpolation only. "
                "Negative values mean the scheduler reaches the same memorized-card "
                "targets faster. Relative regret is the simple average of each "
                "user's time regret AUC divided by that user's covered baseline "
                "time AUC.\n"
            )
            print(
                memory_target_regret_auc_table(
                    rows,
                    env,
                    schedulers,
                    baseline_scheduler="fsrs6",
                )
            )

    print("\n### Policy-point diagnostics\n")
    print(
        "Unweighted policy-point averages describe where the sampled policies lie; "
        "they are not the primary Pareto metrics.\n"
    )
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

    print("\n### Diagnostic best-efficiency point per user\n")
    print(best_summary_table(rows, env, schedulers, mode="efficiency"))
    winners = winner_counts(rows, env, schedulers)
    winner_text = ", ".join(
        f"{scheduler}: {len(user_ids)}/{sum(len(ids) for ids in winners.values())}"
        f" users {user_ids}"
        for scheduler, user_ids in winners.items()
        if user_ids
    )
    print(f"\nBest-efficiency winners: {winner_text}")

    print("\n### Diagnostic max-memory point per user\n")
    print(best_summary_table(rows, env, schedulers, mode="memory"))

    total, counts, users = pareto_counts(rows, env, schedulers)
    pareto_rows = [
        [scheduler, str(counts[scheduler]), str(users[scheduler])]
        for scheduler in schedulers
    ]
    print(f"\n### Pareto frontier by user\n\nTotal frontier points: {total}\n")
    print(markdown_table(["scheduler", "frontier points", "users"], pareto_rows))

    if "fsrs6" in schedulers:
        target_schedulers = tuple(
            scheduler for scheduler in schedulers if scheduler != "fsrs6"
        )
        for target_scheduler in target_schedulers:
            title = "### Per-user hypervolume vs FSRS6 baseline"
            if len(target_schedulers) > 1:
                title = f"{title}: {target_scheduler}"
            print(f"\n{title}\n")
            print(
                hypervolume_summary_table(
                    rows,
                    env,
                    baseline_scheduler="fsrs6",
                    target_scheduler=target_scheduler,
                )
            )


def render_report(args: argparse.Namespace) -> str:
    return render_summary_report(build_analysis_summary(args))


def main() -> int:
    args = parse_args()
    summary = build_analysis_summary(args)
    report = render_summary_report(summary)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(report, encoding="utf-8")
    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

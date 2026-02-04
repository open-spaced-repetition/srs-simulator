from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_END_USER,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_PRIORITY,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
    DEFAULT_SHORT_TERM_LOOPS_LIMIT,
)


def add_user_range_args(
    parser: argparse.ArgumentParser, *, default_end: int | None = None
) -> None:
    if default_end is None:
        default_end = DEFAULT_END_USER
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument(
        "--end-user", type=int, default=default_end, help="Last user id."
    )


def add_retention_range_args(
    parser: argparse.ArgumentParser,
    *,
    start_default: float = 0.50,
    end_default: float = 0.98,
    step_default: float = 0.02,
) -> None:
    parser.add_argument(
        "--start-retention",
        type=float,
        default=start_default,
        help="Start retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=end_default,
        help="End retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=step_default,
        help="Retention step (0-1, rounded to 2 decimals).",
    )


def add_env_sched_args(
    parser: argparse.ArgumentParser,
    *,
    env_default: str,
    sched_default: str,
    env_help: str,
    sched_help: str,
) -> None:
    parser.add_argument("--env", default=env_default, help=env_help)
    parser.add_argument("--sched", default=sched_default, help=sched_help)


def add_common_sim_args(
    parser: argparse.ArgumentParser,
    *,
    days_default: int | None = None,
    deck_default: int | None = None,
    learn_limit_default: int | None = None,
    review_limit_default: int | None = None,
    cost_limit_default: float | None = None,
    seed_default: int | None = None,
    priority_default: str | None = None,
    scheduler_priority_default: str | None = None,
) -> None:
    if days_default is None:
        days_default = DEFAULT_DAYS
    if deck_default is None:
        deck_default = DEFAULT_DECK_SIZE
    if learn_limit_default is None:
        learn_limit_default = DEFAULT_LEARN_LIMIT
    if review_limit_default is None:
        review_limit_default = DEFAULT_REVIEW_LIMIT
    if cost_limit_default is None:
        cost_limit_default = DEFAULT_COST_LIMIT_MINUTES
    if seed_default is None:
        seed_default = DEFAULT_SEED
    if priority_default is None:
        priority_default = DEFAULT_PRIORITY
    if scheduler_priority_default is None:
        scheduler_priority_default = DEFAULT_SCHEDULER_PRIORITY
    parser.add_argument(
        "--days", type=int, default=days_default, help="Simulation days."
    )
    parser.add_argument("--deck", type=int, default=deck_default, help="Deck size.")
    parser.add_argument(
        "--learn-limit",
        type=int,
        default=learn_limit_default,
        help="Max new cards per day (behavior limit).",
    )
    parser.add_argument(
        "--review-limit",
        type=int,
        default=review_limit_default,
        help="Max reviews per day (behavior limit).",
    )
    parser.add_argument(
        "--cost-limit-minutes",
        type=float,
        default=cost_limit_default,
        help="Daily study time limit in minutes (behavior limit).",
    )
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed.")
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default=priority_default,
        help="Action priority passed to simulate.py.",
    )
    parser.add_argument(
        "--scheduler-priority",
        default=scheduler_priority_default,
        help="FSRS6 priority hint passed to simulate.py.",
    )


def add_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    benchmark_result_help: str = "Override benchmark result files (key=value, comma-separated).",
    benchmark_partition_help: str = "Benchmark parameter partition key.",
    srs_benchmark_help: str = "Path to the srs-benchmark repo (used for weights).",
) -> None:
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help=benchmark_result_help,
    )
    parser.add_argument(
        "--benchmark-partition",
        default="0",
        help=benchmark_partition_help,
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help=srs_benchmark_help,
    )


def add_button_usage_arg(
    parser: argparse.ArgumentParser, *, default_path: Path | None
) -> None:
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=default_path,
        help="Path to Anki button usage JSONL for per-user costs/probabilities.",
    )


def add_log_args(
    parser: argparse.ArgumentParser,
    *,
    log_dir_default: Path | None = None,
    include_no_log: bool = True,
    include_no_progress: bool = True,
) -> None:
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=log_dir_default,
        help="Directory to store logs (defaults to logs/retention_sweep).",
    )
    if include_no_log:
        parser.add_argument(
            "--no-log",
            action="store_true",
            help="Disable writing logs to disk.",
        )
    if include_no_progress:
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable tqdm progress bars (useful for parallel runs).",
        )


def add_fuzz_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fuzz",
        action="store_true",
        help="Apply scheduler interval fuzz (Anki-style).",
    )


def add_short_term_args(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str],
    source_help: str,
    learning_help: str,
    relearning_help: str,
    threshold_help: str,
    threshold_default: float = 0.5,
    loops_limit_help: str = "Max short-term review loops per day (per user).",
    loops_limit_default: int | None = None,
) -> None:
    parser.add_argument(
        "--short-term-source",
        choices=list(choices),
        default=None,
        help=source_help,
    )
    parser.add_argument("--learning-steps", default=None, help=learning_help)
    parser.add_argument("--relearning-steps", default=None, help=relearning_help)
    parser.add_argument(
        "--short-term-threshold",
        type=float,
        default=threshold_default,
        help=threshold_help,
    )
    parser.add_argument(
        "--short-term-loops-limit",
        type=int,
        default=(
            DEFAULT_SHORT_TERM_LOOPS_LIMIT
            if loops_limit_default is None
            else loops_limit_default
        ),
        help=loops_limit_help,
    )


def add_torch_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for vectorized engine (e.g. cuda, cuda:0, cpu).",
    )


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def has_flag(args: Sequence[str], flag: str) -> bool:
    for arg in args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def build_retention_command(
    *,
    uv_cmd: str,
    script_path: Path,
    env: str,
    sched: str,
    user_id: int,
    torch_device: str | None = None,
    inject_torch_device: bool = False,
) -> list[str]:
    cmd = [
        uv_cmd,
        "run",
        str(script_path),
        "--env",
        env,
        "--sched",
        sched,
        "--user-id",
        str(user_id),
    ]
    if inject_torch_device and torch_device is not None:
        cmd.extend(["--torch-device", torch_device])
    return cmd


def passthrough_args(argv: Sequence[str]) -> list[str]:
    if "--" in argv:
        return list(argv[argv.index("--") + 1 :])
    return []

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import (
    DEFAULT_COST_LIMIT_MINUTES,
    DEFAULT_DECK_SIZE,
    DEFAULT_DAYS,
    DEFAULT_END_RETENTION,
    DEFAULT_LEARN_LIMIT,
    DEFAULT_PRIORITY,
    DEFAULT_RETENTION_STEP,
    DEFAULT_REVIEW_LIMIT,
    DEFAULT_SCHEDULER_PRIORITY,
    DEFAULT_SEED,
    DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    DEFAULT_START_RETENTION,
)

from experiments.retention_sweep import run_sweep_users_batched


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated wrapper for evaluating FSRS-6-trained SA FSRS-6 "
            "policies in an external LSTM memory environment."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--train-run-root",
        type=Path,
        required=True,
        help="Run root containing train-overfit/train_outputs from FSRS-6 training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LSTM external-eval logs.",
    )
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument("--end-user", type=int, default=8, help="Last user id.")
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.5,
        help="SA objective lambda value to select from the training output root.",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=DEFAULT_START_RETENTION,
        help="First trained baseline desired retention.",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=DEFAULT_END_RETENTION,
        help="Last trained baseline desired retention.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_RETENTION_STEP,
        help="Retention grid step.",
    )
    parser.add_argument(
        "--lane-batch-size",
        type=int,
        default=0,
        help="Lanes per simulation batch. Use 0 to run all lanes in one batch.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for the batched simulation, e.g. cuda or cuda:0.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--learn-limit", type=int, default=DEFAULT_LEARN_LIMIT)
    parser.add_argument("--review-limit", type=int, default=DEFAULT_REVIEW_LIMIT)
    parser.add_argument(
        "--cost-limit-minutes",
        type=float,
        default=DEFAULT_COST_LIMIT_MINUTES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default=DEFAULT_PRIORITY,
    )
    parser.add_argument(
        "--scheduler-priority",
        default=DEFAULT_SCHEDULER_PRIORITY,
    )
    parser.add_argument("--fuzz", action="store_true")
    parser.add_argument(
        "--short-term",
        choices=["off"],
        default="off",
        help="Compatibility option; external LSTM eval remains short-term off.",
    )
    parser.add_argument("--short-term-threshold", type=float, default=0.5)
    parser.add_argument(
        "--short-term-loops-limit",
        type=int,
        default=DEFAULT_SHORT_TERM_LOOPS_LIMIT,
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
    )
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default="0")
    parser.add_argument("--diagnostic-csv-logs", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _batched_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--start-user",
        str(args.start_user),
        "--end-user",
        str(args.end_user),
        "--batch-size",
        str(max(1, args.end_user - args.start_user + 1)),
        "--env",
        "lstm",
        "--sched",
        "fsrs6,sa_fsrs6",
        "--start-retention",
        str(args.start_retention),
        "--end-retention",
        str(args.end_retention),
        "--step",
        str(args.step),
        "--days",
        str(args.days),
        "--deck",
        str(args.deck),
        "--learn-limit",
        str(args.learn_limit),
        "--review-limit",
        str(args.review_limit),
        "--cost-limit-minutes",
        str(args.cost_limit_minutes),
        "--seed",
        str(args.seed),
        "--priority",
        args.priority,
        "--scheduler-priority",
        args.scheduler_priority,
        "--button-usage",
        str(args.button_usage),
        "--benchmark-partition",
        str(args.benchmark_partition),
        "--log-dir",
        str(args.output_dir / "sweep_outputs"),
        "--sa-fsrs6-train-run-root",
        str(args.train_run_root),
        "--sa-fsrs6-lambda-values",
        str(args.lambda_value),
        "--short-term-threshold",
        str(args.short_term_threshold),
        "--short-term-loops-limit",
        str(args.short_term_loops_limit),
    ]
    if args.lane_batch_size > 0:
        argv.extend(["--max-lanes-per-batch", str(args.lane_batch_size)])
    if args.torch_device:
        argv.extend(["--torch-device", args.torch_device])
    if args.srs_benchmark_root is not None:
        argv.extend(["--srs-benchmark-root", str(args.srs_benchmark_root)])
    if args.benchmark_result:
        argv.extend(["--benchmark-result", args.benchmark_result])
    if args.fuzz:
        argv.append("--fuzz")
    if args.diagnostic_csv_logs:
        argv.append("--diagnostic-csv-logs")
    if args.no_log:
        argv.append("--no-log")
    if args.no_progress:
        argv.append("--no-progress")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "evaluate_sa_fsrs6_external_lstm.py is deprecated; use "
        "experiments/retention_sweep/run_sweep_users_batched.py --config instead.",
        file=sys.stderr,
    )
    return run_sweep_users_batched.main(_batched_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())

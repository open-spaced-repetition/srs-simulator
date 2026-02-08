from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.fanout import FanoutJob, create_fanout_bars, run_fanout
from simulator.subprocess_runner import run_command_with_progress

from experiments.retention_sweep.cli_utils import (
    add_user_range_args,
    build_retention_command,
    passthrough_args,
)

from tqdm import tqdm


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run retention_sweep.build_pareto.py for a range of user IDs.",
        allow_abbrev=False,
    )
    add_user_range_args(parser)
    parser.add_argument(
        "--env",
        dest="env",
        default="lstm",
        help="Comma-separated environments passed to build_pareto.py.",
    )
    parser.add_argument(
        "--sched",
        dest="sched",
        default="fsrs6,anki_sm2,memrise,fixed,sspmmc",
        help="Comma-separated schedulers passed to build_pareto.py.",
    )
    parser.add_argument(
        "--short-term",
        choices=["on", "off", "any"],
        default="any",
        help="Short-term filter passed to build_pareto.py.",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched", "any"],
        default="any",
        help="Short-term source filter passed to build_pareto.py.",
    )
    parser.add_argument(
        "--engine",
        choices=["event", "vectorized", "batched", "any"],
        default="any",
        help="Engine filter passed to build_pareto.py.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max parallel users to run (1 keeps sequential behavior).",
    )
    parser.add_argument(
        "--child-progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control child progress bars (auto enables in parallel).",
    )
    parser.add_argument(
        "--show-commands",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control command echoing (auto shows only in sequential runs).",
    )
    parser.add_argument(
        "--compare-short-term",
        action="store_true",
        help="Pass --compare-short-term to build_pareto.py.",
    )
    parser.add_argument(
        "--compare-engine",
        action="store_true",
        help="Pass --compare-engine to build_pareto.py.",
    )
    parser.add_argument(
        "--uv-cmd",
        default="uv",
        help="Command to invoke uv (override if needed).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between users.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first non-zero exit code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_known_args()


def _build_command(
    args: argparse.Namespace, user_id: int, extra_args: list[str]
) -> list[str]:
    script_path = Path("experiments") / "retention_sweep" / "build_pareto.py"
    cmd = build_retention_command(
        uv_cmd=args.uv_cmd,
        script_path=script_path,
        env=args.env,
        sched=args.sched,
        user_id=user_id,
    )
    if args.short_term != "any":
        cmd.extend(["--short-term", args.short_term])
    if args.short_term_source != "any":
        cmd.extend(["--short-term-source", args.short_term_source])
    if args.engine != "any":
        cmd.extend(["--engine", args.engine])
    if args.compare_short_term:
        cmd.append("--compare-short-term")
    if args.compare_engine:
        cmd.append("--compare-engine")
    cmd.extend(extra_args)
    return cmd


def _run_command(
    cmd: list[str],
    env: dict[str, str],
    progress_bar: tqdm | None,
    overall_bar: tqdm | None,
    progress_lock: threading.RLock | None,
) -> int:
    return run_command_with_progress(
        cmd=cmd,
        env=env,
        progress_bar=progress_bar,
        overall_bar=overall_bar,
        progress_lock=progress_lock,
        write_line=progress_bar.write if progress_bar is not None else None,
    )


def main() -> int:
    args, _ = parse_args()
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    extra_args = passthrough_args(sys.argv)

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1.")

    user_ids = list(range(args.start_user, args.end_user + 1))
    parallel = args.max_parallel > 1 and not args.dry_run
    enable_child_progress = args.child_progress == "on" or (
        args.child_progress == "auto" and args.max_parallel > 1
    )
    use_parent_progress = parallel and enable_child_progress
    show_commands = args.show_commands == "on" or (
        args.show_commands == "auto" and args.max_parallel == 1
    )

    env = os.environ.copy()
    bars = create_fanout_bars(
        user_count=len(user_ids),
        show_overall=False,
        overall_total=None,
        use_parent_progress=use_parent_progress,
        max_parallel=args.max_parallel,
    )
    try:

        def build_job(user_id: int, _slot: int | None, _index: int) -> FanoutJob:
            cmd = _build_command(args, user_id, extra_args)
            return FanoutJob(user_id=user_id, cmd=cmd, env=env)

        def run_job(
            job: FanoutJob,
            progress_bar: tqdm | None,
            overall_bar: tqdm | None,
            progress_lock: threading.RLock | None,
        ) -> int:
            return _run_command(
                job.cmd,
                job.env,
                progress_bar,
                overall_bar,
                progress_lock,
            )

        failures, first_failure = run_fanout(
            user_ids=user_ids,
            max_parallel=args.max_parallel,
            dry_run=args.dry_run,
            show_commands=show_commands,
            fail_fast=args.fail_fast,
            sleep_seconds=args.sleep_seconds,
            use_parent_progress=use_parent_progress,
            bars=bars,
            build_job=build_job,
            run_job=run_job,
        )
    finally:
        bars.close()

    if failures:
        print(f"Completed with {failures} failures.")
        if first_failure is not None:
            return first_failure
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

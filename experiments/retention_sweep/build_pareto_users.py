from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import threading

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.fanout import FanoutJob, create_fanout_bars, run_fanout
from simulator.subprocess_runner import run_command_with_progress
from simulator.experiment_infra import ExperimentConfig

from experiments.retention_sweep.cli_utils import (
    add_user_range_args,
    build_retention_command,
    has_flag,
    passthrough_args,
)

from tqdm import tqdm


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Run retention_sweep.build_pareto.py for a range of user IDs.",
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
        "--run-id",
        default=None,
        help="Run id passed to build_pareto.py for run-scoped scheduler filtering.",
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
        choices=["event", "batched", "any"],
        default="any",
        help="Engine filter passed to build_pareto.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed filter passed to build_pareto.py.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Root directory containing retention_sweep JSONL logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-user Pareto JSON/PNG outputs. When set, each user "
            "writes simulation_results_retention_sweep_user_<id>.json here."
        ),
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=0.50,
        help="Minimum desired retention passed to build_pareto.py.",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=0.98,
        help="Maximum desired retention passed to build_pareto.py.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Max parallel users to run (1 keeps sequential behavior).",
    )
    parser.add_argument(
        "--child-progress",
        choices=["auto", "on", "off"],
        default="off",
        help="Unused (build_pareto_users does not display per-worker bars).",
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
        "--no-plot",
        action="store_true",
        help="Pass --no-plot to build_pareto.py.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Pass --hide-labels to build_pareto.py.",
    )
    parser.add_argument(
        "--baseline-dr-manifest",
        type=Path,
        default=None,
        help="Pass a per-user FSRS6 baseline DR manifest to build_pareto.py.",
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
    args, extra = parser.parse_known_args(argv)
    if args.config is not None:
        args = _merge_config_args(cli_args=args, config=args.config, argv=argv)
    return args, extra


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
    config: Path,
    argv: list[str],
) -> argparse.Namespace:
    experiment = ExperimentConfig.from_toml(config)
    build_config = experiment.build_pareto
    sweep_config = experiment.sweep_batched

    if not (has_flag(argv, "--start-user") or has_flag(argv, "--end-user")):
        cli_args.start_user = min(experiment.users.train)
        cli_args.end_user = max(experiment.users.train)

    if not has_flag(argv, "--env"):
        envs = (
            build_config.envs
            or sweep_config.envs
            or (experiment.simulation.environment,)
        )
        cli_args.env = ",".join(envs)
    if not has_flag(argv, "--sched"):
        schedulers = build_config.schedulers or (
            (experiment.baseline.scheduler, *sweep_config.schedulers)
            if sweep_config.schedulers
            else (experiment.baseline.scheduler,)
        )
        cli_args.sched = ",".join(dict.fromkeys(schedulers))
    if not has_flag(argv, "--log-dir"):
        cli_args.log_dir = _resolve_repo_path(
            build_config.log_dir or sweep_config.log_dir
        )
    if not has_flag(argv, "--start-retention"):
        cli_args.start_retention = build_config.start_retention
    if not has_flag(argv, "--end-retention"):
        cli_args.end_retention = build_config.end_retention
    if not has_flag(argv, "--short-term"):
        cli_args.short_term = build_config.short_term
    if not has_flag(argv, "--short-term-source"):
        cli_args.short_term_source = build_config.short_term_source
    if not has_flag(argv, "--engine"):
        cli_args.engine = build_config.engine
    if not has_flag(argv, "--seed"):
        cli_args.seed = experiment.seed
    if not has_flag(argv, "--max-parallel"):
        cli_args.max_parallel = build_config.max_parallel
    if build_config.compare_short_term and not has_flag(argv, "--compare-short-term"):
        cli_args.compare_short_term = True
    if build_config.compare_engine and not has_flag(argv, "--compare-engine"):
        cli_args.compare_engine = True
    if build_config.no_plot and not has_flag(argv, "--no-plot"):
        cli_args.no_plot = True
    if build_config.hide_labels and not has_flag(argv, "--hide-labels"):
        cli_args.hide_labels = True
    if experiment.baseline_dr_selection.manifest is not None and not has_flag(
        argv, "--baseline-dr-manifest"
    ):
        cli_args.baseline_dr_manifest = _resolve_repo_path(
            experiment.baseline_dr_selection.manifest
        )
    return cli_args


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
    if args.log_dir is not None:
        cmd.extend(["--log-dir", str(args.log_dir)])
    cmd.extend(["--start-retention", str(args.start_retention)])
    cmd.extend(["--end-retention", str(args.end_retention)])
    if args.short_term != "any":
        cmd.extend(["--short-term", args.short_term])
    if args.short_term_source != "any":
        cmd.extend(["--short-term-source", args.short_term_source])
    if args.engine != "any":
        cmd.extend(["--engine", args.engine])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.run_id is not None:
        cmd.extend(["--run-id", args.run_id])
    if args.compare_short_term:
        cmd.append("--compare-short-term")
    if args.compare_engine:
        cmd.append("--compare-engine")
    if args.no_plot:
        cmd.append("--no-plot")
    if args.hide_labels:
        cmd.append("--hide-labels")
    if args.baseline_dr_manifest is not None:
        cmd.extend(["--baseline-dr-manifest", str(args.baseline_dr_manifest)])
    if args.output_dir is not None:
        cmd.extend(
            [
                "--results-path",
                str(
                    args.output_dir
                    / f"simulation_results_retention_sweep_user_{user_id}.json"
                ),
                "--plot-dir",
                str(args.output_dir),
            ]
        )
    cmd.extend(extra_args)
    return cmd


def _run_command(
    cmd: list[str],
    env: dict[str, str],
    progress_bar: tqdm | None,
    overall_bar: tqdm | None,
    progress_lock: threading.RLock | None,
    suppress_output: bool,
) -> int:
    write_line = None
    local_bar: tqdm | None = None
    if progress_bar is None and suppress_output:
        local_bar = tqdm(total=0, disable=True)
        progress_bar = local_bar
    if progress_bar is not None:
        if suppress_output:

            def write_line(_line: str) -> None:
                pass

        else:
            write_line = progress_bar.write
    try:
        return run_command_with_progress(
            cmd=cmd,
            env=env,
            progress_bar=progress_bar,
            overall_bar=overall_bar,
            progress_lock=progress_lock,
            write_line=write_line,
        )
    finally:
        if local_bar is not None:
            local_bar.close()


def main() -> int:
    args, extra_args = parse_args()
    if args.run_id is None and args.run_root is not None:
        args.run_id = args.run_root.name
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    if not extra_args:
        extra_args = passthrough_args(sys.argv)

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1.")

    user_ids = list(range(args.start_user, args.end_user + 1))
    parallel = args.max_parallel > 1 and not args.dry_run
    # Build pareto jobs are one-shot; we keep only the users bar.
    use_parent_progress = False
    suppress_child_output = parallel
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
                suppress_child_output,
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

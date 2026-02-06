from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import threading

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.fanout import FanoutJob, create_fanout_bars, run_fanout
from simulator.subprocess_runner import run_command_with_progress
from simulator.scheduler_spec import (
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)
from simulator.retention_sweep.grid import count_dr_steps
from simulator.retention_sweep.overrides import (
    RunSweepOverrides,
    parse_run_sweep_overrides,
)
from simulator.retention_sweep.sspmmc import resolve_sspmmc_policy_paths
from simulator.sweep_utils import (
    parse_cuda_devices,
    parse_env_overrides,
    strip_arg_terminator,
)

from experiments.retention_sweep.cli_utils import (
    add_user_range_args,
    build_retention_command,
    has_flag,
    parse_csv,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run retention_sweep.run_sweep.py for a range of user IDs.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Extra arguments after `--` are forwarded to run_sweep.py.\n"
            "Example:\n"
            "  uv run experiments/retention_sweep/run_sweep_users.py \\\n"
            "    --start-user 1 --end-user 200 --env lstm \\\n"
            "    --sched fsrs6 --max-parallel 3 \\\n"
            "    -- --start-retention 0.50 --end-retention 0.68 --step 0.02\n"
        ),
        allow_abbrev=False,
    )
    add_user_range_args(parser)
    parser.add_argument(
        "--env",
        dest="env",
        default="lstm",
        help="Comma-separated environments passed to run_sweep.py.",
    )
    parser.add_argument(
        "--sched",
        dest="sched",
        default="fsrs6,anki_sm2",
        help="Comma-separated schedulers passed to run_sweep.py.",
    )
    parser.add_argument(
        "--uv-cmd",
        default="uv",
        help="Command to invoke uv (override if needed).",
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
        "--child-summary",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control child summary lines (auto disables in parallel).",
    )
    parser.add_argument(
        "--show-commands",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control command echoing (auto shows only in sequential runs).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between user submissions.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first non-zero exit code.",
    )
    parser.add_argument(
        "--mps-active-thread-percentage",
        type=int,
        default=None,
        help="Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for each subprocess.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute workers across "
            "(e.g. 0,1). Each worker is assigned a device round-robin."
        ),
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help="Extra environment variables (KEY=VALUE) for each subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_known_args()


def _estimate_total_days(
    user_ids: list[int],
    environments: list[str],
    schedulers: list[str],
    overrides: RunSweepOverrides,
    repo_root: Path,
) -> int:
    scheduler_specs = [parse_scheduler_spec(item) for item in schedulers]
    fixed_schedulers = [spec for spec in scheduler_specs if spec[0] == "fixed"]
    has_sspmmc = any(spec[0] == "sspmmc" for spec in scheduler_specs)
    dr_schedulers: list[str] = []
    non_dr_schedulers: list[str] = []
    for name, _, _ in scheduler_specs:
        if name in {"sspmmc", "fixed"}:
            continue
        if scheduler_uses_desired_retention(name):
            if name not in dr_schedulers:
                dr_schedulers.append(name)
        else:
            if name not in non_dr_schedulers:
                non_dr_schedulers.append(name)

    run_dr = bool(dr_schedulers)
    run_sspmmc = has_sspmmc
    run_fixed = bool(fixed_schedulers)
    run_non_dr = bool(non_dr_schedulers)
    dr_steps = count_dr_steps(
        overrides.start_retention, overrides.end_retention, overrides.step
    )
    base_runs = 0
    if run_dr:
        base_runs += dr_steps * len(dr_schedulers)
    if run_non_dr:
        base_runs += len(non_dr_schedulers)
    if run_fixed:
        base_runs += len(fixed_schedulers)

    total_days = 0
    for user_id in user_ids:
        sspmmc_runs = 0
        if run_sspmmc:
            policies = resolve_sspmmc_policy_paths(
                repo_root=repo_root,
                user_id=user_id,
                run_sspmmc=run_sspmmc,
                sspmmc_policies=overrides.sspmmc_policies,
                sspmmc_policy=overrides.sspmmc_policy,
                sspmmc_policy_dir=overrides.sspmmc_policy_dir,
                sspmmc_policy_glob=overrides.sspmmc_policy_glob,
                sspmmc_max=overrides.sspmmc_max,
            )
            sspmmc_runs = len(policies)
        total_runs = base_runs + sspmmc_runs
        total_days += total_runs * len(environments) * overrides.days
    return total_days


def _build_command(
    uv_cmd: str,
    script_path: Path,
    environments: str,
    schedulers: str,
    user_id: int,
    extra_args: list[str],
    disable_progress: bool,
    disable_summary: bool,
    emit_progress_events: bool,
    torch_device: str | None,
    inject_torch_device: bool,
) -> list[str]:
    cmd = build_retention_command(
        uv_cmd=uv_cmd,
        script_path=script_path,
        env=environments,
        sched=schedulers,
        user_id=user_id,
        torch_device=torch_device,
        inject_torch_device=inject_torch_device,
    )
    cmd.extend(extra_args)
    if disable_progress and not has_flag(extra_args, "--no-progress"):
        cmd.append("--no-progress")
    if disable_summary and not has_flag(extra_args, "--no-summary"):
        cmd.append("--no-summary")
    if emit_progress_events and not has_flag(extra_args, "--progress-events"):
        cmd.append("--progress-events")
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
    args, extra_args = parse_args()
    extra_args = strip_arg_terminator(extra_args)
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1.")
    if args.mps_active_thread_percentage is not None and not (
        1 <= args.mps_active_thread_percentage <= 100
    ):
        raise ValueError("--mps-active-thread-percentage must be between 1 and 100.")

    script_path = Path("experiments") / "retention_sweep" / "run_sweep.py"
    sweep_overrides = parse_run_sweep_overrides(extra_args)
    repo_root = Path(__file__).resolve().parents[2]

    parallel = args.max_parallel > 1 and not args.dry_run
    enable_child_progress = args.child_progress == "on" or (
        args.child_progress == "auto" and args.max_parallel > 1
    )
    use_parent_progress = parallel and enable_child_progress
    disable_progress = not enable_child_progress or use_parent_progress
    show_commands = args.show_commands == "on" or (
        args.show_commands == "auto" and args.max_parallel == 1
    )
    enable_child_summary = args.child_summary == "on" or (
        args.child_summary == "auto" and args.max_parallel == 1
    )

    env = os.environ.copy()
    env_overrides = parse_env_overrides(args.set_env)
    cuda_devices = parse_cuda_devices(args.cuda_devices)
    if cuda_devices and "CUDA_VISIBLE_DEVICES" in env_overrides:
        raise ValueError(
            "--cuda-devices cannot be combined with --set-env CUDA_VISIBLE_DEVICES."
        )
    env.update(env_overrides)
    if args.mps_active_thread_percentage is not None:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            args.mps_active_thread_percentage
        )

    user_ids = list(range(args.start_user, args.end_user + 1))
    envs = parse_csv(args.env) or ["lstm"]
    schedulers = parse_csv(args.sched) or ["fsrs6"]
    inject_torch_device = bool(cuda_devices) and not has_flag(
        extra_args, "--torch-device"
    )
    if inject_torch_device:
        extra_args = list(extra_args)
    overall_total = None
    show_overall = use_parent_progress
    if show_overall:
        overall_total = _estimate_total_days(
            user_ids,
            envs,
            schedulers,
            sweep_overrides,
            repo_root,
        )
    bars = create_fanout_bars(
        user_count=len(user_ids),
        show_overall=show_overall,
        overall_total=overall_total,
        use_parent_progress=use_parent_progress,
        max_parallel=args.max_parallel,
    )
    try:

        def build_job(user_id: int, slot: int | None, index: int) -> FanoutJob:
            device = None
            if cuda_devices:
                if slot is None:
                    device = cuda_devices[index % len(cuda_devices)]
                else:
                    device = cuda_devices[slot % len(cuda_devices)]
            worker_env = env
            if device is not None:
                worker_env = env.copy()
                worker_env["CUDA_VISIBLE_DEVICES"] = device
            cmd = _build_command(
                args.uv_cmd,
                script_path,
                args.env,
                args.sched,
                user_id,
                extra_args,
                disable_progress,
                not enable_child_summary or use_parent_progress,
                use_parent_progress,
                torch_device="cuda:0" if device is not None else None,
                inject_torch_device=inject_torch_device,
            )
            prefix = f"CUDA_VISIBLE_DEVICES={device} " if device is not None else ""
            return FanoutJob(
                user_id=user_id, cmd=cmd, env=worker_env, echo_prefix=prefix
            )

        def run_job(
            job: FanoutJob,
            progress_bar: tqdm | None,
            overall_bar: tqdm | None,
            progress_lock: threading.RLock | None,
        ) -> int:
            return _run_command(
                job.cmd, job.env, progress_bar, overall_bar, progress_lock
            )

        failures, first_failure_code = run_fanout(
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

        if failures:
            print(f"Completed with {failures} failures.")
            if args.fail_fast and first_failure_code is not None:
                return first_failure_code
            return 1
        return 0
    finally:
        bars.close()


if __name__ == "__main__":
    raise SystemExit(main())

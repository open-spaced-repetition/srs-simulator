from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import threading

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.scheduler_spec import (
    parse_scheduler_spec,
    scheduler_uses_desired_retention,
)

from experiments.retention_sweep.cli_utils import add_user_range_args, parse_csv


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
    add_user_range_args(parser, default_end=10000)
    parser.add_argument(
        "--env",
        "--environments",
        dest="environments",
        default="lstm",
        help="Comma-separated environments passed to run_sweep.py.",
    )
    parser.add_argument(
        "--sched",
        "--schedulers",
        dest="schedulers",
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


def _parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set-env '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"Invalid --set-env '{item}'. Missing key.")
        overrides[key] = value
    return overrides


def _parse_cuda_devices(raw: str | None) -> list[str]:
    if not raw:
        return []
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    for device in devices:
        if not device.isdigit():
            raise ValueError(
                f"Invalid --cuda-devices entry '{device}'. Expected numeric indices."
            )
    return devices


def _parse_run_sweep_overrides(extra_args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--start-retention", type=float, default=0.70)
    parser.add_argument("--end-retention", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--sspmmc-policy", type=Path, default=None)
    parser.add_argument("--sspmmc-policy-dir", type=Path, default=None)
    parser.add_argument("--sspmmc-policies", default=None)
    parser.add_argument("--sspmmc-policy-glob", default="*.json")
    parser.add_argument("--sspmmc-max", type=int, default=None)
    return parser.parse_known_args(extra_args)[0]


def _strip_arg_terminator(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == "--":
        return extra_args[1:]
    return extra_args


def _resolve_sspmmc_policies(
    overrides: argparse.Namespace,
    repo_root: Path,
    user_id: int,
    run_sspmmc: bool,
) -> list[Path]:
    if overrides.sspmmc_policies:
        paths = [Path(path) for path in _parse_csv(overrides.sspmmc_policies)]
        return [path.resolve() for path in paths]

    if overrides.sspmmc_policy:
        return [overrides.sspmmc_policy.resolve()]

    policy_dir = overrides.sspmmc_policy_dir
    if policy_dir is None and run_sspmmc:
        candidate = (
            repo_root.parent
            / "SSP-MMC-FSRS"
            / "outputs"
            / "policies"
            / f"user_{user_id}"
        )
        if candidate.exists():
            policy_dir = candidate

    if policy_dir is None:
        return []

    policy_dir = policy_dir.resolve()
    paths = sorted(policy_dir.glob(overrides.sspmmc_policy_glob))
    if overrides.sspmmc_max is not None:
        paths = paths[: overrides.sspmmc_max]
    return paths


def _count_dr_steps(start: float, end: float, step: float) -> int:
    if step == 0:
        return 0
    if step > 0 and start > end:
        return 0
    if step < 0 and start < end:
        return 0
    span_abs = abs(end - start)
    step_abs = abs(step)
    return int(math.floor(span_abs / step_abs + 1e-9)) + 1


def _estimate_total_days(
    user_ids: list[int],
    environments: list[str],
    schedulers: list[str],
    overrides: argparse.Namespace,
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
    dr_steps = _count_dr_steps(
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
            policies = _resolve_sspmmc_policies(
                overrides,
                repo_root,
                user_id,
                run_sspmmc,
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
    cmd = [
        uv_cmd,
        "run",
        str(script_path),
        "--env",
        environments,
        "--sched",
        schedulers,
        "--user-id",
        str(user_id),
    ]
    if inject_torch_device and torch_device is not None:
        cmd.extend(["--torch-device", torch_device])
    cmd.extend(extra_args)
    if disable_progress and "--no-progress" not in extra_args:
        cmd.append("--no-progress")
    if disable_summary and "--no-summary" not in extra_args:
        cmd.append("--no-summary")
    if emit_progress_events and "--progress-events" not in extra_args:
        cmd.append("--progress-events")
    return cmd


def _has_arg(extra_args: list[str], flag: str) -> bool:
    for arg in extra_args:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _run_command(
    cmd: list[str],
    env: dict[str, str],
    progress_bar: tqdm | None,
    overall_bar: tqdm | None,
    progress_lock: threading.RLock | None,
) -> int:
    if progress_bar is None:
        result = subprocess.run(cmd, check=False, env=env)
        return result.returncode

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    last_label = None
    overall_label = None
    overall_completed = 0
    if process.stdout is not None:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if progress_lock is not None:
                    with progress_lock:
                        progress_bar.write(line)
                else:
                    progress_bar.write(line)
                continue
            if payload.get("type") != "progress":
                continue
            label = payload.get("label", "")
            completed = int(payload.get("completed", 0))
            total = int(payload.get("total", 0))
            if progress_lock is not None:
                progress_lock.acquire()
            try:
                if overall_bar is not None:
                    if label and label != overall_label:
                        overall_label = label
                        overall_completed = 0
                    if completed < overall_completed:
                        overall_completed = 0
                    delta_overall = completed - overall_completed
                    if delta_overall > 0:
                        overall_bar.update(delta_overall)
                        overall_completed = completed

                if label and label != last_label:
                    reset_total = total if total > 0 else progress_bar.total
                    progress_bar.reset(total=reset_total)
                    progress_bar.set_description_str(label)
                    last_label = label
                    if completed > 0:
                        progress_bar.update(completed)
                    else:
                        progress_bar.refresh()
                    continue
                if total > 0 and progress_bar.total != total:
                    progress_bar.total = total
                if completed < progress_bar.n:
                    progress_bar.reset(total=progress_bar.total)
                    if completed > 0:
                        progress_bar.update(completed)
                    else:
                        progress_bar.refresh()
                elif completed > progress_bar.n:
                    progress_bar.update(completed - progress_bar.n)
            finally:
                if progress_lock is not None:
                    progress_lock.release()
    return process.wait()


def _reset_worker_bar(progress_bar: tqdm, label: str | None = None) -> None:
    progress_bar.reset(total=0)
    if label is not None:
        progress_bar.set_description_str(label)
    progress_bar.refresh()


def main() -> int:
    args, extra_args = parse_args()
    extra_args = _strip_arg_terminator(extra_args)
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1.")
    if args.mps_active_thread_percentage is not None and not (
        1 <= args.mps_active_thread_percentage <= 100
    ):
        raise ValueError("--mps-active-thread-percentage must be between 1 and 100.")

    script_path = Path("experiments") / "retention_sweep" / "run_sweep.py"
    sweep_overrides = _parse_run_sweep_overrides(extra_args)
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
    env_overrides = _parse_env_overrides(args.set_env)
    cuda_devices = _parse_cuda_devices(args.cuda_devices)
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
    envs = parse_csv(args.environments) or ["lstm"]
    schedulers = parse_csv(args.schedulers) or ["fsrs6"]
    inject_torch_device = cuda_devices and not _has_arg(extra_args, "--torch-device")
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
    overall_bar = None
    users_position = 0
    if show_overall:
        overall_bar = tqdm(
            total=overall_total,
            desc="overall",
            unit="day",
            file=sys.stderr,
            ascii=True,
            position=0,
            leave=True,
        )
        users_position = 1

    progress = tqdm(
        total=len(user_ids),
        desc="users",
        unit="user",
        file=sys.stderr,
        ascii=True,
        position=users_position,
        leave=True,
    )
    try:
        if args.dry_run or args.max_parallel == 1:
            failures = 0
            for index, user_id in enumerate(user_ids):
                device = None
                worker_env = env
                if cuda_devices:
                    device = cuda_devices[index % len(cuda_devices)]
                    worker_env = env.copy()
                    worker_env["CUDA_VISIBLE_DEVICES"] = device
                cmd = _build_command(
                    args.uv_cmd,
                    script_path,
                    args.environments,
                    args.schedulers,
                    user_id,
                    extra_args,
                    disable_progress,
                    not enable_child_summary,
                    False,
                    torch_device="cuda:0" if device is not None else None,
                    inject_torch_device=inject_torch_device,
                )
                if args.dry_run or show_commands:
                    prefix = (
                        f"CUDA_VISIBLE_DEVICES={device} " if device is not None else ""
                    )
                    progress.write(f"[{user_id}] {prefix}{' '.join(cmd)}")
                if not args.dry_run:
                    returncode = _run_command(cmd, worker_env, None, None, None)
                    if returncode != 0:
                        failures += 1
                        progress.write(
                            f"[{user_id}] FAILED with exit code {returncode}"
                        )
                        if args.fail_fast:
                            return returncode
                progress.update(1)
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            if failures:
                print(f"Completed with {failures} failures.")
                return 1
            return 0

        failures = 0
        first_failure_code: int | None = None
        stop_scheduling = False
        user_iter = iter(user_ids)
        worker_position_base = users_position + 1
        available_positions = list(
            range(worker_position_base, worker_position_base + args.max_parallel)
        )
        worker_bars: dict[int, tqdm] = {}
        progress_lock = threading.RLock()
        tqdm.set_lock(progress_lock)

        def submit_next(executor: concurrent.futures.Executor) -> bool:
            try:
                user_id = next(user_iter)
            except StopIteration:
                return False
            if not available_positions:
                return False
            position = available_positions.pop(0)
            device = None
            if cuda_devices:
                device_index = (position - worker_position_base) % len(cuda_devices)
                device = cuda_devices[device_index]
            cmd = _build_command(
                args.uv_cmd,
                script_path,
                args.environments,
                args.schedulers,
                user_id,
                extra_args,
                disable_progress,
                not enable_child_summary or use_parent_progress,
                use_parent_progress,
                torch_device="cuda:0" if device is not None else None,
                inject_torch_device=inject_torch_device,
            )
            if show_commands:
                prefix = f"CUDA_VISIBLE_DEVICES={device} " if device is not None else ""
                progress.write(f"[{user_id}] {prefix}{' '.join(cmd)}")
            progress_bar = None
            if use_parent_progress:
                progress_bar = worker_bars.get(position)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=0,
                        desc=f"u{user_id}",
                        unit="day",
                        file=sys.stderr,
                        ascii=True,
                        position=position,
                        leave=False,
                    )
                    worker_bars[position] = progress_bar
                else:
                    _reset_worker_bar(progress_bar, f"u{user_id}")
                progress_bar.refresh()
            worker_env = env
            if device is not None:
                worker_env = env.copy()
                worker_env["CUDA_VISIBLE_DEVICES"] = device
            future = executor.submit(
                _run_command,
                cmd,
                worker_env,
                progress_bar,
                overall_bar,
                progress_lock if use_parent_progress else None,
            )
            pending[future] = (user_id, position)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            return True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel
        ) as executor:
            pending: dict[concurrent.futures.Future[int], tuple[int, int]] = {}
            for _ in range(args.max_parallel):
                if not submit_next(executor):
                    break
            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    user_id, position = pending.pop(future)
                    try:
                        returncode = future.result()
                    except concurrent.futures.CancelledError:
                        progress.update(1)
                        if use_parent_progress and position in worker_bars:
                            _reset_worker_bar(worker_bars[position], "idle")
                        available_positions.append(position)
                        continue
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - unexpected subprocess error
                        failures += 1
                        progress.write(f"[{user_id}] FAILED with exception: {exc}")
                        if first_failure_code is None:
                            first_failure_code = 1
                        if args.fail_fast:
                            stop_scheduling = True
                        progress.update(1)
                        if use_parent_progress and position in worker_bars:
                            _reset_worker_bar(worker_bars[position], "idle")
                        available_positions.append(position)
                        continue
                    if returncode != 0:
                        failures += 1
                        progress.write(
                            f"[{user_id}] FAILED with exit code {returncode}"
                        )
                        if first_failure_code is None:
                            first_failure_code = returncode
                        if args.fail_fast:
                            stop_scheduling = True
                    progress.update(1)
                    if use_parent_progress and position in worker_bars:
                        _reset_worker_bar(worker_bars[position], "idle")
                    available_positions.append(position)
                if stop_scheduling:
                    for future in list(pending):
                        future.cancel()
                    continue
                while len(pending) < args.max_parallel:
                    if not submit_next(executor):
                        break

        if failures:
            print(f"Completed with {failures} failures.")
            if args.fail_fast and first_failure_code is not None:
                return first_failure_code
            return 1
        return 0
    finally:
        progress.close()
        if overall_bar is not None:
            overall_bar.close()
        if use_parent_progress:
            for bar in worker_bars.values():
                bar.close()


if __name__ == "__main__":
    raise SystemExit(main())

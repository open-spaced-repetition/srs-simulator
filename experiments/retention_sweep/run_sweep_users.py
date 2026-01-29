from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import threading

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retention_sweep.run_sweep.py for a range of user IDs.",
    )
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument("--end-user", type=int, default=10000, help="Last user id.")
    parser.add_argument(
        "--step-user",
        type=int,
        default=1,
        help="Step size for user ids.",
    )
    parser.add_argument(
        "--environments",
        default="lstm",
        help="Comma-separated environments passed to run_sweep.py.",
    )
    parser.add_argument(
        "--schedulers",
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
    return parser.parse_args()


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
) -> list[str]:
    cmd = [
        uv_cmd,
        "run",
        str(script_path),
        "--environments",
        environments,
        "--schedulers",
        schedulers,
        "--user-id",
        str(user_id),
    ]
    cmd.extend(extra_args)
    if disable_progress and "--no-progress" not in extra_args:
        cmd.append("--no-progress")
    if disable_summary and "--no-summary" not in extra_args:
        cmd.append("--no-summary")
    if emit_progress_events and "--progress-events" not in extra_args:
        cmd.append("--progress-events")
    return cmd


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
                        if total > 0:
                            overall_bar.total = (overall_bar.total or 0) + total
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


def main() -> int:
    args = parse_args()
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1.")
    if args.mps_active_thread_percentage is not None and not (
        1 <= args.mps_active_thread_percentage <= 100
    ):
        raise ValueError("--mps-active-thread-percentage must be between 1 and 100.")

    script_path = Path("experiments") / "retention_sweep" / "run_sweep.py"
    extra_args = []
    if "--" in sys.argv:
        extra_args = sys.argv[sys.argv.index("--") + 1 :]

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
    env.update(_parse_env_overrides(args.set_env))
    if args.mps_active_thread_percentage is not None:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            args.mps_active_thread_percentage
        )

    user_ids = list(range(args.start_user, args.end_user + 1, args.step_user))
    show_overall = use_parent_progress
    overall_bar = None
    users_position = 0
    if show_overall:
        overall_bar = tqdm(
            total=0,
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
            for user_id in user_ids:
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
                )
                if args.dry_run or show_commands:
                    progress.write(f"[{user_id}] {' '.join(cmd)}")
                if not args.dry_run:
                    returncode = _run_command(cmd, env, None, None, None)
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
            )
            if show_commands:
                progress.write(f"[{user_id}] {' '.join(cmd)}")
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
                    progress_bar.reset(total=0)
                    progress_bar.set_description_str(f"u{user_id}")
            future = executor.submit(
                _run_command,
                cmd,
                env,
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
                            worker_bars[position].close()
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
                            worker_bars[position].close()
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
                        worker_bars[position].close()
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


if __name__ == "__main__":
    raise SystemExit(main())

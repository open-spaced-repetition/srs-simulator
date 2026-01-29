from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path

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
        help="Control child progress bars (auto disables when parallel).",
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
    return cmd


def _run_command(cmd: list[str], env: dict[str, str]) -> int:
    result = subprocess.run(cmd, check=False, env=env)
    return result.returncode


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

    disable_progress = args.child_progress == "off" or (
        args.child_progress == "auto" and args.max_parallel > 1
    )

    env = os.environ.copy()
    env.update(_parse_env_overrides(args.set_env))
    if args.mps_active_thread_percentage is not None:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
            args.mps_active_thread_percentage
        )

    user_ids = list(range(args.start_user, args.end_user + 1, args.step_user))
    progress = tqdm(
        total=len(user_ids),
        desc="users",
        unit="user",
        file=sys.stderr,
        ascii=True,
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
                )
                print(f"[{user_id}] {' '.join(cmd)}")
                if not args.dry_run:
                    returncode = _run_command(cmd, env)
                    if returncode != 0:
                        failures += 1
                        print(f"[{user_id}] FAILED with exit code {returncode}")
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

        def submit_next(executor: concurrent.futures.Executor) -> bool:
            try:
                user_id = next(user_iter)
            except StopIteration:
                return False
            cmd = _build_command(
                args.uv_cmd,
                script_path,
                args.environments,
                args.schedulers,
                user_id,
                extra_args,
                disable_progress,
            )
            print(f"[{user_id}] {' '.join(cmd)}")
            future = executor.submit(_run_command, cmd, env)
            pending[future] = user_id
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            return True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel
        ) as executor:
            pending: dict[concurrent.futures.Future[int], int] = {}
            for _ in range(args.max_parallel):
                if not submit_next(executor):
                    break
            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    user_id = pending.pop(future)
                    try:
                        returncode = future.result()
                    except concurrent.futures.CancelledError:
                        progress.update(1)
                        continue
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - unexpected subprocess error
                        failures += 1
                        print(f"[{user_id}] FAILED with exception: {exc}")
                        if first_failure_code is None:
                            first_failure_code = 1
                        if args.fail_fast:
                            stop_scheduling = True
                        progress.update(1)
                        continue
                    if returncode != 0:
                        failures += 1
                        print(f"[{user_id}] FAILED with exit code {returncode}")
                        if first_failure_code is None:
                            first_failure_code = returncode
                        if args.fail_fast:
                            stop_scheduling = True
                    progress.update(1)
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


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    script_path = Path("experiments") / "retention_sweep" / "run_sweep.py"
    extra_args = []
    if "--" in sys.argv:
        extra_args = sys.argv[sys.argv.index("--") + 1 :]

    failures = 0
    for user_id in range(args.start_user, args.end_user + 1, args.step_user):
        cmd = [
            args.uv_cmd,
            "run",
            str(script_path),
            "--environments",
            args.environments,
            "--schedulers",
            args.schedulers,
            "--user-id",
            str(user_id),
        ]
        cmd.extend(extra_args)
        print(f"[{user_id}] {' '.join(cmd)}")
        if not args.dry_run:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failures += 1
                print(f"[{user_id}] FAILED with exit code {result.returncode}")
                if args.fail_fast:
                    return result.returncode
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if failures:
        print(f"Completed with {failures} failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

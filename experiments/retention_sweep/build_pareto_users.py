from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.retention_sweep.cli_utils import add_user_range_args, passthrough_args


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run retention_sweep.build_pareto.py for a range of user IDs.",
        allow_abbrev=False,
    )
    add_user_range_args(parser, default_end=10000)
    parser.add_argument(
        "--env",
        "--environments",
        dest="environments",
        default="lstm",
        help="Comma-separated environments passed to build_pareto.py.",
    )
    parser.add_argument(
        "--sched",
        "--schedulers",
        dest="schedulers",
        default="fsrs6,anki_sm2,memrise,fixed,sspmmc",
        help="Comma-separated schedulers passed to build_pareto.py.",
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


def main() -> int:
    args, _ = parse_args()
    if args.start_user < 1 or args.end_user < args.start_user:
        raise ValueError("Invalid user range.")

    script_path = Path("experiments") / "retention_sweep" / "build_pareto.py"
    extra_args = passthrough_args(sys.argv)

    failures = 0
    for user_id in range(args.start_user, args.end_user + 1):
        cmd = [
            args.uv_cmd,
            "run",
            str(script_path),
            "--env",
            args.environments,
            "--sched",
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

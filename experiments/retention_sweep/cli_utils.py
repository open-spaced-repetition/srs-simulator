from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def add_user_range_args(parser: argparse.ArgumentParser, *, default_end: int) -> None:
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument(
        "--end-user", type=int, default=default_end, help="Last user id."
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

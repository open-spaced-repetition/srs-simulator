from __future__ import annotations

import argparse
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


def passthrough_args(argv: Sequence[str]) -> list[str]:
    if "--" in argv:
        return list(argv[argv.index("--") + 1 :])
    return []

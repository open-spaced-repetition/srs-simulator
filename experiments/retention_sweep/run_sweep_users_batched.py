from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.batched_sweep.plan import build_batched_sweep_plan
from simulator.batched_sweep.execution import run_batches

from experiments.retention_sweep.cli_utils import (
    add_benchmark_args,
    add_button_usage_arg,
    add_common_sim_args,
    add_env_sched_args,
    add_fuzz_arg,
    add_log_args,
    add_retention_range_args,
    add_short_term_args,
    add_torch_device_arg,
    add_user_range_args,
    parse_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-user retention sweeps with batched vectorized simulation.",
        allow_abbrev=False,
    )
    add_user_range_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of users to simulate in parallel per batch.",
    )
    add_env_sched_args(
        parser,
        env_default="lstm",
        sched_default="fsrs6,anki_sm2,memrise",
        env_help="Comma-separated environments to sweep (lstm, fsrs6, fsrs6_default).",
        sched_help=(
            "Comma-separated schedulers to sweep "
            "(fsrs6, fsrs6_default, fsrs3, lstm, anki_sm2, memrise, fixed)."
        ),
    )
    add_retention_range_args(parser)
    add_common_sim_args(
        parser,
    )
    add_button_usage_arg(parser, default_path=DEFAULT_BUTTON_USAGE_PATH)
    add_benchmark_args(parser)
    add_log_args(
        parser, log_dir_default=None, include_no_log=True, include_no_progress=True
    )
    add_fuzz_arg(parser)
    add_short_term_args(
        parser,
        choices=["steps", "sched"],
        source_help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or sched (LSTM-only short-term intervals)."
        ),
        learning_help="Comma-separated learning steps (minutes) for short-term steps mode.",
        relearning_help="Comma-separated relearning steps (minutes) for short-term steps mode.",
        threshold_help="Short-term cutoff in days (used by sched mode).",
    )
    add_torch_device_arg(parser)
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute batches across "
            "(e.g. 0,1). Each batch is assigned a device round-robin."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    envs = parse_csv(args.env)
    schedulers = parse_csv(args.sched)
    plan = build_batched_sweep_plan(
        repo_root=REPO_ROOT,
        args=args,
        envs=envs,
        schedulers=schedulers,
    )
    overall = None
    if not args.no_progress:
        overall = tqdm(
            total=plan.total_user_days,
            desc="Overall",
            unit="user-day",
            leave=True,
        )

    run_batches(
        args=args,
        ctx=plan.ctx,
        batches=plan.batches,
        devices=plan.devices,
        device=plan.device,
        overall=overall,
    )
    if overall is not None:
        overall.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

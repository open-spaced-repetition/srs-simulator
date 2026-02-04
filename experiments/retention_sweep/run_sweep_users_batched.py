from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent import futures
from multiprocessing import get_context
import queue
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulate as simulate_cli
from simulator.benchmark_loader import load_benchmark_weights, parse_result_overrides
from simulator.button_usage import (
    DEFAULT_BUTTON_USAGE_PATH,
    load_button_usage_config,
    normalize_button_usage,
)
from simulator.math.fsrs import Bounds
from simulator.models.lstm import _resolve_benchmark_weights
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.scheduler_spec import parse_scheduler_spec
from simulator.scheduler_spec import normalize_fixed_interval
from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps, AnkiSM2Scheduler
from simulator.schedulers.fixed import FixedBatchSchedulerOps
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.lstm import LSTMBatchSchedulerOps
from simulator.schedulers.memrise import MemriseScheduler
from simulator.schedulers.memrise import MemriseBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost

from experiments.retention_sweep.cli_utils import add_user_range_args, parse_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-user retention sweeps with batched vectorized simulation.",
        allow_abbrev=False,
    )
    add_user_range_args(parser, default_end=1000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of users to simulate in parallel per batch.",
    )
    parser.add_argument(
        "--env",
        dest="env",
        default="lstm",
        help="Comma-separated environments to sweep (lstm, fsrs6).",
    )
    parser.add_argument(
        "--sched",
        dest="sched",
        default="fsrs6,anki_sm2,memrise",
        help="Comma-separated schedulers to sweep (fsrs6, lstm, anki_sm2, memrise, fixed).",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=0.50,
        help="Start retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=0.98,
        help="End retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Retention step (0-1, rounded to 2 decimals).",
    )
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--deck", type=int, default=10000)
    parser.add_argument("--learn-limit", type=int, default=10)
    parser.add_argument("--review-limit", type=int, default=9999)
    parser.add_argument("--cost-limit-minutes", type=float, default=720.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default="review-first",
    )
    parser.add_argument(
        "--scheduler-priority",
        default="low_retrievability",
        help="FSRS6 priority hint (ignored by other schedulers).",
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL for per-user costs/probabilities.",
    )
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help="Override benchmark result files (key=value, comma-separated).",
    )
    parser.add_argument(
        "--benchmark-partition",
        default="0",
        help="Benchmark parameter partition key.",
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help="Path to the srs-benchmark repo (used for weights).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store logs (defaults to logs/retention_sweep).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing logs to disk.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--fuzz",
        action="store_true",
        help="Apply scheduler interval fuzz (Anki-style).",
    )
    parser.add_argument(
        "--short-term-source",
        choices=["steps", "sched"],
        default=None,
        help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or sched (LSTM-only short-term intervals; not yet supported in batched)."
        ),
    )
    parser.add_argument(
        "--learning-steps",
        default=None,
        help="Comma-separated learning steps (minutes) for short-term steps mode.",
    )
    parser.add_argument(
        "--relearning-steps",
        default=None,
        help="Comma-separated relearning steps (minutes) for short-term steps mode.",
    )
    parser.add_argument(
        "--short-term-threshold",
        type=float,
        default=0.5,
        help="Short-term cutoff in days (used by sched mode).",
    )
    parser.add_argument(
        "--short-term-loops-limit",
        type=int,
        default=None,
        help="Max short-term review loops per day (per user).",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for vectorized engine (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute batches across "
            "(e.g. 0,1). Each batch is assigned a device round-robin."
        ),
    )
    return parser.parse_args()


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


def _chunked(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def _dr_values(start: float, end: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("--step must be non-zero.")
    start = round(start, 2)
    end = round(end, 2)
    values = []
    value = start
    epsilon = abs(step) * 1e-6
    if step > 0 and start > end:
        raise ValueError(
            "--start-retention must be <= --end-retention for positive step."
        )
    if step < 0 and start < end:
        raise ValueError(
            "--start-retention must be >= --end-retention for negative step."
        )
    while (value <= end + epsilon) if step > 0 else (value >= end - epsilon):
        values.append(round(value, 2))
        next_value = round(value + step, 2)
        if next_value == value:
            raise ValueError("Retention step is too small after rounding; use >= 0.01.")
        value = next_value
    return values


def _load_usage(
    user_ids: list[int], button_usage: Path | None
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    learn_costs = []
    review_costs = []
    first_rating_prob = []
    review_rating_prob = []
    learning_rating_prob = []
    relearning_rating_prob = []
    state_rating_costs = []
    for user_id in user_ids:
        config = (
            load_button_usage_config(button_usage, user_id)
            if button_usage is not None
            else None
        )
        usage = normalize_button_usage(config)
        learn_costs.append(usage["learn_costs"])
        review_costs.append(usage["review_costs"])
        first_rating_prob.append(usage["first_rating_prob"])
        review_rating_prob.append(usage["review_rating_prob"])
        learning_rating_prob.append(usage["learning_rating_prob"])
        relearning_rating_prob.append(usage["relearning_rating_prob"])
        state_rating_costs.append(usage["state_rating_costs"])
    return (
        torch.tensor(learn_costs, dtype=torch.float32),
        torch.tensor(review_costs, dtype=torch.float32),
        torch.tensor(first_rating_prob, dtype=torch.float32),
        torch.tensor(review_rating_prob, dtype=torch.float32),
        torch.tensor(learning_rating_prob, dtype=torch.float32),
        torch.tensor(relearning_rating_prob, dtype=torch.float32),
        torch.tensor(state_rating_costs, dtype=torch.float32),
    )


def _resolve_lstm_paths(
    user_ids: list[int], benchmark_root: Path, *, short_term: bool
) -> list[Path]:
    paths = []
    for user_id in user_ids:
        path = _resolve_benchmark_weights(
            user_id, benchmark_root, short_term=short_term
        )
        if path is None:
            raise FileNotFoundError(f"LSTM weights not found for user {user_id}.")
        paths.append(path)
    return paths


def _build_behavior_cost(
    user_count: int,
    *,
    deck_size: int,
    learn_limit: int,
    review_limit: int | None,
    cost_limit_minutes: float | None,
    learn_costs: torch.Tensor,
    review_costs: torch.Tensor,
    first_rating_prob: torch.Tensor,
    review_rating_prob: torch.Tensor,
    learning_rating_prob: torch.Tensor,
    relearning_rating_prob: torch.Tensor,
    state_rating_costs: torch.Tensor,
    short_term: bool,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    max_reviews = review_limit if review_limit is not None else deck_size
    max_cost = cost_limit_minutes * 60.0 if cost_limit_minutes is not None else math.inf
    if short_term:
        learn_costs = state_rating_costs[:, 0]
        review_costs = state_rating_costs[:, 1]
    behavior = MultiUserBehavior(
        attendance_prob=torch.full((user_count,), 1.0),
        lazy_good_bias=torch.zeros(user_count),
        max_new_per_day=torch.full((user_count,), learn_limit, dtype=torch.int64),
        max_reviews_per_day=torch.full((user_count,), max_reviews, dtype=torch.int64),
        max_cost_per_day=torch.full((user_count,), max_cost),
        success_weights=review_rating_prob,
        learning_success_weights=learning_rating_prob,
        relearning_success_weights=relearning_rating_prob,
        first_rating_prob=first_rating_prob,
    )
    cost = MultiUserCost(
        base=torch.zeros(user_count),
        penalty=torch.zeros(user_count),
        learn_costs=learn_costs,
        review_costs=review_costs,
        learning_review_costs=state_rating_costs[:, 0],
        relearning_review_costs=state_rating_costs[:, 2],
    )
    return behavior, cost


def _progress_callback_from_queue(
    progress_queue,
    *,
    multiplier: int,
    device_label: str,
    run_label: str,
    total_days: int,
) -> callable | None:
    if progress_queue is None:
        return None
    last = 0
    progress_queue.put(("start", device_label, run_label, total_days))

    def _update(completed: int, total: int) -> None:
        nonlocal last
        if total <= 0:
            return
        delta = completed - last
        if delta > 0:
            progress_queue.put(("overall", delta * multiplier))
            progress_queue.put(("gpu", device_label, delta))
        last = completed

    return _update


def _simulate_and_log(
    *,
    args: argparse.Namespace,
    batch: list[int],
    env_ops,
    sched_ops,
    behavior: MultiUserBehavior,
    cost_model: MultiUserCost,
    progress: bool,
    progress_queue,
    device_label: str,
    run_label: str,
    environment: str,
    scheduler_name: str,
    scheduler_spec: str,
    desired_retention: float | None,
    fixed_interval: float | None,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
    learning_steps_arg: str | None,
    relearning_steps_arg: str | None,
    log_root: Path,
    batch_log_root: Path,
) -> None:
    progress_callback = _progress_callback_from_queue(
        progress_queue,
        multiplier=len(batch),
        device_label=device_label,
        run_label=run_label,
        total_days=args.days,
    )
    batch_stats: dict[str, list[int]] = {}
    stats_list = simulate_multiuser(
        days=args.days,
        deck_size=args.deck,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        seed=args.seed,
        device=env_ops.device,
        dtype=torch.float32,
        fuzz=args.fuzz,
        priority_mode=args.priority,
        progress=progress,
        progress_label=run_label,
        progress_callback=progress_callback,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        short_term_threshold=args.short_term_threshold,
        short_term_loops_limit=args.short_term_loops_limit,
        batch_stats=batch_stats,
    )
    if batch_stats:
        batch_log_root.mkdir(parents=True, exist_ok=True)
        start_user = batch[0]
        end_user = batch[-1]
        parts = [
            f"env={environment}",
            "engine=batched",
            f"sched={scheduler_name}",
            f"users={start_user}-{end_user}",
        ]
        if desired_retention is not None:
            parts.append(f"ret={desired_retention:.2f}")
        if fixed_interval is not None:
            parts.append(f"ivl={fixed_interval:.2f}")
        if short_term_source:
            parts.append(f"st={short_term_source}")
            if args.short_term_loops_limit is not None:
                parts.append(f"stloops={args.short_term_loops_limit}")
        parts.append(f"seed={args.seed}")
        filename = batch_log_root / f"batch_{'_'.join(parts)}.csv"
        with filename.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["day", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"]
            )
            allocated = batch_stats.get("gpu_peak_allocated_bytes")
            reserved = batch_stats.get("gpu_peak_reserved_bytes")
            total_days = args.days
            for day in range(total_days):
                alloc = allocated[day] if allocated is not None else ""
                resv = reserved[day] if reserved is not None else ""
                writer.writerow([day, alloc, resv])
    if args.no_log:
        return
    for user_id, stats in zip(batch, stats_list):
        user_log_dir = log_root / f"user_{user_id}"
        user_log_dir.mkdir(parents=True, exist_ok=True)
        log_args = argparse.Namespace(
            engine="batched",
            days=args.days,
            deck=args.deck,
            learn_limit=args.learn_limit,
            review_limit=args.review_limit,
            cost_limit_minutes=args.cost_limit_minutes,
            priority=args.priority,
            environment=environment,
            scheduler=scheduler_name,
            scheduler_spec=scheduler_spec,
            user_id=user_id,
            button_usage=str(args.button_usage)
            if args.button_usage is not None
            else None,
            desired_retention=desired_retention,
            scheduler_priority=args.scheduler_priority,
            sspmmc_policy=None,
            fixed_interval=fixed_interval,
            seed=args.seed,
            fuzz=args.fuzz,
            short_term_source=short_term_source,
            learning_steps=learning_steps_arg,
            relearning_steps=relearning_steps_arg,
            short_term_threshold=args.short_term_threshold,
            short_term_loops_limit=args.short_term_loops_limit,
            log_dir=user_log_dir,
            log_reviews=False,
        )
        simulate_cli._write_log(log_args, stats)


def _run_batch_core(
    *,
    args: argparse.Namespace,
    batch: list[int],
    benchmark_root: Path,
    overrides: dict[str, str],
    log_root: Path,
    batch_log_root: Path,
    dr_values: list[float],
    device: torch.device | None,
    progress: bool,
    progress_queue,
    device_label: str,
) -> None:
    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
    ) = _load_usage(batch, args.button_usage)
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        args
    )
    short_term_enabled = bool(short_term_source)
    if short_term_source not in {None, "steps"}:
        raise SystemExit(
            "Batched short-term currently supports --short-term-source steps only."
        )
    learning_steps_arg = (
        ",".join(str(step) for step in learning_steps)
        if short_term_source == "steps"
        else None
    )
    relearning_steps_arg = (
        ",".join(str(step) for step in relearning_steps)
        if short_term_source == "steps"
        else None
    )
    schedulers = parse_csv(args.sched)
    envs = parse_csv(args.env)
    base_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for environment in envs:
        lstm_packed: PackedLSTMWeights | None = None
        if environment == "lstm":
            lstm_paths = _resolve_lstm_paths(
                batch, benchmark_root, short_term=short_term_enabled
            )
            lstm_packed = PackedLSTMWeights.from_paths(
                lstm_paths,
                use_duration_feature=False,
                device=base_device,
                dtype=torch.float32,
            )
            env_ops = LSTMBatchedEnvOps(
                lstm_packed,
                device=lstm_packed.process_0_weight.device,
                dtype=torch.float32,
            )
        elif environment == "fsrs6":
            env_weights = torch.stack(
                [
                    torch.tensor(
                        load_benchmark_weights(
                            repo_root=REPO_ROOT,
                            benchmark_root=benchmark_root,
                            environment="fsrs6",
                            user_id=user_id,
                            partition_key=args.benchmark_partition,
                            overrides=overrides,
                            short_term=short_term_enabled,
                        ),
                        dtype=torch.float32,
                    )
                    for user_id in batch
                ],
                dim=0,
            ).to(base_device)
            env_ops = FSRS6BatchEnvOps(
                weights=env_weights,
                bounds=Bounds(),
                device=env_weights.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported environment '{environment}' in batched run.")

        behavior, cost_model = _build_behavior_cost(
            len(batch),
            deck_size=args.deck,
            learn_limit=args.learn_limit,
            review_limit=args.review_limit,
            cost_limit_minutes=args.cost_limit_minutes,
            learn_costs=learn_costs.to(env_ops.device),
            review_costs=review_costs.to(env_ops.device),
            first_rating_prob=first_rating_prob.to(env_ops.device),
            review_rating_prob=review_rating_prob.to(env_ops.device),
            learning_rating_prob=learning_rating_prob.to(env_ops.device),
            relearning_rating_prob=relearning_rating_prob.to(env_ops.device),
            state_rating_costs=state_rating_costs.to(env_ops.device),
            short_term=short_term_enabled,
        )

        for scheduler_spec in schedulers:
            name, fixed_interval, raw = parse_scheduler_spec(scheduler_spec)
            if name not in {"fsrs6", "anki_sm2", "memrise", "fixed", "lstm"}:
                raise ValueError(f"Unsupported scheduler '{name}' in batched run.")
            label_prefix = f"{environment} u{batch[0]}-{batch[-1]} {name}"

            if name == "fsrs6":
                weights = torch.stack(
                    [
                        torch.tensor(
                            load_benchmark_weights(
                                repo_root=REPO_ROOT,
                                benchmark_root=benchmark_root,
                                environment="fsrs6",
                                user_id=user_id,
                                partition_key=args.benchmark_partition,
                                overrides=overrides,
                                short_term=short_term_enabled,
                            ),
                            dtype=torch.float32,
                        )
                        for user_id in batch
                    ],
                    dim=0,
                ).to(env_ops.device)
                for dr in dr_values:
                    scheduler_ops = FSRS6BatchSchedulerOps(
                        weights=weights,
                        desired_retention=dr,
                        bounds=Bounds(),
                        priority_mode=args.scheduler_priority,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    _simulate_and_log(
                        args=args,
                        batch=batch,
                        env_ops=env_ops,
                        sched_ops=scheduler_ops,
                        behavior=behavior,
                        cost_model=cost_model,
                        progress=progress,
                        progress_queue=progress_queue,
                        device_label=device_label,
                        run_label=f"{label_prefix} dr={dr:.2f}",
                        environment=environment,
                        scheduler_name=name,
                        scheduler_spec=raw,
                        desired_retention=dr,
                        fixed_interval=fixed_interval,
                        short_term_source=short_term_source,
                        learning_steps=learning_steps,
                        relearning_steps=relearning_steps,
                        learning_steps_arg=learning_steps_arg,
                        relearning_steps_arg=relearning_steps_arg,
                        log_root=log_root,
                        batch_log_root=batch_log_root,
                    )
                continue

            if name == "lstm":
                if lstm_packed is None:
                    lstm_paths = _resolve_lstm_paths(
                        batch, benchmark_root, short_term=short_term_enabled
                    )
                    lstm_packed = PackedLSTMWeights.from_paths(
                        lstm_paths,
                        use_duration_feature=False,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                for dr in dr_values:
                    sched_ops = LSTMBatchSchedulerOps(
                        lstm_packed,
                        desired_retention=dr,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    _simulate_and_log(
                        args=args,
                        batch=batch,
                        env_ops=env_ops,
                        sched_ops=sched_ops,
                        behavior=behavior,
                        cost_model=cost_model,
                        progress=progress,
                        progress_queue=progress_queue,
                        device_label=device_label,
                        run_label=f"{label_prefix} dr={dr:.2f}",
                        environment=environment,
                        scheduler_name=name,
                        scheduler_spec=raw,
                        desired_retention=dr,
                        fixed_interval=fixed_interval,
                        short_term_source=short_term_source,
                        learning_steps=learning_steps,
                        relearning_steps=relearning_steps,
                        learning_steps_arg=learning_steps_arg,
                        relearning_steps_arg=relearning_steps_arg,
                        log_root=log_root,
                        batch_log_root=batch_log_root,
                    )
                continue

            if name == "anki_sm2":
                scheduler = AnkiSM2Scheduler()
                sched_ops = AnkiSM2BatchSchedulerOps(
                    graduating_interval=scheduler.graduating_interval,
                    easy_interval=scheduler.easy_interval,
                    easy_bonus=scheduler.easy_bonus,
                    hard_interval_factor=scheduler.hard_interval_factor,
                    ease_start=scheduler.ease_start,
                    ease_min=scheduler.ease_min,
                    ease_max=scheduler.ease_max,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                _simulate_and_log(
                    args=args,
                    batch=batch,
                    env_ops=env_ops,
                    sched_ops=sched_ops,
                    behavior=behavior,
                    cost_model=cost_model,
                    progress=progress,
                    progress_queue=progress_queue,
                    device_label=device_label,
                    run_label=label_prefix,
                    environment=environment,
                    scheduler_name=name,
                    scheduler_spec=raw,
                    desired_retention=None,
                    fixed_interval=fixed_interval,
                    short_term_source=short_term_source,
                    learning_steps=learning_steps,
                    relearning_steps=relearning_steps,
                    learning_steps_arg=learning_steps_arg,
                    relearning_steps_arg=relearning_steps_arg,
                    log_root=log_root,
                    batch_log_root=batch_log_root,
                )
                continue

            if name == "memrise":
                scheduler = MemriseScheduler()
                sched_ops = MemriseBatchSchedulerOps(
                    scheduler,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                _simulate_and_log(
                    args=args,
                    batch=batch,
                    env_ops=env_ops,
                    sched_ops=sched_ops,
                    behavior=behavior,
                    cost_model=cost_model,
                    progress=progress,
                    progress_queue=progress_queue,
                    device_label=device_label,
                    run_label=label_prefix,
                    environment=environment,
                    scheduler_name=name,
                    scheduler_spec=raw,
                    desired_retention=None,
                    fixed_interval=fixed_interval,
                    short_term_source=short_term_source,
                    learning_steps=learning_steps,
                    relearning_steps=relearning_steps,
                    learning_steps_arg=learning_steps_arg,
                    relearning_steps_arg=relearning_steps_arg,
                    log_root=log_root,
                    batch_log_root=batch_log_root,
                )
                continue

            if name == "fixed":
                interval = normalize_fixed_interval(fixed_interval)
                sched_ops = FixedBatchSchedulerOps(
                    interval=interval,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                _simulate_and_log(
                    args=args,
                    batch=batch,
                    env_ops=env_ops,
                    sched_ops=sched_ops,
                    behavior=behavior,
                    cost_model=cost_model,
                    progress=progress,
                    progress_queue=progress_queue,
                    device_label=device_label,
                    run_label=f"{label_prefix} ivl={interval:.2f}",
                    environment=environment,
                    scheduler_name=name,
                    scheduler_spec=raw,
                    desired_retention=None,
                    fixed_interval=interval,
                    short_term_source=short_term_source,
                    learning_steps=learning_steps,
                    relearning_steps=relearning_steps,
                    learning_steps_arg=learning_steps_arg,
                    relearning_steps_arg=relearning_steps_arg,
                    log_root=log_root,
                    batch_log_root=batch_log_root,
                )
                continue


def _run_batch_worker(
    *,
    args: argparse.Namespace,
    batch: list[int],
    benchmark_root: Path,
    overrides: dict[str, str],
    log_root: Path,
    batch_log_root: Path,
    dr_values: list[float],
    device_str: str,
    progress_queue,
) -> None:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _run_batch_core(
        args=args,
        batch=batch,
        benchmark_root=benchmark_root,
        overrides=overrides,
        log_root=log_root,
        batch_log_root=batch_log_root,
        dr_values=dr_values,
        device=device,
        progress=False,
        progress_queue=progress_queue,
        device_label=device_str,
    )


class _LocalProgressQueue:
    def __init__(self, overall: tqdm) -> None:
        self._overall = overall

    def put(self, message) -> None:
        if not isinstance(message, tuple):
            return
        if message[0] != "overall":
            return
        self._overall.update(message[1])


def main() -> int:
    args = parse_args()
    envs = parse_csv(args.env)
    if not envs:
        raise ValueError("No environments specified.")
    for env in envs:
        if env not in {"lstm", "fsrs6"}:
            raise ValueError(
                "Batched retention sweep supports only lstm or fsrs6 environments."
            )
    schedulers = parse_csv(args.sched)
    if not schedulers:
        raise ValueError("No schedulers specified.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.torch_device and args.cuda_devices:
        raise ValueError("--torch-device cannot be combined with --cuda-devices.")

    user_ids = list(range(args.start_user, args.end_user + 1))
    if not user_ids:
        raise ValueError("Empty user range.")

    benchmark_root = args.srs_benchmark_root or (REPO_ROOT.parent / "srs-benchmark")
    benchmark_root = benchmark_root.resolve()
    overrides = parse_result_overrides(args.benchmark_result)

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    log_root.mkdir(parents=True, exist_ok=True)
    batch_log_root = log_root / "batch_logs"
    batch_log_root.mkdir(parents=True, exist_ok=True)

    dr_values = _dr_values(args.start_retention, args.end_retention, args.step)
    devices = _parse_cuda_devices(args.cuda_devices)
    if devices and not torch.cuda.is_available():
        raise ValueError("--cuda-devices was provided but CUDA is not available.")
    device = torch.device(args.torch_device) if args.torch_device else None

    batches = list(_chunked(user_ids, args.batch_size))
    runs_per_batch = 0
    for scheduler in schedulers:
        name, _, _ = parse_scheduler_spec(scheduler)
        if name in {"fsrs6", "lstm"}:
            runs_per_batch += len(dr_values)
        else:
            runs_per_batch += 1
    runs_per_batch *= len(envs)
    total_user_days = sum(len(batch) * args.days * runs_per_batch for batch in batches)
    overall = None
    if not args.no_progress:
        overall = tqdm(
            total=total_user_days,
            desc="Overall",
            unit="user-day",
            leave=True,
        )

    if devices and len(devices) > 1:
        ctx = get_context("spawn")
        manager = ctx.Manager() if overall is not None else None
        progress_queue = manager.Queue() if manager is not None else None
        gpu_bars: dict[str, tqdm] = {}
        if overall is not None:
            for idx, dev in enumerate(devices, start=1):
                label = f"cuda:{dev}"
                gpu_bars[label] = tqdm(
                    total=0,
                    desc=label,
                    unit="day",
                    position=idx,
                    leave=False,
                )

        def _drain_progress_queue() -> None:
            if overall is None or progress_queue is None:
                return
            while True:
                try:
                    message = progress_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(message, tuple):
                    continue
                kind = message[0]
                if kind == "overall":
                    overall.update(message[1])
                    continue
                if kind == "start":
                    _, device_label, run_label, total_days = message
                    bar = gpu_bars.get(device_label)
                    if bar is None:
                        bar = tqdm(
                            total=0,
                            desc=device_label,
                            unit="day",
                            leave=False,
                        )
                        gpu_bars[device_label] = bar
                    bar.reset(total=int(total_days))
                    bar.set_description_str(f"{device_label} {run_label}")
                    bar.refresh()
                    continue
                if kind == "gpu":
                    _, device_label, delta = message
                    bar = gpu_bars.get(device_label)
                    if bar is not None:
                        bar.update(int(delta))
                    continue

        pending: set[futures.Future] = set()
        try:
            with futures.ProcessPoolExecutor(
                max_workers=len(devices),
                mp_context=ctx,
            ) as executor:
                for batch_idx, batch in enumerate(batches):
                    device_str = f"cuda:{devices[batch_idx % len(devices)]}"
                    pending.add(
                        executor.submit(
                            _run_batch_worker,
                            args=args,
                            batch=batch,
                            benchmark_root=benchmark_root,
                            overrides=overrides,
                            log_root=log_root,
                            batch_log_root=batch_log_root,
                            dr_values=dr_values,
                            device_str=device_str,
                            progress_queue=progress_queue,
                        )
                    )
                while pending:
                    done, pending = futures.wait(
                        pending, timeout=0.1, return_when=futures.FIRST_COMPLETED
                    )
                    _drain_progress_queue()
                    for task in done:
                        task.result()
                _drain_progress_queue()
        finally:
            for bar in gpu_bars.values():
                bar.close()
            if manager is not None:
                manager.shutdown()
    else:
        batch_device = torch.device(f"cuda:{devices[0]}") if devices else device
        for batch in batches:
            progress_queue = (
                _LocalProgressQueue(overall) if overall is not None else None
            )
            _run_batch_core(
                args=args,
                batch=batch,
                benchmark_root=benchmark_root,
                overrides=overrides,
                log_root=log_root,
                batch_log_root=batch_log_root,
                dr_values=dr_values,
                device=batch_device,
                progress=not args.no_progress,
                progress_queue=progress_queue,
                device_label=str(batch_device or "device"),
            )
    if overall is not None:
        overall.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

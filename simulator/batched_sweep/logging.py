from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import torch

from simulator.core import SimulationStats
from simulator.vectorized.multiuser_engine import simulate_multiuser
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost


def progress_callback_from_queue(
    progress_queue,
    *,
    multiplier: int,
    device_label: str,
    run_label: str,
    total_days: int,
) -> Callable[[int, int], None] | None:
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


def _write_batch_stats_csv(
    *,
    batch_stats: dict[str, list[int]],
    batch_log_root: Path,
    batch: list[int],
    environment: str,
    scheduler_name: str,
    desired_retention: float | None,
    fixed_interval: float | None,
    short_term_source: str | None,
    short_term_loops_limit: int | None,
    seed: int,
) -> None:
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
        if short_term_loops_limit is not None:
            parts.append(f"stloops={short_term_loops_limit}")
    parts.append(f"seed={seed}")
    filename = batch_log_root / f"batch_{'_'.join(parts)}.csv"
    with filename.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"])
        allocated = batch_stats.get("gpu_peak_allocated_bytes")
        reserved = batch_stats.get("gpu_peak_reserved_bytes")
        total_days = len(allocated) if allocated is not None else len(reserved or [])
        for day in range(total_days):
            alloc = allocated[day] if allocated is not None else ""
            resv = reserved[day] if reserved is not None else ""
            writer.writerow([day, alloc, resv])


def simulate_and_log(
    *,
    write_log: Callable[[argparse.Namespace, SimulationStats], None],
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
    progress_callback = progress_callback_from_queue(
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
        _write_batch_stats_csv(
            batch_stats=batch_stats,
            batch_log_root=batch_log_root,
            batch=batch,
            environment=environment,
            scheduler_name=scheduler_name,
            desired_retention=desired_retention,
            fixed_interval=fixed_interval,
            short_term_source=short_term_source,
            short_term_loops_limit=args.short_term_loops_limit,
            seed=args.seed,
        )
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
        write_log(log_args, stats)

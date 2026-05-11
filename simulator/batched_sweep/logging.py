from __future__ import annotations

import argparse
import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from simulator.core import SimulationStats
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.batched_engine.multiuser_types import MultiUserBehavior, MultiUserCost


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


@dataclass(frozen=True, slots=True)
class BatchedSweepLogLane:
    user_id: int
    log_root: Path
    environment: str
    scheduler_name: str
    scheduler_spec: str
    desired_retention: float | None
    fixed_interval: float | None
    fsrs6_adr_policy: Path | None = None
    fsrs6_adr_baseline_desired_retention: float | None = None
    fsrs6_adr_lambda_value: float | None = None
    fsrs6_ap_policy: Path | None = None
    fsrs6_ap_baseline_desired_retention: float | None = None
    fsrs6_ap_lambda_value: float | None = None
    log_dir: Path | None = None

    @property
    def final_log_dir(self) -> Path:
        return self.log_dir or (self.log_root / f"user_{self.user_id}")


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


def _build_log_args(
    *,
    args: argparse.Namespace,
    user_id: int,
    environment: str,
    scheduler_name: str,
    scheduler_spec: str,
    desired_retention: float | None,
    fixed_interval: float | None,
    short_term_source: str | None,
    learning_steps_arg: str | None,
    relearning_steps_arg: str | None,
    log_dir: Path,
    fsrs6_adr_policy: Path | None = None,
    fsrs6_ap_policy: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
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
        run_id=getattr(args, "run_id", None),
        user_id=user_id,
        button_usage=str(args.button_usage) if args.button_usage is not None else None,
        desired_retention=desired_retention,
        scheduler_priority=args.scheduler_priority,
        sspmmc_policy=None,
        fsrs6_adr_policy=fsrs6_adr_policy,
        fsrs6_adr_baseline_desired_retention=None,
        fsrs6_adr_lambda_value=None,
        fsrs6_ap_policy=fsrs6_ap_policy,
        fsrs6_ap_baseline_desired_retention=None,
        fsrs6_ap_lambda_value=None,
        fixed_interval=fixed_interval,
        seed=args.seed,
        fuzz=args.fuzz,
        short_term_source=short_term_source,
        learning_steps=learning_steps_arg,
        relearning_steps=relearning_steps_arg,
        short_term_threshold=args.short_term_threshold,
        short_term_loops_limit=args.short_term_loops_limit,
        log_dir=log_dir,
        log_reviews=False,
        write_daily_csv=bool(getattr(args, "diagnostic_csv_logs", False)),
    )


def simulate_and_log_lanes(
    *,
    write_log: Callable[[argparse.Namespace, SimulationStats], None],
    args: argparse.Namespace,
    lanes: Sequence[BatchedSweepLogLane],
    env_ops,
    sched_ops,
    behavior: MultiUserBehavior,
    cost_model: MultiUserCost,
    progress: bool,
    progress_queue,
    device_label: str,
    run_label: str,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
    learning_steps_arg: str | None,
    relearning_steps_arg: str | None,
    batch_log_root: Path,
) -> None:
    if not lanes:
        raise ValueError("No sweep lanes were provided.")

    environment = lanes[0].environment
    scheduler_names = {lane.scheduler_name for lane in lanes}
    scheduler_name = lanes[0].scheduler_name if len(scheduler_names) == 1 else "mixed"
    if any(lane.environment != environment for lane in lanes):
        raise ValueError("All sweep lanes must share the same environment.")

    batch = [lane.user_id for lane in lanes]
    progress_callback = progress_callback_from_queue(
        progress_queue,
        multiplier=len(batch),
        device_label=device_label,
        run_label=run_label,
        total_days=args.days,
    )
    diagnostic_csv_logs = bool(getattr(args, "diagnostic_csv_logs", False))
    batch_stats: dict[str, list[int]] | None = {} if diagnostic_csv_logs else None
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
    if diagnostic_csv_logs and batch_stats:
        desired_retentions = [lane.desired_retention for lane in lanes]
        fixed_intervals = [lane.fixed_interval for lane in lanes]
        _write_batch_stats_csv(
            batch_stats=batch_stats,
            batch_log_root=batch_log_root,
            batch=batch,
            environment=environment,
            scheduler_name=scheduler_name,
            desired_retention=desired_retentions[0]
            if desired_retentions
            and all(value == desired_retentions[0] for value in desired_retentions)
            else None,
            fixed_interval=fixed_intervals[0]
            if fixed_intervals
            and all(value == fixed_intervals[0] for value in fixed_intervals)
            else None,
            short_term_source=short_term_source,
            short_term_loops_limit=args.short_term_loops_limit,
            seed=args.seed,
        )
    if args.no_log:
        return
    for lane, stats in zip(lanes, stats_list, strict=True):
        user_log_dir = lane.final_log_dir
        user_log_dir.mkdir(parents=True, exist_ok=True)
        log_args = _build_log_args(
            args=args,
            user_id=lane.user_id,
            environment=lane.environment,
            scheduler_name=lane.scheduler_name,
            scheduler_spec=lane.scheduler_spec,
            desired_retention=lane.desired_retention,
            fixed_interval=lane.fixed_interval,
            short_term_source=short_term_source,
            learning_steps_arg=learning_steps_arg,
            relearning_steps_arg=relearning_steps_arg,
            log_dir=user_log_dir,
            fsrs6_adr_policy=lane.fsrs6_adr_policy,
            fsrs6_ap_policy=lane.fsrs6_ap_policy,
        )
        log_args.fsrs6_adr_baseline_desired_retention = (
            lane.fsrs6_adr_baseline_desired_retention
        )
        log_args.fsrs6_adr_lambda_value = lane.fsrs6_adr_lambda_value
        log_args.fsrs6_ap_baseline_desired_retention = (
            lane.fsrs6_ap_baseline_desired_retention
        )
        log_args.fsrs6_ap_lambda_value = lane.fsrs6_ap_lambda_value
        write_log(log_args, stats)


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
    lanes = [
        BatchedSweepLogLane(
            user_id=user_id,
            log_root=log_root,
            environment=environment,
            scheduler_name=scheduler_name,
            scheduler_spec=scheduler_spec,
            desired_retention=desired_retention,
            fixed_interval=fixed_interval,
            fsrs6_adr_policy=getattr(args, "fsrs6_adr_policy", None),
            fsrs6_ap_policy=getattr(args, "fsrs6_ap_policy", None),
        )
        for user_id in batch
    ]
    simulate_and_log_lanes(
        write_log=write_log,
        args=args,
        lanes=lanes,
        env_ops=env_ops,
        sched_ops=sched_ops,
        behavior=behavior,
        cost_model=cost_model,
        progress=progress,
        progress_queue=progress_queue,
        device_label=device_label,
        run_label=run_label,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        learning_steps_arg=learning_steps_arg,
        relearning_steps_arg=relearning_steps_arg,
        batch_log_root=batch_log_root,
    )

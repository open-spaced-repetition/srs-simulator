from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.batched_sweep.runner import BatchedSweepContext
from simulator.batched_sweep.utils import chunked, dr_values, parse_cuda_devices
from simulator.scheduler_spec import parse_scheduler_spec


SUPPORTED_ENVS = {"lstm", "fsrs6"}
SUPPORTED_SCHEDS = {"fsrs6", "fsrs3", "lstm", "anki_sm2", "memrise", "fixed"}


@dataclass(frozen=True)
class BatchedSweepPlan:
    ctx: BatchedSweepContext
    batches: list[list[int]]
    devices: list[str]
    device: torch.device | None
    total_user_days: int


def build_batched_sweep_plan(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    envs: list[str],
    schedulers: list[str],
) -> BatchedSweepPlan:
    if not envs:
        raise ValueError("No environments specified.")
    for env in envs:
        if env not in SUPPORTED_ENVS:
            raise ValueError(
                "Batched retention sweep supports only lstm or fsrs6 environments."
            )

    if not schedulers:
        raise ValueError("No schedulers specified.")
    for raw in schedulers:
        name, _, _ = parse_scheduler_spec(raw)
        if name not in SUPPORTED_SCHEDS:
            raise ValueError(f"Unsupported scheduler '{name}' in batched run.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.torch_device and args.cuda_devices:
        raise ValueError("--torch-device cannot be combined with --cuda-devices.")

    user_ids = list(range(args.start_user, args.end_user + 1))
    if not user_ids:
        raise ValueError("Empty user range.")

    benchmark_root = resolve_benchmark_root(
        repo_root, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)

    log_root = args.log_dir or (repo_root / "logs" / "retention_sweep")
    log_root.mkdir(parents=True, exist_ok=True)
    batch_log_root = log_root / "batch_logs"
    batch_log_root.mkdir(parents=True, exist_ok=True)

    drs = dr_values(args.start_retention, args.end_retention, args.step)
    devices = parse_cuda_devices(args.cuda_devices)
    if devices and not torch.cuda.is_available():
        raise ValueError("--cuda-devices was provided but CUDA is not available.")
    device = torch.device(args.torch_device) if args.torch_device else None

    batches = list(chunked(user_ids, args.batch_size))

    runs_per_batch = 0
    for scheduler in schedulers:
        name, _, _ = parse_scheduler_spec(scheduler)
        if name in {"fsrs6", "fsrs3", "lstm"}:
            runs_per_batch += len(drs)
        else:
            runs_per_batch += 1
    runs_per_batch *= len(envs)
    total_user_days = sum(
        len(batch) * int(args.days) * runs_per_batch for batch in batches
    )

    ctx = BatchedSweepContext(
        repo_root=repo_root,
        benchmark_root=benchmark_root,
        overrides=overrides,
        log_root=log_root,
        batch_log_root=batch_log_root,
        envs=envs,
        schedulers=schedulers,
        dr_values=drs,
    )

    return BatchedSweepPlan(
        ctx=ctx,
        batches=batches,
        devices=devices,
        device=device,
        total_user_days=int(total_user_days),
    )

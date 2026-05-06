from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.batched_sweep.runner import BatchedSweepContext, _build_sweep_lanes
from simulator.batched_sweep.sa_policy import resolve_sa_fsrs6_policy_specs
from simulator.batched_sweep.utils import chunked, dr_values, parse_cuda_devices
from simulator.scheduler_spec import parse_scheduler_spec


SUPPORTED_ENVS = {"lstm", "fsrs6", "fsrs6_default"}
SUPPORTED_SCHEDS = {
    "fsrs6",
    "fsrs6_default",
    "fsrs3_default",
    "fsrs3",
    "lstm",
    "anki_sm2",
    "memrise",
    "fixed",
    "sa_fsrs6",
}


@dataclass(frozen=True)
class BatchedSweepPlan:
    ctx: BatchedSweepContext
    batches: list[list[int]]
    devices: list[str]
    device: torch.device | None
    total_user_days: int
    total_lanes: int
    example_log_dir: Path | None


_LOG_LAYOUTS = {"user", "sweep"}


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
                "Batched retention sweep supports only lstm, fsrs6, or fsrs6_default environments."
            )

    if not schedulers:
        raise ValueError("No schedulers specified.")
    for raw in schedulers:
        name, _, _ = parse_scheduler_spec(raw)
        if name not in SUPPORTED_SCHEDS:
            raise ValueError(f"Unsupported scheduler '{name}' in batched run.")
        if name == "sa_fsrs6" and not _has_sa_fsrs6_source(args):
            raise ValueError(
                "--sched sa_fsrs6 requires an SA policy source "
                "(--sa-fsrs6-policy, --sa-fsrs6-policy-root, "
                "--sa-fsrs6-train-run-root, or --sa-fsrs6-policy-manifest)."
            )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    max_lanes_per_batch = getattr(args, "max_lanes_per_batch", None)
    if max_lanes_per_batch is not None and max_lanes_per_batch < 1:
        raise ValueError("--max-lanes-per-batch must be >= 1.")
    if args.torch_device and args.cuda_devices:
        raise ValueError("--torch-device cannot be combined with --cuda-devices.")
    log_layout = getattr(args, "log_layout", "user")
    if log_layout not in _LOG_LAYOUTS:
        raise ValueError(f"--log-layout must be one of {sorted(_LOG_LAYOUTS)}.")

    explicit_user_ids = getattr(args, "user_ids", None)
    if explicit_user_ids is not None:
        user_ids = list(explicit_user_ids)
    else:
        user_ids = list(range(args.start_user, args.end_user + 1))
    if not user_ids:
        raise ValueError("Empty user range.")
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("User ids must not contain duplicates.")

    benchmark_root = resolve_benchmark_root(
        repo_root, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)

    log_root = args.log_dir or (repo_root / "logs" / "retention_sweep")
    log_root.mkdir(parents=True, exist_ok=True)
    batch_log_root = log_root / "batch_logs"
    if getattr(args, "diagnostic_csv_logs", False):
        batch_log_root.mkdir(parents=True, exist_ok=True)

    drs = dr_values(args.start_retention, args.end_retention, args.step)
    if any(value <= 0.0 or value >= 1.0 for value in drs):
        raise ValueError("Retention grid values must satisfy 0 < value < 1.")
    devices = parse_cuda_devices(args.cuda_devices)
    if devices and not torch.cuda.is_available():
        raise ValueError("--cuda-devices was provided but CUDA is not available.")
    device = torch.device(args.torch_device) if args.torch_device else None

    batches = list(chunked(user_ids, args.batch_size))

    sa_fsrs6_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "sa_fsrs6" for raw in schedulers):
        if _uses_expanded_sa_fsrs6_source(args):
            sa_fsrs6_policy_specs = resolve_sa_fsrs6_policy_specs(
                user_ids=user_ids,
                dr_values=drs,
                policy_root=getattr(args, "sa_fsrs6_policy_root", None),
                train_run_root=getattr(args, "sa_fsrs6_train_run_root", None),
                policy_manifest=getattr(args, "sa_fsrs6_policy_manifest", None),
                lambda_values=getattr(args, "sa_fsrs6_lambda_values", None),
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
        log_layout=log_layout,
        sa_fsrs6_policy=getattr(args, "sa_fsrs6_policy", None),
        sa_fsrs6_policy_specs=sa_fsrs6_policy_specs,
    )
    total_lanes = 0
    example_log_dir: Path | None = None
    for batch in batches:
        for environment in envs:
            lanes = _build_sweep_lanes(batch=batch, ctx=ctx, environment=environment)
            total_lanes += len(lanes)
            if example_log_dir is None and lanes:
                example_log_dir = lanes[0].final_log_dir
    total_user_days = total_lanes * int(args.days)

    return BatchedSweepPlan(
        ctx=ctx,
        batches=batches,
        devices=devices,
        device=device,
        total_user_days=int(total_user_days),
        total_lanes=int(total_lanes),
        example_log_dir=example_log_dir,
    )


def _has_sa_fsrs6_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "sa_fsrs6_policy",
            "sa_fsrs6_policy_root",
            "sa_fsrs6_train_run_root",
            "sa_fsrs6_policy_manifest",
        )
    )


def _uses_expanded_sa_fsrs6_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "sa_fsrs6_policy_root", None) is not None,
        getattr(args, "sa_fsrs6_train_run_root", None) is not None,
        getattr(args, "sa_fsrs6_policy_manifest", None) is not None,
    ]
    if getattr(args, "sa_fsrs6_policy", None) is not None and any(expanded):
        raise ValueError(
            "--sa-fsrs6-policy cannot be combined with expanded SA policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded SA FSRS-6 policy source: "
            "--sa-fsrs6-policy-root, --sa-fsrs6-train-run-root, or "
            "--sa-fsrs6-policy-manifest."
        )
    return any(expanded)

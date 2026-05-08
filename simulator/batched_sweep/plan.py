from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.batched_sweep.runner import BatchedSweepContext, _build_sweep_lanes
from simulator.batched_sweep.fsrs6_adr_delta_policy import (
    resolve_fsrs6_adr_delta_policy_specs,
)
from simulator.batched_sweep.fsrs6_adr_direct_policy import (
    resolve_fsrs6_adr_direct_policy_specs,
)
from simulator.batched_sweep.fsrs6_adp_policy import (
    resolve_fsrs6_adp_policy_specs,
)
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
    "fsrs6_adr_direct",
    "fsrs6_adr_delta",
    "fsrs6_adp",
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
        if name == "fsrs6_adr_direct" and not _has_fsrs6_adr_direct_source(args):
            raise ValueError(
                "--sched fsrs6_adr_direct requires an FSRS6 ADR Direct policy source "
                "(--fsrs6-adr-direct-policy, --fsrs6-adr-direct-policy-root, "
                "--fsrs6-adr-direct-train-run-root, or --fsrs6-adr-direct-policy-manifest)."
            )
        if name == "fsrs6_adr_delta" and not _has_fsrs6_adr_delta_source(args):
            raise ValueError(
                "--sched fsrs6_adr_delta requires an FSRS6 ADR Delta policy source "
                "(--fsrs6-adr-delta-policy, --fsrs6-adr-delta-policy-root, "
                "--fsrs6-adr-delta-train-run-root, or "
                "--fsrs6-adr-delta-policy-manifest)."
            )
        if name == "fsrs6_adp" and not _has_fsrs6_adp_source(args):
            raise ValueError(
                "--sched fsrs6_adp requires an FSRS6 ADP policy source "
                "(--fsrs6-adp-policy, --fsrs6-adp-policy-root, "
                "--fsrs6-adp-train-run-root, or --fsrs6-adp-policy-manifest)."
            )

    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None and batch_size < 1:
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

    fsrs6_adr_direct_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "fsrs6_adr_direct" for raw in schedulers):
        if _uses_expanded_fsrs6_adr_direct_source(args):
            fsrs6_adr_direct_policy_specs = resolve_fsrs6_adr_direct_policy_specs(
                user_ids=user_ids,
                dr_values=drs,
                policy_root=getattr(args, "fsrs6_adr_direct_policy_root", None),
                train_run_root=getattr(args, "fsrs6_adr_direct_train_run_root", None),
                policy_manifest=getattr(args, "fsrs6_adr_direct_policy_manifest", None),
                lambda_values=getattr(args, "fsrs6_adr_direct_lambda_values", None),
            )
    fsrs6_adr_delta_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "fsrs6_adr_delta" for raw in schedulers):
        if _uses_expanded_fsrs6_adr_delta_source(args):
            fsrs6_adr_delta_policy_specs = resolve_fsrs6_adr_delta_policy_specs(
                user_ids=user_ids,
                policy_root=getattr(args, "fsrs6_adr_delta_policy_root", None),
                train_run_root=getattr(args, "fsrs6_adr_delta_train_run_root", None),
                policy_manifest=getattr(args, "fsrs6_adr_delta_policy_manifest", None),
                lambda_values=getattr(args, "fsrs6_adr_delta_lambda_values", None),
            )
    fsrs6_adp_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "fsrs6_adp" for raw in schedulers):
        if _uses_expanded_fsrs6_adp_source(args):
            fsrs6_adp_policy_specs = resolve_fsrs6_adp_policy_specs(
                user_ids=user_ids,
                dr_values=drs,
                policy_root=getattr(args, "fsrs6_adp_policy_root", None),
                train_run_root=getattr(args, "fsrs6_adp_train_run_root", None),
                policy_manifest=getattr(args, "fsrs6_adp_policy_manifest", None),
                lambda_values=getattr(args, "fsrs6_adp_lambda_values", None),
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
        fsrs6_adr_direct_policy=getattr(args, "fsrs6_adr_direct_policy", None),
        fsrs6_adr_direct_policy_specs=fsrs6_adr_direct_policy_specs,
        fsrs6_adr_delta_policy=getattr(args, "fsrs6_adr_delta_policy", None),
        fsrs6_adr_delta_policy_specs=fsrs6_adr_delta_policy_specs,
        fsrs6_adp_policy=getattr(args, "fsrs6_adp_policy", None),
        fsrs6_adp_policy_specs=fsrs6_adp_policy_specs,
    )
    batches = _build_user_batches(
        user_ids=user_ids,
        batch_size=batch_size,
        max_lanes_per_batch=max_lanes_per_batch,
        ctx=ctx,
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


def _build_user_batches(
    *,
    user_ids: list[int],
    batch_size: int | None,
    max_lanes_per_batch: int | None,
    ctx: BatchedSweepContext,
) -> list[list[int]]:
    if batch_size is not None:
        return list(chunked(user_ids, batch_size))
    if max_lanes_per_batch is None:
        return [user_ids]

    lane_counts_by_user = _lane_counts_by_user(ctx, user_ids)
    batches: list[list[int]] = []
    current: list[int] = []
    current_lanes = 0
    for user_id in user_ids:
        user_lanes = lane_counts_by_user.get(user_id, 0)
        if user_lanes <= 0:
            continue
        if current and current_lanes + user_lanes > max_lanes_per_batch:
            batches.append(current)
            current = []
            current_lanes = 0
        current.append(user_id)
        current_lanes += user_lanes
    if current:
        batches.append(current)
    return batches


def _lane_counts_by_user(
    ctx: BatchedSweepContext, user_ids: list[int]
) -> dict[int, int]:
    lanes_per_user = 0
    counts_by_user: dict[int, int] = {}
    for raw in ctx.schedulers:
        name, _, _ = parse_scheduler_spec(raw)
        if name in {"fsrs6", "fsrs6_default", "fsrs3", "fsrs3_default", "lstm"}:
            lanes_per_user += len(ctx.dr_values)
            continue
        if name == "fsrs6_adr_direct" and ctx.fsrs6_adr_direct_policy_specs:
            for spec in ctx.fsrs6_adr_direct_policy_specs:
                counts_by_user[spec.user_id] = counts_by_user.get(spec.user_id, 0) + 1
            continue
        if name == "fsrs6_adr_delta" and ctx.fsrs6_adr_delta_policy_specs:
            for spec in ctx.fsrs6_adr_delta_policy_specs:
                counts_by_user[spec.user_id] = counts_by_user.get(
                    spec.user_id, 0
                ) + len(ctx.dr_values)
            continue
        if name == "fsrs6_adp" and ctx.fsrs6_adp_policy_specs:
            for spec in ctx.fsrs6_adp_policy_specs:
                counts_by_user[spec.user_id] = counts_by_user.get(spec.user_id, 0) + 1
            continue
        if name == "fsrs6_adr_delta":
            lanes_per_user += len(ctx.dr_values)
            continue
        lanes_per_user += 1
    return {
        user_id: lanes_per_user + counts_by_user.get(user_id, 0) for user_id in user_ids
    }


def _has_fsrs6_adr_direct_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "fsrs6_adr_direct_policy",
            "fsrs6_adr_direct_policy_root",
            "fsrs6_adr_direct_train_run_root",
            "fsrs6_adr_direct_policy_manifest",
        )
    )


def _uses_expanded_fsrs6_adr_direct_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "fsrs6_adr_direct_policy_root", None) is not None,
        getattr(args, "fsrs6_adr_direct_train_run_root", None) is not None,
        getattr(args, "fsrs6_adr_direct_policy_manifest", None) is not None,
    ]
    if getattr(args, "fsrs6_adr_direct_policy", None) is not None and any(expanded):
        raise ValueError(
            "--fsrs6-adr-direct-policy cannot be combined with expanded FSRS6 ADR Direct policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded FSRS6 ADR Direct policy source: "
            "--fsrs6-adr-direct-policy-root, --fsrs6-adr-direct-train-run-root, or "
            "--fsrs6-adr-direct-policy-manifest."
        )
    return any(expanded)


def _has_fsrs6_adr_delta_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "fsrs6_adr_delta_policy",
            "fsrs6_adr_delta_policy_root",
            "fsrs6_adr_delta_train_run_root",
            "fsrs6_adr_delta_policy_manifest",
        )
    )


def _uses_expanded_fsrs6_adr_delta_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "fsrs6_adr_delta_policy_root", None) is not None,
        getattr(args, "fsrs6_adr_delta_train_run_root", None) is not None,
        getattr(args, "fsrs6_adr_delta_policy_manifest", None) is not None,
    ]
    if getattr(args, "fsrs6_adr_delta_policy", None) is not None and any(expanded):
        raise ValueError(
            "--fsrs6-adr-delta-policy cannot be combined with expanded FSRS6 ADR Delta "
            "policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded FSRS6 ADR Delta policy source: "
            "--fsrs6-adr-delta-policy-root, --fsrs6-adr-delta-train-run-root, or "
            "--fsrs6-adr-delta-policy-manifest."
        )
    return any(expanded)


def _has_fsrs6_adp_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "fsrs6_adp_policy",
            "fsrs6_adp_policy_root",
            "fsrs6_adp_train_run_root",
            "fsrs6_adp_policy_manifest",
        )
    )


def _uses_expanded_fsrs6_adp_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "fsrs6_adp_policy_root", None) is not None,
        getattr(args, "fsrs6_adp_train_run_root", None) is not None,
        getattr(args, "fsrs6_adp_policy_manifest", None) is not None,
    ]
    if getattr(args, "fsrs6_adp_policy", None) is not None and any(expanded):
        raise ValueError(
            "--fsrs6-adp-policy cannot be combined with expanded FSRS6 ADP policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded FSRS6 ADP policy source: "
            "--fsrs6-adp-policy-root, --fsrs6-adp-train-run-root, or "
            "--fsrs6-adp-policy-manifest."
        )
    return any(expanded)

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.experiment_infra.baseline_dr_selection import (
    load_baseline_dr_manifest,
)
from simulator.batched_sweep.runner import BatchedSweepContext, _build_sweep_lanes
from simulator.batched_sweep.fsrs6_adr_policy import (
    resolve_fsrs6_adr_policy_specs,
)
from simulator.batched_sweep.fsrs6_ap_policy import (
    resolve_fsrs6_ap_policy_specs,
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
    "fsrs6_adr",
    "fsrs6_ap",
}


@dataclass(frozen=True)
class BatchedSweepPlan:
    ctx: BatchedSweepContext
    batches: list[list[int]]
    batches_by_env: dict[str, list[list[int]]]
    batch_size_by_env: dict[str, int | None]
    max_lanes_per_batch_by_env: dict[str, int | None]
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
        if name == "fsrs6_adr" and not _has_fsrs6_adr_source(args):
            raise ValueError(
                "--sched fsrs6_adr requires an FSRS6 ADR policy source "
                "(--fsrs6-adr-policy, --fsrs6-adr-policy-root, "
                "--fsrs6-adr-train-run-root, or --fsrs6-adr-policy-manifest)."
            )
        if name == "fsrs6_ap" and not _has_fsrs6_ap_source(args):
            raise ValueError(
                "--sched fsrs6_ap requires an FSRS6 AP policy source "
                "(--fsrs6-ap-policy, --fsrs6-ap-policy-root, "
                "--fsrs6-ap-train-run-root, or --fsrs6-ap-policy-manifest)."
            )

    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None and batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    max_lanes_per_batch = getattr(args, "max_lanes_per_batch", None)
    if max_lanes_per_batch is not None and max_lanes_per_batch < 1:
        raise ValueError("--max-lanes-per-batch must be >= 1.")
    _validate_env_batch_overrides(args=args, envs=envs)
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
    fsrs6_dr_values_by_user: dict[int, tuple[float, ...]] = {}
    fsrs6_dr_manifest = getattr(args, "fsrs6_dr_manifest", None)
    if fsrs6_dr_manifest is not None and any(
        parse_scheduler_spec(raw)[0] == "fsrs6" for raw in schedulers
    ):
        manifest = load_baseline_dr_manifest(
            Path(fsrs6_dr_manifest),
            user_ids=user_ids,
        )
        fsrs6_dr_values_by_user = {
            user_id: manifest.values_for_user(user_id) for user_id in user_ids
        }
    devices = parse_cuda_devices(args.cuda_devices)
    if devices and not torch.cuda.is_available():
        raise ValueError("--cuda-devices was provided but CUDA is not available.")
    device = torch.device(args.torch_device) if args.torch_device else None

    fsrs6_adr_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "fsrs6_adr" for raw in schedulers):
        if _uses_expanded_fsrs6_adr_source(args):
            fsrs6_adr_policy_specs = resolve_fsrs6_adr_policy_specs(
                user_ids=user_ids,
                dr_values=drs,
                policy_root=getattr(args, "fsrs6_adr_policy_root", None),
                train_run_root=getattr(args, "fsrs6_adr_train_run_root", None),
                policy_manifest=getattr(args, "fsrs6_adr_policy_manifest", None),
                lambda_values=getattr(args, "fsrs6_adr_lambda_values", None),
            )
    fsrs6_ap_policy_specs = ()
    if any(parse_scheduler_spec(raw)[0] == "fsrs6_ap" for raw in schedulers):
        if _uses_expanded_fsrs6_ap_source(args):
            fsrs6_ap_policy_specs = resolve_fsrs6_ap_policy_specs(
                user_ids=user_ids,
                dr_values=drs,
                policy_root=getattr(args, "fsrs6_ap_policy_root", None),
                train_run_root=getattr(args, "fsrs6_ap_train_run_root", None),
                policy_manifest=getattr(args, "fsrs6_ap_policy_manifest", None),
                lambda_values=getattr(args, "fsrs6_ap_lambda_values", None),
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
        fsrs6_adr_policy=getattr(args, "fsrs6_adr_policy", None),
        fsrs6_adr_policy_specs=fsrs6_adr_policy_specs,
        fsrs6_ap_policy=getattr(args, "fsrs6_ap_policy", None),
        fsrs6_ap_policy_specs=fsrs6_ap_policy_specs,
        fsrs6_dr_values_by_user=fsrs6_dr_values_by_user,
    )
    batches_by_env: dict[str, list[list[int]]] = {}
    batch_size_by_env: dict[str, int | None] = {}
    max_lanes_per_batch_by_env: dict[str, int | None] = {}
    for environment in envs:
        env_batch_size = _env_batch_override_value(
            args=args,
            environment=environment,
            field_name="batch_size",
            default=batch_size,
        )
        env_max_lanes_per_batch = _env_batch_override_value(
            args=args,
            environment=environment,
            field_name="max_lanes_per_batch",
            default=max_lanes_per_batch,
        )
        batch_size_by_env[environment] = env_batch_size
        max_lanes_per_batch_by_env[environment] = env_max_lanes_per_batch
        batches_by_env[environment] = _build_user_batches(
            user_ids=user_ids,
            batch_size=env_batch_size,
            max_lanes_per_batch=env_max_lanes_per_batch,
            ctx=ctx,
        )
    batches = batches_by_env[envs[0]]
    total_lanes = 0
    example_log_dir: Path | None = None
    for environment in envs:
        for batch in batches_by_env[environment]:
            lanes = _build_sweep_lanes(batch=batch, ctx=ctx, environment=environment)
            total_lanes += len(lanes)
            if example_log_dir is None and lanes:
                example_log_dir = lanes[0].final_log_dir
    total_user_days = total_lanes * int(args.days)

    return BatchedSweepPlan(
        ctx=ctx,
        batches=batches,
        batches_by_env=batches_by_env,
        batch_size_by_env=batch_size_by_env,
        max_lanes_per_batch_by_env=max_lanes_per_batch_by_env,
        devices=devices,
        device=device,
        total_user_days=int(total_user_days),
        total_lanes=int(total_lanes),
        example_log_dir=example_log_dir,
    )


def _validate_env_batch_overrides(
    *,
    args: argparse.Namespace,
    envs: list[str],
) -> None:
    overrides = getattr(args, "env_batch_overrides", None) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("Environment batch overrides must be a mapping.")
    for environment, override in overrides.items():
        if environment not in SUPPORTED_ENVS:
            raise ValueError(
                "Environment batch overrides support only lstm, fsrs6, or "
                "fsrs6_default environments."
            )
        if environment not in envs:
            raise ValueError(
                "Environment batch overrides must target environments listed in the sweep."
            )
        if override is None:
            continue
        if not isinstance(override, Mapping):
            for field_name in ("batch_size", "max_lanes_per_batch"):
                value = getattr(override, field_name, None)
                _validate_optional_positive_int(
                    value,
                    f"env_batch_overrides.{environment}.{field_name}",
                )
            continue
        for field_name, value in override.items():
            if field_name not in {"batch_size", "max_lanes_per_batch"}:
                raise ValueError(
                    "Environment batch overrides may contain only batch_size and "
                    "max_lanes_per_batch."
                )
            _validate_optional_positive_int(
                value,
                f"env_batch_overrides.{environment}.{field_name}",
            )


def _env_batch_override_value(
    *,
    args: argparse.Namespace,
    environment: str,
    field_name: str,
    default: int | None,
) -> int | None:
    overrides = getattr(args, "env_batch_overrides", None) or {}
    override = overrides.get(environment) if isinstance(overrides, Mapping) else None
    if override is None:
        return default
    if isinstance(override, Mapping):
        value = override.get(field_name)
    else:
        value = getattr(override, field_name, None)
    if value is None:
        return default
    return _validate_optional_positive_int(
        value,
        f"env_batch_overrides.{environment}.{field_name}",
    )


def _validate_optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return int(value)


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
            if name == "fsrs6" and ctx.fsrs6_dr_values_by_user:
                for user_id in user_ids:
                    counts_by_user[user_id] = counts_by_user.get(user_id, 0) + len(
                        ctx.fsrs6_dr_values_by_user.get(user_id, ())
                    )
                continue
            lanes_per_user += len(ctx.dr_values)
            continue
        if name == "fsrs6_adr" and ctx.fsrs6_adr_policy_specs:
            for spec in ctx.fsrs6_adr_policy_specs:
                counts_by_user[spec.user_id] = counts_by_user.get(spec.user_id, 0) + 1
            continue
        if name == "fsrs6_ap" and ctx.fsrs6_ap_policy_specs:
            for spec in ctx.fsrs6_ap_policy_specs:
                counts_by_user[spec.user_id] = counts_by_user.get(spec.user_id, 0) + 1
            continue
        lanes_per_user += 1
    return {
        user_id: lanes_per_user + counts_by_user.get(user_id, 0) for user_id in user_ids
    }


def _has_fsrs6_adr_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "fsrs6_adr_policy",
            "fsrs6_adr_policy_root",
            "fsrs6_adr_train_run_root",
            "fsrs6_adr_policy_manifest",
        )
    )


def _uses_expanded_fsrs6_adr_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "fsrs6_adr_policy_root", None) is not None,
        getattr(args, "fsrs6_adr_train_run_root", None) is not None,
        getattr(args, "fsrs6_adr_policy_manifest", None) is not None,
    ]
    if getattr(args, "fsrs6_adr_policy", None) is not None and any(expanded):
        raise ValueError(
            "--fsrs6-adr-policy cannot be combined with expanded FSRS6 ADR policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded FSRS6 ADR policy source: "
            "--fsrs6-adr-policy-root, --fsrs6-adr-train-run-root, or "
            "--fsrs6-adr-policy-manifest."
        )
    return any(expanded)


def _has_fsrs6_ap_source(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attr, None) is not None
        for attr in (
            "fsrs6_ap_policy",
            "fsrs6_ap_policy_root",
            "fsrs6_ap_train_run_root",
            "fsrs6_ap_policy_manifest",
        )
    )


def _uses_expanded_fsrs6_ap_source(args: argparse.Namespace) -> bool:
    expanded = [
        getattr(args, "fsrs6_ap_policy_root", None) is not None,
        getattr(args, "fsrs6_ap_train_run_root", None) is not None,
        getattr(args, "fsrs6_ap_policy_manifest", None) is not None,
    ]
    if getattr(args, "fsrs6_ap_policy", None) is not None and any(expanded):
        raise ValueError(
            "--fsrs6-ap-policy cannot be combined with expanded FSRS6 AP policy sources."
        )
    if sum(expanded) > 1:
        raise ValueError(
            "Configure only one expanded FSRS6 AP policy source: "
            "--fsrs6-ap-policy-root, --fsrs6-ap-train-run-root, or "
            "--fsrs6-ap-policy-manifest."
        )
    return any(expanded)

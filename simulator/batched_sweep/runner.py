from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.scheduler_spec import normalize_fixed_interval, parse_scheduler_spec
from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps, AnkiSM2Scheduler
from simulator.schedulers.fixed import FixedBatchSchedulerOps
from simulator.schedulers.fsrs import FSRS3BatchSchedulerOps, FSRS6BatchSchedulerOps
from simulator.schedulers.lstm import LSTMBatchSchedulerOps
from simulator.schedulers.memrise import MemriseBatchSchedulerOps, MemriseScheduler
from simulator.schedulers.fsrs6_adr_delta import FSRS6ADRDeltaBatchSchedulerOps
from simulator.schedulers.fsrs6_adr_direct import FSRS6ADRDirectBatchSchedulerOps
from simulator.fsrs6_adr_delta_policy import FSRS6ADRDeltaPolicy
from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
from simulator.fsrs6_adp_policy import FSRS6ADPPolicy
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.mixed_scheduler import (
    MixedBatchSchedulerOps as _MixedBatchSchedulerOps,
    MixedSchedulerGroup as _MixedSchedulerGroup,
)

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
from simulator.batched_sweep.logging import (
    BatchedSweepLogLane,
    simulate_and_log_lanes,
)
from simulator.batched_sweep.utils import format_id_list
from simulator.batched_sweep.weights import (
    build_default_fsrs3_weights,
    build_default_fsrs6_weights,
    load_fsrs3_weights,
    load_fsrs6_weights,
    resolve_lstm_paths,
)
from simulator.batched_sweep.fsrs6_adr_delta_policy import FSRS6ADRDeltaPolicySpec
from simulator.batched_sweep.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicySpec
from simulator.batched_sweep.fsrs6_adp_policy import FSRS6ADPPolicySpec


@dataclass(frozen=True)
class BatchedSweepContext:
    repo_root: Path
    benchmark_root: Path
    overrides: dict[str, str]
    log_root: Path
    batch_log_root: Path
    envs: list[str]
    schedulers: list[str]
    dr_values: list[float]
    log_layout: str = "user"
    fsrs6_adr_direct_policy: Path | None = None
    fsrs6_adr_direct_policy_specs: tuple[FSRS6ADRDirectPolicySpec, ...] = ()
    fsrs6_adr_delta_policy: Path | None = None
    fsrs6_adr_delta_policy_specs: tuple[FSRS6ADRDeltaPolicySpec, ...] = ()
    fsrs6_adp_policy: Path | None = None
    fsrs6_adp_policy_specs: tuple[FSRS6ADPPolicySpec, ...] = ()


_DR_SCHEDULERS = {"fsrs6", "fsrs6_default", "fsrs3", "fsrs3_default", "lstm"}
_LOG_LAYOUTS = {"user", "sweep"}


def _format_float_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def _lane_log_dir(
    *,
    log_root: Path,
    user_id: int,
    scheduler_subpath: Path,
    log_layout: str,
) -> Path:
    if log_layout == "user":
        return log_root / f"user_{user_id}" / scheduler_subpath
    if log_layout == "sweep":
        return log_root / scheduler_subpath / f"user_{user_id}"
    raise ValueError(f"log_layout must be one of {sorted(_LOG_LAYOUTS)}.")


def _build_dr_grid_lanes(
    *,
    batch: list[int],
    log_root: Path,
    environment: str,
    scheduler_name: str,
    scheduler_spec: str,
    dr_values: list[float],
    fixed_interval: float | None,
    log_layout: str = "user",
) -> list[BatchedSweepLogLane]:
    lanes: list[BatchedSweepLogLane] = []
    for desired_retention in dr_values:
        scheduler_subpath = Path(f"sched_{scheduler_name}") / (
            f"dr_{_format_float_token(desired_retention)}"
        )
        dr_root = log_root / scheduler_subpath
        lanes.extend(
            BatchedSweepLogLane(
                user_id=user_id,
                log_root=dr_root,
                log_dir=_lane_log_dir(
                    log_root=log_root,
                    user_id=user_id,
                    scheduler_subpath=scheduler_subpath,
                    log_layout=log_layout,
                ),
                environment=environment,
                scheduler_name=scheduler_name,
                scheduler_spec=scheduler_spec,
                desired_retention=desired_retention,
                fixed_interval=fixed_interval,
            )
            for user_id in batch
        )
    return lanes


def _build_sweep_lanes(
    *,
    batch: list[int],
    ctx: BatchedSweepContext,
    environment: str,
) -> list[BatchedSweepLogLane]:
    if ctx.log_layout not in _LOG_LAYOUTS:
        raise ValueError(f"log_layout must be one of {sorted(_LOG_LAYOUTS)}.")
    lanes: list[BatchedSweepLogLane] = []
    for scheduler_spec in ctx.schedulers:
        name, fixed_interval, raw = parse_scheduler_spec(scheduler_spec)
        if name in _DR_SCHEDULERS:
            lanes.extend(
                _build_dr_grid_lanes(
                    batch=batch,
                    log_root=ctx.log_root,
                    log_layout=ctx.log_layout,
                    environment=environment,
                    scheduler_name=name,
                    scheduler_spec=raw,
                    dr_values=ctx.dr_values,
                    fixed_interval=None,
                )
            )
            continue

        interval = normalize_fixed_interval(fixed_interval) if name == "fixed" else None
        if name == "fsrs6_adr_direct" and ctx.fsrs6_adr_direct_policy_specs:
            batch_users = set(batch)
            for spec in ctx.fsrs6_adr_direct_policy_specs:
                if spec.user_id not in batch_users:
                    continue
                if spec.baseline_desired_retention is None:
                    policy_token = (
                        f"policy_{spec.policy_index}"
                        if spec.policy_index is not None
                        else f"policy_{spec.path.parent.name}"
                    )
                    scheduler_subpath = Path("sched_fsrs6_adr_direct") / policy_token
                else:
                    dr_token = _format_float_token(spec.baseline_desired_retention)
                    scheduler_subpath = (
                        Path("sched_fsrs6_adr_direct") / f"dr_{dr_token}"
                    )
                if spec.lambda_value is not None:
                    scheduler_subpath = scheduler_subpath / (
                        f"lambda_{_format_float_token(spec.lambda_value)}"
                    )
                scheduler_root = ctx.log_root / scheduler_subpath
                lanes.append(
                    BatchedSweepLogLane(
                        user_id=spec.user_id,
                        log_root=scheduler_root,
                        log_dir=_lane_log_dir(
                            log_root=ctx.log_root,
                            user_id=spec.user_id,
                            scheduler_subpath=scheduler_subpath,
                            log_layout=ctx.log_layout,
                        ),
                        environment=environment,
                        scheduler_name=name,
                        scheduler_spec=raw,
                        desired_retention=None,
                        fixed_interval=None,
                        fsrs6_adr_direct_policy=spec.path,
                        fsrs6_adr_direct_baseline_desired_retention=(
                            spec.baseline_desired_retention
                        ),
                        fsrs6_adr_direct_lambda_value=spec.lambda_value,
                    )
                )
            continue

        if name == "fsrs6_adr_delta" and ctx.fsrs6_adr_delta_policy_specs:
            batch_users = set(batch)
            for spec in ctx.fsrs6_adr_delta_policy_specs:
                if spec.user_id not in batch_users:
                    continue
                for desired_retention in ctx.dr_values:
                    dr_token = _format_float_token(desired_retention)
                    scheduler_subpath = Path("sched_fsrs6_adr_delta") / f"dr_{dr_token}"
                    if spec.lambda_value is not None:
                        scheduler_subpath = scheduler_subpath / (
                            f"lambda_{_format_float_token(spec.lambda_value)}"
                        )
                    scheduler_root = ctx.log_root / scheduler_subpath
                    lanes.append(
                        BatchedSweepLogLane(
                            user_id=spec.user_id,
                            log_root=scheduler_root,
                            log_dir=_lane_log_dir(
                                log_root=ctx.log_root,
                                user_id=spec.user_id,
                                scheduler_subpath=scheduler_subpath,
                                log_layout=ctx.log_layout,
                            ),
                            environment=environment,
                            scheduler_name=name,
                            scheduler_spec=raw,
                            desired_retention=desired_retention,
                            fixed_interval=None,
                            fsrs6_adr_delta_policy=spec.path,
                            fsrs6_adr_delta_lambda_value=spec.lambda_value,
                        )
                    )
            continue

        if name == "fsrs6_adp" and ctx.fsrs6_adp_policy_specs:
            batch_users = set(batch)
            for spec in ctx.fsrs6_adp_policy_specs:
                if spec.user_id not in batch_users:
                    continue
                if spec.baseline_desired_retention is None:
                    policy_token = (
                        f"policy_{spec.policy_index}"
                        if spec.policy_index is not None
                        else f"policy_{spec.path.parent.name}"
                    )
                    scheduler_subpath = Path("sched_fsrs6_adp") / policy_token
                else:
                    dr_token = _format_float_token(spec.baseline_desired_retention)
                    scheduler_subpath = Path("sched_fsrs6_adp") / f"dr_{dr_token}"
                if spec.lambda_value is not None:
                    scheduler_subpath = scheduler_subpath / (
                        f"lambda_{_format_float_token(spec.lambda_value)}"
                    )
                scheduler_root = ctx.log_root / scheduler_subpath
                lanes.append(
                    BatchedSweepLogLane(
                        user_id=spec.user_id,
                        log_root=scheduler_root,
                        log_dir=_lane_log_dir(
                            log_root=ctx.log_root,
                            user_id=spec.user_id,
                            scheduler_subpath=scheduler_subpath,
                            log_layout=ctx.log_layout,
                        ),
                        environment=environment,
                        scheduler_name=name,
                        scheduler_spec=raw,
                        desired_retention=None,
                        fixed_interval=None,
                        fsrs6_adp_policy=spec.path,
                        fsrs6_adp_baseline_desired_retention=(
                            spec.baseline_desired_retention
                        ),
                        fsrs6_adp_lambda_value=spec.lambda_value,
                    )
                )
            continue

        policy = ctx.fsrs6_adr_direct_policy if name == "fsrs6_adr_direct" else None
        dr_policy = ctx.fsrs6_adr_delta_policy if name == "fsrs6_adr_delta" else None
        adp_policy = ctx.fsrs6_adp_policy if name == "fsrs6_adp" else None
        scheduler_subpath = Path(f"sched_{name}")
        if name == "fixed" and interval is not None:
            scheduler_subpath = scheduler_subpath / (
                f"ivl_{_format_float_token(interval)}"
            )
        elif name == "fsrs6_adr_direct" and policy is not None:
            scheduler_subpath = scheduler_subpath / f"policy_{policy.stem}"
        elif name == "fsrs6_adp" and adp_policy is not None:
            scheduler_subpath = scheduler_subpath / f"policy_{adp_policy.stem}"
        if name == "fsrs6_adr_delta" and dr_policy is not None:
            for desired_retention in ctx.dr_values:
                dr_token = _format_float_token(desired_retention)
                dr_subpath = (
                    scheduler_subpath / f"dr_{dr_token}" / (f"policy_{dr_policy.stem}")
                )
                scheduler_root = ctx.log_root / dr_subpath
                lanes.extend(
                    BatchedSweepLogLane(
                        user_id=user_id,
                        log_root=scheduler_root,
                        log_dir=_lane_log_dir(
                            log_root=ctx.log_root,
                            user_id=user_id,
                            scheduler_subpath=dr_subpath,
                            log_layout=ctx.log_layout,
                        ),
                        environment=environment,
                        scheduler_name=name,
                        scheduler_spec=raw,
                        desired_retention=desired_retention,
                        fixed_interval=None,
                        fsrs6_adr_delta_policy=dr_policy,
                    )
                    for user_id in batch
                )
            continue
        scheduler_root = ctx.log_root / scheduler_subpath
        lanes.extend(
            BatchedSweepLogLane(
                user_id=user_id,
                log_root=scheduler_root,
                log_dir=_lane_log_dir(
                    log_root=ctx.log_root,
                    user_id=user_id,
                    scheduler_subpath=scheduler_subpath,
                    log_layout=ctx.log_layout,
                ),
                environment=environment,
                scheduler_name=name,
                scheduler_spec=raw,
                desired_retention=None,
                fixed_interval=interval,
                fsrs6_adr_direct_policy=policy,
                fsrs6_adp_policy=adp_policy,
            )
            for user_id in batch
        )
    return lanes


def _repeat_weights_for_lanes(
    *,
    weights: torch.Tensor,
    active_batch: list[int],
    lanes: list[BatchedSweepLogLane],
) -> torch.Tensor:
    index_by_user_id = {user_id: index for index, user_id in enumerate(active_batch)}
    lane_indices = torch.tensor(
        [index_by_user_id[lane.user_id] for lane in lanes],
        device=weights.device,
        dtype=torch.int64,
    )
    return weights.index_select(0, lane_indices)


def _repeat_lstm_weights_for_lanes(
    *,
    weights: PackedLSTMWeights,
    active_batch: list[int],
    lanes: list[BatchedSweepLogLane],
) -> PackedLSTMWeights:
    index_by_user_id = {user_id: index for index, user_id in enumerate(active_batch)}
    lane_indices = torch.tensor(
        [index_by_user_id[lane.user_id] for lane in lanes],
        device=weights.input_mean.device,
        dtype=torch.int64,
    )
    updates: dict[str, Any] = {"n_users": len(lanes)}
    for field in fields(PackedLSTMWeights):
        value = getattr(weights, field.name)
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == int(weights.n_users)
        ):
            updates[field.name] = value.index_select(0, lane_indices)
    return replace(weights, **updates)


def _required_desired_retention(lane: BatchedSweepLogLane) -> float:
    if lane.desired_retention is None:
        raise ValueError(f"{lane.scheduler_name} lanes require desired_retention.")
    return float(lane.desired_retention)


def _mixed_scheduler_group_key(lane: BatchedSweepLogLane) -> tuple[Any, ...]:
    if lane.scheduler_name in {
        "fsrs6",
        "fsrs6_default",
        "fsrs3",
        "fsrs3_default",
    }:
        return (lane.scheduler_name, lane.scheduler_spec)
    if lane.scheduler_name == "fixed":
        return (lane.scheduler_name, lane.scheduler_spec, lane.fixed_interval)
    if lane.scheduler_name == "fsrs6_adr_direct":
        return (lane.scheduler_name, lane.scheduler_spec)
    if lane.scheduler_name == "fsrs6_adr_delta":
        return (lane.scheduler_name, lane.scheduler_spec)
    return (lane.scheduler_name, lane.scheduler_spec)


def _group_lane_indices(lanes: list[BatchedSweepLogLane]) -> list[list[int]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for lane_index, lane in enumerate(lanes):
        grouped.setdefault(_mixed_scheduler_group_key(lane), []).append(lane_index)
    return list(grouped.values())


def _split_lanes(
    lanes: list[BatchedSweepLogLane],
    max_lanes_per_batch: int | None,
) -> list[list[BatchedSweepLogLane]]:
    if max_lanes_per_batch is None:
        return [lanes]
    if max_lanes_per_batch < 1:
        raise ValueError("--max-lanes-per-batch must be >= 1.")

    lanes_by_user: dict[int, list[BatchedSweepLogLane]] = {}
    user_order: list[int] = []
    for lane in lanes:
        if lane.user_id not in lanes_by_user:
            lanes_by_user[lane.user_id] = []
            user_order.append(lane.user_id)
        lanes_by_user[lane.user_id].append(lane)

    chunks: list[list[BatchedSweepLogLane]] = []
    current: list[BatchedSweepLogLane] = []
    for user_id in user_order:
        user_lanes = lanes_by_user[user_id]
        if current and len(current) + len(user_lanes) > max_lanes_per_batch:
            chunks.append(current)
            current = []
        current.extend(user_lanes)
    if current:
        chunks.append(current)
    return chunks


def _same_adr_policy_bounds(
    lhs: FSRS6ADRDirectPolicy, rhs: FSRS6ADRDirectPolicy
) -> bool:
    return (
        math.isclose(lhs.retention_min, rhs.retention_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            lhs.retention_max, rhs.retention_max, rel_tol=0.0, abs_tol=1e-9
        )
        and math.isclose(lhs.bounds.s_min, rhs.bounds.s_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.s_max, rhs.bounds.s_max, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_min, rhs.bounds.d_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_max, rhs.bounds.d_max, rel_tol=0.0, abs_tol=1e-9)
    )


def _same_adr_delta_policy_bounds(
    lhs: FSRS6ADRDeltaPolicy, rhs: FSRS6ADRDeltaPolicy
) -> bool:
    return (
        lhs.feature_version == rhs.feature_version
        and math.isclose(
            lhs.retention_min, rhs.retention_min, rel_tol=0.0, abs_tol=1e-9
        )
        and math.isclose(
            lhs.retention_max,
            rhs.retention_max,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(lhs.bounds.s_min, rhs.bounds.s_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.s_max, rhs.bounds.s_max, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_min, rhs.bounds.d_min, rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(lhs.bounds.d_max, rhs.bounds.d_max, rel_tol=0.0, abs_tol=1e-9)
    )


def _build_env_ops_for_lanes(
    *,
    environment: str,
    active_batch: list[int],
    lanes: list[BatchedSweepLogLane],
    fsrs_weights: torch.Tensor | None,
    fsrs_default_weights: torch.Tensor | None,
    lstm_packed: PackedLSTMWeights | None,
    device: torch.device,
) -> Any:
    if environment == "lstm":
        if lstm_packed is None:
            raise ValueError("Expected LSTM weights when environment is lstm.")
        lane_lstm = _repeat_lstm_weights_for_lanes(
            weights=lstm_packed,
            active_batch=active_batch,
            lanes=lanes,
        )
        return LSTMBatchedEnvOps(
            lane_lstm,
            device=lane_lstm.process_0_weight.device,
            dtype=torch.float32,
        )

    if environment == "fsrs6":
        if fsrs_weights is None:
            raise ValueError("Expected FSRS-6 weights when environment is fsrs6.")
        lane_weights = _repeat_weights_for_lanes(
            weights=fsrs_weights.to(device),
            active_batch=active_batch,
            lanes=lanes,
        )
        return FSRS6BatchEnvOps(
            weights=lane_weights,
            bounds=Bounds(),
            device=lane_weights.device,
            dtype=torch.float32,
        )

    if environment == "fsrs6_default":
        if fsrs_default_weights is None:
            raise ValueError("Expected default FSRS-6 weights for fsrs6_default.")
        lane_weights = _repeat_weights_for_lanes(
            weights=fsrs_default_weights.to(device),
            active_batch=active_batch,
            lanes=lanes,
        )
        return FSRS6BatchEnvOps(
            weights=lane_weights,
            bounds=Bounds(),
            device=lane_weights.device,
            dtype=torch.float32,
        )

    raise ValueError(f"Unsupported environment '{environment}' in batched run.")


def _build_mixed_scheduler_ops(
    *,
    args: argparse.Namespace,
    active_batch: list[int],
    lanes: list[BatchedSweepLogLane],
    fsrs_weights: torch.Tensor | None,
    fsrs_default_weights: torch.Tensor | None,
    fsrs3_weights: torch.Tensor | None,
    fsrs3_default_weights: torch.Tensor | None,
    lstm_packed: PackedLSTMWeights | None,
    short_term_source: str | None,
    device: torch.device,
) -> _MixedBatchSchedulerOps:
    groups: list[_MixedSchedulerGroup] = []
    for indices in _group_lane_indices(lanes):
        group_lanes = [lanes[index] for index in indices]
        sample = group_lanes[0]
        lane_indices = torch.tensor(indices, device=device, dtype=torch.int64)
        name = sample.scheduler_name

        if name == "fsrs6":
            if fsrs_weights is None:
                raise ValueError("Expected FSRS-6 weights for fsrs6 scheduler.")
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            desired_retention = torch.tensor(
                [_required_desired_retention(lane) for lane in group_lanes],
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS6BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                bounds=Bounds(),
                priority_mode=args.scheduler_priority,
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs6_default":
            if fsrs_default_weights is None:
                raise ValueError(
                    "Expected default FSRS-6 weights for fsrs6_default scheduler."
                )
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs_default_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            desired_retention = torch.tensor(
                [_required_desired_retention(lane) for lane in group_lanes],
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS6BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                bounds=Bounds(),
                priority_mode=args.scheduler_priority,
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs3":
            if fsrs3_weights is None:
                raise ValueError("Expected FSRS-3 weights for fsrs3 scheduler.")
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs3_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            desired_retention = torch.tensor(
                [_required_desired_retention(lane) for lane in group_lanes],
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS3BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                bounds=Bounds(),
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs3_default":
            if fsrs3_default_weights is None:
                raise ValueError(
                    "Expected default FSRS-3 weights for fsrs3_default scheduler."
                )
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs3_default_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            desired_retention = torch.tensor(
                [_required_desired_retention(lane) for lane in group_lanes],
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS3BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                bounds=Bounds(),
                device=device,
                dtype=torch.float32,
            )
        elif name == "lstm":
            if lstm_packed is None:
                raise ValueError("Expected LSTM weights for lstm scheduler.")
            scheduler_weights = _repeat_lstm_weights_for_lanes(
                weights=lstm_packed,
                active_batch=active_batch,
                lanes=group_lanes,
            )
            interval_mode = "float" if short_term_source == "sched" else "integer"
            min_interval = 0.0 if short_term_source == "sched" else 1.0
            desired_retention = torch.tensor(
                [_required_desired_retention(lane) for lane in group_lanes],
                device=device,
                dtype=torch.float32,
            )
            ops = LSTMBatchSchedulerOps(
                scheduler_weights,
                desired_retention=desired_retention,
                min_interval=min_interval,
                interval_mode=interval_mode,
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs6_adr_direct":
            if fsrs_weights is None:
                raise ValueError(
                    "Expected FSRS-6 weights for fsrs6_adr_direct scheduler."
                )
            policy_paths: list[Path] = []
            for lane in group_lanes:
                policy_path = lane.fsrs6_adr_direct_policy
                if policy_path is None:
                    raise ValueError(
                        "--sched fsrs6_adr_direct requires an FSRS6 ADR Direct policy source."
                    )
                policy_paths.append(policy_path)
            policies = [
                FSRS6ADRDirectPolicy.from_json(policy_path)
                for policy_path in policy_paths
            ]
            policy = policies[0]
            for policy_path, candidate in zip(policy_paths, policies, strict=True):
                if not _same_adr_policy_bounds(candidate, policy):
                    raise ValueError(
                        "Batched fsrs6_adr_direct sweep requires identical policy retention "
                        f"and FSRS bounds. Mismatch at {policy_path}."
                    )
            coefficients = torch.tensor(
                [candidate.coefficients for candidate in policies],
                device=device,
                dtype=torch.float32,
            )
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            ops = FSRS6ADRDirectBatchSchedulerOps(
                weights=scheduler_weights,
                policy=policy,
                coefficients=coefficients,
                bounds=policy.bounds,
                priority_mode=args.scheduler_priority,
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs6_adr_delta":
            if fsrs_weights is None:
                raise ValueError(
                    "Expected FSRS-6 weights for fsrs6_adr_delta scheduler."
                )
            policy_paths: list[Path] = []
            desired_retentions: list[float] = []
            for lane in group_lanes:
                policy_path = lane.fsrs6_adr_delta_policy
                if policy_path is None:
                    raise ValueError(
                        "--sched fsrs6_adr_delta requires an FSRS6 ADR Delta policy source."
                    )
                policy_paths.append(policy_path)
                desired_retentions.append(_required_desired_retention(lane))
            policies = [
                FSRS6ADRDeltaPolicy.from_json(policy_path)
                for policy_path in policy_paths
            ]
            policy = policies[0]
            for policy_path, candidate in zip(policy_paths, policies, strict=True):
                if not _same_adr_delta_policy_bounds(candidate, policy):
                    raise ValueError(
                        "Batched fsrs6_adr_delta sweep requires identical policy "
                        "feature versions, retention bounds, and FSRS bounds. "
                        f"Mismatch at {policy_path}."
                    )
            coefficients = torch.tensor(
                [candidate.coefficients for candidate in policies],
                device=device,
                dtype=torch.float32,
            )
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            desired_retention = torch.tensor(
                desired_retentions,
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS6ADRDeltaBatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                policy=policy,
                coefficients=coefficients,
                bounds=Bounds(),
                priority_mode=args.scheduler_priority,
                device=device,
                dtype=torch.float32,
            )
        elif name == "fsrs6_adp":
            policy_paths: list[Path] = []
            for lane in group_lanes:
                policy_path = lane.fsrs6_adp_policy
                if policy_path is None:
                    raise ValueError(
                        "--sched fsrs6_adp requires an FSRS6 ADP policy source."
                    )
                policy_paths.append(policy_path)
            policies = [
                FSRS6ADPPolicy.from_json(policy_path) for policy_path in policy_paths
            ]
            scheduler_weights = torch.tensor(
                [policy.weights for policy in policies],
                device=device,
                dtype=torch.float32,
            )
            desired_retention = torch.tensor(
                [policy.baseline_desired_retention for policy in policies],
                device=device,
                dtype=torch.float32,
            )
            ops = FSRS6BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=desired_retention,
                bounds=Bounds(),
                priority_mode=args.scheduler_priority,
                device=device,
                dtype=torch.float32,
            )
        elif name == "anki_sm2":
            scheduler = AnkiSM2Scheduler()
            ops = AnkiSM2BatchSchedulerOps(
                graduating_interval=scheduler.graduating_interval,
                easy_interval=scheduler.easy_interval,
                easy_bonus=scheduler.easy_bonus,
                hard_interval_factor=scheduler.hard_interval_factor,
                ease_start=scheduler.ease_start,
                ease_min=scheduler.ease_min,
                ease_max=scheduler.ease_max,
                device=device,
                dtype=torch.float32,
            )
        elif name == "memrise":
            ops = MemriseBatchSchedulerOps(
                MemriseScheduler(),
                device=device,
                dtype=torch.float32,
            )
        elif name == "fixed":
            interval = normalize_fixed_interval(sample.fixed_interval)
            ops = FixedBatchSchedulerOps(
                interval=interval,
                device=device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported scheduler '{name}' in batched run.")

        groups.append(_MixedSchedulerGroup(lane_indices=lane_indices, ops=ops))

    return _MixedBatchSchedulerOps(
        groups=groups,
        lane_count=len(lanes),
        device=device,
        dtype=torch.float32,
    )


def run_batch_core(
    *,
    args: argparse.Namespace,
    ctx: BatchedSweepContext,
    batch: list[int],
    device: torch.device | None,
    progress: bool,
    progress_queue,
    device_label: str,
) -> None:
    # Import lazily so this module stays self-contained under `simulator/`.
    import simulate as simulate_cli

    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        args
    )
    short_term_enabled = bool(short_term_source)
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

    schedulers = ctx.schedulers
    envs = ctx.envs

    if short_term_source == "sched":
        for raw in schedulers:
            name, _, _ = parse_scheduler_spec(raw)
            if name != "lstm":
                raise SystemExit(
                    "--short-term-source sched requires --sched lstm in batched mode."
                )

    base_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler_names = [parse_scheduler_spec(raw)[0] for raw in schedulers]

    for environment in envs:
        active_batch = list(batch)
        lstm_packed: PackedLSTMWeights | None = None
        lstm_paths: list[Path] | None = None
        fsrs_weights: torch.Tensor | None = None
        fsrs_default_weights: torch.Tensor | None = None
        fsrs3_weights: torch.Tensor | None = None
        fsrs3_default_weights: torch.Tensor | None = None

        needs_lstm_weights = environment == "lstm" or "lstm" in scheduler_names
        needs_fsrs_weights = (
            environment == "fsrs6"
            or "fsrs6" in scheduler_names
            or "fsrs6_adr_direct" in scheduler_names
            or "fsrs6_adr_delta" in scheduler_names
        )
        needs_fsrs_default = (
            environment == "fsrs6_default" or "fsrs6_default" in scheduler_names
        )
        needs_fsrs3_weights = "fsrs3" in scheduler_names
        needs_fsrs3_default = "fsrs3_default" in scheduler_names

        if needs_lstm_weights:
            lstm_paths, active_batch = resolve_lstm_paths(
                active_batch, ctx.benchmark_root, short_term=short_term_enabled
            )
            if not active_batch:
                logging.warning(
                    "Skipping environment '%s' for users %s: no LSTM weights found.",
                    environment,
                    format_id_list(batch),
                )
                continue

        if needs_fsrs_weights:
            fsrs_weights, fsrs_users = load_fsrs6_weights(
                repo_root=ctx.repo_root,
                user_ids=active_batch,
                benchmark_root=ctx.benchmark_root,
                benchmark_partition=args.benchmark_partition,
                overrides=ctx.overrides,
                short_term=short_term_enabled,
                device=base_device,
            )
            if not fsrs_users:
                logging.warning(
                    "Skipping environment '%s' for users %s: no FSRS-6 weights found.",
                    environment,
                    format_id_list(batch),
                )
                continue
            if len(fsrs_users) != len(active_batch):
                if lstm_paths is not None:
                    path_map = {
                        user_id: path for user_id, path in zip(active_batch, lstm_paths)
                    }
                    lstm_paths = [path_map[user_id] for user_id in fsrs_users]
                active_batch = fsrs_users

        if needs_fsrs3_weights:
            fsrs3_weights, fsrs3_users = load_fsrs3_weights(
                repo_root=ctx.repo_root,
                user_ids=active_batch,
                benchmark_root=ctx.benchmark_root,
                benchmark_partition=args.benchmark_partition,
                overrides=ctx.overrides,
                short_term=short_term_enabled,
                device=base_device,
            )
            if not fsrs3_users:
                logging.warning(
                    "Skipping environment '%s' for users %s: no FSRS-3 weights found.",
                    environment,
                    format_id_list(batch),
                )
                continue
            if len(fsrs3_users) != len(active_batch):
                idx_map = {user_id: idx for idx, user_id in enumerate(active_batch)}
                keep_idx = torch.tensor(
                    [idx_map[user_id] for user_id in fsrs3_users],
                    dtype=torch.int64,
                    device=base_device,
                )
                if lstm_paths is not None:
                    lstm_paths = [
                        lstm_paths[idx_map[user_id]] for user_id in fsrs3_users
                    ]
                if fsrs_weights is not None:
                    fsrs_weights = fsrs_weights.index_select(0, keep_idx)
                active_batch = fsrs3_users

        if needs_fsrs_default:
            fsrs_default_weights = build_default_fsrs6_weights(
                user_ids=active_batch,
                device=base_device,
            )
        if needs_fsrs3_default:
            fsrs3_default_weights = build_default_fsrs3_weights(
                user_ids=active_batch,
                device=base_device,
            )

        if lstm_paths is not None and lstm_packed is None and needs_lstm_weights:
            lstm_packed = PackedLSTMWeights.from_paths(
                lstm_paths,
                use_duration_feature=False,
                device=base_device,
                dtype=torch.float32,
            )

        lanes = _build_sweep_lanes(
            batch=active_batch,
            ctx=ctx,
            environment=environment,
        )
        if not lanes:
            continue

        scheduler_label = ",".join(scheduler_names)
        lane_chunks = _split_lanes(
            lanes,
            getattr(args, "max_lanes_per_batch", None),
        )
        for chunk_index, lane_chunk in enumerate(lane_chunks, start=1):
            env_ops = _build_env_ops_for_lanes(
                environment=environment,
                active_batch=active_batch,
                lanes=lane_chunk,
                fsrs_weights=fsrs_weights,
                fsrs_default_weights=fsrs_default_weights,
                lstm_packed=lstm_packed,
                device=base_device,
            )
            lane_user_ids = [lane.user_id for lane in lane_chunk]
            (
                learn_costs,
                review_costs,
                first_rating_prob,
                review_rating_prob,
                learning_rating_prob,
                relearning_rating_prob,
                state_rating_costs,
                review_markov_success_weights,
            ) = load_usage(lane_user_ids, args.button_usage)

            behavior, cost_model = build_behavior_cost(
                len(lane_user_ids),
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
                review_markov_success_weights=review_markov_success_weights.to(
                    env_ops.device
                ),
                short_term=short_term_enabled,
            )
            sched_ops = _build_mixed_scheduler_ops(
                args=args,
                active_batch=active_batch,
                lanes=lane_chunk,
                fsrs_weights=fsrs_weights,
                fsrs_default_weights=fsrs_default_weights,
                fsrs3_weights=fsrs3_weights,
                fsrs3_default_weights=fsrs3_default_weights,
                lstm_packed=lstm_packed,
                short_term_source=short_term_source,
                device=env_ops.device,
            )
            chunk_label = (
                f" chunk={chunk_index}/{len(lane_chunks)}"
                if len(lane_chunks) > 1
                else ""
            )
            simulate_and_log_lanes(
                write_log=simulate_cli._write_log,
                args=args,
                lanes=lane_chunk,
                env_ops=env_ops,
                sched_ops=sched_ops,
                behavior=behavior,
                cost_model=cost_model,
                progress=progress,
                progress_queue=progress_queue,
                device_label=device_label,
                run_label=(
                    f"{environment} u{active_batch[0]}-{active_batch[-1]} "
                    f"sched={scheduler_label} lanes={len(lane_chunk)}{chunk_label}"
                ),
                short_term_source=short_term_source,
                learning_steps=learning_steps,
                relearning_steps=relearning_steps,
                learning_steps_arg=learning_steps_arg,
                relearning_steps_arg=relearning_steps_arg,
                batch_log_root=ctx.batch_log_root,
            )

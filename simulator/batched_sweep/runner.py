from __future__ import annotations

import argparse
import logging
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
from simulator.schedulers.sa_fsrs6 import SAFSRS6BatchSchedulerOps
from simulator.sa_fsrs6_policy import SAFSRS6Policy
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
    sa_fsrs6_policy: Path | None = None


_DR_SCHEDULERS = {"fsrs6", "fsrs6_default", "fsrs3", "fsrs3_default", "lstm"}


def _format_float_token(value: float) -> str:
    token = format(value, ".12g")
    return token.replace("-", "neg_").replace("+", "").replace(".", "p")


def _build_dr_grid_lanes(
    *,
    batch: list[int],
    log_root: Path,
    environment: str,
    scheduler_name: str,
    scheduler_spec: str,
    dr_values: list[float],
    fixed_interval: float | None,
) -> list[BatchedSweepLogLane]:
    lanes: list[BatchedSweepLogLane] = []
    for desired_retention in dr_values:
        dr_root = (
            log_root
            / f"sched_{scheduler_name}"
            / f"dr_{_format_float_token(desired_retention)}"
        )
        lanes.extend(
            BatchedSweepLogLane(
                user_id=user_id,
                log_root=dr_root,
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
    lanes: list[BatchedSweepLogLane] = []
    for scheduler_spec in ctx.schedulers:
        name, fixed_interval, raw = parse_scheduler_spec(scheduler_spec)
        if name in _DR_SCHEDULERS:
            lanes.extend(
                _build_dr_grid_lanes(
                    batch=batch,
                    log_root=ctx.log_root,
                    environment=environment,
                    scheduler_name=name,
                    scheduler_spec=raw,
                    dr_values=ctx.dr_values,
                    fixed_interval=None,
                )
            )
            continue

        interval = normalize_fixed_interval(fixed_interval) if name == "fixed" else None
        policy = ctx.sa_fsrs6_policy if name == "sa_fsrs6" else None
        scheduler_root = ctx.log_root / f"sched_{name}"
        if name == "fixed" and interval is not None:
            scheduler_root = scheduler_root / f"ivl_{_format_float_token(interval)}"
        elif name == "sa_fsrs6" and policy is not None:
            scheduler_root = scheduler_root / f"policy_{policy.stem}"
        lanes.extend(
            BatchedSweepLogLane(
                user_id=user_id,
                log_root=scheduler_root,
                environment=environment,
                scheduler_name=name,
                scheduler_spec=raw,
                desired_retention=None,
                fixed_interval=interval,
                sa_fsrs6_policy=policy,
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
    if lane.scheduler_name in {"fsrs6", "fsrs6_default"}:
        return (lane.scheduler_name, lane.scheduler_spec)
    if lane.scheduler_name in {"fsrs3", "fsrs3_default", "lstm"}:
        return (
            lane.scheduler_name,
            lane.scheduler_spec,
            _required_desired_retention(lane),
        )
    if lane.scheduler_name == "fixed":
        return (lane.scheduler_name, lane.scheduler_spec, lane.fixed_interval)
    if lane.scheduler_name == "sa_fsrs6":
        return (lane.scheduler_name, lane.scheduler_spec, lane.sa_fsrs6_policy)
    return (lane.scheduler_name, lane.scheduler_spec)


def _group_lane_indices(lanes: list[BatchedSweepLogLane]) -> list[list[int]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for lane_index, lane in enumerate(lanes):
        grouped.setdefault(_mixed_scheduler_group_key(lane), []).append(lane_index)
    return list(grouped.values())


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
    policies: dict[Path, SAFSRS6Policy] = {}
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
            ops = FSRS3BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=_required_desired_retention(sample),
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
            ops = FSRS3BatchSchedulerOps(
                weights=scheduler_weights,
                desired_retention=_required_desired_retention(sample),
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
            ops = LSTMBatchSchedulerOps(
                scheduler_weights,
                desired_retention=_required_desired_retention(sample),
                min_interval=min_interval,
                interval_mode=interval_mode,
                device=device,
                dtype=torch.float32,
            )
        elif name == "sa_fsrs6":
            if fsrs_weights is None:
                raise ValueError("Expected FSRS-6 weights for sa_fsrs6 scheduler.")
            policy_path = sample.sa_fsrs6_policy
            if policy_path is None:
                raise ValueError("--sched sa_fsrs6 requires --sa-fsrs6-policy.")
            policy = policies.get(policy_path)
            if policy is None:
                policy = SAFSRS6Policy.from_json(policy_path)
                policies[policy_path] = policy
            scheduler_weights = _repeat_weights_for_lanes(
                weights=fsrs_weights.to(device),
                active_batch=active_batch,
                lanes=group_lanes,
            )
            ops = SAFSRS6BatchSchedulerOps(
                weights=scheduler_weights,
                policy=policy,
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
            or "sa_fsrs6" in scheduler_names
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

        env_ops = _build_env_ops_for_lanes(
            environment=environment,
            active_batch=active_batch,
            lanes=lanes,
            fsrs_weights=fsrs_weights,
            fsrs_default_weights=fsrs_default_weights,
            lstm_packed=lstm_packed,
            device=base_device,
        )
        lane_user_ids = [lane.user_id for lane in lanes]
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
            lanes=lanes,
            fsrs_weights=fsrs_weights,
            fsrs_default_weights=fsrs_default_weights,
            fsrs3_weights=fsrs3_weights,
            fsrs3_default_weights=fsrs3_default_weights,
            lstm_packed=lstm_packed,
            short_term_source=short_term_source,
            device=env_ops.device,
        )
        scheduler_label = ",".join(scheduler_names)
        simulate_and_log_lanes(
            write_log=simulate_cli._write_log,
            args=args,
            lanes=lanes,
            env_ops=env_ops,
            sched_ops=sched_ops,
            behavior=behavior,
            cost_model=cost_model,
            progress=progress,
            progress_queue=progress_queue,
            device_label=device_label,
            run_label=(
                f"{environment} u{active_batch[0]}-{active_batch[-1]} "
                f"sched={scheduler_label} lanes={len(lanes)}"
            ),
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
            learning_steps_arg=learning_steps_arg,
            relearning_steps_arg=relearning_steps_arg,
            batch_log_root=ctx.batch_log_root,
        )

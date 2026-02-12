from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

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
from simulator.short_term_config import resolve_short_term_config

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
from simulator.batched_sweep.logging import simulate_and_log
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
    dr_values = ctx.dr_values

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
        needs_fsrs_weights = environment == "fsrs6" or "fsrs6" in scheduler_names
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

        (
            learn_costs,
            review_costs,
            first_rating_prob,
            review_rating_prob,
            learning_rating_prob,
            relearning_rating_prob,
            state_rating_costs,
        ) = load_usage(active_batch, args.button_usage)

        if environment == "lstm":
            if lstm_paths is None:
                raise ValueError("Expected LSTM weights when environment is lstm.")
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
            if fsrs_weights is None:
                raise ValueError("Expected FSRS-6 weights when environment is fsrs6.")
            env_weights = fsrs_weights.to(base_device)
            env_ops = FSRS6BatchEnvOps(
                weights=env_weights,
                bounds=Bounds(),
                device=env_weights.device,
                dtype=torch.float32,
            )
        elif environment == "fsrs6_default":
            if fsrs_default_weights is None:
                raise ValueError("Expected default FSRS-6 weights for fsrs6_default.")
            env_weights = fsrs_default_weights.to(base_device)
            env_ops = FSRS6BatchEnvOps(
                weights=env_weights,
                bounds=Bounds(),
                device=env_weights.device,
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported environment '{environment}' in batched run.")

        if lstm_paths is not None and lstm_packed is None and "lstm" in scheduler_names:
            lstm_packed = PackedLSTMWeights.from_paths(
                lstm_paths,
                use_duration_feature=False,
                device=env_ops.device,
                dtype=torch.float32,
            )

        behavior, cost_model = build_behavior_cost(
            len(active_batch),
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
            if name not in {
                "fsrs6",
                "fsrs6_default",
                "fsrs3_default",
                "fsrs3",
                "anki_sm2",
                "memrise",
                "fixed",
                "lstm",
            }:
                raise ValueError(f"Unsupported scheduler '{name}' in batched run.")
            label_prefix = f"{environment} u{active_batch[0]}-{active_batch[-1]} {name}"

            if name == "fsrs6":
                if fsrs_weights is None:
                    raise ValueError("Expected FSRS-6 weights for fsrs6 scheduler.")
                weights = fsrs_weights.to(env_ops.device)
                for dr in dr_values:
                    scheduler_ops = FSRS6BatchSchedulerOps(
                        weights=weights,
                        desired_retention=dr,
                        bounds=Bounds(),
                        priority_mode=args.scheduler_priority,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    simulate_and_log(
                        write_log=simulate_cli._write_log,
                        args=args,
                        batch=active_batch,
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
                        log_root=ctx.log_root,
                        batch_log_root=ctx.batch_log_root,
                    )
                continue

            if name == "fsrs6_default":
                if fsrs_default_weights is None:
                    raise ValueError(
                        "Expected default FSRS-6 weights for fsrs6_default scheduler."
                    )
                weights = fsrs_default_weights.to(env_ops.device)
                for dr in dr_values:
                    scheduler_ops = FSRS6BatchSchedulerOps(
                        weights=weights,
                        desired_retention=dr,
                        bounds=Bounds(),
                        priority_mode=args.scheduler_priority,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    simulate_and_log(
                        write_log=simulate_cli._write_log,
                        args=args,
                        batch=active_batch,
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
                        log_root=ctx.log_root,
                        batch_log_root=ctx.batch_log_root,
                    )
                continue

            if name == "fsrs3":
                if fsrs3_weights is None:
                    raise ValueError("Expected FSRS-3 weights for fsrs3 scheduler.")
                weights = fsrs3_weights.to(env_ops.device)
                for dr in dr_values:
                    scheduler_ops = FSRS3BatchSchedulerOps(
                        weights=weights,
                        desired_retention=dr,
                        bounds=Bounds(),
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    simulate_and_log(
                        write_log=simulate_cli._write_log,
                        args=args,
                        batch=active_batch,
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
                        log_root=ctx.log_root,
                        batch_log_root=ctx.batch_log_root,
                    )
                continue

            if name == "fsrs3_default":
                if fsrs3_default_weights is None:
                    raise ValueError(
                        "Expected default FSRS-3 weights for fsrs3_default scheduler."
                    )
                weights = fsrs3_default_weights.to(env_ops.device)
                for dr in dr_values:
                    scheduler_ops = FSRS3BatchSchedulerOps(
                        weights=weights,
                        desired_retention=dr,
                        bounds=Bounds(),
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    simulate_and_log(
                        write_log=simulate_cli._write_log,
                        args=args,
                        batch=active_batch,
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
                        log_root=ctx.log_root,
                        batch_log_root=ctx.batch_log_root,
                    )
                continue

            if name == "lstm":
                if lstm_packed is None:
                    raise ValueError("Expected LSTM weights for lstm scheduler.")
                for dr in dr_values:
                    interval_mode = (
                        "float" if short_term_source == "sched" else "integer"
                    )
                    min_interval = 0.0 if short_term_source == "sched" else 1.0
                    sched_ops = LSTMBatchSchedulerOps(
                        lstm_packed,
                        desired_retention=dr,
                        min_interval=min_interval,
                        interval_mode=interval_mode,
                        device=env_ops.device,
                        dtype=torch.float32,
                    )
                    simulate_and_log(
                        write_log=simulate_cli._write_log,
                        args=args,
                        batch=active_batch,
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
                        log_root=ctx.log_root,
                        batch_log_root=ctx.batch_log_root,
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
                simulate_and_log(
                    write_log=simulate_cli._write_log,
                    args=args,
                    batch=active_batch,
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
                    log_root=ctx.log_root,
                    batch_log_root=ctx.batch_log_root,
                )
                continue

            if name == "memrise":
                scheduler = MemriseScheduler()
                sched_ops = MemriseBatchSchedulerOps(
                    scheduler,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                simulate_and_log(
                    write_log=simulate_cli._write_log,
                    args=args,
                    batch=active_batch,
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
                    log_root=ctx.log_root,
                    batch_log_root=ctx.batch_log_root,
                )
                continue

            if name == "fixed":
                interval = normalize_fixed_interval(fixed_interval)
                sched_ops = FixedBatchSchedulerOps(
                    interval=interval,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                simulate_and_log(
                    write_log=simulate_cli._write_log,
                    args=args,
                    batch=active_batch,
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
                    log_root=ctx.log_root,
                    batch_log_root=ctx.batch_log_root,
                )
                continue

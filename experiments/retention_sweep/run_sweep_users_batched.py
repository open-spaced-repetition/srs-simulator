from __future__ import annotations

import argparse
import math
import sys
from concurrent import futures
from multiprocessing import get_context
import queue
from pathlib import Path
import logging
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulate as simulate_cli
from simulator.benchmark_loader import parse_result_overrides
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.scheduler_spec import parse_scheduler_spec
from simulator.scheduler_spec import normalize_fixed_interval
from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps, AnkiSM2Scheduler
from simulator.schedulers.fixed import FixedBatchSchedulerOps
from simulator.schedulers.fsrs import FSRS3BatchSchedulerOps, FSRS6BatchSchedulerOps
from simulator.schedulers.lstm import LSTMBatchSchedulerOps
from simulator.schedulers.memrise import MemriseScheduler
from simulator.schedulers.memrise import MemriseBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.batched_sweep.utils import (
    chunked as _chunked,
    dr_values as _dr_values,
    format_id_list as _format_id_list,
    parse_cuda_devices as _parse_cuda_devices,
)
from simulator.batched_sweep.behavior_cost import (
    build_behavior_cost as _build_behavior_cost,
    load_usage as _load_usage,
)
from simulator.batched_sweep.logging import simulate_and_log as _simulate_and_log
from simulator.batched_sweep.weights import (
    load_fsrs3_weights as _load_fsrs3_weights,
    load_fsrs6_weights as _load_fsrs6_weights,
    resolve_lstm_paths as _resolve_lstm_paths,
)

from experiments.retention_sweep.cli_utils import (
    add_benchmark_args,
    add_button_usage_arg,
    add_common_sim_args,
    add_env_sched_args,
    add_fuzz_arg,
    add_log_args,
    add_retention_range_args,
    add_short_term_args,
    add_torch_device_arg,
    add_user_range_args,
    parse_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-user retention sweeps with batched vectorized simulation.",
        allow_abbrev=False,
    )
    add_user_range_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of users to simulate in parallel per batch.",
    )
    add_env_sched_args(
        parser,
        env_default="lstm",
        sched_default="fsrs6,anki_sm2,memrise",
        env_help="Comma-separated environments to sweep (lstm, fsrs6).",
        sched_help=(
            "Comma-separated schedulers to sweep "
            "(fsrs6, fsrs3, lstm, anki_sm2, memrise, fixed)."
        ),
    )
    add_retention_range_args(parser)
    add_common_sim_args(
        parser,
    )
    add_button_usage_arg(parser, default_path=DEFAULT_BUTTON_USAGE_PATH)
    add_benchmark_args(parser)
    add_log_args(
        parser, log_dir_default=None, include_no_log=True, include_no_progress=True
    )
    add_fuzz_arg(parser)
    add_short_term_args(
        parser,
        choices=["steps", "sched"],
        source_help=(
            "Short-term scheduling source: steps (Anki-style learning steps) "
            "or sched (LSTM-only short-term intervals)."
        ),
        learning_help="Comma-separated learning steps (minutes) for short-term steps mode.",
        relearning_help="Comma-separated relearning steps (minutes) for short-term steps mode.",
        threshold_help="Short-term cutoff in days (used by sched mode).",
    )
    add_torch_device_arg(parser)
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute batches across "
            "(e.g. 0,1). Each batch is assigned a device round-robin."
        ),
    )
    return parser.parse_args()


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
    schedulers = parse_csv(args.sched)
    envs = parse_csv(args.env)
    if short_term_source == "sched":
        for raw in schedulers:
            name, _, _ = parse_scheduler_spec(raw)
            if name != "lstm":
                raise SystemExit(
                    "--short-term-source sched requires --sched lstm in batched mode."
                )
    base_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler_names = []
    for raw in schedulers:
        name, _, _ = parse_scheduler_spec(raw)
        scheduler_names.append(name)

    for environment in envs:
        active_batch = list(batch)
        lstm_packed: PackedLSTMWeights | None = None
        lstm_paths: list[Path] | None = None
        fsrs_weights: torch.Tensor | None = None
        fsrs3_weights: torch.Tensor | None = None
        needs_lstm_weights = environment == "lstm" or "lstm" in scheduler_names
        needs_fsrs_weights = environment == "fsrs6" or "fsrs6" in scheduler_names
        needs_fsrs3_weights = "fsrs3" in scheduler_names
        if needs_lstm_weights:
            lstm_paths, active_batch = _resolve_lstm_paths(
                active_batch, benchmark_root, short_term=short_term_enabled
            )
            if not active_batch:
                logging.warning(
                    "Skipping environment '%s' for users %s: no LSTM weights found.",
                    environment,
                    _format_id_list(batch),
                )
                continue
        if needs_fsrs_weights:
            fsrs_weights, fsrs_users = _load_fsrs6_weights(
                repo_root=REPO_ROOT,
                user_ids=active_batch,
                benchmark_root=benchmark_root,
                benchmark_partition=args.benchmark_partition,
                overrides=overrides,
                short_term=short_term_enabled,
                device=base_device,
            )
            if not fsrs_users:
                logging.warning(
                    "Skipping environment '%s' for users %s: no FSRS-6 weights found.",
                    environment,
                    _format_id_list(batch),
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
            fsrs3_weights, fsrs3_users = _load_fsrs3_weights(
                repo_root=REPO_ROOT,
                user_ids=active_batch,
                benchmark_root=benchmark_root,
                benchmark_partition=args.benchmark_partition,
                overrides=overrides,
                short_term=short_term_enabled,
                device=base_device,
            )
            if not fsrs3_users:
                logging.warning(
                    "Skipping environment '%s' for users %s: no FSRS-3 weights found.",
                    environment,
                    _format_id_list(batch),
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
        (
            learn_costs,
            review_costs,
            first_rating_prob,
            review_rating_prob,
            learning_rating_prob,
            relearning_rating_prob,
            state_rating_costs,
        ) = _load_usage(active_batch, args.button_usage)
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
        else:
            raise ValueError(f"Unsupported environment '{environment}' in batched run.")

        if lstm_paths is not None and lstm_packed is None and "lstm" in scheduler_names:
            lstm_packed = PackedLSTMWeights.from_paths(
                lstm_paths,
                use_duration_feature=False,
                device=env_ops.device,
                dtype=torch.float32,
            )

        behavior, cost_model = _build_behavior_cost(
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
            if name not in {"fsrs6", "fsrs3", "anki_sm2", "memrise", "fixed", "lstm"}:
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
                    _simulate_and_log(
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
                        log_root=log_root,
                        batch_log_root=batch_log_root,
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
                    _simulate_and_log(
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
                        log_root=log_root,
                        batch_log_root=batch_log_root,
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
                    _simulate_and_log(
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
        if name in {"fsrs6", "fsrs3", "lstm"}:
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

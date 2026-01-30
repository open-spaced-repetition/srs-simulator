from __future__ import annotations

import argparse
import math
import sys
from concurrent import futures
from multiprocessing import get_context
import queue
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import simulate as simulate_cli
from simulator.benchmark_loader import load_benchmark_weights, parse_result_overrides
from simulator.button_usage import (
    DEFAULT_BUTTON_USAGE_PATH,
    load_button_usage_config,
    normalize_button_usage,
)
from simulator.math.fsrs import Bounds
from simulator.models.lstm import _resolve_benchmark_weights
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.scheduler_spec import parse_scheduler_spec
from simulator.schedulers.anki_sm2 import AnkiSM2Scheduler
from simulator.schedulers.memrise import MemriseScheduler
from simulator.vectorized.multiuser import (
    AnkiSM2BatchSchedulerOps,
    FSRS6BatchSchedulerOps,
    MemriseBatchSchedulerOps,
    MultiUserBehavior,
    MultiUserCost,
    simulate_multiuser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-user retention sweeps with batched vectorized simulation."
    )
    parser.add_argument("--start-user", type=int, default=1, help="First user id.")
    parser.add_argument("--end-user", type=int, default=1000, help="Last user id.")
    parser.add_argument(
        "--step-user",
        type=int,
        default=1,
        help="Step size for user ids.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of users to simulate in parallel per batch.",
    )
    parser.add_argument(
        "--environments",
        default="lstm",
        help="Comma-separated environments to sweep (only lstm supported).",
    )
    parser.add_argument(
        "--schedulers",
        default="fsrs6,anki_sm2,memrise",
        help="Comma-separated schedulers to sweep.",
    )
    parser.add_argument(
        "--start-retention",
        type=float,
        default=0.70,
        help="Start retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--end-retention",
        type=float,
        default=0.99,
        help="End retention (0-1, rounded to 2 decimals).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Retention step (0-1, rounded to 2 decimals).",
    )
    parser.add_argument("--days", type=int, default=1825)
    parser.add_argument("--deck", type=int, default=10000)
    parser.add_argument("--learn-limit", type=int, default=10)
    parser.add_argument("--review-limit", type=int, default=9999)
    parser.add_argument("--cost-limit-minutes", type=float, default=720.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--priority",
        choices=["review-first", "new-first"],
        default="review-first",
    )
    parser.add_argument(
        "--scheduler-priority",
        default="low_retrievability",
        help="FSRS6 priority hint (ignored by other schedulers).",
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL for per-user costs/probabilities.",
    )
    parser.add_argument(
        "--benchmark-result",
        default=None,
        help="Override benchmark result files (key=value, comma-separated).",
    )
    parser.add_argument(
        "--benchmark-partition",
        default="0",
        help="Benchmark parameter partition key.",
    )
    parser.add_argument(
        "--srs-benchmark-root",
        type=Path,
        default=None,
        help="Path to the srs-benchmark repo (used for weights).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to store logs (defaults to logs/retention_sweep).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing logs to disk.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Torch device for vectorized engine (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help=(
            "Comma-separated CUDA device indices to distribute batches across "
            "(e.g. 0,1). Each batch is assigned a device round-robin."
        ),
    )
    return parser.parse_args()


def _parse_cuda_devices(raw: str | None) -> list[str]:
    if not raw:
        return []
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    for device in devices:
        if not device.isdigit():
            raise ValueError(
                f"Invalid --cuda-devices entry '{device}'. Expected numeric indices."
            )
    return devices


def _chunked(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


def _dr_values(start: float, end: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("--step must be non-zero.")
    start = round(start, 2)
    end = round(end, 2)
    values = []
    value = start
    epsilon = abs(step) * 1e-6
    if step > 0 and start > end:
        raise ValueError(
            "--start-retention must be <= --end-retention for positive step."
        )
    if step < 0 and start < end:
        raise ValueError(
            "--start-retention must be >= --end-retention for negative step."
        )
    while (value <= end + epsilon) if step > 0 else (value >= end - epsilon):
        values.append(round(value, 2))
        next_value = round(value + step, 2)
        if next_value == value:
            raise ValueError("Retention step is too small after rounding; use >= 0.01.")
        value = next_value
    return values


def _load_usage(
    user_ids: list[int], button_usage: Path | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    learn_costs = []
    review_costs = []
    first_rating_prob = []
    review_rating_prob = []
    for user_id in user_ids:
        config = (
            load_button_usage_config(button_usage, user_id)
            if button_usage is not None
            else None
        )
        usage = normalize_button_usage(config)
        learn_costs.append(usage["learn_costs"])
        review_costs.append(usage["review_costs"])
        first_rating_prob.append(usage["first_rating_prob"])
        review_rating_prob.append(usage["review_rating_prob"])
    return (
        torch.tensor(learn_costs, dtype=torch.float32),
        torch.tensor(review_costs, dtype=torch.float32),
        torch.tensor(first_rating_prob, dtype=torch.float32),
        torch.tensor(review_rating_prob, dtype=torch.float32),
    )


def _resolve_lstm_paths(user_ids: list[int], benchmark_root: Path) -> list[Path]:
    paths = []
    for user_id in user_ids:
        path = _resolve_benchmark_weights(user_id, benchmark_root)
        if path is None:
            raise FileNotFoundError(f"LSTM weights not found for user {user_id}.")
        paths.append(path)
    return paths


def _build_behavior_cost(
    user_count: int,
    *,
    deck_size: int,
    learn_limit: int,
    review_limit: int | None,
    cost_limit_minutes: float | None,
    learn_costs: torch.Tensor,
    review_costs: torch.Tensor,
    first_rating_prob: torch.Tensor,
    review_rating_prob: torch.Tensor,
) -> tuple[MultiUserBehavior, MultiUserCost]:
    max_reviews = review_limit if review_limit is not None else deck_size
    max_cost = cost_limit_minutes * 60.0 if cost_limit_minutes is not None else math.inf
    behavior = MultiUserBehavior(
        attendance_prob=torch.full((user_count,), 1.0),
        lazy_good_bias=torch.zeros(user_count),
        max_new_per_day=torch.full((user_count,), learn_limit, dtype=torch.int64),
        max_reviews_per_day=torch.full((user_count,), max_reviews, dtype=torch.int64),
        max_cost_per_day=torch.full((user_count,), max_cost),
        success_weights=review_rating_prob,
        first_rating_prob=first_rating_prob,
    )
    cost = MultiUserCost(
        base=torch.zeros(user_count),
        penalty=torch.zeros(user_count),
        learn_costs=learn_costs,
        review_costs=review_costs,
    )
    return behavior, cost


def _progress_callback_from_queue(
    progress_queue,
    *,
    multiplier: int,
    device_label: str,
    run_label: str,
    total_days: int,
) -> callable | None:
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


def _run_batch_core(
    *,
    args: argparse.Namespace,
    batch: list[int],
    benchmark_root: Path,
    overrides: dict[str, str],
    log_root: Path,
    dr_values: list[float],
    device: torch.device | None,
    progress: bool,
    progress_queue,
    device_label: str,
) -> None:
    lstm_paths = _resolve_lstm_paths(batch, benchmark_root)
    packed = PackedLSTMWeights.from_paths(
        lstm_paths,
        use_duration_feature=False,
        device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float32,
    )
    env_ops = LSTMBatchedEnvOps(
        packed,
        device=packed.process_0_weight.device,
        dtype=torch.float32,
    )
    learn_costs, review_costs, first_rating_prob, review_rating_prob = _load_usage(
        batch, args.button_usage
    )
    behavior, cost_model = _build_behavior_cost(
        len(batch),
        deck_size=args.deck,
        learn_limit=args.learn_limit,
        review_limit=args.review_limit,
        cost_limit_minutes=args.cost_limit_minutes,
        learn_costs=learn_costs.to(env_ops.device),
        review_costs=review_costs.to(env_ops.device),
        first_rating_prob=first_rating_prob.to(env_ops.device),
        review_rating_prob=review_rating_prob.to(env_ops.device),
    )

    schedulers = [item.strip() for item in args.schedulers.split(",") if item.strip()]
    for scheduler_spec in schedulers:
        name, fixed_interval, raw = parse_scheduler_spec(scheduler_spec)
        if name not in {"fsrs6", "anki_sm2", "memrise"}:
            raise ValueError(f"Unsupported scheduler '{name}' in batched run.")
        label_prefix = f"u{batch[0]}-{batch[-1]} {name}"

        if name == "fsrs6":
            weights = torch.stack(
                [
                    torch.tensor(
                        load_benchmark_weights(
                            repo_root=REPO_ROOT,
                            benchmark_root=benchmark_root,
                            environment="fsrs6",
                            user_id=user_id,
                            partition_key=args.benchmark_partition,
                            overrides=overrides,
                        ),
                        dtype=torch.float32,
                    )
                    for user_id in batch
                ],
                dim=0,
            ).to(env_ops.device)
            for dr in dr_values:
                progress_callback = _progress_callback_from_queue(
                    progress_queue,
                    multiplier=len(batch),
                    device_label=device_label,
                    run_label=f"{label_prefix} dr={dr:.2f}",
                    total_days=args.days,
                )
                scheduler_ops = FSRS6BatchSchedulerOps(
                    weights=weights,
                    desired_retention=dr,
                    bounds=Bounds(),
                    priority_mode=args.scheduler_priority,
                    device=env_ops.device,
                    dtype=torch.float32,
                )
                stats_list = simulate_multiuser(
                    days=args.days,
                    deck_size=args.deck,
                    env_ops=env_ops,
                    sched_ops=scheduler_ops,
                    behavior=behavior,
                    cost_model=cost_model,
                    seed=args.seed,
                    device=env_ops.device,
                    dtype=torch.float32,
                    priority_mode=args.priority,
                    progress=progress,
                    progress_label=f"{label_prefix} dr={dr:.2f}",
                    progress_callback=progress_callback,
                )
                if not args.no_log:
                    for user_id, stats in zip(batch, stats_list):
                        user_log_dir = log_root / f"user_{user_id}"
                        user_log_dir.mkdir(parents=True, exist_ok=True)
                        log_args = argparse.Namespace(
                            engine="vectorized",
                            days=args.days,
                            deck=args.deck,
                            learn_limit=args.learn_limit,
                            review_limit=args.review_limit,
                            cost_limit_minutes=args.cost_limit_minutes,
                            priority=args.priority,
                            environment="lstm",
                            scheduler=name,
                            scheduler_spec=raw,
                            user_id=user_id,
                            button_usage=str(args.button_usage)
                            if args.button_usage is not None
                            else None,
                            desired_retention=dr,
                            scheduler_priority=args.scheduler_priority,
                            sspmmc_policy=None,
                            fixed_interval=fixed_interval,
                            seed=args.seed,
                            log_dir=user_log_dir,
                            log_reviews=False,
                        )
                        simulate_cli._write_log(log_args, stats)
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
            progress_callback = _progress_callback_from_queue(
                progress_queue,
                multiplier=len(batch),
                device_label=device_label,
                run_label=label_prefix,
                total_days=args.days,
            )
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
                priority_mode=args.priority,
                progress=progress,
                progress_label=label_prefix,
                progress_callback=progress_callback,
            )
            if not args.no_log:
                for user_id, stats in zip(batch, stats_list):
                    user_log_dir = log_root / f"user_{user_id}"
                    user_log_dir.mkdir(parents=True, exist_ok=True)
                    log_args = argparse.Namespace(
                        engine="vectorized",
                        days=args.days,
                        deck=args.deck,
                        learn_limit=args.learn_limit,
                        review_limit=args.review_limit,
                        cost_limit_minutes=args.cost_limit_minutes,
                        priority=args.priority,
                        environment="lstm",
                        scheduler=name,
                        scheduler_spec=raw,
                        user_id=user_id,
                        button_usage=str(args.button_usage)
                        if args.button_usage is not None
                        else None,
                        desired_retention=None,
                        scheduler_priority=args.scheduler_priority,
                        sspmmc_policy=None,
                        fixed_interval=fixed_interval,
                        seed=args.seed,
                        log_dir=user_log_dir,
                        log_reviews=False,
                    )
                    simulate_cli._write_log(log_args, stats)
            continue

        if name == "memrise":
            scheduler = MemriseScheduler()
            sched_ops = MemriseBatchSchedulerOps(
                scheduler,
                device=env_ops.device,
                dtype=torch.float32,
            )
            progress_callback = _progress_callback_from_queue(
                progress_queue,
                multiplier=len(batch),
                device_label=device_label,
                run_label=label_prefix,
                total_days=args.days,
            )
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
                priority_mode=args.priority,
                progress=progress,
                progress_label=label_prefix,
                progress_callback=progress_callback,
            )
            if not args.no_log:
                for user_id, stats in zip(batch, stats_list):
                    user_log_dir = log_root / f"user_{user_id}"
                    user_log_dir.mkdir(parents=True, exist_ok=True)
                    log_args = argparse.Namespace(
                        engine="vectorized",
                        days=args.days,
                        deck=args.deck,
                        learn_limit=args.learn_limit,
                        review_limit=args.review_limit,
                        cost_limit_minutes=args.cost_limit_minutes,
                        priority=args.priority,
                        environment="lstm",
                        scheduler=name,
                        scheduler_spec=raw,
                        user_id=user_id,
                        button_usage=str(args.button_usage)
                        if args.button_usage is not None
                        else None,
                        desired_retention=None,
                        scheduler_priority=args.scheduler_priority,
                        sspmmc_policy=None,
                        fixed_interval=fixed_interval,
                        seed=args.seed,
                        log_dir=user_log_dir,
                        log_reviews=False,
                    )
                    simulate_cli._write_log(log_args, stats)
            continue


def _run_batch_worker(
    *,
    args: argparse.Namespace,
    batch: list[int],
    benchmark_root: Path,
    overrides: dict[str, str],
    log_root: Path,
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
    envs = [item.strip() for item in args.environments.split(",") if item.strip()]
    if envs != ["lstm"]:
        raise ValueError("Batched retention sweep currently supports only lstm env.")
    schedulers = [item.strip() for item in args.schedulers.split(",") if item.strip()]
    if not schedulers:
        raise ValueError("No schedulers specified.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.torch_device and args.cuda_devices:
        raise ValueError("--torch-device cannot be combined with --cuda-devices.")

    user_ids = list(range(args.start_user, args.end_user + 1, args.step_user))
    if not user_ids:
        raise ValueError("Empty user range.")

    benchmark_root = args.srs_benchmark_root or (REPO_ROOT.parent / "srs-benchmark")
    benchmark_root = benchmark_root.resolve()
    overrides = parse_result_overrides(args.benchmark_result)

    log_root = args.log_dir or (REPO_ROOT / "logs" / "retention_sweep")
    log_root.mkdir(parents=True, exist_ok=True)

    dr_values = _dr_values(args.start_retention, args.end_retention, args.step)
    devices = _parse_cuda_devices(args.cuda_devices)
    if devices and not torch.cuda.is_available():
        raise ValueError("--cuda-devices was provided but CUDA is not available.")
    device = torch.device(args.torch_device) if args.torch_device else None

    batches = list(_chunked(user_ids, args.batch_size))
    runs_per_batch = 0
    for scheduler in schedulers:
        if scheduler == "fsrs6":
            runs_per_batch += len(dr_values)
        else:
            runs_per_batch += 1
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

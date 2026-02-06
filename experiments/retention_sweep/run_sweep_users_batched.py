from __future__ import annotations

import argparse
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

from simulator.benchmark_loader import parse_result_overrides
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.scheduler_spec import parse_scheduler_spec
from simulator.batched_sweep.utils import (
    chunked as _chunked,
    dr_values as _dr_values,
    format_id_list as _format_id_list,
    parse_cuda_devices as _parse_cuda_devices,
)
from simulator.batched_sweep.runner import BatchedSweepContext, run_batch_core

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
    ctx = BatchedSweepContext(
        repo_root=REPO_ROOT,
        benchmark_root=benchmark_root,
        overrides=overrides,
        log_root=log_root,
        batch_log_root=batch_log_root,
        envs=parse_csv(args.env),
        schedulers=parse_csv(args.sched),
        dr_values=dr_values,
    )
    run_batch_core(
        args=args,
        ctx=ctx,
        batch=batch,
        device=device,
        progress=progress,
        progress_queue=progress_queue,
        device_label=device_label,
    )


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

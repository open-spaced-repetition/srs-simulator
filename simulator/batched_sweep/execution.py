from __future__ import annotations

import argparse
from concurrent import futures
from multiprocessing import get_context
import queue as queue_mod

import torch
from tqdm import tqdm

from simulator.batched_sweep.runner import BatchedSweepContext, run_batch_core


class LocalProgressQueue:
    def __init__(self, overall: tqdm) -> None:
        self._overall = overall

    def put(self, message) -> None:
        if not isinstance(message, tuple):
            return
        if message[0] != "overall":
            return
        self._overall.update(message[1])


def _drain_progress_queue(
    *,
    overall: tqdm,
    progress_queue,
    gpu_bars: dict[str, tqdm],
) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            message = progress_queue.get_nowait()
        except queue_mod.Empty:
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


def _run_batch_worker(
    *,
    args: argparse.Namespace,
    ctx: BatchedSweepContext,
    batch: list[int],
    device_str: str,
    progress_queue,
) -> None:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run_batch_core(
        args=args,
        ctx=ctx,
        batch=batch,
        device=device,
        progress=False,
        progress_queue=progress_queue,
        device_label=device_str,
    )


def run_batches(
    *,
    args: argparse.Namespace,
    ctx: BatchedSweepContext,
    batches: list[list[int]],
    devices: list[str],
    device: torch.device | None,
    overall: tqdm | None,
) -> None:
    # Multi-GPU: one batch per process, one process per device, progress is queued back
    # to the parent for both Overall and per-GPU status bars.
    if devices and len(devices) > 1:
        mp_ctx = get_context("spawn")
        manager = mp_ctx.Manager() if overall is not None else None
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
        try:
            pending: set[futures.Future] = set()
            with futures.ProcessPoolExecutor(
                max_workers=len(devices),
                mp_context=mp_ctx,
            ) as executor:
                for batch_idx, batch in enumerate(batches):
                    device_str = f"cuda:{devices[batch_idx % len(devices)]}"
                    pending.add(
                        executor.submit(
                            _run_batch_worker,
                            args=args,
                            ctx=ctx,
                            batch=batch,
                            device_str=device_str,
                            progress_queue=progress_queue,
                        )
                    )
                while pending:
                    done, pending = futures.wait(
                        pending, timeout=0.1, return_when=futures.FIRST_COMPLETED
                    )
                    if overall is not None:
                        _drain_progress_queue(
                            overall=overall,
                            progress_queue=progress_queue,
                            gpu_bars=gpu_bars,
                        )
                    for task in done:
                        task.result()
                if overall is not None:
                    _drain_progress_queue(
                        overall=overall,
                        progress_queue=progress_queue,
                        gpu_bars=gpu_bars,
                    )
        finally:
            for bar in gpu_bars.values():
                bar.close()
            if manager is not None:
                manager.shutdown()
        return

    # Single device: run in-process with an optional local queue for Overall updates.
    batch_device = torch.device(f"cuda:{devices[0]}") if devices else device
    for batch in batches:
        progress_queue = LocalProgressQueue(overall) if overall is not None else None
        run_batch_core(
            args=args,
            ctx=ctx,
            batch=batch,
            device=batch_device,
            progress=overall is not None,
            progress_queue=progress_queue,
            device_label=str(batch_device or "device"),
        )

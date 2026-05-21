from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path
from typing import Any

from simulator.experiment_infra.gpu_monitor import (
    GpuMonitor,
    GpuMonitorSummary,
    disabled_monitor_summary,
)
from experiments.single_card_tradeoff.oracles.dp_cache import (
    oracle_dp_cache_performance_payload,
)


def add_run_monitoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu-monitor-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable GPU memory monitoring. Defaults to enabled when "
            "--torch-device resolves to cuda."
        ),
    )
    parser.add_argument(
        "--gpu-monitor-interval-seconds",
        type=float,
        default=2.0,
        help="GPU monitor sampling interval in seconds.",
    )


def register_run_monitor(
    args: argparse.Namespace,
    *,
    device: Any,
    output_dir: Path,
    stage_name: str,
) -> None:
    interval_seconds = float(args.gpu_monitor_interval_seconds)
    if interval_seconds <= 0.0:
        raise SystemExit("--gpu-monitor-interval-seconds must be > 0.")

    enabled = args.gpu_monitor_enabled
    device_name = str(device)
    if enabled is None:
        enabled = device_name.startswith("cuda")

    monitor = (
        GpuMonitor(
            output_dir=output_dir / "gpu_monitor",
            interval_seconds=interval_seconds,
        )
        if enabled
        else None
    )
    if monitor is not None:
        monitor.start()

    start = time.perf_counter()
    stopped = False

    def stop() -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        summary = (
            monitor.stop()
            if monitor is not None
            else disabled_monitor_summary(output_dir / "gpu_monitor")
        )
        _write_performance_summary(
            output_dir / "performance_summary.json",
            stage_name=stage_name,
            device=device_name,
            elapsed_runtime_s=time.perf_counter() - start,
            gpu_monitor_summary=summary,
        )

    atexit.register(stop)


def _write_performance_summary(
    path: Path,
    *,
    stage_name: str,
    device: str,
    elapsed_runtime_s: float,
    gpu_monitor_summary: GpuMonitorSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_name": stage_name,
        "command": sys.argv,
        "device": device,
        "elapsed_runtime_s": elapsed_runtime_s,
        "gpu_monitor_enabled": gpu_monitor_summary.enabled,
        "gpu_monitor_summary_path": str(gpu_monitor_summary.summary_path)
        if gpu_monitor_summary.enabled
        else None,
        "gpu_monitor_jsonl_path": str(gpu_monitor_summary.jsonl_path)
        if gpu_monitor_summary.enabled
        else None,
        "gpu_monitor_sample_count": gpu_monitor_summary.sample_count,
        "gpu_monitor_shared_memory_peak_single_adapter_bytes": (
            gpu_monitor_summary.shared_memory_peak_single_adapter_bytes
        ),
        "gpu_monitor_shared_memory_peak_summed_bytes": (
            gpu_monitor_summary.shared_memory_peak_summed_bytes
        ),
        "gpu_monitor_shared_memory_spill_threshold_bytes": (
            gpu_monitor_summary.shared_memory_spill_threshold_bytes
        ),
        "gpu_monitor_shared_memory_spill_detected": (
            gpu_monitor_summary.shared_memory_spill_detected
        ),
        "gpu_monitor_nvidia_smi_peak_memory_used_mib": (
            gpu_monitor_summary.nvidia_smi_peak_memory_used_mib
        ),
        "gpu_monitor_nvidia_smi_peak_utilization_percent": (
            gpu_monitor_summary.nvidia_smi_peak_utilization_percent
        ),
        **oracle_dp_cache_performance_payload(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

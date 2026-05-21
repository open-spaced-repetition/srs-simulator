from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill import (
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_HIDDEN_SIZE,
    DEFAULT_DISTILL_NETWORK_DEPTH,
    DEFAULT_DISTILL_OBS_MODE,
    DEFAULT_DISTILL_SUPERVISION,
    DEFAULT_EVAL_PARTICLES,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
)
from experiments.single_card_tradeoff.core.defaults import DEFAULT_TARGET_RETENTIONS
from experiments.single_card_tradeoff.cli.uvfa_ppo import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_TRAIN_ENVS,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.experiment_infra.gpu_monitor import (
    GpuMonitor,
    GpuMonitorSummary,
    disabled_monitor_summary,
)
from simulator.scheduler_spec import format_float


DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/oracle_stationary_finite_cpu_gpu_benchmark"
)
RUN_FIELDS = [
    "device",
    "device_label",
    "repeat",
    "seed",
    "wall_runtime_s",
    "train_runtime_s",
    "eval_runtime_s",
    "eval_runtime_mean_s",
    "eval_row_count",
    "parameter_count",
    "final_ce_loss",
    "train_teacher_action_agreement",
    "eval_teacher_action_agreement",
    "returncode",
    "results_path",
    "policy_path",
    "stdout_path",
    "stderr_path",
    "performance_summary_path",
    "gpu_monitor_summary_path",
    "gpu_monitor_sample_count",
    "gpu_monitor_shared_memory_peak_single_adapter_bytes",
    "gpu_monitor_shared_memory_peak_summed_bytes",
    "gpu_monitor_shared_memory_spill_detected",
    "gpu_monitor_nvidia_smi_peak_memory_used_mib",
    "gpu_monitor_nvidia_smi_peak_utilization_percent",
]
SUMMARY_FIELDS = [
    "device",
    "device_label",
    "repeat_count",
    "wall_runtime_s_mean",
    "wall_runtime_s_std",
    "cpu_relative_speedup",
    "train_runtime_s_mean",
    "eval_runtime_s_mean",
    "eval_runtime_mean_s_mean",
    "parameter_count",
    "final_ce_loss_mean",
    "train_teacher_action_agreement_mean",
    "eval_teacher_action_agreement_mean",
    "gpu_monitor_sample_count_max",
    "gpu_monitor_shared_memory_peak_single_adapter_bytes_max",
    "gpu_monitor_shared_memory_peak_summed_bytes_max",
    "gpu_monitor_shared_memory_spill_detected",
    "gpu_monitor_nvidia_smi_peak_memory_used_mib_max",
    "gpu_monitor_nvidia_smi_peak_utilization_percent_max",
]


@dataclass(frozen=True)
class BenchmarkCell:
    device: str
    repeat: int
    seed: int
    run_dir: Path
    results_path: Path
    policy_path: Path
    stdout_path: Path
    stderr_path: Path
    performance_summary_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark oracle_stationary_finite_distill.py on CPU and CUDA using "
            "the formal default single-card workload."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--env", default="fsrs6_default")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--deck-scale", type=int, default=DEFAULT_DECK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cost-weights",
        default=",".join(
            format_float(value)
            for value in DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
    )
    parser.add_argument("--train-envs", type=int, default=DEFAULT_TRAIN_ENVS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--supervision", default=DEFAULT_DISTILL_SUPERVISION)
    parser.add_argument(
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    )
    parser.add_argument("--obs-mode", default=DEFAULT_DISTILL_OBS_MODE)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument(
        "--network-depth", type=int, default=DEFAULT_DISTILL_NETWORK_DEPTH
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_DISTILL_HIDDEN_SIZE)
    parser.add_argument(
        "--oracle-s-grid-size", type=int, default=DEFAULT_ORACLE_S_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-d-grid-size", type=int, default=DEFAULT_ORACLE_D_GRID_SIZE
    )
    parser.add_argument(
        "--oracle-stationary-finite-max-iterations",
        type=int,
        default=DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--oracle-stationary-finite-tolerance",
        type=float,
        default=DEFAULT_STATIONARY_FINITE_TOLERANCE,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--eval-particles", type=int, default=DEFAULT_EVAL_PARTICLES)
    parser.add_argument(
        "--gpu-monitor-interval-seconds",
        type=float,
        default=2.0,
        help="Sampling interval for CUDA run GPU memory monitoring.",
    )
    parser.add_argument(
        "--no-gpu-monitor-enabled",
        action="store_true",
        help="Disable CUDA GPU memory monitoring.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def parse_devices(raw: str) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must include at least one device.")
    return devices


def device_label(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


def aggregate_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_device: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_device.setdefault(str(row["device"]), []).append(row)

    cpu_rows = by_device.get("cpu", [])
    cpu_mean = _mean(_float_values(cpu_rows, "wall_runtime_s")) if cpu_rows else None
    summaries: list[dict[str, Any]] = []
    for device, device_rows in by_device.items():
        wall_values = _float_values(device_rows, "wall_runtime_s")
        train_values = _float_values(device_rows, "train_runtime_s")
        eval_values = _float_values(device_rows, "eval_runtime_s")
        eval_mean_values = _float_values(device_rows, "eval_runtime_mean_s")
        wall_mean = _mean(wall_values)
        summary = {
            "device": device,
            "device_label": str(device_rows[0]["device_label"]),
            "repeat_count": len(device_rows),
            "wall_runtime_s_mean": wall_mean,
            "wall_runtime_s_std": _sample_std(wall_values),
            "cpu_relative_speedup": (
                cpu_mean / wall_mean
                if cpu_mean is not None and wall_mean > 0.0
                else None
            ),
            "train_runtime_s_mean": _mean(train_values),
            "eval_runtime_s_mean": _mean(eval_values),
            "eval_runtime_mean_s_mean": _mean(eval_mean_values),
            "parameter_count": int(device_rows[0]["parameter_count"]),
            "final_ce_loss_mean": _mean(_float_values(device_rows, "final_ce_loss")),
            "train_teacher_action_agreement_mean": _mean(
                _float_values(device_rows, "train_teacher_action_agreement")
            ),
            "eval_teacher_action_agreement_mean": _mean(
                _float_values(device_rows, "eval_teacher_action_agreement")
            ),
            "gpu_monitor_sample_count_max": _max_optional_int(
                device_rows, "gpu_monitor_sample_count"
            ),
            "gpu_monitor_shared_memory_peak_single_adapter_bytes_max": (
                _max_optional_int(
                    device_rows,
                    "gpu_monitor_shared_memory_peak_single_adapter_bytes",
                )
            ),
            "gpu_monitor_shared_memory_peak_summed_bytes_max": _max_optional_int(
                device_rows,
                "gpu_monitor_shared_memory_peak_summed_bytes",
            ),
            "gpu_monitor_shared_memory_spill_detected": _any_optional_bool(
                device_rows,
                "gpu_monitor_shared_memory_spill_detected",
            ),
            "gpu_monitor_nvidia_smi_peak_memory_used_mib_max": _max_optional_float(
                device_rows,
                "gpu_monitor_nvidia_smi_peak_memory_used_mib",
            ),
            "gpu_monitor_nvidia_smi_peak_utilization_percent_max": (
                _max_optional_float(
                    device_rows,
                    "gpu_monitor_nvidia_smi_peak_utilization_percent",
                )
            ),
        }
        summaries.append(summary)
    return sorted(
        summaries, key=lambda row: (str(row["device"]) != "cpu", row["device"])
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0.")
    if args.gpu_monitor_interval_seconds <= 0.0:
        raise SystemExit("--gpu-monitor-interval-seconds must be > 0.")
    devices = parse_devices(args.devices)
    _validate_devices(devices)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for device in devices:
        for repeat in range(1, args.repeats + 1):
            cell = _cell(args, device=device, repeat=repeat)
            row = _run_cell(args, cell)
            run_rows.append(row)
            _write_csv(args.out_dir / "runs.csv", run_rows, RUN_FIELDS)
            _write_csv(
                args.out_dir / "summary.csv",
                aggregate_summary_rows(run_rows),
                SUMMARY_FIELDS,
            )
            _write_metadata(args, devices=devices)

    print(f"Wrote {args.out_dir / 'runs.csv'}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    print(f"Wrote {args.out_dir / 'metadata.json'}")
    return 0


def _validate_devices(devices: Sequence[str]) -> None:
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")


def _cell(args: argparse.Namespace, *, device: str, repeat: int) -> BenchmarkCell:
    label = device_label(device)
    run_dir = args.out_dir / label / f"repeat_{repeat}"
    return BenchmarkCell(
        device=device,
        repeat=repeat,
        seed=args.seed,
        run_dir=run_dir,
        results_path=run_dir / "results.csv",
        policy_path=run_dir / "policy.pt",
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        performance_summary_path=run_dir / "performance_summary.json",
    )


def _run_cell(args: argparse.Namespace, cell: BenchmarkCell) -> dict[str, Any]:
    cell.run_dir.mkdir(parents=True, exist_ok=True)
    command = _child_command(args, cell)
    monitor = _start_monitor(args, cell)
    start = time.perf_counter()
    with (
        cell.stdout_path.open("w", encoding="utf-8") as stdout,
        cell.stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    wall_runtime_s = time.perf_counter() - start
    gpu_summary = _stop_monitor(monitor, cell)
    if completed.returncode != 0:
        _write_performance_summary(
            cell,
            command=command,
            wall_runtime_s=wall_runtime_s,
            returncode=completed.returncode,
            gpu_summary=gpu_summary,
        )
        raise SystemExit(
            f"Benchmark cell failed for device={cell.device} repeat={cell.repeat}. "
            f"See {cell.stderr_path}."
        )

    row = _run_row(
        cell,
        command=command,
        wall_runtime_s=wall_runtime_s,
        returncode=completed.returncode,
        gpu_summary=gpu_summary,
    )
    _write_performance_summary(
        cell,
        command=command,
        wall_runtime_s=wall_runtime_s,
        returncode=completed.returncode,
        gpu_summary=gpu_summary,
    )
    return row


def _child_command(args: argparse.Namespace, cell: BenchmarkCell) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.single_card_tradeoff.cli.oracle_stationary_finite_distill",
        "--env",
        args.env,
        "--days",
        str(args.days),
        "--deck-scale",
        str(args.deck_scale),
        "--seed",
        str(cell.seed),
        "--torch-device",
        cell.device,
        "--cost-weights",
        args.cost_weights,
        "--action-retentions",
        args.action_retentions,
        "--train-envs",
        str(args.train_envs),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--learning-rate",
        str(args.learning_rate),
        "--supervision",
        args.supervision,
        "--table-samples-per-weight",
        str(args.table_samples_per_weight),
        "--obs-mode",
        args.obs_mode,
        "--network",
        args.network,
        "--network-depth",
        str(args.network_depth),
        "--hidden-size",
        str(args.hidden_size),
        "--oracle-s-grid-size",
        str(args.oracle_s_grid_size),
        "--oracle-d-grid-size",
        str(args.oracle_d_grid_size),
        "--oracle-stationary-finite-max-iterations",
        str(args.oracle_stationary_finite_max_iterations),
        "--oracle-stationary-finite-tolerance",
        str(args.oracle_stationary_finite_tolerance),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--eval-particles",
        str(args.eval_particles),
        "--out",
        str(cell.results_path),
        "--model-out",
        str(cell.policy_path),
    ]
    if args.no_progress:
        command.append("--no-progress")
    return command


def _start_monitor(
    args: argparse.Namespace,
    cell: BenchmarkCell,
) -> GpuMonitor | None:
    if not cell.device.startswith("cuda") or args.no_gpu_monitor_enabled:
        return None
    monitor = GpuMonitor(
        output_dir=cell.run_dir / "gpu_monitor",
        interval_seconds=float(args.gpu_monitor_interval_seconds),
    )
    monitor.start()
    return monitor


def _stop_monitor(
    monitor: GpuMonitor | None,
    cell: BenchmarkCell,
) -> GpuMonitorSummary:
    if monitor is None:
        return disabled_monitor_summary(cell.run_dir / "gpu_monitor")
    return monitor.stop()


def _run_row(
    cell: BenchmarkCell,
    *,
    command: Sequence[str],
    wall_runtime_s: float,
    returncode: int,
    gpu_summary: GpuMonitorSummary,
) -> dict[str, Any]:
    checkpoint = torch.load(cell.policy_path, map_location="cpu", weights_only=False)
    result_rows = _read_csv_rows(cell.results_path)
    eval_runtime_values = [float(row["runtime_s"]) for row in result_rows]
    parameter_count = int(
        sum(value.numel() for value in checkpoint["model_state_dict"].values())
    )
    return {
        "device": cell.device,
        "device_label": device_label(cell.device),
        "repeat": cell.repeat,
        "seed": cell.seed,
        "wall_runtime_s": wall_runtime_s,
        "train_runtime_s": float(checkpoint["train_runtime_s"]),
        "eval_runtime_s": sum(eval_runtime_values),
        "eval_runtime_mean_s": _mean(eval_runtime_values),
        "eval_row_count": len(result_rows),
        "parameter_count": parameter_count,
        "final_ce_loss": float(checkpoint["final_ce_loss"]),
        "train_teacher_action_agreement": float(
            checkpoint["final_teacher_action_agreement"]
        ),
        "eval_teacher_action_agreement": float(
            checkpoint["eval_teacher_action_agreement"]
        ),
        "returncode": returncode,
        "results_path": _display_path(cell.results_path),
        "policy_path": _display_path(cell.policy_path),
        "stdout_path": _display_path(cell.stdout_path),
        "stderr_path": _display_path(cell.stderr_path),
        "performance_summary_path": _display_path(cell.performance_summary_path),
        "gpu_monitor_summary_path": _display_path(gpu_summary.summary_path)
        if gpu_summary.enabled
        else None,
        "gpu_monitor_sample_count": gpu_summary.sample_count,
        "gpu_monitor_shared_memory_peak_single_adapter_bytes": (
            gpu_summary.shared_memory_peak_single_adapter_bytes
        ),
        "gpu_monitor_shared_memory_peak_summed_bytes": (
            gpu_summary.shared_memory_peak_summed_bytes
        ),
        "gpu_monitor_shared_memory_spill_detected": (
            gpu_summary.shared_memory_spill_detected
        ),
        "gpu_monitor_nvidia_smi_peak_memory_used_mib": (
            gpu_summary.nvidia_smi_peak_memory_used_mib
        ),
        "gpu_monitor_nvidia_smi_peak_utilization_percent": (
            gpu_summary.nvidia_smi_peak_utilization_percent
        ),
    }


def _write_performance_summary(
    cell: BenchmarkCell,
    *,
    command: Sequence[str],
    wall_runtime_s: float,
    returncode: int,
    gpu_summary: GpuMonitorSummary,
) -> None:
    payload = {
        "type": "oracle-stationary-finite-cpu-gpu-benchmark-cell",
        "schema_version": 1,
        "generated_at": _timestamp(),
        "device": cell.device,
        "repeat": cell.repeat,
        "seed": cell.seed,
        "command": list(command),
        "returncode": returncode,
        "wall_runtime_s": wall_runtime_s,
        "gpu_monitor": gpu_summary.to_dict(),
    }
    cell.performance_summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_metadata(args: argparse.Namespace, *, devices: Sequence[str]) -> None:
    payload = {
        "type": "oracle-stationary-finite-cpu-gpu-benchmark",
        "schema_version": 1,
        "generated_at": _timestamp(),
        "command": sys.argv,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "settings": {
            "devices": list(devices),
            "repeats": args.repeats,
            "env": args.env,
            "days": args.days,
            "deck_scale": args.deck_scale,
            "seed": args.seed,
            "cost_weights": args.cost_weights,
            "action_retentions": args.action_retentions,
            "train_envs": args.train_envs,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "supervision": args.supervision,
            "table_samples_per_weight": args.table_samples_per_weight,
            "obs_mode": args.obs_mode,
            "network": args.network,
            "network_depth": args.network_depth,
            "hidden_size": args.hidden_size,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "eval_particles": args.eval_particles,
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path):
        return _display_path(value)
    return value


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _float_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (
        sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
    ) ** 0.5


def _max_optional_int(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    values = [int(row[key]) for row in rows if row.get(key) not in (None, "")]
    return max(values) if values else None


def _max_optional_float(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return max(values) if values else None


def _any_optional_bool(rows: Sequence[Mapping[str, Any]], key: str) -> bool | None:
    values: list[bool] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, bool):
            values.append(raw)
        else:
            values.append(str(raw).lower() == "true")
    return any(values) if values else None


if __name__ == "__main__":
    raise SystemExit(main())

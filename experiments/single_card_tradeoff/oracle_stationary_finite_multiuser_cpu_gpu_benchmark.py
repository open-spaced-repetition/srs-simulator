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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.single_card_tradeoff.oracle_stationary_finite_distill import (
    DEFAULT_DISTILL_EPOCHS,
    DEFAULT_DISTILL_HIDDEN_SIZE,
    DEFAULT_DISTILL_NETWORK_DEPTH,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_STATIONARY_FINITE_DISTILL_COST_WEIGHTS,
    DEFAULT_STATIONARY_FINITE_MAX_ITERATIONS,
    DEFAULT_STATIONARY_FINITE_TOLERANCE,
    DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
)
from experiments.single_card_tradeoff.oracle_stationary_finite_distill_multiuser import (
    DEFAULT_EVAL_PARTICLES,
    DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
    DEFAULT_TRAIN_ENVS_PER_USER,
    DEFAULT_USER_IDS,
)
from experiments.single_card_tradeoff.tradeoff import (
    DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS,
    DEFAULT_TARGET_RETENTIONS,
)
from experiments.single_card_tradeoff.uvfa_ppo import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_NETWORK,
    DEFAULT_ORACLE_D_GRID_SIZE,
    DEFAULT_ORACLE_S_GRID_SIZE,
)
from simulator.defaults import DEFAULT_DAYS, DEFAULT_DECK_SIZE, DEFAULT_SEED
from simulator.scheduler_spec import format_float


DEFAULT_OUT_DIR = Path(
    "artifacts/single_card_tradeoff/"
    "oracle_stationary_finite_multiuser_cpu_gpu_benchmark"
)
RUN_FIELDS = [
    "device",
    "device_label",
    "repeat",
    "seed",
    "user_count",
    "user_ids",
    "wall_runtime_s",
    "setup_runtime_s",
    "teacher_runtime_s",
    "train_runtime_s",
    "agreement_runtime_s",
    "eval_runtime_s",
    "total_runtime_s",
    "params_per_user",
    "ensemble_trainable_params",
    "mean_final_ce_loss",
    "mean_train_teacher_action_agreement",
    "mean_eval_teacher_action_agreement",
    "returncode",
    "results_path",
    "summary_path",
    "train_summary_path",
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
    "user_count",
    "user_ids",
    "wall_runtime_s_mean",
    "wall_runtime_s_std",
    "cpu_relative_speedup",
    "setup_runtime_s_mean",
    "teacher_runtime_s_mean",
    "train_runtime_s_mean",
    "agreement_runtime_s_mean",
    "eval_runtime_s_mean",
    "total_runtime_s_mean",
    "params_per_user",
    "ensemble_trainable_params",
    "mean_final_ce_loss",
    "mean_train_teacher_action_agreement",
    "mean_eval_teacher_action_agreement",
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
    stdout_path: Path
    stderr_path: Path
    performance_summary_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark oracle_stationary_finite_distill_multiuser.py on CPU and "
            "CUDA with per-user stationary finite distill models."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--env", default="fsrs6")
    parser.add_argument(
        "--user-ids",
        default=",".join(str(user_id) for user_id in DEFAULT_USER_IDS),
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=Path("../Anki-button-usage/button_usage.jsonl"),
    )
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
        "--eval-cost-weights",
        default=",".join(
            format_float(value) for value in DEFAULT_SCALARIZATION_EVAL_COST_WEIGHTS
        ),
    )
    parser.add_argument(
        "--action-retentions",
        default=",".join(format_float(value) for value in DEFAULT_TARGET_RETENTIONS),
    )
    parser.add_argument(
        "--train-envs-per-user",
        type=int,
        default=DEFAULT_TRAIN_ENVS_PER_USER,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_DISTILL_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--per-user-supervision", default="uniform_table")
    parser.add_argument(
        "--table-samples-per-weight",
        type=int,
        default=DEFAULT_TABLE_SAMPLES_PER_WEIGHT,
    )
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
        "--oracle-teacher-user-batch-size",
        type=int,
        default=DEFAULT_ORACLE_TEACHER_USER_BATCH_SIZE,
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
    )
    parser.add_argument("--no-gpu-monitor-enabled", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def parse_devices(raw: str) -> list[str]:
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must include at least one device.")
    return devices


def parse_user_ids(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--user-ids must include at least one user.")
    return values


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
        wall_mean = _mean(wall_values)
        summaries.append(
            {
                "device": device,
                "device_label": str(device_rows[0]["device_label"]),
                "repeat_count": len(device_rows),
                "user_count": int(device_rows[0]["user_count"]),
                "user_ids": str(device_rows[0]["user_ids"]),
                "wall_runtime_s_mean": wall_mean,
                "wall_runtime_s_std": _sample_std(wall_values),
                "cpu_relative_speedup": (
                    cpu_mean / wall_mean
                    if cpu_mean is not None and wall_mean > 0.0
                    else None
                ),
                "setup_runtime_s_mean": _mean(
                    _float_values(device_rows, "setup_runtime_s")
                ),
                "teacher_runtime_s_mean": _mean(
                    _float_values(device_rows, "teacher_runtime_s")
                ),
                "train_runtime_s_mean": _mean(
                    _float_values(device_rows, "train_runtime_s")
                ),
                "agreement_runtime_s_mean": _mean(
                    _float_values(device_rows, "agreement_runtime_s")
                ),
                "eval_runtime_s_mean": _mean(
                    _float_values(device_rows, "eval_runtime_s")
                ),
                "total_runtime_s_mean": _mean(
                    _float_values(device_rows, "total_runtime_s")
                ),
                "params_per_user": int(device_rows[0]["params_per_user"]),
                "ensemble_trainable_params": int(
                    device_rows[0]["ensemble_trainable_params"]
                ),
                "mean_final_ce_loss": _mean(
                    _float_values(device_rows, "mean_final_ce_loss")
                ),
                "mean_train_teacher_action_agreement": _mean(
                    _float_values(device_rows, "mean_train_teacher_action_agreement")
                ),
                "mean_eval_teacher_action_agreement": _mean(
                    _float_values(device_rows, "mean_eval_teacher_action_agreement")
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
        )
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
    user_ids = parse_user_ids(args.user_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    for device in devices:
        for repeat in range(1, args.repeats + 1):
            cell = _cell(args, device=device, repeat=repeat)
            row = _run_cell(args, cell, user_ids=user_ids)
            run_rows.append(row)
            _write_csv(args.out_dir / "runs.csv", run_rows, RUN_FIELDS)
            _write_csv(
                args.out_dir / "summary.csv",
                aggregate_summary_rows(run_rows),
                SUMMARY_FIELDS,
            )
            _write_metadata(args, devices=devices, user_ids=user_ids)

    print(f"Wrote {args.out_dir / 'runs.csv'}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    print(f"Wrote {args.out_dir / 'metadata.json'}")
    return 0


def _validate_devices(devices: Sequence[str]) -> None:
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")


def _cell(args: argparse.Namespace, *, device: str, repeat: int) -> BenchmarkCell:
    run_dir = args.out_dir / device_label(device) / f"repeat_{repeat}"
    return BenchmarkCell(
        device=device,
        repeat=repeat,
        seed=args.seed,
        run_dir=run_dir,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        performance_summary_path=run_dir / "performance_summary.json",
    )


def _run_cell(
    args: argparse.Namespace,
    cell: BenchmarkCell,
    *,
    user_ids: Sequence[int],
) -> dict[str, Any]:
    cell.run_dir.mkdir(parents=True, exist_ok=True)
    command = _child_command(args, cell)
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
    if completed.returncode != 0:
        raise SystemExit(
            f"Benchmark cell failed for device={cell.device} repeat={cell.repeat}. "
            f"See {cell.stderr_path}."
        )
    performance_summary = _read_performance_summary(cell)
    row = _run_row(
        cell,
        user_ids=user_ids,
        command=command,
        wall_runtime_s=wall_runtime_s,
        returncode=completed.returncode,
        performance_summary=performance_summary,
    )
    return row


def _child_command(args: argparse.Namespace, cell: BenchmarkCell) -> list[str]:
    command = [
        sys.executable,
        str(
            REPO_ROOT / "experiments/single_card_tradeoff/"
            "oracle_stationary_finite_distill_multiuser.py"
        ),
        "--env",
        args.env,
        "--user-ids",
        args.user_ids,
        "--button-usage",
        str(args.button_usage),
        "--per-user-models",
        "--per-user-supervision",
        args.per_user_supervision,
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
        "--eval-cost-weights",
        args.eval_cost_weights,
        "--action-retentions",
        args.action_retentions,
        "--train-envs-per-user",
        str(args.train_envs_per_user),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--learning-rate",
        str(args.learning_rate),
        "--table-samples-per-weight",
        str(args.table_samples_per_weight),
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
        "--oracle-teacher-user-batch-size",
        str(args.oracle_teacher_user_batch_size),
        "--oracle-stationary-finite-max-iterations",
        str(args.oracle_stationary_finite_max_iterations),
        "--oracle-stationary-finite-tolerance",
        str(args.oracle_stationary_finite_tolerance),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--eval-particles",
        str(args.eval_particles),
        "--gpu-monitor-interval-seconds",
        str(args.gpu_monitor_interval_seconds),
        "--out-dir",
        str(cell.run_dir),
    ]
    if args.no_gpu_monitor_enabled:
        command.append("--no-gpu-monitor-enabled")
    if args.no_progress:
        command.append("--no-progress")
    return command


def _run_row(
    cell: BenchmarkCell,
    *,
    user_ids: Sequence[int],
    command: Sequence[str],
    wall_runtime_s: float,
    returncode: int,
    performance_summary: Mapping[str, Any],
) -> dict[str, Any]:
    train_rows = _read_csv_rows(cell.run_dir / "train_summary.csv")
    if not train_rows:
        raise RuntimeError(f"Missing train summary rows in {cell.run_dir}.")
    first = train_rows[0]
    gpu_monitor_summary_path = performance_summary.get("gpu_monitor_summary_path")
    return {
        "device": cell.device,
        "device_label": device_label(cell.device),
        "repeat": cell.repeat,
        "seed": cell.seed,
        "user_count": len(user_ids),
        "user_ids": ",".join(str(user_id) for user_id in user_ids),
        "wall_runtime_s": wall_runtime_s,
        "setup_runtime_s": float(first["setup_runtime_s"]),
        "teacher_runtime_s": float(first["teacher_runtime_s"]),
        "train_runtime_s": float(first["train_runtime_s"]),
        "agreement_runtime_s": float(first["agreement_runtime_s"]),
        "eval_runtime_s": float(first["eval_runtime_s"]),
        "total_runtime_s": float(first["total_runtime_s"]),
        "params_per_user": int(first["params_per_user"]),
        "ensemble_trainable_params": int(first["ensemble_trainable_params"]),
        "mean_final_ce_loss": _mean(_float_values(train_rows, "final_ce_loss")),
        "mean_train_teacher_action_agreement": _mean(
            _float_values(train_rows, "final_teacher_action_agreement")
        ),
        "mean_eval_teacher_action_agreement": _mean(
            _float_values(train_rows, "eval_teacher_action_agreement")
        ),
        "returncode": returncode,
        "results_path": _display_path(cell.run_dir / "results.csv"),
        "summary_path": _display_path(cell.run_dir / "summary.csv"),
        "train_summary_path": _display_path(cell.run_dir / "train_summary.csv"),
        "stdout_path": _display_path(cell.stdout_path),
        "stderr_path": _display_path(cell.stderr_path),
        "performance_summary_path": _display_path(cell.performance_summary_path),
        "gpu_monitor_summary_path": _display_path(
            _artifact_path(gpu_monitor_summary_path)
        )
        if isinstance(gpu_monitor_summary_path, str) and gpu_monitor_summary_path
        else None,
        "gpu_monitor_sample_count": performance_summary.get("gpu_monitor_sample_count"),
        "gpu_monitor_shared_memory_peak_single_adapter_bytes": (
            performance_summary.get(
                "gpu_monitor_shared_memory_peak_single_adapter_bytes"
            )
        ),
        "gpu_monitor_shared_memory_peak_summed_bytes": (
            performance_summary.get("gpu_monitor_shared_memory_peak_summed_bytes")
        ),
        "gpu_monitor_shared_memory_spill_detected": (
            performance_summary.get("gpu_monitor_shared_memory_spill_detected")
        ),
        "gpu_monitor_nvidia_smi_peak_memory_used_mib": (
            performance_summary.get("gpu_monitor_nvidia_smi_peak_memory_used_mib")
        ),
        "gpu_monitor_nvidia_smi_peak_utilization_percent": (
            performance_summary.get("gpu_monitor_nvidia_smi_peak_utilization_percent")
        ),
    }


def _read_performance_summary(cell: BenchmarkCell) -> dict[str, Any]:
    with cell.performance_summary_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected JSON object at {cell.performance_summary_path}.")
    return raw


def _artifact_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _write_metadata(
    args: argparse.Namespace,
    *,
    devices: Sequence[str],
    user_ids: Sequence[int],
) -> None:
    payload = {
        "type": "oracle-stationary-finite-multiuser-cpu-gpu-benchmark",
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
            "user_ids": list(user_ids),
            "button_usage": str(args.button_usage),
            "days": args.days,
            "deck_scale": args.deck_scale,
            "seed": args.seed,
            "cost_weights": args.cost_weights,
            "eval_cost_weights": args.eval_cost_weights,
            "action_retentions": args.action_retentions,
            "train_envs_per_user": args.train_envs_per_user,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "per_user_supervision": args.per_user_supervision,
            "table_samples_per_weight": args.table_samples_per_weight,
            "network": args.network,
            "network_depth": args.network_depth,
            "hidden_size": args.hidden_size,
            "oracle_s_grid_size": args.oracle_s_grid_size,
            "oracle_d_grid_size": args.oracle_d_grid_size,
            "oracle_teacher_user_batch_size": args.oracle_teacher_user_batch_size,
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

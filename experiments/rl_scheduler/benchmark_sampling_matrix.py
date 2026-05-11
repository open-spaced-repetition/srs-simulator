from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.benchmark_sampling import (
    LSTM_MAX_BATCH_OFF,
    LstmMaxBatchOverride,
    _format_lstm_max_batch_override,
    _parse_ints,
)
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH


BENCHMARK_SCRIPT = Path("experiments/rl_scheduler/benchmark_sampling.py")
COMMON_LANE_SHAPES: dict[int, tuple[tuple[int, int], ...]] = {
    256: ((8, 32), (16, 16), (32, 8), (64, 4), (128, 2)),
    512: ((8, 64), (16, 32), (32, 16), (64, 8), (128, 4)),
    1024: ((8, 128), (16, 64), (32, 32), (64, 16), (128, 8)),
}
FSRS6_EXTENDED_LANE_SHAPES: dict[int, tuple[tuple[int, int], ...]] = {
    2048: ((8, 256), (16, 128), (32, 64), (64, 32), (128, 16)),
    4096: ((8, 512), (16, 256), (32, 128), (64, 64), (128, 32)),
    8192: ((8, 1024), (16, 512), (32, 256), (64, 128), (128, 64)),
}
LSTM_MAX_BATCH_GRID: tuple[LstmMaxBatchOverride, ...] = (
    1024,
    2048,
    4096,
    8192,
    20000,
    LSTM_MAX_BATCH_OFF,
)
COMPETITIVE_REVIEWS_PER_SECOND_FRACTION = 0.90
MATRIX_CSV_FIELDS = (
    "phase",
    "status",
    "cell_id",
    "environment",
    "matrix_group",
    "user_count",
    "candidate_count",
    "total_lanes",
    "lstm_max_batch",
    "effective_lstm_max_batch",
    "repeats",
    "warmup",
    "mean_seconds",
    "mean_lanes_per_second",
    "mean_reviews_per_second",
    "mean_reviews_per_lane",
    "max_torch_peak_reserved_memory_bytes",
    "max_nvidia_smi_peak_memory_used_mib",
    "returncode",
    "failure_kind",
    "summary_path",
)


@dataclass(frozen=True, slots=True)
class SamplingMatrixCell:
    environment: Literal["fsrs6", "lstm"]
    matrix_group: Literal["common", "fsrs6_extended"]
    user_count: int
    candidate_count: int
    lstm_max_batch: LstmMaxBatchOverride | None = None

    @property
    def total_lanes(self) -> int:
        return self.user_count * self.candidate_count

    @property
    def cell_id(self) -> str:
        parts = [
            self.environment,
            self.matrix_group,
            f"lanes{self.total_lanes}",
            f"{self.user_count}x{self.candidate_count}",
        ]
        if self.environment == "lstm":
            parts.append(f"batch{self.lstm_max_batch_label or 'default'}")
        return "_".join(parts)

    @property
    def lstm_max_batch_label(self) -> str | None:
        if self.environment != "lstm":
            return None
        if self.lstm_max_batch is None:
            return None
        return str(_format_lstm_max_batch_override(self.lstm_max_batch))

    @property
    def effective_lstm_max_batch(self) -> int | None:
        if self.environment != "lstm":
            return None
        if self.lstm_max_batch == LSTM_MAX_BATCH_OFF:
            return None
        if isinstance(self.lstm_max_batch, int):
            return self.lstm_max_batch
        return 20000

    @property
    def lane_shape(self) -> str:
        return f"{self.user_count}x{self.candidate_count}"

    def base_record(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "environment": self.environment,
            "matrix_group": self.matrix_group,
            "user_count": self.user_count,
            "candidate_count": self.candidate_count,
            "total_lanes": self.total_lanes,
            "lstm_max_batch": self.lstm_max_batch_label,
            "effective_lstm_max_batch": self.effective_lstm_max_batch,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-DR ADR sampling benchmark matrix by invoking "
            "benchmark_sampling.py once per cell."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "experiments/rl_scheduler/configs/fsrs6_adr_linear_portfolio_users_1_8.toml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl_scheduler/sampling_benchmark_matrix"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--users",
        default="1..128",
        help="Inclusive range like 1..128 or comma-separated user ids.",
    )
    parser.add_argument("--matrix-repeats", type=int, default=1)
    parser.add_argument("--confirmation-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--fixed-desired-retention", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--nvidia-smi-sample-interval", type=float, default=0.5)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write planned cells without launching benchmark subprocesses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix_repeats < 1:
        raise SystemExit("--matrix-repeats must be >= 1.")
    if args.confirmation_repeats < 1:
        raise SystemExit("--confirmation-repeats must be >= 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")
    if args.timeout_seconds <= 0.0:
        raise SystemExit("--timeout-seconds must be > 0.")

    users = _parse_users(args.users)
    cells = generate_matrix_cells()
    max_user_count = max(cell.user_count for cell in cells)
    if len(users) < max_user_count:
        raise SystemExit(f"--users must contain at least {max_user_count} users.")

    run_id = args.run_id or datetime.now(UTC).strftime(
        "sampling_benchmark_matrix_%Y%m%d_%H%M%S"
    )
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "matrix_summary.json"
    csv_path = run_root / "matrix_summary.csv"

    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    if args.dry_run:
        matrix_records = [
            {
                "phase": "matrix",
                "status": "planned",
                "repeats": args.matrix_repeats,
                "warmup": args.warmup,
                **cell.base_record(),
            }
            for cell in cells
        ]
        confirmation_records: list[dict[str, Any]] = []
    else:
        matrix_records = [
            run_cell(
                cell,
                args=args,
                users=users,
                output_dir=run_root / "cells",
                phase="matrix",
                repeats=args.matrix_repeats,
            )
            for cell in cells
        ]
        confirmation_records = []
        if not args.skip_confirmation:
            confirmation_cells = select_confirmation_cells(matrix_records)
            confirmation_records = [
                run_cell(
                    cell,
                    args=args,
                    users=users,
                    output_dir=run_root / "confirmation",
                    phase="confirmation",
                    repeats=args.confirmation_repeats,
                )
                for cell in confirmation_cells
            ]

    summary = {
        "schema_version": 1,
        "created_at": started_at,
        "finished_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "output_dir": str(run_root),
        "config_path": str(args.config),
        "fixed_desired_retention": args.fixed_desired_retention,
        "users": users,
        "matrix_repeats": args.matrix_repeats,
        "confirmation_repeats": args.confirmation_repeats,
        "warmup": args.warmup,
        "matrix_cell_count": len(cells),
        "matrix_records": matrix_records,
        "confirmation_records": confirmation_records,
        "notes": [
            "Each record is complete only when status is ok and summary_path points "
            "to a benchmark_sampling.py summary.json.",
            "LSTM cells set SRS_LSTM_MAX_BATCH in the child process environment.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_matrix_csv(csv_path, [*matrix_records, *confirmation_records])
    stdout_summary = dict(summary)
    stdout_summary["matrix_records"] = f"omitted from stdout; see {csv_path}"
    stdout_summary["confirmation_records"] = f"omitted from stdout; see {csv_path}"
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))
    return 0


def generate_matrix_cells() -> list[SamplingMatrixCell]:
    cells: list[SamplingMatrixCell] = []
    for _total_lanes, shapes in COMMON_LANE_SHAPES.items():
        for user_count, candidate_count in shapes:
            cells.append(
                SamplingMatrixCell(
                    environment="fsrs6",
                    matrix_group="common",
                    user_count=user_count,
                    candidate_count=candidate_count,
                )
            )
            for max_batch in LSTM_MAX_BATCH_GRID:
                cells.append(
                    SamplingMatrixCell(
                        environment="lstm",
                        matrix_group="common",
                        user_count=user_count,
                        candidate_count=candidate_count,
                        lstm_max_batch=max_batch,
                    )
                )
    for _total_lanes, shapes in FSRS6_EXTENDED_LANE_SHAPES.items():
        for user_count, candidate_count in shapes:
            cells.append(
                SamplingMatrixCell(
                    environment="fsrs6",
                    matrix_group="fsrs6_extended",
                    user_count=user_count,
                    candidate_count=candidate_count,
                )
            )
    return cells


def run_cell(
    cell: SamplingMatrixCell,
    *,
    args: argparse.Namespace,
    users: Sequence[int],
    output_dir: Path,
    phase: Literal["matrix", "confirmation"],
    repeats: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess_log_dir = output_dir / "_subprocess_logs"
    subprocess_log_dir.mkdir(parents=True, exist_ok=True)
    child_run_id = f"{phase}_{cell.cell_id}"
    command = _benchmark_command(
        cell,
        args=args,
        users=users,
        output_dir=output_dir,
        child_run_id=child_run_id,
        repeats=repeats,
    )
    env = os.environ.copy()
    if cell.environment == "lstm" and cell.lstm_max_batch_label is not None:
        env["SRS_LSTM_MAX_BATCH"] = cell.lstm_max_batch_label

    stdout_path = subprocess_log_dir / f"{child_run_id}.stdout.txt"
    stderr_path = subprocess_log_dir / f"{child_run_id}.stderr.txt"
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(args.timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_timeout_output(exc.stdout)
        stderr = _coerce_timeout_output(exc.stderr)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        return {
            "phase": phase,
            "status": "failed",
            "failure_kind": "timeout",
            "repeats": repeats,
            "warmup": args.warmup,
            "elapsed_seconds": time.perf_counter() - started,
            "returncode": None,
            "command": command,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            **cell.base_record(),
        }

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    summary_path = output_dir / child_run_id / "summary.json"
    base = {
        "phase": phase,
        "repeats": repeats,
        "warmup": args.warmup,
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
        **cell.base_record(),
    }
    if result.returncode != 0:
        return {
            **base,
            "status": "failed",
            "failure_kind": _classify_failure(result.stdout, result.stderr),
        }
    if not summary_path.exists():
        return {**base, "status": "failed", "failure_kind": "missing_summary"}

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {**base, "status": "failed", "failure_kind": "invalid_summary"}

    aggregate = summary.get("aggregate")
    setup = summary.get("setup")
    if not isinstance(aggregate, list) or len(aggregate) != 1:
        return {**base, "status": "failed", "failure_kind": "invalid_aggregate"}
    aggregate_record = aggregate[0]
    setup_record = setup[0] if isinstance(setup, list) and setup else {}
    return {
        **base,
        "status": "ok",
        "failure_kind": None,
        "setup_seconds": setup_record.get("setup_seconds"),
        "mean_seconds": aggregate_record.get("mean_seconds"),
        "median_seconds": aggregate_record.get("median_seconds"),
        "min_seconds": aggregate_record.get("min_seconds"),
        "max_seconds": aggregate_record.get("max_seconds"),
        "mean_lanes_per_second": aggregate_record.get("mean_lanes_per_second"),
        "mean_lane_days_per_second": aggregate_record.get("mean_lane_days_per_second"),
        "mean_reviews_per_second": aggregate_record.get("mean_reviews_per_second"),
        "mean_reviews_per_lane": aggregate_record.get("mean_reviews_per_lane"),
        "mean_metric_total_reviews": aggregate_record.get("mean_metric_total_reviews"),
        "mean_metric_total_lapses": aggregate_record.get("mean_metric_total_lapses"),
        "mean_metric_total_cost": aggregate_record.get("mean_metric_total_cost"),
        "max_torch_peak_reserved_memory_bytes": aggregate_record.get(
            "max_torch_peak_reserved_memory_bytes"
        ),
        "max_nvidia_smi_peak_memory_used_mib": aggregate_record.get(
            "max_nvidia_smi_peak_memory_used_mib"
        ),
        "max_nvidia_smi_peak_utilization_gpu_percent": aggregate_record.get(
            "max_nvidia_smi_peak_utilization_gpu_percent"
        ),
        "max_nvidia_smi_peak_utilization_memory_percent": aggregate_record.get(
            "max_nvidia_smi_peak_utilization_memory_percent"
        ),
    }


def _benchmark_command(
    cell: SamplingMatrixCell,
    *,
    args: argparse.Namespace,
    users: Sequence[int],
    output_dir: Path,
    child_run_id: str,
    repeats: int,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        str(BENCHMARK_SCRIPT),
        "--config",
        str(args.config),
        "--output-dir",
        str(output_dir),
        "--run-id",
        child_run_id,
        "--environment",
        cell.environment,
        "--candidate-mode",
        "fixed-dr",
        "--fixed-desired-retention",
        str(args.fixed_desired_retention),
        "--users",
        ",".join(str(user) for user in users),
        "--lane-shapes",
        cell.lane_shape,
        "--repeats",
        str(repeats),
        "--warmup",
        str(args.warmup),
        "--button-usage",
        str(args.button_usage),
        "--nvidia-smi-sample-interval",
        str(args.nvidia_smi_sample_interval),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.torch_device is not None:
        command.extend(["--torch-device", str(args.torch_device)])
    if args.srs_benchmark_root is not None:
        command.extend(["--srs-benchmark-root", str(args.srs_benchmark_root)])
    if args.benchmark_result is not None:
        command.extend(["--benchmark-result", str(args.benchmark_result)])
    if args.benchmark_partition is not None:
        command.extend(["--benchmark-partition", str(args.benchmark_partition)])
    if cell.environment == "lstm" and cell.lstm_max_batch_label is not None:
        command.extend(["--lstm-max-batch", cell.lstm_max_batch_label])
    return command


def select_confirmation_cells(
    records: Sequence[dict[str, Any]],
) -> list[SamplingMatrixCell]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        grouped.setdefault(
            (str(record["environment"]), int(record["total_lanes"])), []
        ).append(record)

    selected: list[SamplingMatrixCell] = []
    seen: set[str] = set()
    for (_environment, _total_lanes), items in sorted(grouped.items()):
        fastest = min(items, key=lambda item: float(item["mean_seconds"]))
        best_reviews = max(items, key=_reviews_per_second)
        best_rate = _reviews_per_second(best_reviews)
        competitive = [
            item
            for item in items
            if _reviews_per_second(item)
            >= best_rate * COMPETITIVE_REVIEWS_PER_SECOND_FRACTION
        ]
        lowest_memory = min(competitive or items, key=_memory_score)
        for record in (fastest, best_reviews, lowest_memory):
            cell = _cell_from_record(record)
            if cell.cell_id in seen:
                continue
            seen.add(cell.cell_id)
            selected.append(cell)
    return selected


def _cell_from_record(record: dict[str, Any]) -> SamplingMatrixCell:
    raw_lstm_max_batch = record.get("lstm_max_batch")
    lstm_max_batch: LstmMaxBatchOverride | None
    if record["environment"] == "lstm":
        if raw_lstm_max_batch == LSTM_MAX_BATCH_OFF:
            lstm_max_batch = LSTM_MAX_BATCH_OFF
        elif raw_lstm_max_batch is None:
            lstm_max_batch = None
        else:
            lstm_max_batch = int(raw_lstm_max_batch)
    else:
        lstm_max_batch = None
    return SamplingMatrixCell(
        environment=cast(Literal["fsrs6", "lstm"], record["environment"]),
        matrix_group=cast(Literal["common", "fsrs6_extended"], record["matrix_group"]),
        user_count=int(record["user_count"]),
        candidate_count=int(record["candidate_count"]),
        lstm_max_batch=lstm_max_batch,
    )


def _reviews_per_second(record: dict[str, Any]) -> float:
    raw = record.get("mean_reviews_per_second")
    if raw is not None:
        return float(raw)
    return float(record.get("mean_lanes_per_second") or 0.0)


def _memory_score(record: dict[str, Any]) -> float:
    raw_torch = record.get("max_torch_peak_reserved_memory_bytes")
    if raw_torch is not None:
        return float(raw_torch)
    raw_nvidia = record.get("max_nvidia_smi_peak_memory_used_mib")
    if raw_nvidia is not None:
        return float(raw_nvidia) * 1024.0 * 1024.0
    return float("inf")


def _parse_users(value: str) -> list[int]:
    stripped = value.strip()
    if ".." not in stripped:
        return _parse_ints(stripped)
    start_raw, end_raw = stripped.split("..", 1)
    start = int(start_raw.strip())
    end = int(end_raw.strip())
    if start > end:
        raise ValueError("--users range start must be <= end.")
    return list(range(start, end + 1))


def _write_matrix_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MATRIX_CSV_FIELDS)
        for record in records:
            writer.writerow([record.get(field) for field in MATRIX_CSV_FIELDS])


def _classify_failure(stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "out of memory" in text or "cuda oom" in text or "cuda error: out" in text:
        return "oom"
    return "error"


def _coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

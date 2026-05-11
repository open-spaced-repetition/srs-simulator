from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import MutableMapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    PolicySearchSettings,
    _baseline_dr_values,
    _build_bundle,
    _policy_feature_version,
    _read_training_policy_search,
)
from experiments.rl_scheduler.portfolio_training_common import generator_for_job
from experiments.rl_scheduler.train_fsrs6_adr_portfolio import (
    PortfolioSettings,
    _constant_retention_coefficients,
    _evaluate_adr_coefficients,
    _mutate_coefficients,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.lstm_utils import resolve_lstm_max_batch_size
from simulator.short_term_config import resolve_short_term_config


DEFAULT_USER_COUNTS = (1, 2, 4, 8)
DEFAULT_CANDIDATE_COUNTS = (16, 32, 64, 128)
LSTM_MAX_BATCH_OFF: Literal["off"] = "off"
type LstmMaxBatchOverride = int | Literal["off"]
CSV_FIELDS = (
    "phase",
    "user_count",
    "candidate_count",
    "total_lanes",
    "effective_lstm_max_batch",
    "repeat_index",
    "seconds",
    "lanes_per_second",
    "lane_days_per_second",
    "seconds_per_lane",
    "reviews_per_second",
    "reviews_per_lane",
    "torch_peak_allocated_memory_bytes",
    "torch_peak_reserved_memory_bytes",
    "nvidia_smi_peak_memory_used_mib",
    "nvidia_smi_peak_utilization_gpu_percent",
    "nvidia_smi_peak_utilization_memory_percent",
    "metric_checksum",
    "metric_total_reviews",
    "metric_total_lapses",
    "metric_total_cost",
)


class NvidiaSmiSampler:
    def __init__(self, *, enabled: bool, interval_seconds: float) -> None:
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> NvidiaSmiSampler:
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self._thread = None

    def summary(self) -> dict[str, int | None]:
        if not self.samples:
            return {
                "nvidia_smi_peak_memory_used_mib": None,
                "nvidia_smi_peak_utilization_gpu_percent": None,
                "nvidia_smi_peak_utilization_memory_percent": None,
                "nvidia_smi_sample_count": 0,
            }
        return {
            "nvidia_smi_peak_memory_used_mib": max(
                sample["memory_used_mib"] for sample in self.samples
            ),
            "nvidia_smi_peak_utilization_gpu_percent": max(
                sample["utilization_gpu_percent"] for sample in self.samples
            ),
            "nvidia_smi_peak_utilization_memory_percent": max(
                sample["utilization_memory_percent"] for sample in self.samples
            ),
            "nvidia_smi_sample_count": len(self.samples),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _nvidia_smi_sample()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FSRS6 ADR portfolio candidate sampling by timing batched "
            "simulate_multiuser evaluations at several user/candidate lane shapes."
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
        default=Path("artifacts/rl_scheduler/sampling_benchmark"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--users",
        default=None,
        help="Comma-separated user ids. Defaults to the config training users.",
    )
    parser.add_argument(
        "--user-counts",
        default=",".join(str(item) for item in DEFAULT_USER_COUNTS),
        help="Comma-separated prefixes of --users to benchmark.",
    )
    parser.add_argument(
        "--candidate-counts",
        default=",".join(str(item) for item in DEFAULT_CANDIDATE_COUNTS),
        help="Comma-separated candidates per user for each simulated sample batch.",
    )
    parser.add_argument(
        "--lane-shapes",
        default=None,
        help=(
            "Optional comma-separated user_count x candidate_count pairs, for "
            "example '8x128,16x64'. When set, skips the user/candidate grid."
        ),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-device", default=None)
    parser.add_argument(
        "--candidate-mode",
        choices=("portfolio", "fixed-dr"),
        default="portfolio",
        help=(
            "How to construct candidate policies. 'portfolio' uses seed DRs plus "
            "mutations; 'fixed-dr' repeats one constant desired-retention policy."
        ),
    )
    parser.add_argument(
        "--fixed-desired-retention",
        type=float,
        default=None,
        help=(
            "Desired retention used by --candidate-mode fixed-dr. Defaults to the "
            "largest configured portfolio seed retention value."
        ),
    )
    parser.add_argument(
        "--environment",
        choices=("fsrs6", "lstm"),
        default=None,
        help="Override simulation.environment from the TOML config.",
    )
    parser.add_argument(
        "--lstm-max-batch",
        type=_parse_lstm_max_batch,
        default=None,
        help=(
            "Override SRS_LSTM_MAX_BATCH for --environment lstm. Use an integer "
            "or 'off' to disable LSTM chunking."
        ),
    )
    parser.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument(
        "--nvidia-smi-sample-interval",
        type=float,
        default=0.5,
        help="Seconds between nvidia-smi samples while timing. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-total-lanes",
        type=int,
        default=None,
        help="Skip shapes whose user_count * candidate_count exceeds this value.",
    )
    parser.add_argument(
        "--print-records",
        action="store_true",
        help="Print full per-repeat records to stdout. They are always written to disk.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0.")
    if args.nvidia_smi_sample_interval < 0.0:
        raise SystemExit("--nvidia-smi-sample-interval must be >= 0.")

    config = ExperimentConfig.from_toml(args.config)
    if args.environment is not None:
        config = replace(
            config,
            simulation=replace(config.simulation, environment=args.environment),
        )
    effective_lstm_max_batch = _apply_lstm_max_batch_override(
        environment=config.simulation.environment,
        override=cast(LstmMaxBatchOverride | None, args.lstm_max_batch),
    )
    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    if args.torch_device is not None:
        settings = PolicySearchSettings(
            coefficient_min=settings.coefficient_min,
            coefficient_max=settings.coefficient_max,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            baseline_desired_retention=settings.baseline_desired_retention,
            torch_device=args.torch_device,
            short_term_threshold=settings.short_term_threshold,
            short_term_loops_limit=settings.short_term_loops_limit,
        )
    device = torch.device(settings.torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    raw_training_policy = dict(_read_training_policy_search(args.config))
    baseline_dr_values = _baseline_dr_values(raw_training_policy, settings)
    feature_version = _policy_feature_version(raw_training_policy)
    portfolio = PortfolioSettings.from_mapping(
        config.training_portfolio,
        settings=settings,
        default_seed_retention_values=baseline_dr_values,
    )
    fixed_desired_retention = None
    if args.candidate_mode == "fixed-dr":
        fixed_desired_retention = _resolve_fixed_desired_retention(
            args.fixed_desired_retention,
            portfolio=portfolio,
            settings=settings,
        )
    users = _parse_ints(args.users) if args.users else list(config.users.train)
    if args.lane_shapes:
        lane_shapes = _parse_lane_shapes(args.lane_shapes)
        user_counts = sorted(
            {user_count for user_count, _candidate_count in lane_shapes}
        )
        candidate_counts = sorted(
            {candidate_count for _user_count, candidate_count in lane_shapes}
        )
    else:
        user_counts = _parse_ints(args.user_counts)
        candidate_counts = _parse_ints(args.candidate_counts)
        lane_shapes = [
            (user_count, candidate_count)
            for user_count in user_counts
            for candidate_count in candidate_counts
        ]
    for user_count in user_counts:
        if user_count < 1 or user_count > len(users):
            raise SystemExit(
                f"user_count {user_count} must be between 1 and {len(users)}."
            )
    for candidate_count in candidate_counts:
        if candidate_count < 1:
            raise SystemExit("candidate counts must be >= 1.")
    for user_count, candidate_count in lane_shapes:
        if user_count < 1 or user_count > len(users):
            raise SystemExit(
                f"user_count {user_count} must be between 1 and {len(users)}."
            )
        if candidate_count < 1:
            raise SystemExit("candidate counts must be >= 1.")

    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_policy.get("learning_steps"),
        relearning_steps=raw_training_policy.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )

    run_id = args.run_id or datetime.now(UTC).strftime(
        "sampling_benchmark_%Y%m%d_%H%M%S"
    )
    run_root = args.output_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    samples_path = run_root / "samples.jsonl"
    summary_path = run_root / "summary.json"
    csv_path = run_root / "samples.csv"

    seed = int(args.seed if args.seed is not None else config.seed)
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    all_records: list[dict[str, Any]] = []
    setup_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    with samples_path.open("w", encoding="utf-8") as jsonl_handle:
        for user_count, candidate_count in lane_shapes:
            selected_users = users[:user_count]
            total_lanes = user_count * candidate_count
            if args.max_total_lanes is not None and total_lanes > args.max_total_lanes:
                continue
            shape = {
                "user_count": user_count,
                "candidate_count": candidate_count,
                "total_lanes": total_lanes,
                "effective_lstm_max_batch": effective_lstm_max_batch,
                "users": selected_users,
            }
            coefficients_started = time.perf_counter()
            coefficients_by_job = _candidate_coefficients_by_job(
                user_ids=selected_users,
                candidate_count=candidate_count,
                settings=settings,
                portfolio=portfolio,
                feature_version=feature_version,
                device=device,
                seed=seed,
                candidate_mode=args.candidate_mode,
                fixed_desired_retention=fixed_desired_retention,
            )
            coefficient_generation_seconds = time.perf_counter() - coefficients_started

            lane_user_ids = [
                user_id
                for user_id in selected_users
                for _candidate in range(candidate_count)
            ]
            _reset_cuda_peak(device)
            _sync(device)
            setup_started = time.perf_counter()
            bundle = _build_bundle(
                config=config,
                settings=settings,
                lane_user_ids=lane_user_ids,
                benchmark_root=benchmark_root,
                overrides=overrides,
                benchmark_partition=args.benchmark_partition,
                button_usage=args.button_usage,
                device=device,
                short_term_source=short_term_source,
                learning_steps=learning_steps,
                relearning_steps=relearning_steps,
            )
            _sync(device)
            setup_seconds = time.perf_counter() - setup_started
            setup_record = {
                "phase": "setup",
                **shape,
                "repeat_index": None,
                "seconds": setup_seconds,
                "coefficient_generation_seconds": coefficient_generation_seconds,
                **_throughput(
                    seconds=setup_seconds,
                    total_lanes=total_lanes,
                    days=config.simulation.days,
                ),
                **_torch_memory(device),
            }
            _write_record(jsonl_handle, setup_record)
            setup_records.append(setup_record)
            all_records.append(setup_record)

            for warmup_index in range(args.warmup):
                _sync(device)
                warmup_started = time.perf_counter()
                _evaluate_adr_coefficients(
                    config=config,
                    settings=settings,
                    bundle=bundle,
                    coefficients_by_job=coefficients_by_job,
                    feature_version=feature_version,
                    seed=seed + warmup_index,
                )
                _sync(device)
                warmup_record = {
                    "phase": "warmup",
                    **shape,
                    "repeat_index": warmup_index,
                    "seconds": time.perf_counter() - warmup_started,
                }
                _write_record(jsonl_handle, warmup_record)
                all_records.append(warmup_record)

            for repeat_index in range(args.repeats):
                _reset_cuda_peak(device)
                _sync(device)
                sampler = NvidiaSmiSampler(
                    enabled=(
                        device.type == "cuda" and args.nvidia_smi_sample_interval > 0.0
                    ),
                    interval_seconds=max(args.nvidia_smi_sample_interval, 0.001),
                )
                with sampler:
                    started = time.perf_counter()
                    metrics_by_job = _evaluate_adr_coefficients(
                        config=config,
                        settings=settings,
                        bundle=bundle,
                        coefficients_by_job=coefficients_by_job,
                        feature_version=feature_version,
                        seed=seed + 1000 + repeat_index,
                    )
                    _sync(device)
                    seconds = time.perf_counter() - started
                metric_summary = _metric_summary(metrics_by_job)
                record = {
                    "phase": "evaluate",
                    **shape,
                    "repeat_index": repeat_index,
                    "seconds": seconds,
                    **metric_summary,
                    **_workload_throughput(
                        total_reviews=int(metric_summary["metric_total_reviews"]),
                        seconds=seconds,
                        total_lanes=total_lanes,
                    ),
                    **_throughput(
                        seconds=seconds,
                        total_lanes=total_lanes,
                        days=config.simulation.days,
                    ),
                    **_torch_memory(device),
                    **sampler.summary(),
                }
                _write_record(jsonl_handle, record)
                eval_records.append(record)
                all_records.append(record)

            del bundle
            _clear_cuda_cache(device)

    summary = {
        "schema_version": 1,
        "created_at": started_at,
        "finished_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config_path": str(args.config),
        "run_id": run_id,
        "output_dir": str(run_root),
        "samples_path": str(samples_path),
        "csv_path": str(csv_path),
        "device": str(device),
        "environment": config.simulation.environment,
        "days": config.simulation.days,
        "deck": config.simulation.deck,
        "feature_version": feature_version,
        "candidate_mode": args.candidate_mode,
        "fixed_desired_retention": fixed_desired_retention,
        "lstm_max_batch": _format_lstm_max_batch_override(
            cast(LstmMaxBatchOverride | None, args.lstm_max_batch)
        ),
        "effective_lstm_max_batch": effective_lstm_max_batch,
        "users": users,
        "user_counts": user_counts,
        "candidate_counts": candidate_counts,
        "lane_shapes": [
            {"user_count": user_count, "candidate_count": candidate_count}
            for user_count, candidate_count in lane_shapes
        ],
        "repeats": args.repeats,
        "warmup": args.warmup,
        "portfolio": asdict(portfolio),
        "settings": asdict(settings),
        "records": all_records,
        "aggregate": _aggregate_eval_records(eval_records),
        "setup": _aggregate_setup_records(setup_records),
        "notes": [
            "evaluate records time ADR candidate simulation only; setup records time "
            "bundle construction for the lane shape.",
            "nvidia-smi does not expose shared GPU memory on this Linux host; "
            "dedicated FB memory is recorded as memory.used.",
            "effective_lstm_max_batch is null for non-LSTM environments and for "
            "LSTM runs where SRS_LSTM_MAX_BATCH is disabled.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(csv_path, all_records)
    stdout_summary = dict(summary)
    if not args.print_records:
        stdout_summary["records"] = f"omitted from stdout; see {samples_path}"
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))
    return 0


def _candidate_coefficients_by_job(
    *,
    user_ids: Sequence[int],
    candidate_count: int,
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    feature_version: str,
    device: torch.device,
    seed: int,
    candidate_mode: str = "portfolio",
    fixed_desired_retention: float | None = None,
) -> list[list[tuple[float, ...]]]:
    if candidate_mode == "fixed-dr":
        if fixed_desired_retention is None:
            raise ValueError(
                "fixed_desired_retention is required for fixed-dr candidate mode."
            )
        coefficients = _constant_retention_coefficients(
            desired_retention=fixed_desired_retention,
            settings=settings,
            feature_version=feature_version,
        )
        return [
            [coefficients for _candidate in range(candidate_count)]
            for _user_id in user_ids
        ]
    if candidate_mode != "portfolio":
        raise ValueError(f"Unsupported candidate mode: {candidate_mode}")
    seed_coefficients = [
        _constant_retention_coefficients(
            desired_retention=dr,
            settings=settings,
            feature_version=feature_version,
        )
        for dr in portfolio.seed_retention_values or ()
    ]
    if not seed_coefficients:
        raise ValueError("portfolio seed_retention_values must not be empty.")
    all_coefficients: list[list[tuple[float, ...]]] = []
    for user_id in user_ids:
        generator = generator_for_job(device=device, seed=seed, user_id=user_id)
        job_coefficients: list[tuple[float, ...]] = []
        for index in range(candidate_count):
            base = seed_coefficients[index % len(seed_coefficients)]
            if index < len(seed_coefficients):
                coefficients = base
            else:
                coefficients = _mutate_coefficients(
                    base,
                    mutation_scale=portfolio.mutation_scale,
                    coefficient_min=settings.coefficient_min,
                    coefficient_max=settings.coefficient_max,
                    device=device,
                    generator=generator,
                )
            job_coefficients.append(coefficients)
        all_coefficients.append(job_coefficients)
    return all_coefficients


def _aggregate_eval_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(
            (int(record["user_count"]), int(record["candidate_count"])), []
        ).append(record)
    aggregate: list[dict[str, Any]] = []
    for (user_count, candidate_count), items in sorted(grouped.items()):
        seconds = [float(item["seconds"]) for item in items]
        lanes_per_second = [float(item["lanes_per_second"]) for item in items]
        lane_days_per_second = [float(item["lane_days_per_second"]) for item in items]
        aggregate.append(
            {
                "user_count": user_count,
                "candidate_count": candidate_count,
                "total_lanes": user_count * candidate_count,
                "effective_lstm_max_batch": items[0].get("effective_lstm_max_batch"),
                "repeats": len(items),
                "mean_seconds": statistics.fmean(seconds),
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "mean_lanes_per_second": statistics.fmean(lanes_per_second),
                "mean_lane_days_per_second": statistics.fmean(lane_days_per_second),
                "mean_reviews_per_second": statistics.fmean(
                    float(item.get("reviews_per_second") or 0.0) for item in items
                ),
                "mean_reviews_per_lane": statistics.fmean(
                    float(item.get("reviews_per_lane") or 0.0) for item in items
                ),
                "max_torch_peak_reserved_memory_bytes": max(
                    _optional_int(item.get("torch_peak_reserved_memory_bytes")) or 0
                    for item in items
                ),
                "max_nvidia_smi_peak_memory_used_mib": max(
                    _optional_int(item.get("nvidia_smi_peak_memory_used_mib")) or 0
                    for item in items
                ),
                "max_nvidia_smi_peak_utilization_gpu_percent": max(
                    _optional_int(item.get("nvidia_smi_peak_utilization_gpu_percent"))
                    or 0
                    for item in items
                ),
                "max_nvidia_smi_peak_utilization_memory_percent": max(
                    _optional_int(
                        item.get("nvidia_smi_peak_utilization_memory_percent")
                    )
                    or 0
                    for item in items
                ),
                "mean_metric_total_reviews": statistics.fmean(
                    float(item.get("metric_total_reviews") or 0.0) for item in items
                ),
                "mean_metric_total_lapses": statistics.fmean(
                    float(item.get("metric_total_lapses") or 0.0) for item in items
                ),
                "mean_metric_total_cost": statistics.fmean(
                    float(item.get("metric_total_cost") or 0.0) for item in items
                ),
            }
        )
    return aggregate


def _aggregate_setup_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "user_count": int(record["user_count"]),
            "candidate_count": int(record["candidate_count"]),
            "total_lanes": int(record["total_lanes"]),
            "effective_lstm_max_batch": record.get("effective_lstm_max_batch"),
            "setup_seconds": float(record["seconds"]),
            "coefficient_generation_seconds": float(
                record["coefficient_generation_seconds"]
            ),
            "torch_peak_reserved_memory_bytes": record.get(
                "torch_peak_reserved_memory_bytes"
            ),
        }
        for record in records
    ]


def _write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for record in records:
            writer.writerow([record.get(field) for field in CSV_FIELDS])


def _throughput(*, seconds: float, total_lanes: int, days: int) -> dict[str, float]:
    denom = max(seconds, 1e-12)
    return {
        "lanes_per_second": total_lanes / denom,
        "lane_days_per_second": total_lanes * days / denom,
        "seconds_per_lane": seconds / total_lanes,
    }


def _workload_throughput(
    *, total_reviews: int, seconds: float, total_lanes: int
) -> dict[str, float]:
    return {
        "reviews_per_second": total_reviews / max(seconds, 1e-12),
        "reviews_per_lane": total_reviews / total_lanes,
    }


def _metric_summary(metrics_by_job: Sequence[Sequence[Any]]) -> dict[str, float | int]:
    flat = [
        metrics for metrics_by_user in metrics_by_job for metrics in metrics_by_user
    ]
    return {
        "metric_checksum": float(
            sum(
                metrics.memorized_average + metrics.memorized_per_minute
                for metrics in flat
            )
        ),
        "metric_total_reviews": int(sum(metrics.total_reviews for metrics in flat)),
        "metric_total_lapses": int(sum(metrics.total_lapses for metrics in flat)),
        "metric_total_cost": float(sum(metrics.total_cost for metrics in flat)),
    }


def _parse_lstm_max_batch(value: str) -> LstmMaxBatchOverride:
    stripped = value.strip().lower()
    if stripped in {"off", "none"}:
        return LSTM_MAX_BATCH_OFF
    try:
        parsed = int(stripped)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--lstm-max-batch must be a positive integer or 'off'."
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "--lstm-max-batch must be a positive integer or 'off'."
        )
    return parsed


def _apply_lstm_max_batch_override(
    *,
    environment: str,
    override: LstmMaxBatchOverride | None,
    environ: MutableMapping[str, str] | None = None,
) -> int | None:
    if environment != "lstm":
        if override is not None:
            raise SystemExit("--lstm-max-batch can only be used with environment=lstm.")
        return None
    if override is None:
        return resolve_lstm_max_batch_size(None)
    writable_environ = os.environ if environ is None else environ
    if override == LSTM_MAX_BATCH_OFF:
        writable_environ["SRS_LSTM_MAX_BATCH"] = LSTM_MAX_BATCH_OFF
        return None
    writable_environ["SRS_LSTM_MAX_BATCH"] = str(override)
    return resolve_lstm_max_batch_size(override)


def _format_lstm_max_batch_override(
    override: LstmMaxBatchOverride | None,
) -> int | str | None:
    if override is None:
        return None
    return override


def _resolve_fixed_desired_retention(
    value: float | None,
    *,
    portfolio: PortfolioSettings,
    settings: PolicySearchSettings,
) -> float:
    if value is None:
        retention_values = portfolio.seed_retention_values or ()
        if not retention_values:
            raise ValueError(
                "portfolio seed_retention_values must not be empty when fixed DR is omitted."
            )
        value = max(float(item) for item in retention_values)
    fixed = float(value)
    if not settings.retention_min <= fixed <= settings.retention_max:
        raise ValueError(
            "fixed desired retention must be inside the configured retention bounds."
        )
    return fixed


def _parse_ints(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    parsed: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parsed.append(int(stripped))
    if not parsed:
        raise ValueError("At least one integer is required.")
    return parsed


def _parse_lane_shapes(value: str) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for item in value.split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if "x" not in stripped:
            raise ValueError(
                f"Invalid lane shape '{item}'. Expected user_count x candidate_count."
            )
        raw_user_count, raw_candidate_count = stripped.split("x", 1)
        parsed.append((int(raw_user_count.strip()), int(raw_candidate_count.strip())))
    if not parsed:
        raise ValueError("At least one lane shape is required.")
    return parsed


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _torch_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "torch_current_allocated_memory_bytes": None,
            "torch_current_reserved_memory_bytes": None,
            "torch_peak_allocated_memory_bytes": None,
            "torch_peak_reserved_memory_bytes": None,
        }
    return {
        "torch_current_allocated_memory_bytes": int(
            torch.cuda.memory_allocated(device)
        ),
        "torch_current_reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
        "torch_peak_allocated_memory_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "torch_peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _nvidia_smi_sample() -> dict[str, int] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0:1]
    if not first_line:
        return None
    values = [item.strip() for item in first_line[0].split(",")]
    if len(values) != 4:
        return None
    try:
        return {
            "memory_used_mib": int(values[0]),
            "memory_free_mib": int(values[1]),
            "utilization_gpu_percent": int(values[2]),
            "utilization_memory_percent": int(values[3]),
        }
    except ValueError:
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())

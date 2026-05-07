from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_fsrs6_adr_direct import (
    SASettings,
    _build_bundle,
    _evaluate_sa_candidates,
    _read_training_sa,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.fsrs6_adr_direct_policy import FSRS6ADRDirectPolicy
from simulator.short_term_config import resolve_short_term_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune FSRS6 ADR Direct effective lanes by timing one candidate evaluation "
            "per lane count."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument(
        "--candidate-lanes",
        default="8,16,32,64,128",
        help="Comma-separated effective lane counts to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for lane_tuning_summary.json.",
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        help="Override training.sa.torch_device for the tuning run.",
    )
    parser.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE_PATH)
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument(
        "--stop-after-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop lane sweep after the first CUDA OOM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the resolved tuning plan without running simulations.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)
    lanes = _parse_lane_values(args.candidate_lanes)
    user_id = args.user_id if args.user_id is not None else config.users.train[0]
    settings = SASettings.from_mapping(config.training_sa)
    if args.torch_device is not None:
        settings = replace(settings, torch_device=args.torch_device)
    output_dir = args.output_dir or (
        REPO_ROOT / "artifacts" / "rl_scheduler" / "lane_tuning" / config.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "lane_tuning_summary.json"

    summary: dict[str, Any] = {
        "type": "fsrs6-adr-direct-lane-tuning",
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "user_id": user_id,
        "candidate_lanes": lanes,
        "settings": asdict(settings),
        "simulation": config.simulation.to_dict(),
        "dry_run": args.dry_run,
        "results": [],
        "selected": None,
        "command": sys.argv,
    }
    if args.dry_run:
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    raw_training_sa = _read_training_sa(args.config)
    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_sa.get("learning_steps"),
        relearning_steps=raw_training_sa.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )

    for lanes_value in lanes:
        record = _run_lane_probe(
            config=config,
            settings=replace(settings, chains=lanes_value),
            user_id=user_id,
            lanes=lanes_value,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=args.benchmark_partition,
            button_usage=args.button_usage,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        summary["results"].append(record)
        _write_json(summary_path, summary)
        if record["status"] == "oom" and args.stop_after_oom:
            break

    summary["selected"] = _select_fastest_passed(summary["results"])
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selected"] is not None else 1


def _run_lane_probe(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    user_id: int,
    lanes: int,
    benchmark_root: Path,
    overrides: dict[str, str],
    benchmark_partition: str | None,
    button_usage: Path | None,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
) -> dict[str, Any]:
    device = torch.device(settings.torch_device)
    try:
        bundle = _build_bundle(
            config=config,
            settings=settings,
            user_id=user_id,
            lanes=lanes,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        coefficients = _baseline_coefficients(settings, lanes, bundle.device)
        if bundle.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(bundle.device)
            torch.cuda.synchronize(bundle.device)
        started = time.monotonic()
        metrics = _evaluate_sa_candidates(
            config=config,
            settings=settings,
            bundle=bundle,
            coefficients=coefficients,
            seed=config.seed,
        )
        if bundle.device.type == "cuda":
            torch.cuda.synchronize(bundle.device)
        elapsed = time.monotonic() - started
        candidate_days = config.simulation.days * lanes
        return {
            "lanes": lanes,
            "status": "passed",
            "elapsed_seconds": elapsed,
            "candidate_days": candidate_days,
            "candidate_days_per_second": candidate_days / max(elapsed, 1e-9),
            "device": str(bundle.device),
            "gpu": _gpu_snapshot(bundle.device),
            "metrics": _summarize_metrics(metrics),
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "lanes": lanes,
            "status": "oom",
            "error": str(exc),
            "device": str(device),
            "gpu": _gpu_snapshot(device),
        }


def _baseline_coefficients(
    settings: SASettings,
    lanes: int,
    device: torch.device,
) -> torch.Tensor:
    policy = FSRS6ADRDirectPolicy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
    )
    return torch.tensor(policy.coefficients, dtype=torch.float32, device=device).repeat(
        lanes, 1
    )


def _summarize_metrics(metrics: list[Any]) -> dict[str, float]:
    if not metrics:
        return {}
    count = float(len(metrics))
    return {
        "memorized_average_mean": sum(item.memorized_average for item in metrics)
        / count,
        "time_average_mean": sum(item.time_average for item in metrics) / count,
        "memorized_per_minute_mean": sum(item.memorized_per_minute for item in metrics)
        / count,
    }


def _gpu_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "current_allocated_memory_bytes": int(torch.cuda.memory_allocated(device)),
        "current_reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _parse_lane_values(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("--candidate-lanes values must be >= 1.")
        values.append(value)
    if not values:
        raise ValueError("--candidate-lanes must contain at least one value.")
    if len(set(values)) != len(values):
        raise ValueError("--candidate-lanes must not contain duplicates.")
    return values


def _select_fastest_passed(results: list[Any]) -> dict[str, Any] | None:
    passed = [
        item
        for item in results
        if isinstance(item, dict) and item.get("status") == "passed"
    ]
    if not passed:
        return None
    fastest = max(passed, key=lambda item: item["candidate_days_per_second"])
    return {
        "lanes": fastest["lanes"],
        "candidate_days_per_second": fastest["candidate_days_per_second"],
        "elapsed_seconds": fastest["elapsed_seconds"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

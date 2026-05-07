from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_fsrs6_adr_direct import (
    CandidateMetrics,
    SASettings,
    SimulationBundle,
    TrainingProgress,
    _build_bundle,
    _clamp_coefficients,
    _float_token,
    _git_commit,
    _int,
    _metrics_from_stats,
    _read_training_sa,
    _relative_gain,
    _score,
    _temperature,
    _write_json,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.math.fsrs import Bounds
from simulator.fsrs6_adr_direct_policy import FEATURE_VERSION, FSRS6ADRDirectPolicy
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.fsrs6_adr_direct import FSRS6ADRDirectBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser


@dataclass(frozen=True, slots=True)
class DRTrainingResult:
    baseline_desired_retention: float
    baseline: CandidateMetrics
    best_coefficients: torch.Tensor
    best: CandidateMetrics
    best_score: float
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train FSRS6 ADR Direct policies for a desired-retention grid inside one "
            "process using batch lanes."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL.",
    )
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument("--training-command-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress(output_dir / "training_progress.jsonl")
    progress.write(
        "started",
        config_path=str(args.config),
        user_id=args.user_id,
        lambda_value=args.lambda_value,
    )

    config = ExperimentConfig.from_toml(args.config)
    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = _read_training_sa(args.config)
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    progress.write(
        "config_loaded",
        settings=asdict(settings),
        simulation=config.simulation.to_dict(),
        seed=config.seed,
        baseline_desired_retention_values=list(baseline_dr_values),
        dr_batch_size=dr_batch_size,
    )

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
    device = torch.device(settings.torch_device)
    progress.write("device_resolved", device=device, torch_device=str(device))

    baseline_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=len(baseline_dr_values),
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baseline_metrics = _evaluate_fsrs6_baselines(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        baseline_dr_values=baseline_dr_values,
        seed=config.seed,
    )
    baselines_by_dr = dict(zip(baseline_dr_values, baseline_metrics, strict=True))
    progress.write(
        "baselines_evaluated",
        device=baseline_bundle.device,
        effective_lanes=len(baseline_dr_values),
        baseline_desired_retention_values=list(baseline_dr_values),
        metrics=[
            {
                "baseline_desired_retention": dr,
                **asdict(metrics),
            }
            for dr, metrics in baselines_by_dr.items()
        ],
    )
    del baseline_bundle
    _clear_cuda_cache(device)

    results: list[DRTrainingResult] = []
    for chunk_start, chunk_dr_values in _iter_chunks(baseline_dr_values, dr_batch_size):
        chunk_lanes = len(chunk_dr_values) * settings.chains
        train_bundle = _build_bundle(
            config=config,
            settings=settings,
            user_id=args.user_id,
            lanes=chunk_lanes,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=args.benchmark_partition,
            button_usage=args.button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        progress.write(
            "train_bundle_built",
            device=train_bundle.device,
            chunk_start=chunk_start,
            chunk_size=len(chunk_dr_values),
            baseline_desired_retention_values=list(chunk_dr_values),
            effective_lanes=chunk_lanes,
        )
        chunk_baselines = [baselines_by_dr[dr] for dr in chunk_dr_values]
        chunk_results = _anneal_dr_chunk(
            config=config,
            settings=settings,
            bundle=train_bundle,
            baseline_dr_values=chunk_dr_values,
            baselines=chunk_baselines,
            lambda_value=args.lambda_value,
            progress=progress,
            seed=config.seed + chunk_start,
        )
        results.extend(chunk_results)
        del train_bundle
        _clear_cuda_cache(device)

    artifact_paths = _write_grid_artifacts(
        output_dir=output_dir,
        config=config,
        config_path=args.config,
        settings=settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
        training_command_path=args.training_command_path,
        results=results,
    )
    passed_count = sum(1 for result in results if result.passed)
    progress.write(
        "artifacts_written",
        device=device,
        policies=len(results),
        passed_policies=passed_count,
        artifact_paths=[str(path) for path in artifact_paths],
    )
    return 0 if passed_count > 0 else 1


def _baseline_dr_values(
    raw_training_sa: Mapping[str, Any],
    settings: SASettings,
) -> tuple[float, ...]:
    raw_values = raw_training_sa.get("baseline_desired_retention_values")
    if raw_values is None:
        values = (settings.baseline_desired_retention,)
    else:
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise ValueError(
                "training.sa.baseline_desired_retention_values must be an array."
            )
        values = tuple(
            _float(item, "training.sa.baseline_desired_retention_values")
            for item in raw_values
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "training.sa.baseline_desired_retention_values must not contain duplicates."
        )
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "training.sa.baseline_desired_retention_values must be inside "
                "the retention bounds."
            )
    return values


def _dr_batch_size(raw_training_sa: Mapping[str, Any], value_count: int) -> int:
    default = min(max(value_count, 1), 4)
    return min(
        _int(
            raw_training_sa.get("dr_batch_size", default),
            "training.sa.dr_batch_size",
            1,
        ),
        value_count,
    )


def _evaluate_fsrs6_baselines(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    seed: int,
) -> list[CandidateMetrics]:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            baseline_dr_values,
            device=bundle.device,
            dtype=torch.float32,
        ),
        bounds=Bounds(),
        priority_mode=config.simulation.scheduler_priority,
        device=bundle.device,
        dtype=torch.float32,
    )
    stats = simulate_multiuser(
        days=config.simulation.days,
        deck_size=config.simulation.deck,
        env_ops=bundle.env_ops,
        sched_ops=sched_ops,
        behavior=bundle.behavior,
        cost_model=bundle.cost_model,
        seed=seed,
        device=bundle.device,
        dtype=torch.float32,
        fuzz=config.simulation.fuzz,
        priority_mode=config.simulation.priority,
        progress=False,
        short_term_source=bundle.short_term_source,
        learning_steps=bundle.learning_steps,
        relearning_steps=bundle.relearning_steps,
        short_term_threshold=settings.short_term_threshold,
        short_term_loops_limit=settings.short_term_loops_limit,
    )
    return [_metrics_from_stats(item) for item in stats]


def _anneal_dr_chunk(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    baselines: list[CandidateMetrics],
    lambda_value: float,
    progress: TrainingProgress,
    seed: int,
) -> list[DRTrainingResult]:
    device = bundle.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    current = _initial_coefficients(
        baseline_dr_values=baseline_dr_values,
        settings=settings,
        device=device,
        generator=generator,
    )
    lane_dr_indices = [
        dr_index
        for dr_index in range(len(baseline_dr_values))
        for _ in range(settings.chains)
    ]
    current_metrics = _evaluate_sa_candidates(
        config=config,
        settings=settings,
        bundle=bundle,
        coefficients=current,
        seed=seed,
    )
    current_scores = _candidate_scores(
        metrics=current_metrics,
        baselines=baselines,
        lane_dr_indices=lane_dr_indices,
        lambda_value=lambda_value,
        device=device,
    )
    best_coefficients, best_metrics, best_scores = _best_by_dr(
        coefficients=current,
        metrics=current_metrics,
        scores=current_scores,
        chains=settings.chains,
        dr_count=len(baseline_dr_values),
    )
    histories: list[list[dict[str, float]]] = [
        [] for _ in range(len(baseline_dr_values))
    ]
    progress.write(
        "initial_candidates_evaluated",
        device=device,
        baseline_desired_retention_values=list(baseline_dr_values),
        effective_lanes=current.shape[0],
        best_by_dr=_progress_best_by_dr(
            baseline_dr_values=baseline_dr_values,
            baselines=baselines,
            best_metrics=best_metrics,
            best_scores=best_scores,
        ),
    )

    for iteration in range(settings.iterations):
        temp = _temperature(settings, iteration)
        proposal = _clamp_coefficients(
            current
            + torch.randn(current.shape, device=device, generator=generator)
            * settings.proposal_scale,
            settings,
        )
        proposal_metrics = _evaluate_sa_candidates(
            config=config,
            settings=settings,
            bundle=bundle,
            coefficients=proposal,
            seed=seed,
        )
        proposal_scores = _candidate_scores(
            metrics=proposal_metrics,
            baselines=baselines,
            lane_dr_indices=lane_dr_indices,
            lambda_value=lambda_value,
            device=device,
        )
        delta = proposal_scores - current_scores
        accept_prob = torch.exp(delta / max(temp, 1e-9))
        accept = (delta >= 0) | (
            torch.rand(delta.shape, device=device, generator=generator) < accept_prob
        )
        accepted_count = int(accept.sum().item())
        if accept.any():
            current[accept] = proposal[accept]
            current_scores[accept] = proposal_scores[accept]
            for idx in torch.nonzero(accept, as_tuple=False).flatten().tolist():
                current_metrics[int(idx)] = proposal_metrics[int(idx)]

        for dr_index, baseline_dr in enumerate(baseline_dr_values):
            start = dr_index * settings.chains
            end = start + settings.chains
            local_idx = int(torch.argmax(current_scores[start:end]).item())
            flat_idx = start + local_idx
            score = float(current_scores[flat_idx].item())
            if score > best_scores[dr_index]:
                best_scores[dr_index] = score
                best_coefficients[dr_index] = current[flat_idx].detach().clone()
                best_metrics[dr_index] = current_metrics[flat_idx]
            history_entry = {
                "iteration": float(iteration),
                "temperature": float(temp),
                "best_score": best_scores[dr_index],
                "best_relative_memorized_gain": _relative_gain(
                    best_metrics[dr_index].memorized_average,
                    baselines[dr_index].memorized_average,
                ),
                "best_relative_efficiency_gain": _relative_gain(
                    best_metrics[dr_index].memorized_per_minute,
                    baselines[dr_index].memorized_per_minute,
                ),
                "baseline_desired_retention": float(baseline_dr),
            }
            histories[dr_index].append(history_entry)

        progress.write(
            "annealing_iteration",
            device=device,
            iteration=iteration,
            temperature=float(temp),
            accepted_count=accepted_count,
            baseline_desired_retention_values=list(baseline_dr_values),
            effective_lanes=current.shape[0],
            best_by_dr=_progress_best_by_dr(
                baseline_dr_values=baseline_dr_values,
                baselines=baselines,
                best_metrics=best_metrics,
                best_scores=best_scores,
            ),
        )

    results: list[DRTrainingResult] = []
    for dr_index, baseline_dr in enumerate(baseline_dr_values):
        rel_mem = _relative_gain(
            best_metrics[dr_index].memorized_average,
            baselines[dr_index].memorized_average,
        )
        rel_eff = _relative_gain(
            best_metrics[dr_index].memorized_per_minute,
            baselines[dr_index].memorized_per_minute,
        )
        results.append(
            DRTrainingResult(
                baseline_desired_retention=baseline_dr,
                baseline=baselines[dr_index],
                best_coefficients=best_coefficients[dr_index].detach().cpu(),
                best=best_metrics[dr_index],
                best_score=best_scores[dr_index],
                history=histories[dr_index],
                passed=rel_mem > 0.0 and rel_eff > 0.0,
            )
        )
    return results


def _initial_coefficients(
    *,
    baseline_dr_values: tuple[float, ...],
    settings: SASettings,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    base_coefficients = [
        FSRS6ADRDirectPolicy.baseline(
            desired_retention=baseline_dr,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
        ).coefficients
        for baseline_dr in baseline_dr_values
    ]
    current = torch.tensor(base_coefficients, dtype=torch.float32, device=device)
    current = current[:, None, :].repeat(1, settings.chains, 1)
    if settings.chains > 1:
        current[:, 1:, :] = _clamp_coefficients(
            current[:, 1:, :]
            + torch.randn(
                current[:, 1:, :].shape,
                device=device,
                generator=generator,
            )
            * settings.proposal_scale,
            settings,
        )
    return current.reshape(len(baseline_dr_values) * settings.chains, 6)


def _evaluate_sa_candidates(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    coefficients: torch.Tensor,
    seed: int,
) -> list[CandidateMetrics]:
    template = FSRS6ADRDirectPolicy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
    )
    sched_ops = FSRS6ADRDirectBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=template,
        coefficients=coefficients,
        bounds=Bounds(),
        priority_mode=config.simulation.scheduler_priority,
        device=bundle.device,
        dtype=torch.float32,
    )
    stats = simulate_multiuser(
        days=config.simulation.days,
        deck_size=config.simulation.deck,
        env_ops=bundle.env_ops,
        sched_ops=sched_ops,
        behavior=bundle.behavior,
        cost_model=bundle.cost_model,
        seed=seed,
        device=bundle.device,
        dtype=torch.float32,
        fuzz=config.simulation.fuzz,
        priority_mode=config.simulation.priority,
        progress=False,
        short_term_source=bundle.short_term_source,
        learning_steps=bundle.learning_steps,
        relearning_steps=bundle.relearning_steps,
        short_term_threshold=settings.short_term_threshold,
        short_term_loops_limit=settings.short_term_loops_limit,
    )
    return [_metrics_from_stats(item) for item in stats]


def _candidate_scores(
    *,
    metrics: list[CandidateMetrics],
    baselines: list[CandidateMetrics],
    lane_dr_indices: list[int],
    lambda_value: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            _score(metrics[index], baselines[lane_dr_indices[index]], lambda_value)
            for index in range(len(metrics))
        ],
        device=device,
        dtype=torch.float32,
    )


def _best_by_dr(
    *,
    coefficients: torch.Tensor,
    metrics: list[CandidateMetrics],
    scores: torch.Tensor,
    chains: int,
    dr_count: int,
) -> tuple[list[torch.Tensor], list[CandidateMetrics], list[float]]:
    best_coefficients: list[torch.Tensor] = []
    best_metrics: list[CandidateMetrics] = []
    best_scores: list[float] = []
    for dr_index in range(dr_count):
        start = dr_index * chains
        end = start + chains
        local_idx = int(torch.argmax(scores[start:end]).item())
        flat_idx = start + local_idx
        best_coefficients.append(coefficients[flat_idx].detach().clone())
        best_metrics.append(metrics[flat_idx])
        best_scores.append(float(scores[flat_idx].item()))
    return best_coefficients, best_metrics, best_scores


def _write_grid_artifacts(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: SASettings,
    user_id: int,
    lambda_value: float,
    training_command_path: Path | None,
    results: list[DRTrainingResult],
) -> list[Path]:
    artifact_paths: list[Path] = []
    grid_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "policy_count": len(results),
        "passed_policy_count": sum(1 for result in results if result.passed),
        "baseline_desired_retention_values": [
            result.baseline_desired_retention for result in results
        ],
        "results": [],
    }
    for result in results:
        dr = result.baseline_desired_retention
        dr_token = _float_token(dr)
        result_dir = output_dir / f"dr_{dr_token}"
        result_dir.mkdir(parents=True, exist_ok=True)
        policy = FSRS6ADRDirectPolicy(
            coefficients=tuple(float(v) for v in result.best_coefficients.tolist()),
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            baseline_desired_retention=dr,
            title=f"fsrs6_adr_direct_u{user_id}_dr_{dr:.2f}_lambda_{lambda_value:g}",
        )
        policy_path = result_dir / "policy.json"
        policy.write_json(policy_path)

        rel_mem = _relative_gain(
            result.best.memorized_average,
            result.baseline.memorized_average,
        )
        rel_eff = _relative_gain(
            result.best.memorized_per_minute,
            result.baseline.memorized_per_minute,
        )
        effective_settings = {
            **asdict(settings),
            "baseline_desired_retention": dr,
        }
        metrics = {
            "passed_overfit_gate": result.passed,
            "gate": {
                "memorized_average_gt_baseline": rel_mem > 0.0,
                "memorized_per_minute_gt_baseline": rel_eff > 0.0,
                "relative_memorized_gain": rel_mem,
                "relative_efficiency_gain": rel_eff,
            },
            "baseline": asdict(result.baseline),
            "best": asdict(result.best),
            "best_score": result.best_score,
            "settings": effective_settings,
            "history": result.history,
        }
        metrics_path = result_dir / "metrics.json"
        _write_json(metrics_path, metrics)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "scheduler-policy",
            "artifact_id": _artifact_id(user_id, lambda_value, dr, config.seed),
            "family": config.family,
            "scheduler_name": "fsrs6_adr_direct",
            "environment": config.simulation.environment,
            "engine": config.simulation.engine,
            "training_user_ids": [user_id],
            "validation_user_ids": list(config.users.validation),
            "seed": config.seed,
            "policy_path": "policy.json",
            "feature_version": FEATURE_VERSION,
            "action_space": "sd_retention_function",
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "code_commit": _git_commit(),
            "lambda_value": lambda_value,
            "baseline_desired_retention": dr,
            "config_snapshot_path": str(config_path.resolve()),
            "training_command_path": str(training_command_path)
            if training_command_path
            else None,
            "metrics_path": "metrics.json",
            "capabilities": ["event", "vectorized", "batched"],
        }
        metadata_path = result_dir / "metadata.json"
        _write_json(metadata_path, metadata)
        artifact_paths.append(metadata_path)
        grid_summary["results"].append(
            {
                "baseline_desired_retention": dr,
                "passed_overfit_gate": result.passed,
                "relative_memorized_gain": rel_mem,
                "relative_efficiency_gain": rel_eff,
                "policy_path": str(policy_path),
                "metadata_path": str(metadata_path),
                "metrics_path": str(metrics_path),
            }
        )
    _write_json(output_dir / "grid_metrics.json", grid_summary)
    return artifact_paths


def _progress_best_by_dr(
    *,
    baseline_dr_values: tuple[float, ...],
    baselines: list[CandidateMetrics],
    best_metrics: list[CandidateMetrics],
    best_scores: list[float],
) -> list[dict[str, float]]:
    return [
        {
            "baseline_desired_retention": float(baseline_dr),
            "best_score": best_scores[index],
            "best_relative_memorized_gain": _relative_gain(
                best_metrics[index].memorized_average,
                baselines[index].memorized_average,
            ),
            "best_relative_efficiency_gain": _relative_gain(
                best_metrics[index].memorized_per_minute,
                baselines[index].memorized_per_minute,
            ),
        }
        for index, baseline_dr in enumerate(baseline_dr_values)
    ]


def _iter_chunks(
    values: tuple[float, ...],
    chunk_size: int,
) -> Sequence[tuple[int, tuple[float, ...]]]:
    return [
        (start, values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]


def _artifact_id(
    user_id: int,
    lambda_value: float,
    baseline_desired_retention: float,
    seed: int,
) -> str:
    lambda_token = _float_token(lambda_value)
    dr_token = _float_token(baseline_desired_retention)
    return f"fsrs6-adr-direct-user-{user_id}-dr-{dr_token}-lambda-{lambda_token}-seed-{seed}"


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must contain only numbers.")
    return float(value)


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

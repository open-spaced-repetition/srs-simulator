from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

import cma
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.train_cmaes_fsrs6_adr_delta import (
    CMAESSettings,
    _optimizer_seed,
)
from experiments.rl_scheduler.train_fsrs6_adr_direct import (
    CandidateMetrics,
    SASettings,
    TrainingProgress,
    _artifact_id as _adr_artifact_id,
    _build_bundle,
    _git_commit,
    _metrics_from_stats,
    _passes_overfit_gate,
    _read_training_sa,
    _relative_gain,
    _relative_gain_gate_metrics,
    _score,
    _write_json,
)
from experiments.rl_scheduler.train_fsrs6_adr_direct_dr_grid import (
    _baseline_dr_values,
    _float_token,
    _iter_chunks,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_adp_policy import (
    DEFAULT_WEIGHT_DELTA_SCALE,
    FEATURE_VERSION,
    FSRS6_ADP_DEFAULT_STDDEV,
    FSRS6_ADP_WEIGHT_BOUNDS,
    FSRS6ADPPolicy,
    WEIGHT_COUNT,
)
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser


@dataclass(frozen=True, slots=True)
class ADPSettings:
    dr_batch_size: int
    weight_delta_scale: float = DEFAULT_WEIGHT_DELTA_SCALE

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        *,
        raw_training_sa: Mapping[str, Any],
        dr_count: int,
    ) -> ADPSettings:
        raw = dict(config.training_adp)
        raw_dr_batch_size = raw.get(
            "dr_batch_size", raw_training_sa.get("dr_batch_size")
        )
        if raw_dr_batch_size is None:
            dr_batch_size = dr_count
        else:
            dr_batch_size = _int(raw_dr_batch_size, "training.adp.dr_batch_size", 1)
            dr_batch_size = min(dr_batch_size, dr_count)
        return cls(
            dr_batch_size=max(1, dr_batch_size),
            weight_delta_scale=_float(
                raw.get("weight_delta_scale", DEFAULT_WEIGHT_DELTA_SCALE),
                "training.adp.weight_delta_scale",
                0.0,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dr_batch_size": self.dr_batch_size,
            "weight_delta_scale": self.weight_delta_scale,
        }


@dataclass(frozen=True, slots=True)
class ADPTrainJob:
    user_id: int
    lambda_value: float
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ADPTrainingResult:
    baseline_desired_retention: float
    baseline: CandidateMetrics
    best: CandidateMetrics
    best_score: float
    base_weights: tuple[float, ...]
    best_weights: tuple[float, ...]
    best_delta: tuple[float, ...]
    best_search_vector: tuple[float, ...]
    clipped_dimensions: int
    history: list[dict[str, float]]
    passed: bool
    optimizer_seed: int


@dataclass(frozen=True, slots=True)
class ADPTrainJobResult:
    job: ADPTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path | None


@dataclass(frozen=True, slots=True)
class _CommonContext:
    raw_training_sa: dict[str, Any]
    short_term_source: str | None
    learning_steps: list[float]
    relearning_steps: list[float]
    device: torch.device
    benchmark_root: Path
    overrides: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FSRS6 ADP adaptive-parameter scheduler policies with CMA-ES.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-command-path", type=Path, default=None)
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)
    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = dict(_read_training_sa(args.config))
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    adp_settings = ADPSettings.from_config(
        config,
        raw_training_sa=raw_training_sa,
        dr_count=len(baseline_dr_values),
    )
    optimizer_settings = optimizer_settings_from_mapping(config.training_optimizer)
    ctx = _common_context(
        config=config,
        config_path=args.config,
        repo_root=REPO_ROOT,
        settings=settings,
        srs_benchmark_root=args.srs_benchmark_root,
        benchmark_result=args.benchmark_result,
    )
    job = ADPTrainJob(
        user_id=args.user_id,
        lambda_value=args.lambda_value,
        output_dir=args.output_dir,
        command_record_path=args.training_command_path,
    )
    results = run_training_jobs(
        jobs=[job],
        config=config,
        config_path=args.config,
        repo_root=REPO_ROOT,
        settings=settings,
        adp_settings=adp_settings,
        optimizer_settings=optimizer_settings,
        ctx=ctx,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        baseline_dr_values=baseline_dr_values,
    )
    return 0 if results and results[0].passed else 1


def optimizer_settings_from_mapping(raw: Mapping[str, Any]) -> CMAESSettings:
    return CMAESSettings.from_mapping(
        raw,
        coefficient_count=WEIGHT_COUNT,
        coefficient_min=-2.0,
        coefficient_max=2.0,
    )


def run_training_batch_jobs(
    *,
    jobs: Sequence[ADPTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[ADPTrainJobResult]:
    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = dict(_read_training_sa(config_path))
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    adp_settings = ADPSettings.from_config(
        config,
        raw_training_sa=raw_training_sa,
        dr_count=len(baseline_dr_values),
    )
    optimizer_settings = optimizer_settings_from_mapping(config.training_optimizer)
    ctx = _common_context(
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        settings=settings,
        srs_benchmark_root=None,
        benchmark_result=None,
    )
    return run_training_jobs(
        jobs=jobs,
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        settings=settings,
        adp_settings=adp_settings,
        optimizer_settings=optimizer_settings,
        ctx=ctx,
        benchmark_partition=None,
        button_usage=DEFAULT_BUTTON_USAGE_PATH,
        baseline_dr_values=baseline_dr_values,
    )


def run_training_jobs(
    *,
    jobs: Sequence[ADPTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    settings: SASettings,
    adp_settings: ADPSettings,
    optimizer_settings: CMAESSettings,
    ctx: _CommonContext,
    benchmark_partition: str | None,
    button_usage: Path | None,
    baseline_dr_values: tuple[float, ...],
) -> list[ADPTrainJobResult]:
    if not jobs:
        return []
    progresses = _progress_for_jobs(jobs=jobs, config_path=config_path)
    for job, progress in zip(jobs, progresses, strict=True):
        optimizer_seed = _optimizer_seed(
            config=config,
            settings=optimizer_settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            adp=adp_settings.to_dict(),
            optimizer=optimizer_settings.to_dict(),
            optimizer_seed=optimizer_seed,
            feature_version=FEATURE_VERSION,
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            baseline_desired_retention_values=list(baseline_dr_values),
        )
        progress.write(
            "device_resolved",
            device=ctx.device,
            torch_device=str(ctx.device),
        )

    baseline_bundle, baselines_by_job = _evaluate_baseline_grid(
        config=config,
        settings=settings,
        jobs=jobs,
        baseline_dr_values=baseline_dr_values,
        ctx=ctx,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
    )
    for baselines, progress in zip(baselines_by_job, progresses, strict=True):
        progress.write(
            "baselines_evaluated",
            device=baseline_bundle.device,
            effective_lanes=len(baseline_dr_values),
            batch_effective_lanes=len(jobs) * len(baseline_dr_values),
            metrics=[
                {"baseline_desired_retention": dr, **asdict(metric)}
                for dr, metric in zip(baseline_dr_values, baselines, strict=True)
            ],
        )
    del baseline_bundle
    _clear_cuda_cache(ctx.device)

    baselines_by_job_dr = [
        dict(zip(baseline_dr_values, baselines, strict=True))
        for baselines in baselines_by_job
    ]
    results_by_job: list[list[ADPTrainingResult]] = [[] for _job in jobs]

    for chunk_start, chunk_dr_values in _iter_chunks(
        baseline_dr_values,
        adp_settings.dr_batch_size,
    ):
        lane_user_ids = [
            job.user_id
            for job in jobs
            for _dr in chunk_dr_values
            for _candidate in range(optimizer_settings.population_size)
        ]
        train_bundle = _build_bundle(
            config=config,
            settings=settings,
            lane_user_ids=lane_user_ids,
            benchmark_root=ctx.benchmark_root,
            overrides=ctx.overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=ctx.device,
            short_term_source=ctx.short_term_source,
            learning_steps=ctx.learning_steps,
            relearning_steps=ctx.relearning_steps,
        )
        for progress in progresses:
            progress.write(
                "train_bundle_built",
                device=train_bundle.device,
                chunk_start=chunk_start,
                chunk_size=len(chunk_dr_values),
                baseline_desired_retention_values=list(chunk_dr_values),
                effective_lanes=len(chunk_dr_values)
                * optimizer_settings.population_size,
                batch_effective_lanes=len(lane_user_ids),
            )

        optimizer_seeds_by_job_dr: list[list[int]] = []
        strategies = []
        for job in jobs:
            base_seed = _optimizer_seed(
                config=config,
                settings=optimizer_settings,
                user_id=job.user_id,
                lambda_value=job.lambda_value,
            )
            job_seeds = [
                _dr_optimizer_seed(base_seed, chunk_start + dr_index)
                for dr_index, _dr in enumerate(chunk_dr_values)
            ]
            optimizer_seeds_by_job_dr.append(job_seeds)
            strategies.append(
                [
                    _make_strategy(
                        optimizer_settings=optimizer_settings,
                        seed=seed,
                    )
                    for seed in job_seeds
                ]
            )
        histories: list[list[list[dict[str, float]]]] = [
            [[] for _dr in chunk_dr_values] for _job in jobs
        ]
        best_vectors: list[list[torch.Tensor | None]] = [
            [None for _dr in chunk_dr_values] for _job in jobs
        ]
        best_weights: list[list[tuple[float, ...] | None]] = [
            [None for _dr in chunk_dr_values] for _job in jobs
        ]
        best_metrics: list[list[CandidateMetrics | None]] = [
            [None for _dr in chunk_dr_values] for _job in jobs
        ]
        best_scores: list[list[float]] = [
            [float("-inf") for _dr in chunk_dr_values] for _job in jobs
        ]

        for generation in range(optimizer_settings.generations):
            solutions_by_job_dr: list[list[list[list[float]]]] = []
            for job_index, job in enumerate(jobs):
                job_solutions = []
                for dr_index, _baseline_dr in enumerate(chunk_dr_values):
                    solutions = [
                        list(map(float, item))
                        for item in strategies[job_index][dr_index].ask()
                    ]
                    if len(solutions) != optimizer_settings.population_size:
                        raise RuntimeError(
                            "CMA-ES returned an unexpected population size: "
                            f"{len(solutions)} != "
                            f"{optimizer_settings.population_size}."
                        )
                    if generation == 0:
                        solutions[0] = list(optimizer_settings.initial_mean)
                    job_solutions.append(solutions)
                solutions_by_job_dr.append(job_solutions)

            search_vectors = torch.tensor(
                solutions_by_job_dr,
                device=train_bundle.device,
                dtype=torch.float32,
            )
            metrics_by_job_dr, weights_by_job_dr = _evaluate_adp_candidates(
                config=config,
                settings=settings,
                adp_settings=adp_settings,
                bundle=train_bundle,
                jobs=jobs,
                chunk_dr_values=chunk_dr_values,
                search_vectors=search_vectors,
                seed=config.seed + chunk_start,
            )

            for job_index, job in enumerate(jobs):
                for dr_index, baseline_dr in enumerate(chunk_dr_values):
                    baseline = baselines_by_job_dr[job_index][baseline_dr]
                    metrics = metrics_by_job_dr[job_index][dr_index]
                    scores = [
                        _score(metric, baseline, job.lambda_value) for metric in metrics
                    ]
                    strategies[job_index][dr_index].tell(
                        solutions_by_job_dr[job_index][dr_index],
                        [-score for score in scores],
                    )
                    generation_best_idx = max(
                        range(len(scores)),
                        key=scores.__getitem__,
                    )
                    generation_best = metrics[generation_best_idx]
                    generation_best_score = float(scores[generation_best_idx])
                    if generation_best_score > best_scores[job_index][dr_index]:
                        best_scores[job_index][dr_index] = generation_best_score
                        best_vectors[job_index][dr_index] = (
                            search_vectors[job_index, dr_index, generation_best_idx]
                            .detach()
                            .clone()
                        )
                        best_weights[job_index][dr_index] = tuple(
                            float(value)
                            for value in weights_by_job_dr[job_index][dr_index][
                                generation_best_idx
                            ]
                        )
                        best_metrics[job_index][dr_index] = generation_best
                    history_entry = {
                        "generation": float(generation),
                        "baseline_desired_retention": float(baseline_dr),
                        "sigma": float(strategies[job_index][dr_index].sigma),
                        "best_score": float(best_scores[job_index][dr_index]),
                        "generation_best_score": generation_best_score,
                        "mean_score": float(sum(scores) / max(len(scores), 1)),
                        "generation_best_relative_memorized_gain": _relative_gain(
                            generation_best.memorized_average,
                            baseline.memorized_average,
                        ),
                        "generation_best_relative_efficiency_gain": _relative_gain(
                            generation_best.memorized_per_minute,
                            baseline.memorized_per_minute,
                        ),
                    }
                    histories[job_index][dr_index].append(history_entry)

                progresses[job_index].write(
                    "cmaes_generation",
                    device=train_bundle.device,
                    generation=generation,
                    baseline_desired_retention_values=list(chunk_dr_values),
                    effective_lanes=len(chunk_dr_values)
                    * optimizer_settings.population_size,
                    batch_effective_lanes=len(lane_user_ids),
                    best_by_dr=[
                        {
                            "baseline_desired_retention": float(dr),
                            "best_score": best_scores[job_index][dr_index],
                        }
                        for dr_index, dr in enumerate(chunk_dr_values)
                    ],
                )

        base_weights_by_job = _base_weights_by_job(
            bundle=train_bundle,
            jobs=jobs,
            chunk_dr_values=chunk_dr_values,
            population_size=optimizer_settings.population_size,
        )
        for job_index, _job in enumerate(jobs):
            for dr_index, baseline_dr in enumerate(chunk_dr_values):
                vector = best_vectors[job_index][dr_index]
                weights = best_weights[job_index][dr_index]
                metric = best_metrics[job_index][dr_index]
                if vector is None or weights is None or metric is None:
                    raise RuntimeError("CMA-ES did not evaluate any ADP candidates.")
                baseline = baselines_by_job_dr[job_index][baseline_dr]
                base_weights = base_weights_by_job[job_index]
                delta = tuple(
                    weight - base_weight
                    for weight, base_weight in zip(weights, base_weights)
                )
                rel_mem = _relative_gain(
                    metric.memorized_average,
                    baseline.memorized_average,
                )
                rel_eff = _relative_gain(
                    metric.memorized_per_minute,
                    baseline.memorized_per_minute,
                )
                results_by_job[job_index].append(
                    ADPTrainingResult(
                        baseline_desired_retention=baseline_dr,
                        baseline=baseline,
                        best=metric,
                        best_score=best_scores[job_index][dr_index],
                        base_weights=base_weights,
                        best_weights=weights,
                        best_delta=delta,
                        best_search_vector=tuple(
                            float(value) for value in vector.detach().cpu().tolist()
                        ),
                        clipped_dimensions=_clipped_dimension_count(
                            base_weights=base_weights,
                            search_vector=tuple(
                                float(value) for value in vector.detach().cpu().tolist()
                            ),
                            weights=weights,
                            weight_delta_scale=adp_settings.weight_delta_scale,
                        ),
                        history=histories[job_index][dr_index],
                        passed=_passes_overfit_gate(rel_mem, rel_eff),
                        optimizer_seed=optimizer_seeds_by_job_dr[job_index][dr_index],
                    )
                )

    outcomes: list[ADPTrainJobResult] = []
    for job_index, job in enumerate(jobs):
        artifact_paths = _write_grid_artifacts(
            output_dir=job.output_dir,
            config=config,
            config_path=config_path,
            settings=settings,
            adp_settings=adp_settings,
            optimizer_settings=optimizer_settings,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
            training_command_path=job.command_record_path,
            results=results_by_job[job_index],
        )
        passed_count = sum(1 for result in results_by_job[job_index] if result.passed)
        progresses[job_index].write(
            "artifacts_written",
            device=ctx.device,
            policies=len(results_by_job[job_index]),
            passed_policies=passed_count,
            artifact_paths=[str(path) for path in artifact_paths],
        )
        outcomes.append(
            ADPTrainJobResult(
                job=job,
                passed=passed_count > 0,
                artifact_paths=tuple(artifact_paths),
                progress_path=progresses[job_index].path,
            )
        )
    return outcomes


def _common_context(
    *,
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    settings: SASettings,
    srs_benchmark_root: Path | None,
    benchmark_result: str | None,
) -> _CommonContext:
    raw_training_sa = dict(_read_training_sa(config_path))
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_sa.get("learning_steps"),
        relearning_steps=raw_training_sa.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    return _CommonContext(
        raw_training_sa=raw_training_sa,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
        device=torch.device(settings.torch_device),
        benchmark_root=resolve_benchmark_root(repo_root, srs_benchmark_root).resolve(),
        overrides=parse_result_overrides(benchmark_result),
    )


def _progress_for_jobs(
    *,
    jobs: Sequence[ADPTrainJob],
    config_path: Path,
) -> list[TrainingProgress]:
    progresses: list[TrainingProgress] = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(job.output_dir / "training_progress.jsonl")
        progress.write(
            "started",
            config_path=str(config_path),
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        progresses.append(progress)
    return progresses


def _evaluate_baseline_grid(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    jobs: Sequence[ADPTrainJob],
    baseline_dr_values: tuple[float, ...],
    ctx: _CommonContext,
    benchmark_partition: str | None,
    button_usage: Path | None,
) -> tuple[Any, list[list[CandidateMetrics]]]:
    lane_user_ids = [job.user_id for job in jobs for _dr in baseline_dr_values]
    bundle = _build_bundle(
        config=config,
        settings=settings,
        lane_user_ids=lane_user_ids,
        benchmark_root=ctx.benchmark_root,
        overrides=ctx.overrides,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
        device=ctx.device,
        short_term_source=ctx.short_term_source,
        learning_steps=ctx.learning_steps,
        relearning_steps=ctx.relearning_steps,
    )
    desired = torch.tensor(
        [dr for _job in jobs for dr in baseline_dr_values],
        device=bundle.device,
        dtype=torch.float32,
    )
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=desired,
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
        seed=config.seed,
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
    metrics = [_metrics_from_stats(item) for item in stats]
    by_job = [
        metrics[index * len(baseline_dr_values) : (index + 1) * len(baseline_dr_values)]
        for index in range(len(jobs))
    ]
    return bundle, by_job


def _evaluate_adp_candidates(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    adp_settings: ADPSettings,
    bundle: Any,
    jobs: Sequence[ADPTrainJob],
    chunk_dr_values: tuple[float, ...],
    search_vectors: torch.Tensor,
    seed: int,
) -> tuple[list[list[list[CandidateMetrics]]], list[list[list[tuple[float, ...]]]]]:
    job_count = len(jobs)
    dr_count = len(chunk_dr_values)
    population_size = int(search_vectors.shape[2])
    flat_vectors = search_vectors.reshape(job_count * dr_count * population_size, -1)
    scheduler_weights = _decode_weight_delta_tensor(
        base_weights=bundle.scheduler_weights,
        search_vectors=flat_vectors,
        weight_delta_scale=adp_settings.weight_delta_scale,
    )
    desired = torch.tensor(
        [
            dr
            for _job in jobs
            for dr in chunk_dr_values
            for _candidate in range(population_size)
        ],
        device=bundle.device,
        dtype=torch.float32,
    )
    sched_ops = FSRS6BatchSchedulerOps(
        weights=scheduler_weights,
        desired_retention=desired,
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
    flat_metrics = [_metrics_from_stats(item) for item in stats]
    flat_weights = [
        tuple(float(value) for value in row)
        for row in scheduler_weights.detach().cpu().tolist()
    ]
    metrics_by_job_dr: list[list[list[CandidateMetrics]]] = []
    weights_by_job_dr: list[list[list[tuple[float, ...]]]] = []
    for job_index in range(job_count):
        job_metrics = []
        job_weights = []
        for dr_index in range(dr_count):
            start = (job_index * dr_count + dr_index) * population_size
            end = start + population_size
            job_metrics.append(flat_metrics[start:end])
            job_weights.append(flat_weights[start:end])
        metrics_by_job_dr.append(job_metrics)
        weights_by_job_dr.append(job_weights)
    return metrics_by_job_dr, weights_by_job_dr


def _decode_weight_delta_tensor(
    *,
    base_weights: torch.Tensor,
    search_vectors: torch.Tensor,
    weight_delta_scale: float,
) -> torch.Tensor:
    stddev = torch.tensor(
        FSRS6_ADP_DEFAULT_STDDEV,
        device=base_weights.device,
        dtype=base_weights.dtype,
    )
    lower = torch.tensor(
        [item[0] for item in FSRS6_ADP_WEIGHT_BOUNDS],
        device=base_weights.device,
        dtype=base_weights.dtype,
    )
    upper = torch.tensor(
        [item[1] for item in FSRS6_ADP_WEIGHT_BOUNDS],
        device=base_weights.device,
        dtype=base_weights.dtype,
    )
    raw = base_weights + search_vectors.to(base_weights.dtype) * stddev * float(
        weight_delta_scale
    )
    return torch.clamp(raw, min=lower, max=upper)


def _base_weights_by_job(
    *,
    bundle: Any,
    jobs: Sequence[ADPTrainJob],
    chunk_dr_values: tuple[float, ...],
    population_size: int,
) -> list[tuple[float, ...]]:
    weights = bundle.scheduler_weights.detach().cpu()
    lanes_per_job = len(chunk_dr_values) * population_size
    return [
        tuple(float(value) for value in weights[job_index * lanes_per_job].tolist())
        for job_index in range(len(jobs))
    ]


def _make_strategy(
    *,
    optimizer_settings: CMAESSettings,
    seed: int,
) -> Any:
    return cma.CMAEvolutionStrategy(
        list(optimizer_settings.initial_mean),
        optimizer_settings.sigma0,
        {
            "bounds": [
                list(optimizer_settings.bounds[0]),
                list(optimizer_settings.bounds[1]),
            ],
            "popsize": optimizer_settings.population_size,
            "seed": seed,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
        },
    )


def _dr_optimizer_seed(base_seed: int, dr_index: int) -> int:
    return int((base_seed + 10_007 * dr_index) % (2**32 - 1))


def _write_grid_artifacts(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: SASettings,
    adp_settings: ADPSettings,
    optimizer_settings: CMAESSettings,
    user_id: int,
    lambda_value: float,
    training_command_path: Path | None,
    results: Sequence[ADPTrainingResult],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    grid_summary: dict[str, Any] = {
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
        optimizer = {
            **optimizer_settings.to_dict(),
            "seed_resolved": result.optimizer_seed,
        }
        policy = FSRS6ADPPolicy(
            base_weights=result.base_weights,
            weights=result.best_weights,
            delta=result.best_delta,
            search_vector=result.best_search_vector,
            baseline_desired_retention=dr,
            weight_delta_scale=adp_settings.weight_delta_scale,
            title=f"fsrs6_adp_u{user_id}_dr_{dr:.2f}_lambda_{lambda_value:g}",
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
        metrics_path = result_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "passed_overfit_gate": result.passed,
                "gate": _relative_gain_gate_metrics(rel_mem, rel_eff),
                "baseline": asdict(result.baseline),
                "best": asdict(result.best),
                "best_score": result.best_score,
                "feature_version": FEATURE_VERSION,
                "optimizer": optimizer,
                "settings": asdict(settings),
                "adp": adp_settings.to_dict(),
                "base_weights": list(result.base_weights),
                "best_weights": list(result.best_weights),
                "best_delta": list(result.best_delta),
                "best_search_vector": list(result.best_search_vector),
                "clipped_dimensions": result.clipped_dimensions,
                "history": result.history,
            },
        )
        metadata_path = result_dir / "metadata.json"
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "scheduler-policy",
                "artifact_id": _artifact_id(
                    user_id,
                    lambda_value,
                    dr,
                    config.seed,
                ),
                "family": config.family,
                "scheduler_name": "fsrs6_adp",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": FEATURE_VERSION,
                "action_space": "fsrs6_weight_delta",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "lambda_value": lambda_value,
                "baseline_desired_retention": dr,
                "config_snapshot_path": str(config_path.resolve()),
                "training_command_path": str(training_command_path)
                if training_command_path
                else None,
                "metrics_path": "metrics.json",
                "optimizer": "cma_es",
                "optimizer_settings": optimizer,
                "capabilities": ["event", "vectorized", "batched"],
            },
        )
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
                "clipped_dimensions": result.clipped_dimensions,
                "optimizer_seed": result.optimizer_seed,
            }
        )
    _write_json(output_dir / "grid_metrics.json", grid_summary)
    return artifact_paths


def _artifact_id(
    user_id: int,
    lambda_value: float,
    baseline_desired_retention: float,
    seed: int,
) -> str:
    return _adr_artifact_id(
        user_id,
        lambda_value,
        baseline_desired_retention,
        seed,
    ).replace("fsrs6-adr-direct", "fsrs6-adp")


def _clipped_dimension_count(
    *,
    base_weights: tuple[float, ...],
    search_vector: tuple[float, ...],
    weights: tuple[float, ...],
    weight_delta_scale: float,
) -> int:
    count = 0
    for base, offset, stddev, actual, (lower, upper) in zip(
        base_weights,
        search_vector,
        FSRS6_ADP_DEFAULT_STDDEV,
        weights,
        FSRS6_ADP_WEIGHT_BOUNDS,
    ):
        raw = base + offset * stddev * weight_delta_scale
        if raw < lower or raw > upper or abs(raw - actual) > 1e-6:
            count += 1
    return count


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _float(value: Any, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return result


def _int(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

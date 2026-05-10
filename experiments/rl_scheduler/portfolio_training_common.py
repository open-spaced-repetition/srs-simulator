from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    TrainingProgress,
    _baseline_dr_values,
    _build_bundle,
    _evaluate_fsrs6_baseline_grid,
    _float,
    _read_training_policy_search,
)
from experiments.rl_scheduler.portfolio_selection import (
    DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    LightweightSelectionPool,
    ObjectivePoint,
    SelectionPayload,
    SelectionTask,
    frontier_candidate_count,
    objective_exclusive_hypervolume_contributions,
    objective_hypervolume_2d,
    point_from_metrics,
    reference_point,
    selection_executor as common_selection_executor,
    selection_payload_from_objective_candidates,
    selection_process_pool_enabled as common_selection_process_pool_enabled,
    selection_process_pool_worker_count as common_selection_process_pool_worker_count,
    select_portfolio_child_indices,
    select_survivors_for_generation,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig
from simulator.short_term_config import resolve_short_term_config


SELECTION_PROCESS_POOL_ENV = "FSRS6_PORTFOLIO_SELECTION_PROCESS_POOL"
SELECTION_PROCESS_POOL_WORKERS_ENV = "FSRS6_PORTFOLIO_SELECTION_WORKERS"


@dataclass(frozen=True, slots=True)
class PortfolioFamilyAdapter:
    settings_from_mapping: Callable[..., Any]
    build_family_context: Callable[..., Any]
    progress_payload: Callable[..., Mapping[str, Any]]
    initial_populations: Callable[..., tuple[list[list[Any]], list[int]]]
    prepare_family_state: Callable[..., Any]
    evaluate_candidates: Callable[..., list[list[Any]]]
    make_offspring: Callable[..., tuple[list[Any], int]]
    selected_child_from_candidate: Callable[..., Any]
    build_result: Callable[..., Any]
    write_artifacts: Callable[..., list[Path]]
    build_outcome: Callable[..., Any]
    selection_enabled_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_ENV,)
    selection_worker_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_WORKERS_ENV,)
    selection_default_min_jobs: int = DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS
    selection_default_workers: int = DEFAULT_SELECTION_PROCESS_POOL_WORKERS


def run_portfolio_train_jobs(
    *,
    jobs: Sequence[Any],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    adapter: PortfolioFamilyAdapter,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[Any]:
    if not jobs:
        return []

    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search = dict(_read_training_policy_search(config_path))
    baseline_dr_values = _baseline_dr_values(raw_training_policy_search, settings)
    family_context = adapter.build_family_context(
        config=config,
        raw_training_policy_search=raw_training_policy_search,
        baseline_dr_values=baseline_dr_values,
    )
    portfolio = adapter.settings_from_mapping(
        config.training_portfolio,
        settings=settings,
        default_seed_retention_values=baseline_dr_values,
    )
    device = torch.device(settings.torch_device)
    benchmark_root = resolve_benchmark_root(repo_root, srs_benchmark_root).resolve()
    overrides = parse_result_overrides(benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_policy_search.get("learning_steps"),
        relearning_steps=raw_training_policy_search.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )

    progresses = progress_for_jobs(
        jobs=jobs,
        config_path=config_path,
        execution_mode=execution_mode,
    )
    for progress, job in zip(progresses, jobs, strict=True):
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            portfolio=asdict(portfolio),
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            user_id=job.user_id,
            **dict(adapter.progress_payload(family_context=family_context)),
        )
        progress.write("device_resolved", device=device, torch_device=str(device))

    baseline_bundle = _build_bundle(
        config=config,
        settings=settings,
        lane_user_ids=[job.user_id for job in jobs for _dr in baseline_dr_values],
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baseline_metrics_by_job = _evaluate_fsrs6_baseline_grid(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        baseline_dr_values=baseline_dr_values,
        job_count=len(jobs),
        seed=config.seed,
    )
    for progress, baselines in zip(progresses, baseline_metrics_by_job, strict=True):
        progress.write(
            "baseline_grid_evaluated",
            device=baseline_bundle.device,
            effective_lanes=len(baseline_dr_values),
            batch_effective_lanes=len(jobs) * len(baseline_dr_values),
            baseline_desired_retention_values=list(baseline_dr_values),
            metrics=[
                {"baseline_desired_retention": dr, **asdict(metrics)}
                for dr, metrics in zip(baseline_dr_values, baselines, strict=True)
            ],
        )
    del baseline_bundle
    clear_cuda_cache(device)

    populations, next_candidate_ids = adapter.initial_populations(
        jobs=jobs,
        settings=settings,
        portfolio=portfolio,
        family_context=family_context,
        device=device,
        seed=config.seed,
    )
    initial_bundle = _build_bundle(
        config=config,
        settings=settings,
        lane_user_ids=[
            job.user_id
            for job in jobs
            for _candidate in range(portfolio.population_size)
        ],
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    family_state = adapter.prepare_family_state(
        bundle=initial_bundle,
        jobs=jobs,
        portfolio=portfolio,
        family_context=family_context,
    )
    populations = adapter.evaluate_candidates(
        config=config,
        settings=settings,
        portfolio=portfolio,
        family_context=family_context,
        family_state=family_state,
        bundle=initial_bundle,
        candidates_by_job=populations,
        seed=config.seed,
    )
    for progress in progresses:
        progress.write(
            "initial_population_evaluated",
            device=initial_bundle.device,
            effective_lanes=portfolio.population_size,
            batch_effective_lanes=len(jobs) * portfolio.population_size,
        )
    del initial_bundle
    clear_cuda_cache(device)

    offspring_bundle = _build_bundle(
        config=config,
        settings=settings,
        lane_user_ids=[
            job.user_id
            for job in jobs
            for _candidate in range(portfolio.offspring_size)
        ],
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=benchmark_partition,
        button_usage=button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    generators = [
        generator_for_job(
            device=offspring_bundle.device,
            seed=config.seed,
            user_id=job.user_id,
        )
        for job in jobs
    ]
    history_by_job: list[list[dict[str, float]]] = [[] for _job in jobs]
    baseline_points_by_job = [
        [point_from_metrics(metrics) for metrics in baselines]
        for baselines in baseline_metrics_by_job
    ]
    references = [
        reference_point(
            baseline_points,
            margin_fraction=portfolio.reference_margin_fraction,
        )
        for baseline_points in baseline_points_by_job
    ]
    baseline_hv = [
        objective_hypervolume_2d(
            baseline_points,
            reference=references[index],
        )
        for index, baseline_points in enumerate(baseline_points_by_job)
    ]

    executor = selection_executor(
        len(jobs),
        enabled_env_vars=adapter.selection_enabled_env_vars,
        worker_env_vars=adapter.selection_worker_env_vars,
        default_min_jobs=adapter.selection_default_min_jobs,
        default_workers=adapter.selection_default_workers,
    )
    try:
        for generation in range(portfolio.generations):
            offspring_by_job: list[list[Any]] = []
            for job_index, generator in enumerate(generators):
                offspring, next_id = adapter.make_offspring(
                    population=populations[job_index],
                    next_candidate_id=next_candidate_ids[job_index],
                    settings=settings,
                    portfolio=portfolio,
                    family_context=family_context,
                    device=offspring_bundle.device,
                    generator=generator,
                )
                next_candidate_ids[job_index] = next_id
                offspring_by_job.append(offspring)
            evaluation_started = time.perf_counter()
            offspring_by_job = adapter.evaluate_candidates(
                config=config,
                settings=settings,
                portfolio=portfolio,
                family_context=family_context,
                family_state=family_state,
                bundle=offspring_bundle,
                candidates_by_job=offspring_by_job,
                seed=config.seed,
            )
            offspring_evaluation_seconds = time.perf_counter() - evaluation_started

            selection_tasks: list[SelectionTask[Any]] = []
            for job_index in range(len(jobs)):
                candidates = tuple(
                    [*populations[job_index], *offspring_by_job[job_index]]
                )
                selection_tasks.append(
                    SelectionTask(
                        candidates=candidates,
                        payload=selection_payload(
                            baseline_points=baseline_points_by_job[job_index],
                            candidates=candidates,
                            population_size=portfolio.population_size,
                            reference=references[job_index],
                        ),
                    )
                )

            selection_started = time.perf_counter()
            populations, selection_worker_seconds_by_job = (
                select_survivors_for_generation(
                    tasks=selection_tasks, executor=executor
                )
            )
            selection_seconds = time.perf_counter() - selection_started

            for job_index in range(len(jobs)):
                post_selection_started = time.perf_counter()
                candidate_points = [
                    point_from_metrics(candidate.metrics)
                    for candidate in populations[job_index]
                ]
                baseline_points = baseline_points_by_job[job_index]
                current_hv = objective_hypervolume_2d(
                    [*baseline_points, *candidate_points],
                    reference=references[job_index],
                )
                contributions = objective_exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=references[job_index],
                )
                current_frontier_candidate_count = frontier_candidate_count(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                )
                post_selection_metrics_seconds = (
                    time.perf_counter() - post_selection_started
                )
                entry = {
                    "generation": float(generation),
                    "baseline_hypervolume": baseline_hv[job_index],
                    "portfolio_hypervolume": current_hv,
                    "hypervolume_improvement": current_hv - baseline_hv[job_index],
                    "max_candidate_contribution": max(contributions)
                    if contributions
                    else 0.0,
                    "frontier_candidate_count": float(current_frontier_candidate_count),
                    "offspring_evaluation_seconds": offspring_evaluation_seconds,
                    "selection_seconds": selection_seconds,
                    "selection_worker_seconds": selection_worker_seconds_by_job[
                        job_index
                    ],
                    "post_selection_metrics_seconds": post_selection_metrics_seconds,
                }
                history_by_job[job_index].append(entry)
                progresses[job_index].write(
                    "sms_emoa_generation",
                    device=offspring_bundle.device,
                    effective_lanes=portfolio.offspring_size,
                    batch_effective_lanes=len(jobs) * portfolio.offspring_size,
                    **entry,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    del offspring_bundle
    clear_cuda_cache(device)

    results: list[Any] = []
    for job_index, job in enumerate(jobs):
        baseline_points = baseline_points_by_job[job_index]
        candidate_points = [
            point_from_metrics(candidate.metrics)
            for candidate in populations[job_index]
        ]
        final_population_hv = objective_hypervolume_2d(
            [*baseline_points, *candidate_points],
            reference=references[job_index],
        )
        final_population_hv_delta = final_population_hv - baseline_hv[job_index]
        selected = select_portfolio_children(
            baseline_points=baseline_points,
            candidates=populations[job_index],
            portfolio_size=portfolio.portfolio_size,
            reference=references[job_index],
            selected_child_from_candidate=adapter.selected_child_from_candidate,
        )
        selected_points = [
            point_from_metrics(child.candidate.metrics) for child in selected
        ]
        selected_hv = objective_hypervolume_2d(
            [*baseline_points, *selected_points],
            reference=references[job_index],
        )
        selected_hv_delta = selected_hv - baseline_hv[job_index]
        results.append(
            adapter.build_result(
                job=job,
                job_index=job_index,
                baseline_desired_retention_values=baseline_dr_values,
                baseline_metrics=baseline_metrics_by_job[job_index],
                baseline_hypervolume=baseline_hv[job_index],
                portfolio_hypervolume=selected_hv,
                hypervolume_improvement=selected_hv_delta,
                final_population_hypervolume=final_population_hv,
                final_population_hypervolume_improvement=final_population_hv_delta,
                reference_point=references[job_index],
                selected_children=selected,
                final_population=populations[job_index],
                family_state=family_state,
                history=history_by_job[job_index],
                passed=selected_hv_delta > portfolio.hv_epsilon,
            )
        )

    outcomes: list[Any] = []
    for result, progress in zip(results, progresses, strict=True):
        artifact_paths = adapter.write_artifacts(
            result=result,
            config=config,
            config_path=config_path,
            settings=settings,
            family_context=family_context,
            portfolio=portfolio,
        )
        progress.write(
            "artifacts_written",
            device=device,
            passed=result.passed,
            portfolio_path=str(result.job.output_dir / "portfolio.json"),
            portfolio_metrics_path=str(
                result.job.output_dir / "portfolio_metrics.json"
            ),
            child_artifact_paths=[str(path) for path in artifact_paths],
        )
        outcomes.append(
            adapter.build_outcome(
                job=result.job,
                passed=result.passed,
                artifact_paths=tuple(artifact_paths),
                progress_path=progress.path,
                error=None,
            )
        )
    return outcomes


def selection_payload(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[Any],
    population_size: int,
    reference: ObjectivePoint,
) -> SelectionPayload:
    return selection_payload_from_objective_candidates(
        baseline_points=baseline_points,
        candidates=candidates,
        population_size=population_size,
        reference=reference,
    )


def selection_process_pool_worker_count(
    job_count: int,
    *,
    worker_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_WORKERS_ENV,),
    default_workers: int = DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
) -> int:
    return common_selection_process_pool_worker_count(
        job_count,
        worker_env_vars=worker_env_vars,
        default_workers=default_workers,
    )


def selection_process_pool_enabled(
    job_count: int,
    *,
    enabled_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_ENV,),
    default_min_jobs: int = DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
) -> bool:
    return common_selection_process_pool_enabled(
        job_count,
        enabled_env_vars=enabled_env_vars,
        default_min_jobs=default_min_jobs,
    )


def selection_executor(
    job_count: int,
    *,
    enabled_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_ENV,),
    worker_env_vars: Sequence[str] = (SELECTION_PROCESS_POOL_WORKERS_ENV,),
    default_min_jobs: int = DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    default_workers: int = DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
) -> LightweightSelectionPool | None:
    return common_selection_executor(
        job_count,
        enabled_env_vars=enabled_env_vars,
        worker_env_vars=worker_env_vars,
        default_min_jobs=default_min_jobs,
        default_workers=default_workers,
    )


def select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[Any],
    portfolio_size: int,
    reference: ObjectivePoint,
    selected_child_from_candidate: Callable[..., Any],
) -> list[Any]:
    selections = select_portfolio_child_indices(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
    )
    return [
        selected_child_from_candidate(
            portfolio_index=selection.portfolio_index,
            candidate=candidates[selection.candidate_index],
            hypervolume_contribution=selection.hypervolume_contribution,
            pareto_rank=selection.pareto_rank,
        )
        for selection in selections
    ]


def progress_for_jobs(
    *,
    jobs: Sequence[Any],
    config_path: Path,
    execution_mode: str,
) -> list[TrainingProgress]:
    progresses: list[TrainingProgress] = []
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        progress = TrainingProgress(job.output_dir / "training_progress.jsonl")
        progress.write(
            "started",
            config_path=str(config_path),
            user_id=job.user_id,
            execution_mode=execution_mode,
        )
        progresses.append(progress)
    return progresses


def generator_for_job(
    *,
    device: torch.device,
    seed: int,
    user_id: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(user_id) * 1009)
    return generator


def zero_metrics() -> CandidateMetrics:
    return CandidateMetrics(
        memorized_average=0.0,
        time_average=0.0,
        memorized_per_minute=0.0,
        total_reviews=0,
        total_lapses=0,
        total_cost=0.0,
    )


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array.")
    values = tuple(
        _float(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    )
    if not values:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return values


def optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return float_tuple(value, field_name)


__all__ = [
    "ObjectivePoint",
    "PortfolioFamilyAdapter",
    "SelectionTask",
    "clear_cuda_cache",
    "float_tuple",
    "generator_for_job",
    "optional_float_tuple",
    "progress_for_jobs",
    "run_portfolio_train_jobs",
    "selection_executor",
    "selection_payload",
    "selection_process_pool_enabled",
    "selection_process_pool_worker_count",
    "select_portfolio_children",
    "zero_metrics",
]

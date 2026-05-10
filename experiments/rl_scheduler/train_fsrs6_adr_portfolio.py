from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    TrainingProgress,
    _baseline_dr_values,
    _build_bundle,
    _evaluate_fsrs6_baseline_grid,
    _float,
    _float_token,
    _git_commit,
    _int,
    _metrics_from_stats,
    _policy_feature_version,
    _read_training_policy_search,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    LightweightSelectionPool,
    ObjectivePoint,
    SelectionPayload,
    SelectionTask,
    frontier_candidate_count as _frontier_candidate_count,
    objective_baseline_aware_candidate_ranks as _baseline_aware_candidate_ranks,
    objective_dominates as dominates,
    objective_exclusive_hypervolume_contributions as exclusive_hypervolume_contributions,
    objective_hypervolume_2d as hypervolume_2d,
    objective_non_dominated_indices as non_dominated_indices,
    point_from_metrics,
    reference_point,
    selection_executor as _common_selection_executor,
    selection_payload_from_objective_candidates,
    selection_process_pool_enabled as _common_selection_process_pool_enabled,
    selection_process_pool_worker_count as _common_selection_process_pool_worker_count,
    select_portfolio_child_indices,
    select_sms_emoa_survivors,
    select_survivors_for_generation as _common_select_survivors_for_generation,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.batched_engine.multiuser_engine import simulate_multiuser


_SELECTION_PROCESS_POOL_ENV = "FSRS6_ADR_PORTFOLIO_SELECTION_PROCESS_POOL"
_SELECTION_PROCESS_POOL_WORKERS_ENV = "FSRS6_ADR_PORTFOLIO_SELECTION_WORKERS"
_SELECTION_ENV_VARS = (
    _SELECTION_PROCESS_POOL_ENV,
    "FSRS6_PORTFOLIO_SELECTION_PROCESS_POOL",
)
_SELECTION_WORKER_ENV_VARS = (
    _SELECTION_PROCESS_POOL_WORKERS_ENV,
    "FSRS6_PORTFOLIO_SELECTION_WORKERS",
)


@dataclass(frozen=True, slots=True)
class PortfolioSettings:
    algorithm: str = "sms_emoa"
    population_size: int = 16
    generations: int = 4
    offspring_size: int = 8
    portfolio_size: int = 4
    mutation_scale: float = 0.35
    reference_margin_fraction: float = 0.05
    hv_epsilon: float = 0.0
    seed_retention_values: tuple[float, ...] | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        settings: PolicySearchSettings,
        default_seed_retention_values: Sequence[float],
    ) -> PortfolioSettings:
        defaults = cls()
        seed_retention_values = _optional_float_tuple(
            raw.get("seed_retention_values"),
            "training.portfolio.seed_retention_values",
        )
        if seed_retention_values is None:
            seed_retention_values = tuple(
                float(item) for item in default_seed_retention_values
            )
        for value in seed_retention_values:
            if not (settings.retention_min <= value <= settings.retention_max):
                raise ValueError(
                    "training.portfolio.seed_retention_values must be inside "
                    "training.policy_search retention bounds."
                )
        algorithm = raw.get("algorithm", defaults.algorithm)
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("training.portfolio.algorithm must be a non-empty string.")
        return cls(
            algorithm=algorithm.strip(),
            population_size=_int(
                raw.get("population_size", defaults.population_size),
                "training.portfolio.population_size",
                1,
            ),
            generations=_int(
                raw.get("generations", defaults.generations),
                "training.portfolio.generations",
                0,
            ),
            offspring_size=_int(
                raw.get("offspring_size", defaults.offspring_size),
                "training.portfolio.offspring_size",
                1,
            ),
            portfolio_size=_int(
                raw.get("portfolio_size", defaults.portfolio_size),
                "training.portfolio.portfolio_size",
                1,
            ),
            mutation_scale=_float(
                raw.get("mutation_scale", defaults.mutation_scale),
                "training.portfolio.mutation_scale",
                0.0,
            ),
            reference_margin_fraction=_float(
                raw.get(
                    "reference_margin_fraction",
                    defaults.reference_margin_fraction,
                ),
                "training.portfolio.reference_margin_fraction",
                0.0,
            ),
            hv_epsilon=_float(
                raw.get("hv_epsilon", defaults.hv_epsilon),
                "training.portfolio.hv_epsilon",
                0.0,
            ),
            seed_retention_values=seed_retention_values,
        )

    def __post_init__(self) -> None:
        if self.algorithm != "sms_emoa":
            raise ValueError("training.portfolio.algorithm must be 'sms_emoa'.")
        if self.reference_margin_fraction < 0.0:
            raise ValueError("reference_margin_fraction must be >= 0.")
        if self.hv_epsilon < 0.0:
            raise ValueError("hv_epsilon must be >= 0.")
        if self.seed_retention_values is not None and not self.seed_retention_values:
            raise ValueError("seed_retention_values must not be empty.")


@dataclass(frozen=True, slots=True)
class PortfolioTrainJob:
    user_id: int
    lambda_value: float
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PortfolioTrainOutcome:
    job: PortfolioTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioCandidate:
    candidate_id: int
    coefficients: tuple[float, ...]
    metrics: CandidateMetrics

    @property
    def point(self) -> ObjectivePoint:
        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedPortfolioChild:
    portfolio_index: int
    candidate: PortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


_SelectionTask = SelectionTask


@dataclass(frozen=True, slots=True)
class UserPortfolioResult:
    job: PortfolioTrainJob
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    portfolio_hypervolume: float
    hypervolume_improvement: float
    final_population_hypervolume: float
    final_population_hypervolume_improvement: float
    reference_point: ObjectivePoint
    selected_children: list[SelectedPortfolioChild]
    final_population: list[PortfolioCandidate]
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FSRS6 ADR policy portfolio with SMS-EMOA.",
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
    config = ExperimentConfig.from_toml(args.config)
    outcomes = run_portfolio_train_jobs(
        jobs=[
            PortfolioTrainJob(
                user_id=args.user_id,
                lambda_value=args.lambda_value,
                output_dir=args.output_dir,
                command_record_path=args.training_command_path,
            )
        ],
        config=config,
        config_path=args.config,
        repo_root=REPO_ROOT,
        button_usage=args.button_usage,
        srs_benchmark_root=args.srs_benchmark_root,
        benchmark_result=args.benchmark_result,
        benchmark_partition=args.benchmark_partition,
        execution_mode="subprocess",
    )
    return 0 if outcomes and outcomes[0].passed else 1


def run_portfolio_train_jobs(
    *,
    jobs: Sequence[PortfolioTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[PortfolioTrainOutcome]:
    if not jobs:
        return []
    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search = dict(_read_training_policy_search(config_path))
    feature_version = _policy_feature_version(raw_training_policy_search)
    baseline_dr_values = _baseline_dr_values(raw_training_policy_search, settings)
    portfolio = PortfolioSettings.from_mapping(
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

    progresses = _progress_for_jobs(
        jobs=jobs,
        config_path=config_path,
        execution_mode=execution_mode,
    )
    for progress, job in zip(progresses, jobs, strict=True):
        progress.write(
            "config_loaded",
            settings=asdict(settings),
            portfolio=asdict(portfolio),
            feature_version=feature_version,
            simulation=config.simulation.to_dict(),
            seed=config.seed,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
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
    _clear_cuda_cache(device)

    populations, next_candidate_ids = _initial_populations(
        jobs=jobs,
        settings=settings,
        portfolio=portfolio,
        feature_version=feature_version,
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
    initial_metrics = _evaluate_adr_coefficients(
        config=config,
        settings=settings,
        bundle=initial_bundle,
        coefficients_by_job=[
            [candidate.coefficients for candidate in population]
            for population in populations
        ],
        feature_version=feature_version,
        seed=config.seed,
    )
    populations = [
        [
            PortfolioCandidate(
                candidate_id=population[index].candidate_id,
                coefficients=population[index].coefficients,
                metrics=metrics_by_candidate[index],
            )
            for index in range(len(population))
        ]
        for population, metrics_by_candidate in zip(
            populations, initial_metrics, strict=True
        )
    ]
    for progress in progresses:
        progress.write(
            "initial_population_evaluated",
            device=initial_bundle.device,
            effective_lanes=portfolio.population_size,
            batch_effective_lanes=len(jobs) * portfolio.population_size,
        )
    del initial_bundle
    _clear_cuda_cache(device)

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
        _generator_for_job(
            device=offspring_bundle.device,
            seed=config.seed,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
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
        hypervolume_2d(
            baseline_points,
            reference=references[index],
        )
        for index, baseline_points in enumerate(baseline_points_by_job)
    ]

    selection_executor = _selection_executor(len(jobs))
    try:
        for generation in range(portfolio.generations):
            offspring_coefficients: list[list[tuple[float, ...]]] = []
            offspring_ids: list[list[int]] = []
            for job_index, generator in enumerate(generators):
                coefficients, candidate_ids, next_id = _make_offspring(
                    population=populations[job_index],
                    next_candidate_id=next_candidate_ids[job_index],
                    offspring_size=portfolio.offspring_size,
                    mutation_scale=portfolio.mutation_scale,
                    coefficient_min=settings.coefficient_min,
                    coefficient_max=settings.coefficient_max,
                    device=offspring_bundle.device,
                    generator=generator,
                )
                next_candidate_ids[job_index] = next_id
                offspring_coefficients.append(coefficients)
                offspring_ids.append(candidate_ids)
            evaluation_started = time.perf_counter()
            offspring_metrics = _evaluate_adr_coefficients(
                config=config,
                settings=settings,
                bundle=offspring_bundle,
                coefficients_by_job=offspring_coefficients,
                feature_version=feature_version,
                seed=config.seed,
            )
            offspring_evaluation_seconds = time.perf_counter() - evaluation_started

            selection_tasks: list[SelectionTask[PortfolioCandidate]] = []
            for job_index in range(len(jobs)):
                offspring = [
                    PortfolioCandidate(
                        candidate_id=offspring_ids[job_index][candidate_index],
                        coefficients=offspring_coefficients[job_index][candidate_index],
                        metrics=offspring_metrics[job_index][candidate_index],
                    )
                    for candidate_index in range(portfolio.offspring_size)
                ]
                candidates = tuple([*populations[job_index], *offspring])
                selection_tasks.append(
                    _SelectionTask(
                        candidates=candidates,
                        payload=_selection_payload(
                            baseline_points=baseline_points_by_job[job_index],
                            candidates=candidates,
                            population_size=portfolio.population_size,
                            reference=references[job_index],
                        ),
                    )
                )

            selection_started = time.perf_counter()
            populations, selection_worker_seconds_by_job = (
                _select_survivors_for_generation(
                    tasks=selection_tasks,
                    executor=selection_executor,
                )
            )
            selection_seconds = time.perf_counter() - selection_started

            for job_index in range(len(jobs)):
                post_selection_started = time.perf_counter()
                candidate_points = [
                    candidate.point for candidate in populations[job_index]
                ]
                current_hv = hypervolume_2d(
                    [*baseline_points_by_job[job_index], *candidate_points],
                    reference=references[job_index],
                )
                contributions = exclusive_hypervolume_contributions(
                    baseline_points=baseline_points_by_job[job_index],
                    candidate_points=candidate_points,
                    reference=references[job_index],
                )
                frontier_candidate_count = _frontier_candidate_count(
                    baseline_points=baseline_points_by_job[job_index],
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
                    "frontier_candidate_count": float(frontier_candidate_count),
                    "offspring_evaluation_seconds": offspring_evaluation_seconds,
                    "selection_seconds": selection_seconds,
                    "selection_worker_seconds": selection_worker_seconds_by_job[
                        job_index
                    ],
                    "post_selection_metrics_seconds": (post_selection_metrics_seconds),
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
        if selection_executor is not None:
            selection_executor.shutdown()
    del offspring_bundle
    _clear_cuda_cache(device)

    results: list[UserPortfolioResult] = []
    for job_index, job in enumerate(jobs):
        baseline_points = [
            point_from_metrics(metrics)
            for metrics in baseline_metrics_by_job[job_index]
        ]
        candidate_points = [candidate.point for candidate in populations[job_index]]
        final_population_hv = hypervolume_2d(
            [*baseline_points, *candidate_points],
            reference=references[job_index],
        )
        final_population_hv_delta = final_population_hv - baseline_hv[job_index]
        selected = _select_portfolio_children(
            baseline_points=baseline_points,
            candidates=populations[job_index],
            portfolio_size=portfolio.portfolio_size,
            reference=references[job_index],
        )
        selected_points = [child.candidate.point for child in selected]
        selected_hv = hypervolume_2d(
            [*baseline_points, *selected_points],
            reference=references[job_index],
        )
        selected_hv_delta = selected_hv - baseline_hv[job_index]
        results.append(
            UserPortfolioResult(
                job=job,
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
                history=history_by_job[job_index],
                passed=selected_hv_delta > portfolio.hv_epsilon,
            )
        )

    outcomes: list[PortfolioTrainOutcome] = []
    for result, progress in zip(results, progresses, strict=True):
        artifact_paths = _write_portfolio_artifacts(
            result=result,
            config=config,
            config_path=config_path,
            settings=settings,
            portfolio=portfolio,
            feature_version=feature_version,
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
            PortfolioTrainOutcome(
                job=result.job,
                passed=result.passed,
                artifact_paths=tuple(artifact_paths),
                progress_path=progress.path,
            )
        )
    return outcomes


def _selection_payload(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[PortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> SelectionPayload:
    return selection_payload_from_objective_candidates(
        baseline_points=baseline_points,
        candidates=candidates,
        population_size=population_size,
        reference=reference,
    )


def _selection_process_pool_worker_count(job_count: int) -> int:
    return _common_selection_process_pool_worker_count(
        job_count,
        worker_env_vars=_SELECTION_WORKER_ENV_VARS,
        default_workers=DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    )


def _selection_process_pool_enabled(job_count: int) -> bool:
    return _common_selection_process_pool_enabled(
        job_count,
        enabled_env_vars=_SELECTION_ENV_VARS,
        default_min_jobs=DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    )


def _selection_executor(job_count: int) -> LightweightSelectionPool | None:
    return _common_selection_executor(
        job_count,
        enabled_env_vars=_SELECTION_ENV_VARS,
        worker_env_vars=_SELECTION_WORKER_ENV_VARS,
        default_min_jobs=DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
        default_workers=DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    )


def _select_survivors_for_generation(
    *,
    tasks: Sequence[SelectionTask[PortfolioCandidate]],
    executor: LightweightSelectionPool | None,
) -> tuple[list[list[PortfolioCandidate]], list[float]]:
    return _common_select_survivors_for_generation(tasks=tasks, executor=executor)


def _evaluate_adr_coefficients(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    coefficients_by_job: Sequence[Sequence[tuple[float, ...]]],
    feature_version: str,
    seed: int,
) -> list[list[CandidateMetrics]]:
    candidate_count = len(coefficients_by_job[0])
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")
    if any(len(row) != candidate_count for row in coefficients_by_job):
        raise ValueError("All jobs must evaluate the same number of candidates.")
    flat_coefficients = torch.tensor(
        [
            coefficients
            for job_coefficients in coefficients_by_job
            for coefficients in job_coefficients
        ],
        device=bundle.device,
        dtype=torch.float32,
    )
    template = FSRS6ADRPolicy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    sched_ops = FSRS6ADRBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=template,
        coefficients=flat_coefficients,
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
    metrics = [_metrics_from_stats(item) for item in stats]
    return [
        metrics[index * candidate_count : (index + 1) * candidate_count]
        for index in range(len(coefficients_by_job))
    ]


def _initial_populations(
    *,
    jobs: Sequence[PortfolioTrainJob],
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    feature_version: str,
    device: torch.device,
    seed: int,
) -> tuple[list[list[PortfolioCandidate]], list[int]]:
    seed_coefficients = [
        _constant_retention_coefficients(
            desired_retention=dr,
            settings=settings,
            feature_version=feature_version,
        )
        for dr in portfolio.seed_retention_values or ()
    ]
    populations: list[list[PortfolioCandidate]] = []
    next_ids: list[int] = []
    for job in jobs:
        generator = _generator_for_job(
            device=device,
            seed=seed,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        candidates: list[PortfolioCandidate] = []
        for index in range(portfolio.population_size):
            if index < len(seed_coefficients):
                coefficients = seed_coefficients[index]
            else:
                base = seed_coefficients[index % len(seed_coefficients)]
                coefficients = _mutate_coefficients(
                    base,
                    mutation_scale=portfolio.mutation_scale,
                    coefficient_min=settings.coefficient_min,
                    coefficient_max=settings.coefficient_max,
                    device=device,
                    generator=generator,
                )
            candidates.append(
                PortfolioCandidate(
                    candidate_id=index,
                    coefficients=coefficients,
                    metrics=_zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[PortfolioCandidate],
    next_candidate_id: int,
    offspring_size: int,
    mutation_scale: float,
    coefficient_min: float,
    coefficient_max: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[tuple[float, ...]], list[int], int]:
    coefficients: list[tuple[float, ...]] = []
    candidate_ids: list[int] = []
    for _index in range(offspring_size):
        parent_index = int(
            torch.randint(
                len(population),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        coefficients.append(
            _mutate_coefficients(
                population[parent_index].coefficients,
                mutation_scale=mutation_scale,
                coefficient_min=coefficient_min,
                coefficient_max=coefficient_max,
                device=device,
                generator=generator,
            )
        )
        candidate_ids.append(next_candidate_id)
        next_candidate_id += 1
    return coefficients, candidate_ids, next_candidate_id


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[PortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedPortfolioChild]:
    selections = select_portfolio_child_indices(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
    )
    return [
        SelectedPortfolioChild(
            portfolio_index=selection.portfolio_index,
            candidate=candidates[selection.candidate_index],
            hypervolume_contribution=selection.hypervolume_contribution,
            pareto_rank=selection.pareto_rank,
        )
        for selection in selections
    ]


def _write_portfolio_artifacts(
    *,
    result: UserPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    feature_version: str,
) -> list[Path]:
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_id = _portfolio_id(
        user_id=result.job.user_id,
        lambda_value=result.job.lambda_value,
        seed=config.seed,
    )
    artifact_paths: list[Path] = []
    child_summaries: list[dict[str, Any]] = []
    for child in result.selected_children:
        child_dir = output_dir / "policies" / f"policy_{child.portfolio_index}"
        child_dir.mkdir(parents=True, exist_ok=True)
        policy = FSRS6ADRPolicy(
            coefficients=child.candidate.coefficients,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            baseline_desired_retention=None,
            feature_version=feature_version,
            title=(
                f"fsrs6_adr_portfolio_u{result.job.user_id}_"
                f"policy_{child.portfolio_index}"
            ),
        )
        policy_path = child_dir / "policy.json"
        policy.write_json(policy_path)
        metrics_path = child_dir / "metrics.json"
        _write_json(
            metrics_path,
            {
                "candidate_id": child.candidate.candidate_id,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "pareto_rank": child.pareto_rank,
                "hypervolume_contribution": child.hypervolume_contribution,
                "metrics": asdict(child.candidate.metrics),
                "coefficients": list(child.candidate.coefficients),
            },
        )
        metadata_path = child_dir / "metadata.json"
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "scheduler-policy",
                "artifact_id": (f"{portfolio_id}-policy-{child.portfolio_index}"),
                "family": config.family,
                "scheduler_name": "fsrs6_adr",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": feature_version,
                "action_space": "sd_retention_function_portfolio_child",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "lambda_value": result.job.lambda_value,
                "baseline_desired_retention": None,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "hypervolume_contribution": child.hypervolume_contribution,
                "training_objective": "hypervolume",
                "config_snapshot_path": str(config_path.resolve()),
                "training_command_path": str(result.job.command_record_path)
                if result.job.command_record_path
                else None,
                "metrics_path": "metrics.json",
                "capabilities": ["event", "batched"],
            },
        )
        artifact_paths.append(metadata_path)
        child_summaries.append(
            {
                "portfolio_index": child.portfolio_index,
                "candidate_id": child.candidate.candidate_id,
                "policy_path": str(policy_path),
                "metadata_path": str(metadata_path),
                "metrics_path": str(metrics_path),
                "hypervolume_contribution": child.hypervolume_contribution,
                "pareto_rank": child.pareto_rank,
                "metrics": asdict(child.candidate.metrics),
            }
        )

    _write_json(
        output_dir / "portfolio.json",
        {
            "schema_version": SCHEMA_VERSION,
            "portfolio_id": portfolio_id,
            "scheduler_name": "fsrs6_adr",
            "training_user_ids": [result.job.user_id],
            "lambda_value": result.job.lambda_value,
            "baseline_desired_retention": None,
            "algorithm": portfolio.algorithm,
            "training_objective": "hypervolume",
            "passed": result.passed,
            "child_count": len(child_summaries),
            "children": child_summaries,
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
    )
    _write_json(
        output_dir / "portfolio_metrics.json",
        {
            "passed": result.passed,
            "hv_epsilon": portfolio.hv_epsilon,
            "baseline_hypervolume": result.baseline_hypervolume,
            "portfolio_hypervolume": result.portfolio_hypervolume,
            "hypervolume_improvement": result.hypervolume_improvement,
            "final_population_hypervolume": result.final_population_hypervolume,
            "final_population_hypervolume_improvement": (
                result.final_population_hypervolume_improvement
            ),
            "reference_point": asdict(result.reference_point),
            "baseline_desired_retention_values": list(
                result.baseline_desired_retention_values
            ),
            "baseline_metrics": [
                {
                    "baseline_desired_retention": dr,
                    **asdict(metrics),
                }
                for dr, metrics in zip(
                    result.baseline_desired_retention_values,
                    result.baseline_metrics,
                    strict=True,
                )
            ],
            "selected_child_count": len(result.selected_children),
            "final_population_size": len(result.final_population),
            "selection_algorithm": "greedy_subset_hypervolume",
            "settings": asdict(settings),
            "portfolio_settings": asdict(portfolio),
            "history": result.history,
        },
    )
    return artifact_paths


def _progress_for_jobs(
    *,
    jobs: Sequence[PortfolioTrainJob],
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
            lambda_value=job.lambda_value,
            execution_mode=execution_mode,
        )
        progresses.append(progress)
    return progresses


def _constant_retention_coefficients(
    *,
    desired_retention: float,
    settings: PolicySearchSettings,
    feature_version: str,
) -> tuple[float, ...]:
    policy = FSRS6ADRPolicy.baseline(
        desired_retention=desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    return _clamp_coefficient_tuple(
        policy.coefficients,
        coefficient_min=settings.coefficient_min,
        coefficient_max=settings.coefficient_max,
    )


def _mutate_coefficients(
    coefficients: tuple[float, ...],
    *,
    mutation_scale: float,
    coefficient_min: float,
    coefficient_max: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, ...]:
    base = torch.tensor(coefficients, device=device, dtype=torch.float32)
    mutated = (
        base
        + torch.randn(
            base.shape,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
        * mutation_scale
    )
    return _clamp_coefficient_tuple(
        tuple(float(value) for value in mutated.detach().cpu().tolist()),
        coefficient_min=coefficient_min,
        coefficient_max=coefficient_max,
    )


def _clamp_coefficient_tuple(
    coefficients: Sequence[float],
    *,
    coefficient_min: float,
    coefficient_max: float,
) -> tuple[float, ...]:
    return tuple(
        min(coefficient_max, max(coefficient_min, float(value)))
        for value in coefficients
    )


def _generator_for_job(
    *,
    device: torch.device,
    seed: int,
    user_id: int,
    lambda_value: float,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    lambda_token = int(round(float(lambda_value) * 1_000_000))
    generator.manual_seed(int(seed) + int(user_id) * 1009 + lambda_token * 9173)
    return generator


def _portfolio_id(*, user_id: int, lambda_value: float, seed: int) -> str:
    return (
        f"fsrs6-adr-portfolio-user-{user_id}-"
        f"lambda-{_float_token(lambda_value)}-seed-{seed}"
    )


def _zero_metrics() -> CandidateMetrics:
    return CandidateMetrics(
        memorized_average=0.0,
        time_average=0.0,
        memorized_per_minute=0.0,
        total_reviews=0,
        total_lapses=0,
        total_cost=0.0,
    )


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
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


def _optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return _float_tuple(value, field_name)


if __name__ == "__main__":
    raise SystemExit(main())

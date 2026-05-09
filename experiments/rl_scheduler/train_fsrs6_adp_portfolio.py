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

from experiments.rl_scheduler.train_cmaes_fsrs6_adp import (
    ADPSettings,
    _clipped_dimension_count,
    _decode_weight_delta_tensor,
)
from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    TrainingProgress,
    _build_bundle,
    _float,
    _float_token,
    _git_commit,
    _int,
    _metrics_from_stats,
    _read_training_policy_search,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    LightweightSelectionPool,
    SelectionPayload,
    SelectionPoint,
    SelectionTask,
    select_sms_emoa_survivor_indices,
    select_survivors_for_generation,
    selection_executor,
    selection_payload_from_candidate_metrics,
)
from experiments.rl_scheduler.train_fsrs6_adr_direct_portfolio import (
    ObjectivePoint,
    dominates,
    exclusive_hypervolume_contributions,
    hypervolume_2d,
    non_dominated_indices,
    point_from_metrics,
    reference_point,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_adp_policy import FEATURE_VERSION, FSRS6ADPPolicy, WEIGHT_COUNT
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser


@dataclass(frozen=True, slots=True)
class ADPPortfolioSettings:
    algorithm: str = "sms_emoa"
    population_size: int = 16
    generations: int = 4
    offspring_size: int = 8
    portfolio_size: int = 4
    mutation_scale: float = 0.35
    retention_mutation_scale: float = 0.02
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
    ) -> ADPPortfolioSettings:
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
            retention_mutation_scale=_float(
                raw.get(
                    "retention_mutation_scale",
                    defaults.retention_mutation_scale,
                ),
                "training.portfolio.retention_mutation_scale",
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
class ADPPortfolioTrainJob:
    user_id: int
    lambda_value: float
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ADPPortfolioTrainOutcome:
    job: ADPPortfolioTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ADPPortfolioCandidate:
    candidate_id: int
    desired_retention: float
    search_vector: tuple[float, ...]
    weights: tuple[float, ...]
    metrics: CandidateMetrics

    @property
    def point(self) -> ObjectivePoint:
        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedADPPortfolioChild:
    portfolio_index: int
    candidate: ADPPortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


@dataclass(frozen=True, slots=True)
class UserADPPortfolioResult:
    job: ADPPortfolioTrainJob
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    portfolio_hypervolume: float
    hypervolume_improvement: float
    final_population_hypervolume: float
    final_population_hypervolume_improvement: float
    reference_point: ObjectivePoint
    selected_children: list[SelectedADPPortfolioChild]
    final_population: list[ADPPortfolioCandidate]
    base_weights: tuple[float, ...]
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FSRS6 ADP policy portfolio with SMS-EMOA.",
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
            ADPPortfolioTrainJob(
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
    jobs: Sequence[ADPPortfolioTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[ADPPortfolioTrainOutcome]:
    if not jobs:
        return []
    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search = dict(_read_training_policy_search(config_path))
    baseline_dr_values = _baseline_dr_values(raw_training_policy_search, settings)
    adp_settings = ADPSettings.from_config(
        config,
        raw_training_policy_search=raw_training_policy_search,
        dr_count=len(baseline_dr_values),
    )
    portfolio = ADPPortfolioSettings.from_mapping(
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
            adp=adp_settings.to_dict(),
            portfolio=asdict(portfolio),
            feature_version=FEATURE_VERSION,
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
    base_weights_by_job = _base_weights_by_job(
        bundle=initial_bundle,
        job_count=len(jobs),
        candidate_count=portfolio.population_size,
    )
    initial_metrics, initial_weights = _evaluate_adp_portfolio_candidates(
        config=config,
        settings=settings,
        adp_settings=adp_settings,
        bundle=initial_bundle,
        desired_retentions_by_job=[
            [candidate.desired_retention for candidate in population]
            for population in populations
        ],
        search_vectors_by_job=[
            [candidate.search_vector for candidate in population]
            for population in populations
        ],
        seed=config.seed,
    )
    populations = [
        [
            ADPPortfolioCandidate(
                candidate_id=population[index].candidate_id,
                desired_retention=population[index].desired_retention,
                search_vector=population[index].search_vector,
                weights=weights_by_candidate[index],
                metrics=metrics_by_candidate[index],
            )
            for index in range(len(population))
        ]
        for population, metrics_by_candidate, weights_by_candidate in zip(
            populations, initial_metrics, initial_weights, strict=True
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
    selection_pool: LightweightSelectionPool | None = selection_executor(len(jobs))
    try:
        for generation in range(portfolio.generations):
            offspring_desired: list[list[float]] = []
            offspring_vectors: list[list[tuple[float, ...]]] = []
            offspring_ids: list[list[int]] = []
            for job_index, generator in enumerate(generators):
                desired, vectors, candidate_ids, next_id = _make_offspring(
                    population=populations[job_index],
                    next_candidate_id=next_candidate_ids[job_index],
                    offspring_size=portfolio.offspring_size,
                    mutation_scale=portfolio.mutation_scale,
                    retention_mutation_scale=portfolio.retention_mutation_scale,
                    retention_min=settings.retention_min,
                    retention_max=settings.retention_max,
                    device=offspring_bundle.device,
                    generator=generator,
                )
                next_candidate_ids[job_index] = next_id
                offspring_desired.append(desired)
                offspring_vectors.append(vectors)
                offspring_ids.append(candidate_ids)
            evaluation_started = time.perf_counter()
            offspring_metrics, offspring_weights = _evaluate_adp_portfolio_candidates(
                config=config,
                settings=settings,
                adp_settings=adp_settings,
                bundle=offspring_bundle,
                desired_retentions_by_job=offspring_desired,
                search_vectors_by_job=offspring_vectors,
                seed=config.seed + generation + 1,
            )
            offspring_evaluation_seconds = time.perf_counter() - evaluation_started

            selection_tasks: list[SelectionTask[ADPPortfolioCandidate]] = []
            for job_index in range(len(jobs)):
                offspring = [
                    ADPPortfolioCandidate(
                        candidate_id=offspring_ids[job_index][candidate_index],
                        desired_retention=offspring_desired[job_index][candidate_index],
                        search_vector=offspring_vectors[job_index][candidate_index],
                        weights=offspring_weights[job_index][candidate_index],
                        metrics=offspring_metrics[job_index][candidate_index],
                    )
                    for candidate_index in range(portfolio.offspring_size)
                ]
                candidates = tuple([*populations[job_index], *offspring])
                selection_tasks.append(
                    SelectionTask(
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
                select_survivors_for_generation(
                    tasks=selection_tasks,
                    executor=selection_pool,
                )
            )
            selection_seconds = time.perf_counter() - selection_started

            for job_index in range(len(jobs)):
                post_selection_started = time.perf_counter()
                candidate_points = [
                    candidate.point for candidate in populations[job_index]
                ]
                baseline_points = baseline_points_by_job[job_index]
                current_hv = hypervolume_2d(
                    [*baseline_points, *candidate_points],
                    reference=references[job_index],
                )
                contributions = exclusive_hypervolume_contributions(
                    baseline_points=baseline_points,
                    candidate_points=candidate_points,
                    reference=references[job_index],
                )
                frontier_candidate_count = _frontier_candidate_count(
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
        if selection_pool is not None:
            selection_pool.shutdown()
    del offspring_bundle
    _clear_cuda_cache(device)

    results: list[UserADPPortfolioResult] = []
    for job_index, job in enumerate(jobs):
        baseline_points = baseline_points_by_job[job_index]
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
            UserADPPortfolioResult(
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
                base_weights=base_weights_by_job[job_index],
                history=history_by_job[job_index],
                passed=selected_hv_delta > portfolio.hv_epsilon,
            )
        )

    outcomes: list[ADPPortfolioTrainOutcome] = []
    for result, progress in zip(results, progresses, strict=True):
        artifact_paths = _write_portfolio_artifacts(
            result=result,
            config=config,
            config_path=config_path,
            settings=settings,
            adp_settings=adp_settings,
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
            ADPPortfolioTrainOutcome(
                job=result.job,
                passed=result.passed,
                artifact_paths=tuple(artifact_paths),
                progress_path=progress.path,
            )
        )
    return outcomes


def select_sms_emoa_survivors(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[ADPPortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> list[ADPPortfolioCandidate]:
    payload = _selection_payload(
        baseline_points=baseline_points,
        candidates=candidates,
        population_size=population_size,
        reference=reference,
    )
    survivor_indices = select_sms_emoa_survivor_indices(payload)
    return [candidates[index] for index in survivor_indices]


def _selection_point(point: ObjectivePoint) -> SelectionPoint:
    return SelectionPoint(
        memorized_average=point.memorized_average,
        negative_time_average=point.negative_time_average,
    )


def _selection_points(points: Sequence[ObjectivePoint]) -> list[SelectionPoint]:
    return [_selection_point(point) for point in points]


def _selection_payload(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[ADPPortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> SelectionPayload:
    return selection_payload_from_candidate_metrics(
        baseline_points=_selection_points(baseline_points),
        candidates=candidates,
        population_size=population_size,
        reference=_selection_point(reference),
    )


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[ADPPortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedADPPortfolioChild]:
    candidate_points = [candidate.point for candidate in candidates]
    ranks = _baseline_aware_candidate_ranks(
        baseline_points=baseline_points,
        candidate_points=candidate_points,
    )
    remaining = set(range(len(candidates)))
    selected_indices: list[int] = []
    current_points = list(baseline_points)
    current_hv = hypervolume_2d(current_points, reference=reference)
    children: list[SelectedADPPortfolioChild] = []
    while remaining and len(children) < portfolio_size:
        best_index = max(
            remaining,
            key=lambda index: (
                hypervolume_2d(
                    [*current_points, candidate_points[index]],
                    reference=reference,
                )
                - current_hv,
                -ranks[index],
                candidates[index].metrics.memorized_average,
                -candidates[index].metrics.time_average,
                -candidates[index].candidate_id,
            ),
        )
        next_hv = hypervolume_2d(
            [*current_points, candidate_points[best_index]],
            reference=reference,
        )
        contribution = max(0.0, next_hv - current_hv)
        current_hv = next_hv
        current_points.append(candidate_points[best_index])
        selected_indices.append(best_index)
        remaining.remove(best_index)
        candidate = candidates[best_index]
        children.append(
            SelectedADPPortfolioChild(
                portfolio_index=len(selected_indices) - 1,
                candidate=candidate,
                hypervolume_contribution=contribution,
                pareto_rank=ranks[best_index],
            )
        )
    return children


def _baseline_aware_candidate_ranks(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidate_points: Sequence[ObjectivePoint],
) -> list[int]:
    ranks = [-1 for _candidate in candidate_points]
    baseline_dominated = {
        index
        for index, candidate in enumerate(candidate_points)
        if any(dominates(baseline, candidate) for baseline in baseline_points)
    }
    remaining = [
        index
        for index in range(len(candidate_points))
        if index not in baseline_dominated
    ]
    rank = 0
    while remaining:
        layer_points = [
            *baseline_points,
            *[candidate_points[index] for index in remaining],
        ]
        nd = non_dominated_indices(layer_points)
        selected = [
            remaining[index - len(baseline_points)]
            for index in nd
            if index >= len(baseline_points)
        ]
        if not selected:
            break
        selected_set = set(selected)
        for index in selected:
            ranks[index] = rank
        remaining = [index for index in remaining if index not in selected_set]
        rank += 1
    worst_rank = rank + len(candidate_points) + 1
    for index in range(len(candidate_points)):
        if ranks[index] < 0:
            ranks[index] = worst_rank
    return ranks


def _frontier_candidate_count(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidate_points: Sequence[ObjectivePoint],
) -> int:
    points = [*baseline_points, *candidate_points]
    nd = non_dominated_indices(points)
    return sum(1 for index in nd if index >= len(baseline_points))


def _baseline_dr_values(
    raw_training_policy_search: Mapping[str, Any],
    settings: PolicySearchSettings,
) -> tuple[float, ...]:
    raw_values = raw_training_policy_search.get("baseline_desired_retention_values")
    if raw_values is None:
        values = (settings.baseline_desired_retention,)
    else:
        values = _float_tuple(
            raw_values,
            "training.policy_search.baseline_desired_retention_values",
        )
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "training.policy_search.baseline_desired_retention_values must be inside "
                "training.policy_search retention bounds."
            )
    return values


def _evaluate_fsrs6_baseline_grid(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    baseline_dr_values: tuple[float, ...],
    job_count: int,
    seed: int,
) -> list[list[CandidateMetrics]]:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            [dr for _job in range(job_count) for dr in baseline_dr_values],
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
    metrics = [_metrics_from_stats(item) for item in stats]
    dr_count = len(baseline_dr_values)
    return [
        metrics[index * dr_count : (index + 1) * dr_count] for index in range(job_count)
    ]


def _evaluate_adp_portfolio_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    adp_settings: ADPSettings,
    bundle: Any,
    desired_retentions_by_job: Sequence[Sequence[float]],
    search_vectors_by_job: Sequence[Sequence[tuple[float, ...]]],
    seed: int,
) -> tuple[list[list[CandidateMetrics]], list[list[tuple[float, ...]]]]:
    candidate_count = len(search_vectors_by_job[0])
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")
    if any(len(row) != candidate_count for row in search_vectors_by_job):
        raise ValueError("All jobs must evaluate the same number of candidates.")
    if any(len(row) != candidate_count for row in desired_retentions_by_job):
        raise ValueError("All jobs must evaluate the same number of candidates.")
    flat_vectors = torch.tensor(
        [vector for job_vectors in search_vectors_by_job for vector in job_vectors],
        device=bundle.device,
        dtype=torch.float32,
    )
    scheduler_weights = _decode_weight_delta_tensor(
        base_weights=bundle.scheduler_weights,
        search_vectors=flat_vectors,
        weight_delta_scale=adp_settings.weight_delta_scale,
    )
    desired = torch.tensor(
        [
            desired_retention
            for job_desired_retentions in desired_retentions_by_job
            for desired_retention in job_desired_retentions
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
    return (
        [
            flat_metrics[index * candidate_count : (index + 1) * candidate_count]
            for index in range(len(search_vectors_by_job))
        ],
        [
            flat_weights[index * candidate_count : (index + 1) * candidate_count]
            for index in range(len(search_vectors_by_job))
        ],
    )


def _initial_populations(
    *,
    jobs: Sequence[ADPPortfolioTrainJob],
    settings: PolicySearchSettings,
    portfolio: ADPPortfolioSettings,
    device: torch.device,
    seed: int,
) -> tuple[list[list[ADPPortfolioCandidate]], list[int]]:
    seed_genomes = [
        (float(dr), _zero_search_vector())
        for dr in portfolio.seed_retention_values or ()
    ]
    populations: list[list[ADPPortfolioCandidate]] = []
    next_ids: list[int] = []
    for job in jobs:
        generator = _generator_for_job(
            device=device,
            seed=seed,
            user_id=job.user_id,
            lambda_value=job.lambda_value,
        )
        candidates: list[ADPPortfolioCandidate] = []
        for index in range(portfolio.population_size):
            if index < len(seed_genomes):
                desired_retention, search_vector = seed_genomes[index]
            else:
                base_desired, base_vector = seed_genomes[index % len(seed_genomes)]
                desired_retention, search_vector = _mutate_genome(
                    desired_retention=base_desired,
                    search_vector=base_vector,
                    mutation_scale=portfolio.mutation_scale,
                    retention_mutation_scale=portfolio.retention_mutation_scale,
                    retention_min=settings.retention_min,
                    retention_max=settings.retention_max,
                    device=device,
                    generator=generator,
                )
            candidates.append(
                ADPPortfolioCandidate(
                    candidate_id=index,
                    desired_retention=desired_retention,
                    search_vector=search_vector,
                    weights=_zero_search_vector(),
                    metrics=_zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[ADPPortfolioCandidate],
    next_candidate_id: int,
    offspring_size: int,
    mutation_scale: float,
    retention_mutation_scale: float,
    retention_min: float,
    retention_max: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[float], list[tuple[float, ...]], list[int], int]:
    desired_retentions: list[float] = []
    search_vectors: list[tuple[float, ...]] = []
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
        desired_retention, search_vector = _mutate_genome(
            desired_retention=population[parent_index].desired_retention,
            search_vector=population[parent_index].search_vector,
            mutation_scale=mutation_scale,
            retention_mutation_scale=retention_mutation_scale,
            retention_min=retention_min,
            retention_max=retention_max,
            device=device,
            generator=generator,
        )
        desired_retentions.append(desired_retention)
        search_vectors.append(search_vector)
        candidate_ids.append(next_candidate_id)
        next_candidate_id += 1
    return desired_retentions, search_vectors, candidate_ids, next_candidate_id


def _mutate_genome(
    *,
    desired_retention: float,
    search_vector: Sequence[float],
    mutation_scale: float,
    retention_mutation_scale: float,
    retention_min: float,
    retention_max: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, tuple[float, ...]]:
    base = torch.tensor(tuple(search_vector), device=device, dtype=torch.float32)
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
    retention_delta = (
        float(
            torch.randn(
                (), device=device, generator=generator, dtype=torch.float32
            ).item()
        )
        * retention_mutation_scale
    )
    return (
        _clip_retention(
            desired_retention + retention_delta,
            retention_min=retention_min,
            retention_max=retention_max,
        ),
        _search_vector_tuple(mutated.detach().cpu().tolist()),
    )


def _clip_retention(
    desired_retention: float,
    *,
    retention_min: float,
    retention_max: float,
) -> float:
    return min(retention_max, max(retention_min, float(desired_retention)))


def _search_vector_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != WEIGHT_COUNT:
        raise ValueError(f"search_vector must contain {WEIGHT_COUNT} values.")
    return result


def _zero_search_vector() -> tuple[float, ...]:
    return (0.0,) * WEIGHT_COUNT


def _base_weights_by_job(
    *,
    bundle: Any,
    job_count: int,
    candidate_count: int,
) -> list[tuple[float, ...]]:
    weights = bundle.scheduler_weights.detach().cpu()
    return [
        tuple(float(value) for value in weights[job_index * candidate_count].tolist())
        for job_index in range(job_count)
    ]


def _write_portfolio_artifacts(
    *,
    result: UserADPPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    adp_settings: ADPSettings,
    portfolio: ADPPortfolioSettings,
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
        delta = tuple(
            weight - base_weight
            for weight, base_weight in zip(child.candidate.weights, result.base_weights)
        )
        policy = FSRS6ADPPolicy(
            base_weights=result.base_weights,
            weights=child.candidate.weights,
            delta=delta,
            search_vector=child.candidate.search_vector,
            baseline_desired_retention=child.candidate.desired_retention,
            weight_delta_scale=adp_settings.weight_delta_scale,
            title=(
                f"fsrs6_adp_portfolio_u{result.job.user_id}_"
                f"policy_{child.portfolio_index}"
            ),
        )
        policy_path = child_dir / "policy.json"
        policy.write_json(policy_path)
        metrics_path = child_dir / "metrics.json"
        clipped_dimensions = _clipped_dimension_count(
            base_weights=result.base_weights,
            search_vector=child.candidate.search_vector,
            weights=child.candidate.weights,
            weight_delta_scale=adp_settings.weight_delta_scale,
        )
        _write_json(
            metrics_path,
            {
                "candidate_id": child.candidate.candidate_id,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "pareto_rank": child.pareto_rank,
                "scheduler_desired_retention": child.candidate.desired_retention,
                "hypervolume_contribution": child.hypervolume_contribution,
                "baseline_hypervolume": result.baseline_hypervolume,
                "portfolio_hypervolume": result.portfolio_hypervolume,
                "hypervolume_improvement": result.hypervolume_improvement,
                "final_population_hypervolume": result.final_population_hypervolume,
                "metrics": asdict(child.candidate.metrics),
                "base_weights": list(result.base_weights),
                "weights": list(child.candidate.weights),
                "delta": list(delta),
                "search_vector": list(child.candidate.search_vector),
                "clipped_dimensions": clipped_dimensions,
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
                "scheduler_name": "fsrs6_adp",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": FEATURE_VERSION,
                "action_space": "fsrs6_adp_weight_delta_portfolio_child",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "lambda_value": result.job.lambda_value,
                "baseline_desired_retention": None,
                "scheduler_desired_retention": child.candidate.desired_retention,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "hypervolume_contribution": child.hypervolume_contribution,
                "training_objective": "hypervolume",
                "config_snapshot_path": str(config_path.resolve()),
                "training_command_path": str(result.job.command_record_path)
                if result.job.command_record_path
                else None,
                "metrics_path": "metrics.json",
                "optimizer": "sms_emoa",
                "capabilities": ["event", "vectorized", "batched"],
            },
        )
        artifact_paths.append(metadata_path)
        child_summaries.append(
            {
                "portfolio_index": child.portfolio_index,
                "candidate_id": child.candidate.candidate_id,
                "scheduler_desired_retention": child.candidate.desired_retention,
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
            "scheduler_name": "fsrs6_adp",
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
            "adp_settings": adp_settings.to_dict(),
            "portfolio_settings": asdict(portfolio),
            "history": result.history,
        },
    )
    return artifact_paths


def _progress_for_jobs(
    *,
    jobs: Sequence[ADPPortfolioTrainJob],
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
        f"fsrs6-adp-portfolio-user-{user_id}-"
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

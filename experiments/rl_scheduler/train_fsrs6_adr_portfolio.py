from __future__ import annotations

import argparse
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

from experiments.rl_scheduler.policy_search_common import (
    CandidateMetrics,
    PolicySearchSettings,
    _float,
    _git_commit,
    _int,
    _metrics_from_stats,
    _policy_feature_version,
    _relative_path_string,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
    LightweightSelectionPool,
    ObjectivePoint,
    SelectionTask,
    objective_exclusive_hypervolume_contributions as exclusive_hypervolume_contributions,
    objective_hypervolume_2d as hypervolume_2d,
    objective_non_dominated_indices as non_dominated_indices,
    reference_point,
    select_sms_emoa_survivors,
)
from experiments.rl_scheduler.portfolio_training_common import (
    PortfolioFamilyAdapter,
    clear_cuda_cache as _common_clear_cuda_cache,
    float_tuple as _common_float_tuple,
    generator_for_job,
    optional_float_tuple as _common_optional_float_tuple,
    progress_for_jobs,
    run_portfolio_train_jobs as _run_common_portfolio_train_jobs,
    selection_executor as _common_selection_executor,
    selection_payload as _common_selection_payload,
    selection_process_pool_enabled as _common_selection_process_pool_enabled,
    selection_process_pool_worker_count as _common_selection_process_pool_worker_count,
    select_portfolio_children as _common_select_portfolio_children,
    select_survivors_for_generation as _common_select_survivors_for_generation,
    zero_metrics,
)
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_adr_policy import FSRS6ADRPolicy
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs6_adr import FSRS6ADRBatchSchedulerOps


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
        from experiments.rl_scheduler.portfolio_selection import point_from_metrics

        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedPortfolioChild:
    portfolio_index: int
    candidate: PortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


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


@dataclass(frozen=True, slots=True)
class _ADRFamilyContext:
    feature_version: str


_SelectionTask = SelectionTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FSRS6 ADR policy portfolio with SMS-EMOA.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
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
    return _run_common_portfolio_train_jobs(
        jobs=jobs,
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        adapter=_ADAPTER,
        button_usage=button_usage,
        srs_benchmark_root=srs_benchmark_root,
        benchmark_result=benchmark_result,
        benchmark_partition=benchmark_partition,
        execution_mode=execution_mode,
    )


def _build_family_context(
    *,
    config: ExperimentConfig,
    raw_training_policy_search: Mapping[str, Any],
    baseline_dr_values: Sequence[float],
) -> _ADRFamilyContext:
    del config, baseline_dr_values
    return _ADRFamilyContext(
        feature_version=_policy_feature_version(raw_training_policy_search)
    )


def _progress_payload(*, family_context: _ADRFamilyContext) -> Mapping[str, Any]:
    return {"feature_version": family_context.feature_version}


def _prepare_family_state(
    *,
    bundle: Any,
    jobs: Sequence[PortfolioTrainJob],
    portfolio: PortfolioSettings,
    family_context: _ADRFamilyContext,
) -> None:
    del bundle, jobs, portfolio, family_context
    return None


def _evaluate_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    family_context: _ADRFamilyContext,
    family_state: None,
    bundle: Any,
    candidates_by_job: Sequence[Sequence[PortfolioCandidate]],
    seed: int,
) -> list[list[PortfolioCandidate]]:
    del portfolio, family_state
    metrics_by_job = _evaluate_adr_coefficients(
        config=config,
        settings=settings,
        bundle=bundle,
        coefficients_by_job=[
            [candidate.coefficients for candidate in candidates]
            for candidates in candidates_by_job
        ],
        feature_version=family_context.feature_version,
        seed=seed,
    )
    return [
        [
            PortfolioCandidate(
                candidate_id=candidates[index].candidate_id,
                coefficients=candidates[index].coefficients,
                metrics=metrics_by_job[job_index][index],
            )
            for index in range(len(candidates))
        ]
        for job_index, candidates in enumerate(candidates_by_job)
    ]


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
    family_context: _ADRFamilyContext,
    device: torch.device,
    seed: int,
) -> tuple[list[list[PortfolioCandidate]], list[int]]:
    seed_coefficients = [
        _constant_retention_coefficients(
            desired_retention=dr,
            settings=settings,
            feature_version=family_context.feature_version,
        )
        for dr in portfolio.seed_retention_values or ()
    ]
    populations: list[list[PortfolioCandidate]] = []
    next_ids: list[int] = []
    for job in jobs:
        generator = generator_for_job(device=device, seed=seed, user_id=job.user_id)
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
                    metrics=zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[PortfolioCandidate],
    next_candidate_id: int,
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    family_context: _ADRFamilyContext,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[PortfolioCandidate], int]:
    del family_context
    candidates: list[PortfolioCandidate] = []
    for _index in range(portfolio.offspring_size):
        parent_index = int(
            torch.randint(
                len(population),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        coefficients = _mutate_coefficients(
            population[parent_index].coefficients,
            mutation_scale=portfolio.mutation_scale,
            coefficient_min=settings.coefficient_min,
            coefficient_max=settings.coefficient_max,
            device=device,
            generator=generator,
        )
        candidates.append(
            PortfolioCandidate(
                candidate_id=next_candidate_id,
                coefficients=coefficients,
                metrics=zero_metrics(),
            )
        )
        next_candidate_id += 1
    return candidates, next_candidate_id


def _selected_child_from_candidate(
    *,
    portfolio_index: int,
    candidate: PortfolioCandidate,
    hypervolume_contribution: float,
    pareto_rank: int,
) -> SelectedPortfolioChild:
    return SelectedPortfolioChild(
        portfolio_index=portfolio_index,
        candidate=candidate,
        hypervolume_contribution=hypervolume_contribution,
        pareto_rank=pareto_rank,
    )


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[PortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedPortfolioChild]:
    return _common_select_portfolio_children(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
        selected_child_from_candidate=_selected_child_from_candidate,
    )


def _build_result(
    *,
    job: PortfolioTrainJob,
    job_index: int,
    baseline_desired_retention_values: tuple[float, ...],
    baseline_metrics: list[CandidateMetrics],
    baseline_hypervolume: float,
    portfolio_hypervolume: float,
    hypervolume_improvement: float,
    final_population_hypervolume: float,
    final_population_hypervolume_improvement: float,
    reference_point: ObjectivePoint,
    selected_children: list[SelectedPortfolioChild],
    final_population: list[PortfolioCandidate],
    family_state: None,
    history: list[dict[str, float]],
    passed: bool,
) -> UserPortfolioResult:
    del job_index, family_state
    return UserPortfolioResult(
        job=job,
        baseline_desired_retention_values=baseline_desired_retention_values,
        baseline_metrics=baseline_metrics,
        baseline_hypervolume=baseline_hypervolume,
        portfolio_hypervolume=portfolio_hypervolume,
        hypervolume_improvement=hypervolume_improvement,
        final_population_hypervolume=final_population_hypervolume,
        final_population_hypervolume_improvement=(
            final_population_hypervolume_improvement
        ),
        reference_point=reference_point,
        selected_children=selected_children,
        final_population=final_population,
        history=history,
        passed=passed,
    )


def _write_portfolio_artifacts(
    *,
    result: UserPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    portfolio: PortfolioSettings,
    family_context: _ADRFamilyContext | None = None,
    feature_version: str | None = None,
) -> list[Path]:
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_id = _portfolio_id(user_id=result.job.user_id, seed=config.seed)
    effective_feature_version = (
        family_context.feature_version
        if family_context is not None
        else feature_version
    )
    if effective_feature_version is None:
        raise ValueError("feature_version is required for ADR portfolio artifacts.")
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
            feature_version=effective_feature_version,
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
        metadata_dir = metadata_path.parent
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "scheduler-policy",
                "artifact_id": f"{portfolio_id}-policy-{child.portfolio_index}",
                "family": config.family,
                "scheduler_name": "fsrs6_adr",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": effective_feature_version,
                "action_space": "sd_retention_function_portfolio_child",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "baseline_desired_retention": None,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "hypervolume_contribution": child.hypervolume_contribution,
                "training_objective": "hypervolume",
                "config_snapshot_path": _relative_path_string(
                    config_path,
                    base=metadata_dir,
                ),
                "training_command_path": _relative_path_string(
                    result.job.command_record_path,
                    base=metadata_dir,
                )
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
                "policy_path": _relative_path_string(policy_path, base=output_dir),
                "metadata_path": _relative_path_string(
                    metadata_path,
                    base=output_dir,
                ),
                "metrics_path": _relative_path_string(metrics_path, base=output_dir),
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
) -> torch.Generator:
    return generator_for_job(device=device, seed=seed, user_id=user_id)


def _portfolio_id(*, user_id: int, seed: int) -> str:
    return f"fsrs6-adr-portfolio-user-{user_id}-seed-{seed}"


def _zero_metrics() -> CandidateMetrics:
    return zero_metrics()


def _clear_cuda_cache(device: torch.device) -> None:
    _common_clear_cuda_cache(device)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    return _common_float_tuple(value, field_name)


def _optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    return _common_optional_float_tuple(value, field_name)


def _selection_payload(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[PortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> Any:
    return _common_selection_payload(
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


def _build_outcome(
    *,
    job: PortfolioTrainJob,
    passed: bool,
    artifact_paths: tuple[Path, ...],
    progress_path: Path,
    error: str | None,
) -> PortfolioTrainOutcome:
    return PortfolioTrainOutcome(
        job=job,
        passed=passed,
        artifact_paths=artifact_paths,
        progress_path=progress_path,
        error=error,
    )


_progress_for_jobs = progress_for_jobs


_ADAPTER = PortfolioFamilyAdapter(
    settings_from_mapping=PortfolioSettings.from_mapping,
    build_family_context=_build_family_context,
    progress_payload=_progress_payload,
    initial_populations=_initial_populations,
    prepare_family_state=_prepare_family_state,
    evaluate_candidates=_evaluate_candidates,
    make_offspring=_make_offspring,
    selected_child_from_candidate=_selected_child_from_candidate,
    build_result=_build_result,
    write_artifacts=_write_portfolio_artifacts,
    build_outcome=_build_outcome,
    selection_enabled_env_vars=_SELECTION_ENV_VARS,
    selection_worker_env_vars=_SELECTION_WORKER_ENV_VARS,
    selection_default_min_jobs=DEFAULT_SELECTION_PROCESS_POOL_MIN_JOBS,
    selection_default_workers=DEFAULT_SELECTION_PROCESS_POOL_WORKERS,
)


if __name__ == "__main__":
    raise SystemExit(main())

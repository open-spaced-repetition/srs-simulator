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
    _relative_path_string,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    ObjectivePoint,
    SelectionTask,
)
from experiments.rl_scheduler.portfolio_training_common import (
    PortfolioFamilyAdapter,
    clear_cuda_cache as _common_clear_cuda_cache,
    float_tuple as _common_float_tuple,
    generator_for_job,
    optional_float_tuple as _common_optional_float_tuple,
    progress_for_jobs,
    run_portfolio_train_jobs as _run_common_portfolio_train_jobs,
    selection_payload as _common_selection_payload,
    select_portfolio_children as _common_select_portfolio_children,
    zero_metrics,
)
from experiments.rl_scheduler.train_cmaes_fsrs6_ap import (
    APSettings,
    _clipped_dimension_count,
    _decode_weight_delta_tensor,
)
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_ap_policy import FEATURE_VERSION, FSRS6APPolicy, WEIGHT_COUNT
from simulator.math.fsrs import Bounds
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps


@dataclass(frozen=True, slots=True)
class APPortfolioSettings:
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
    ) -> APPortfolioSettings:
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
class APPortfolioTrainJob:
    user_id: int
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class APPortfolioTrainOutcome:
    job: APPortfolioTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path
    error: str | None = None


@dataclass(frozen=True, slots=True)
class APPortfolioCandidate:
    candidate_id: int
    desired_retention: float
    search_vector: tuple[float, ...]
    weights: tuple[float, ...]
    metrics: CandidateMetrics

    @property
    def point(self) -> ObjectivePoint:
        from experiments.rl_scheduler.portfolio_selection import point_from_metrics

        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedAPPortfolioChild:
    portfolio_index: int
    candidate: APPortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


@dataclass(frozen=True, slots=True)
class UserAPPortfolioResult:
    job: APPortfolioTrainJob
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    portfolio_hypervolume: float
    hypervolume_improvement: float
    final_population_hypervolume: float
    final_population_hypervolume_improvement: float
    reference_point: ObjectivePoint
    selected_children: list[SelectedAPPortfolioChild]
    final_population: list[APPortfolioCandidate]
    base_weights: tuple[float, ...]
    history: list[dict[str, float]]
    passed: bool


@dataclass(frozen=True, slots=True)
class _APFamilyContext:
    ap_settings: APSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an FSRS6 AP policy portfolio with SMS-EMOA.",
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
            APPortfolioTrainJob(
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
    jobs: Sequence[APPortfolioTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[APPortfolioTrainOutcome]:
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
) -> _APFamilyContext:
    return _APFamilyContext(
        ap_settings=APSettings.from_config(
            config,
            raw_training_policy_search=raw_training_policy_search,
            dr_count=len(baseline_dr_values),
        )
    )


def _progress_payload(*, family_context: _APFamilyContext) -> Mapping[str, Any]:
    return {
        "ap": family_context.ap_settings.to_dict(),
        "feature_version": FEATURE_VERSION,
    }


def _prepare_family_state(
    *,
    bundle: Any,
    jobs: Sequence[APPortfolioTrainJob],
    portfolio: APPortfolioSettings,
    family_context: _APFamilyContext,
) -> list[tuple[float, ...]]:
    del family_context
    return _base_weights_by_job(
        bundle=bundle,
        job_count=len(jobs),
        candidate_count=portfolio.population_size,
    )


def _evaluate_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    portfolio: APPortfolioSettings,
    family_context: _APFamilyContext,
    family_state: list[tuple[float, ...]],
    bundle: Any,
    candidates_by_job: Sequence[Sequence[APPortfolioCandidate]],
    seed: int,
) -> list[list[APPortfolioCandidate]]:
    del portfolio, family_state
    metrics_by_job, weights_by_job = _evaluate_ap_portfolio_candidates(
        config=config,
        settings=settings,
        ap_settings=family_context.ap_settings,
        bundle=bundle,
        desired_retentions_by_job=[
            [candidate.desired_retention for candidate in candidates]
            for candidates in candidates_by_job
        ],
        search_vectors_by_job=[
            [candidate.search_vector for candidate in candidates]
            for candidates in candidates_by_job
        ],
        seed=seed,
    )
    return [
        [
            APPortfolioCandidate(
                candidate_id=candidates[index].candidate_id,
                desired_retention=candidates[index].desired_retention,
                search_vector=candidates[index].search_vector,
                weights=weights_by_job[job_index][index],
                metrics=metrics_by_job[job_index][index],
            )
            for index in range(len(candidates))
        ]
        for job_index, candidates in enumerate(candidates_by_job)
    ]


def _evaluate_ap_portfolio_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    ap_settings: APSettings,
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
        weight_delta_scale=ap_settings.weight_delta_scale,
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
    jobs: Sequence[APPortfolioTrainJob],
    settings: PolicySearchSettings,
    portfolio: APPortfolioSettings,
    family_context: _APFamilyContext,
    seed_retention_values_by_job: Sequence[Sequence[float]],
    device: torch.device,
    seed: int,
) -> tuple[list[list[APPortfolioCandidate]], list[int]]:
    del family_context
    populations: list[list[APPortfolioCandidate]] = []
    next_ids: list[int] = []
    for job, seed_retention_values in zip(
        jobs,
        seed_retention_values_by_job,
        strict=True,
    ):
        seed_genomes = [
            (float(dr), _zero_search_vector()) for dr in seed_retention_values
        ]
        generator = generator_for_job(device=device, seed=seed, user_id=job.user_id)
        candidates: list[APPortfolioCandidate] = []
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
                APPortfolioCandidate(
                    candidate_id=index,
                    desired_retention=desired_retention,
                    search_vector=search_vector,
                    weights=_zero_search_vector(),
                    metrics=zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[APPortfolioCandidate],
    next_candidate_id: int,
    settings: PolicySearchSettings,
    portfolio: APPortfolioSettings,
    family_context: _APFamilyContext,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[APPortfolioCandidate], int]:
    del family_context
    candidates: list[APPortfolioCandidate] = []
    for _index in range(portfolio.offspring_size):
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
            mutation_scale=portfolio.mutation_scale,
            retention_mutation_scale=portfolio.retention_mutation_scale,
            retention_min=settings.retention_min,
            retention_max=settings.retention_max,
            device=device,
            generator=generator,
        )
        candidates.append(
            APPortfolioCandidate(
                candidate_id=next_candidate_id,
                desired_retention=desired_retention,
                search_vector=search_vector,
                weights=_zero_search_vector(),
                metrics=zero_metrics(),
            )
        )
        next_candidate_id += 1
    return candidates, next_candidate_id


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


def _selected_child_from_candidate(
    *,
    portfolio_index: int,
    candidate: APPortfolioCandidate,
    hypervolume_contribution: float,
    pareto_rank: int,
) -> SelectedAPPortfolioChild:
    return SelectedAPPortfolioChild(
        portfolio_index=portfolio_index,
        candidate=candidate,
        hypervolume_contribution=hypervolume_contribution,
        pareto_rank=pareto_rank,
    )


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[APPortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedAPPortfolioChild]:
    return _common_select_portfolio_children(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
        selected_child_from_candidate=_selected_child_from_candidate,
    )


def _build_result(
    *,
    job: APPortfolioTrainJob,
    job_index: int,
    baseline_desired_retention_values: tuple[float, ...],
    baseline_metrics: list[CandidateMetrics],
    baseline_hypervolume: float,
    portfolio_hypervolume: float,
    hypervolume_improvement: float,
    final_population_hypervolume: float,
    final_population_hypervolume_improvement: float,
    reference_point: ObjectivePoint,
    selected_children: list[SelectedAPPortfolioChild],
    final_population: list[APPortfolioCandidate],
    family_state: list[tuple[float, ...]],
    history: list[dict[str, float]],
    passed: bool,
) -> UserAPPortfolioResult:
    return UserAPPortfolioResult(
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
        base_weights=family_state[job_index],
        history=history,
        passed=passed,
    )


def _write_portfolio_artifacts_adapter(
    *,
    result: UserAPPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    family_context: _APFamilyContext,
    portfolio: APPortfolioSettings,
) -> list[Path]:
    return _write_portfolio_artifacts(
        result=result,
        config=config,
        config_path=config_path,
        settings=settings,
        ap_settings=family_context.ap_settings,
        portfolio=portfolio,
    )


def _write_portfolio_artifacts(
    *,
    result: UserAPPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    ap_settings: APSettings,
    portfolio: APPortfolioSettings,
) -> list[Path]:
    del settings
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_id = _portfolio_id(user_id=result.job.user_id, seed=config.seed)
    artifact_paths: list[Path] = []
    child_summaries: list[dict[str, Any]] = []
    for child in result.selected_children:
        child_dir = output_dir / "policies" / f"policy_{child.portfolio_index}"
        child_dir.mkdir(parents=True, exist_ok=True)
        delta = tuple(
            weight - base_weight
            for weight, base_weight in zip(child.candidate.weights, result.base_weights)
        )
        policy = FSRS6APPolicy(
            base_weights=result.base_weights,
            weights=child.candidate.weights,
            delta=delta,
            search_vector=child.candidate.search_vector,
            baseline_desired_retention=child.candidate.desired_retention,
            weight_delta_scale=ap_settings.weight_delta_scale,
            title=(
                f"fsrs6_ap_portfolio_u{result.job.user_id}_"
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
            weight_delta_scale=ap_settings.weight_delta_scale,
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
        metadata_dir = metadata_path.parent
        _write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_kind": "scheduler-policy",
                "artifact_id": f"{portfolio_id}-policy-{child.portfolio_index}",
                "family": config.family,
                "scheduler_name": "fsrs6_ap",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": FEATURE_VERSION,
                "action_space": "fsrs6_ap_weight_delta_portfolio_child",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "baseline_desired_retention": None,
                "scheduler_desired_retention": child.candidate.desired_retention,
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
                "optimizer": "sms_emoa",
                "capabilities": ["event", "batched"],
            },
        )
        artifact_paths.append(metadata_path)
        child_summaries.append(
            {
                "portfolio_index": child.portfolio_index,
                "candidate_id": child.candidate.candidate_id,
                "scheduler_desired_retention": child.candidate.desired_retention,
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
            "scheduler_name": "fsrs6_ap",
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
            "ap_settings": ap_settings.to_dict(),
            "portfolio_settings": asdict(portfolio),
            "history": result.history,
        },
    )
    return artifact_paths


def _portfolio_id(*, user_id: int, seed: int) -> str:
    return f"fsrs6-ap-portfolio-user-{user_id}-seed-{seed}"


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
    candidates: Sequence[APPortfolioCandidate],
    population_size: int,
    reference: ObjectivePoint,
) -> Any:
    return _common_selection_payload(
        baseline_points=baseline_points,
        candidates=candidates,
        population_size=population_size,
        reference=reference,
    )


def _build_outcome(
    *,
    job: APPortfolioTrainJob,
    passed: bool,
    artifact_paths: tuple[Path, ...],
    progress_path: Path,
    error: str | None,
) -> APPortfolioTrainOutcome:
    return APPortfolioTrainOutcome(
        job=job,
        passed=passed,
        artifact_paths=artifact_paths,
        progress_path=progress_path,
        error=error,
    )


_progress_for_jobs = progress_for_jobs


_ADAPTER = PortfolioFamilyAdapter(
    settings_from_mapping=APPortfolioSettings.from_mapping,
    build_family_context=_build_family_context,
    progress_payload=_progress_payload,
    initial_populations=_initial_populations,
    prepare_family_state=_prepare_family_state,
    evaluate_candidates=_evaluate_candidates,
    make_offspring=_make_offspring,
    selected_child_from_candidate=_selected_child_from_candidate,
    build_result=_build_result,
    write_artifacts=_write_portfolio_artifacts_adapter,
    build_outcome=_build_outcome,
)


if __name__ == "__main__":
    raise SystemExit(main())

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
from experiments.rl_scheduler.portfolio_selection import ObjectivePoint
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
from simulator.anki_sm2_ap_policy import (
    ANKI_SM2_AP_DEFAULT_PARAMS,
    ANKI_SM2_AP_DEFAULT_STDDEV,
    ANKI_SM2_AP_PARAM_BOUNDS,
    DEFAULT_PARAMETER_DELTA_SCALE,
    FEATURE_VERSION,
    PARAM_COUNT,
    AnkiSM2APPolicy,
)
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.schedulers.anki_sm2 import AnkiSM2BatchSchedulerOps, AnkiSM2Scheduler


@dataclass(frozen=True, slots=True)
class AnkiSM2APSettings:
    parameter_delta_scale: float = DEFAULT_PARAMETER_DELTA_SCALE

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> AnkiSM2APSettings:
        raw = config.training_ap
        return cls(
            parameter_delta_scale=_float(
                raw.get("parameter_delta_scale", DEFAULT_PARAMETER_DELTA_SCALE),
                "training.ap.parameter_delta_scale",
                0.0,
            )
        )

    def to_dict(self) -> dict[str, float]:
        return {"parameter_delta_scale": self.parameter_delta_scale}


@dataclass(frozen=True, slots=True)
class AnkiSM2APPortfolioSettings:
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
    ) -> AnkiSM2APPortfolioSettings:
        del settings, default_seed_retention_values
        defaults = cls()
        seed_retention_values = _optional_float_tuple(
            raw.get("seed_retention_values"),
            "training.portfolio.seed_retention_values",
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


@dataclass(frozen=True, slots=True)
class AnkiSM2APPortfolioTrainJob:
    user_id: int
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AnkiSM2APPortfolioTrainOutcome:
    job: AnkiSM2APPortfolioTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AnkiSM2APPortfolioCandidate:
    candidate_id: int
    search_vector: tuple[float, ...]
    params: tuple[float, ...]
    metrics: CandidateMetrics

    @property
    def point(self) -> ObjectivePoint:
        from experiments.rl_scheduler.portfolio_selection import point_from_metrics

        return point_from_metrics(self.metrics)


@dataclass(frozen=True, slots=True)
class SelectedAnkiSM2APPortfolioChild:
    portfolio_index: int
    candidate: AnkiSM2APPortfolioCandidate
    hypervolume_contribution: float
    pareto_rank: int


@dataclass(frozen=True, slots=True)
class UserAnkiSM2APPortfolioResult:
    job: AnkiSM2APPortfolioTrainJob
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    portfolio_hypervolume: float
    hypervolume_improvement: float
    final_population_hypervolume: float
    final_population_hypervolume_improvement: float
    reference_point: ObjectivePoint
    selected_children: list[SelectedAnkiSM2APPortfolioChild]
    final_population: list[AnkiSM2APPortfolioCandidate]
    base_params: tuple[float, ...]
    history: list[dict[str, float]]
    passed: bool


@dataclass(frozen=True, slots=True)
class _AnkiSM2APFamilyContext:
    ap_settings: AnkiSM2APSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Anki SM2 AP policy portfolio with SMS-EMOA.",
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
            AnkiSM2APPortfolioTrainJob(
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
    jobs: Sequence[AnkiSM2APPortfolioTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
    execution_mode: str = "in_process_batch",
) -> list[AnkiSM2APPortfolioTrainOutcome]:
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
) -> _AnkiSM2APFamilyContext:
    del raw_training_policy_search, baseline_dr_values
    return _AnkiSM2APFamilyContext(ap_settings=AnkiSM2APSettings.from_config(config))


def _progress_payload(*, family_context: _AnkiSM2APFamilyContext) -> Mapping[str, Any]:
    return {
        "ap": family_context.ap_settings.to_dict(),
        "feature_version": FEATURE_VERSION,
        "param_bounds": [list(bounds) for bounds in ANKI_SM2_AP_PARAM_BOUNDS],
    }


def _prepare_family_state(
    *,
    bundle: Any,
    jobs: Sequence[AnkiSM2APPortfolioTrainJob],
    portfolio: AnkiSM2APPortfolioSettings,
    family_context: _AnkiSM2APFamilyContext,
) -> list[tuple[float, ...]]:
    del bundle, portfolio, family_context
    return [ANKI_SM2_AP_DEFAULT_PARAMS for _job in jobs]


def _evaluate_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    portfolio: AnkiSM2APPortfolioSettings,
    family_context: _AnkiSM2APFamilyContext,
    family_state: list[tuple[float, ...]],
    bundle: Any,
    candidates_by_job: Sequence[Sequence[AnkiSM2APPortfolioCandidate]],
    seed: int,
) -> list[list[AnkiSM2APPortfolioCandidate]]:
    del portfolio, family_state
    metrics_by_job, params_by_job = _evaluate_anki_sm2_ap_portfolio_candidates(
        config=config,
        settings=settings,
        ap_settings=family_context.ap_settings,
        bundle=bundle,
        search_vectors_by_job=[
            [candidate.search_vector for candidate in candidates]
            for candidates in candidates_by_job
        ],
        seed=seed,
    )
    return [
        [
            AnkiSM2APPortfolioCandidate(
                candidate_id=candidates[index].candidate_id,
                search_vector=candidates[index].search_vector,
                params=params_by_job[job_index][index],
                metrics=metrics_by_job[job_index][index],
            )
            for index in range(len(candidates))
        ]
        for job_index, candidates in enumerate(candidates_by_job)
    ]


def _evaluate_anki_sm2_ap_portfolio_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    ap_settings: AnkiSM2APSettings,
    bundle: Any,
    search_vectors_by_job: Sequence[Sequence[tuple[float, ...]]],
    seed: int,
) -> tuple[list[list[CandidateMetrics]], list[list[tuple[float, ...]]]]:
    candidate_count = len(search_vectors_by_job[0])
    if candidate_count < 1:
        raise ValueError("At least one candidate is required.")
    if any(len(row) != candidate_count for row in search_vectors_by_job):
        raise ValueError("All jobs must evaluate the same number of candidates.")
    flat_vectors = torch.tensor(
        [vector for job_vectors in search_vectors_by_job for vector in job_vectors],
        device=bundle.device,
        dtype=torch.float32,
    )
    scheduler_params = _decode_parameter_delta_tensor(
        search_vectors=flat_vectors,
        parameter_delta_scale=ap_settings.parameter_delta_scale,
    )
    default_scheduler = AnkiSM2Scheduler()
    sched_ops = AnkiSM2BatchSchedulerOps(
        graduating_interval=scheduler_params[:, 0],
        easy_interval=scheduler_params[:, 1],
        ease_start=scheduler_params[:, 2],
        easy_bonus=scheduler_params[:, 3],
        hard_interval_factor=scheduler_params[:, 4],
        new_interval_factor=scheduler_params[:, 5],
        interval_multiplier=scheduler_params[:, 6],
        ease_min=default_scheduler.ease_min,
        ease_max=default_scheduler.ease_max,
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
    flat_params = [
        tuple(float(value) for value in row)
        for row in scheduler_params.detach().cpu().tolist()
    ]
    return (
        [
            flat_metrics[index * candidate_count : (index + 1) * candidate_count]
            for index in range(len(search_vectors_by_job))
        ],
        [
            flat_params[index * candidate_count : (index + 1) * candidate_count]
            for index in range(len(search_vectors_by_job))
        ],
    )


def _decode_parameter_delta_tensor(
    *,
    search_vectors: torch.Tensor,
    parameter_delta_scale: float,
) -> torch.Tensor:
    base = torch.tensor(
        ANKI_SM2_AP_DEFAULT_PARAMS,
        device=search_vectors.device,
        dtype=search_vectors.dtype,
    )
    stddev = torch.tensor(
        ANKI_SM2_AP_DEFAULT_STDDEV,
        device=search_vectors.device,
        dtype=search_vectors.dtype,
    )
    bounds = torch.tensor(
        ANKI_SM2_AP_PARAM_BOUNDS,
        device=search_vectors.device,
        dtype=search_vectors.dtype,
    )
    raw = base + search_vectors * stddev * float(parameter_delta_scale)
    return torch.minimum(torch.maximum(raw, bounds[:, 0]), bounds[:, 1])


def _initial_populations(
    *,
    jobs: Sequence[AnkiSM2APPortfolioTrainJob],
    settings: PolicySearchSettings,
    portfolio: AnkiSM2APPortfolioSettings,
    family_context: _AnkiSM2APFamilyContext,
    seed_retention_values_by_job: Sequence[Sequence[float]],
    device: torch.device,
    seed: int,
) -> tuple[list[list[AnkiSM2APPortfolioCandidate]], list[int]]:
    del settings, family_context, seed_retention_values_by_job
    populations: list[list[AnkiSM2APPortfolioCandidate]] = []
    next_ids: list[int] = []
    for job in jobs:
        generator = generator_for_job(device=device, seed=seed, user_id=job.user_id)
        candidates: list[AnkiSM2APPortfolioCandidate] = []
        for index in range(portfolio.population_size):
            search_vector = (
                _zero_search_vector()
                if index == 0
                else _mutate_search_vector(
                    _zero_search_vector(),
                    mutation_scale=portfolio.mutation_scale,
                    device=device,
                    generator=generator,
                )
            )
            candidates.append(
                AnkiSM2APPortfolioCandidate(
                    candidate_id=index,
                    search_vector=search_vector,
                    params=ANKI_SM2_AP_DEFAULT_PARAMS,
                    metrics=zero_metrics(),
                )
            )
        populations.append(candidates)
        next_ids.append(portfolio.population_size)
    return populations, next_ids


def _make_offspring(
    *,
    population: Sequence[AnkiSM2APPortfolioCandidate],
    next_candidate_id: int,
    settings: PolicySearchSettings,
    portfolio: AnkiSM2APPortfolioSettings,
    family_context: _AnkiSM2APFamilyContext,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[AnkiSM2APPortfolioCandidate], int]:
    del settings, family_context
    candidates: list[AnkiSM2APPortfolioCandidate] = []
    for _index in range(portfolio.offspring_size):
        parent_index = int(
            torch.randint(
                len(population),
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        search_vector = _mutate_search_vector(
            population[parent_index].search_vector,
            mutation_scale=portfolio.mutation_scale,
            device=device,
            generator=generator,
        )
        candidates.append(
            AnkiSM2APPortfolioCandidate(
                candidate_id=next_candidate_id,
                search_vector=search_vector,
                params=ANKI_SM2_AP_DEFAULT_PARAMS,
                metrics=zero_metrics(),
            )
        )
        next_candidate_id += 1
    return candidates, next_candidate_id


def _mutate_search_vector(
    search_vector: Sequence[float],
    *,
    mutation_scale: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float, ...]:
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
    return _search_vector_tuple(mutated.detach().cpu().tolist())


def _search_vector_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != PARAM_COUNT:
        raise ValueError(f"search_vector must contain {PARAM_COUNT} values.")
    return result


def _zero_search_vector() -> tuple[float, ...]:
    return (0.0,) * PARAM_COUNT


def _selected_child_from_candidate(
    *,
    portfolio_index: int,
    candidate: AnkiSM2APPortfolioCandidate,
    hypervolume_contribution: float,
    pareto_rank: int,
) -> SelectedAnkiSM2APPortfolioChild:
    return SelectedAnkiSM2APPortfolioChild(
        portfolio_index=portfolio_index,
        candidate=candidate,
        hypervolume_contribution=hypervolume_contribution,
        pareto_rank=pareto_rank,
    )


def _select_portfolio_children(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[AnkiSM2APPortfolioCandidate],
    portfolio_size: int,
    reference: ObjectivePoint,
) -> list[SelectedAnkiSM2APPortfolioChild]:
    return _common_select_portfolio_children(
        baseline_points=baseline_points,
        candidates=candidates,
        portfolio_size=portfolio_size,
        reference=reference,
        selected_child_from_candidate=_selected_child_from_candidate,
    )


def _build_result(
    *,
    job: AnkiSM2APPortfolioTrainJob,
    job_index: int,
    baseline_desired_retention_values: tuple[float, ...],
    baseline_metrics: list[CandidateMetrics],
    baseline_hypervolume: float,
    portfolio_hypervolume: float,
    hypervolume_improvement: float,
    final_population_hypervolume: float,
    final_population_hypervolume_improvement: float,
    reference_point: ObjectivePoint,
    selected_children: list[SelectedAnkiSM2APPortfolioChild],
    final_population: list[AnkiSM2APPortfolioCandidate],
    family_state: list[tuple[float, ...]],
    history: list[dict[str, float]],
    passed: bool,
) -> UserAnkiSM2APPortfolioResult:
    return UserAnkiSM2APPortfolioResult(
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
        base_params=family_state[job_index],
        history=history,
        passed=passed,
    )


def _write_portfolio_artifacts_adapter(
    *,
    result: UserAnkiSM2APPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    family_context: _AnkiSM2APFamilyContext,
    portfolio: AnkiSM2APPortfolioSettings,
) -> list[Path]:
    del settings
    return _write_portfolio_artifacts(
        result=result,
        config=config,
        config_path=config_path,
        ap_settings=family_context.ap_settings,
        portfolio=portfolio,
    )


def _write_portfolio_artifacts(
    *,
    result: UserAnkiSM2APPortfolioResult,
    config: ExperimentConfig,
    config_path: Path,
    ap_settings: AnkiSM2APSettings,
    portfolio: AnkiSM2APPortfolioSettings,
) -> list[Path]:
    output_dir = result.job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_id = _portfolio_id(user_id=result.job.user_id, seed=config.seed)
    artifact_paths: list[Path] = []
    child_summaries: list[dict[str, Any]] = []
    for child in result.selected_children:
        child_dir = output_dir / "policies" / f"policy_{child.portfolio_index}"
        child_dir.mkdir(parents=True, exist_ok=True)
        delta = tuple(
            param - base_param
            for param, base_param in zip(
                child.candidate.params,
                result.base_params,
                strict=True,
            )
        )
        policy = AnkiSM2APPolicy(
            base_params=result.base_params,
            params=child.candidate.params,
            delta=delta,
            search_vector=child.candidate.search_vector,
            parameter_delta_scale=ap_settings.parameter_delta_scale,
            title=(
                f"anki_sm2_ap_portfolio_u{result.job.user_id}_"
                f"policy_{child.portfolio_index}"
            ),
        )
        policy_path = child_dir / "policy.json"
        policy.write_json(policy_path)
        metrics_path = child_dir / "metrics.json"
        clipped_dimensions = _clipped_dimension_count(
            search_vector=child.candidate.search_vector,
            params=child.candidate.params,
            parameter_delta_scale=ap_settings.parameter_delta_scale,
        )
        _write_json(
            metrics_path,
            {
                "candidate_id": child.candidate.candidate_id,
                "portfolio_id": portfolio_id,
                "portfolio_index": child.portfolio_index,
                "pareto_rank": child.pareto_rank,
                "hypervolume_contribution": child.hypervolume_contribution,
                "baseline_hypervolume": result.baseline_hypervolume,
                "portfolio_hypervolume": result.portfolio_hypervolume,
                "hypervolume_improvement": result.hypervolume_improvement,
                "final_population_hypervolume": result.final_population_hypervolume,
                "metrics": asdict(child.candidate.metrics),
                "base_params": list(result.base_params),
                "scheduler_params": list(child.candidate.params),
                "scheduler_params_by_name": policy.params_dict(),
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
                "scheduler_name": "anki_sm2_ap",
                "environment": config.simulation.environment,
                "engine": config.simulation.engine,
                "training_user_ids": [result.job.user_id],
                "validation_user_ids": list(config.users.validation),
                "seed": config.seed,
                "policy_path": "policy.json",
                "feature_version": FEATURE_VERSION,
                "action_space": "anki_sm2_ap_params_portfolio_child",
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "code_commit": _git_commit(),
                "baseline_desired_retention": None,
                "scheduler_params": list(child.candidate.params),
                "scheduler_params_by_name": policy.params_dict(),
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
                "policy_path": _relative_path_string(policy_path, base=output_dir),
                "metadata_path": _relative_path_string(
                    metadata_path,
                    base=output_dir,
                ),
                "metrics_path": _relative_path_string(metrics_path, base=output_dir),
                "hypervolume_contribution": child.hypervolume_contribution,
                "pareto_rank": child.pareto_rank,
                "scheduler_params": list(child.candidate.params),
                "metrics": asdict(child.candidate.metrics),
            }
        )

    _write_json(
        output_dir / "portfolio.json",
        {
            "schema_version": SCHEMA_VERSION,
            "portfolio_id": portfolio_id,
            "scheduler_name": "anki_sm2_ap",
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


def _clipped_dimension_count(
    *,
    search_vector: Sequence[float],
    params: Sequence[float],
    parameter_delta_scale: float,
) -> int:
    raw_params = tuple(
        base + offset * stddev * float(parameter_delta_scale)
        for base, offset, stddev in zip(
            ANKI_SM2_AP_DEFAULT_PARAMS,
            search_vector,
            ANKI_SM2_AP_DEFAULT_STDDEV,
            strict=True,
        )
    )
    return sum(
        1
        for raw, actual, (lower, upper) in zip(
            raw_params,
            params,
            ANKI_SM2_AP_PARAM_BOUNDS,
            strict=True,
        )
        if abs(raw - actual) > 1e-6
        and (abs(actual - lower) <= 1e-6 or abs(actual - upper) <= 1e-6)
    )


def _portfolio_id(*, user_id: int, seed: int) -> str:
    return f"anki-sm2-ap-portfolio-user-{user_id}-seed-{seed}"


def _clear_cuda_cache(device: torch.device) -> None:
    _common_clear_cuda_cache(device)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    return _common_float_tuple(value, field_name)


def _optional_float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    return _common_optional_float_tuple(value, field_name)


def _selection_payload(
    *,
    baseline_points: Sequence[ObjectivePoint],
    candidates: Sequence[AnkiSM2APPortfolioCandidate],
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
    job: AnkiSM2APPortfolioTrainJob,
    passed: bool,
    artifact_paths: tuple[Path, ...],
    progress_path: Path,
    error: str | None,
) -> AnkiSM2APPortfolioTrainOutcome:
    return AnkiSM2APPortfolioTrainOutcome(
        job=job,
        passed=passed,
        artifact_paths=artifact_paths,
        progress_path=progress_path,
        error=error,
    )


_progress_for_jobs = progress_for_jobs


_ADAPTER = PortfolioFamilyAdapter(
    settings_from_mapping=AnkiSM2APPortfolioSettings.from_mapping,
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

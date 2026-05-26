from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

import cma
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.rl_scheduler.policy_search_common import (
    CMAESSettings,
    CandidateMetrics,
    PolicySearchSettings,
    TrainingProgress,
    _baseline_dr_values,
    _build_bundle,
    _git_commit,
    _metrics_from_stats,
    _optimizer_seed as _adr_optimizer_seed,
    _read_training_policy_search,
    _relative_path_string,
    _write_json,
)
from experiments.rl_scheduler.portfolio_selection import (
    ObjectivePoint,
    objective_hypervolume_2d,
    point_from_metrics,
    reference_point,
)
from simulator.batched_engine.multiuser_engine import simulate_multiuser
from simulator.batched_sweep.fsrs6_cost_adr_policy import DEFAULT_COST_WEIGHTS
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.experiment_infra.baseline_dr_selection import load_baseline_dr_manifest
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.fsrs6_cost_conditioned_adr_policy import (
    ACTION_HEAD_INTERVAL,
    FEATURE_VERSION_INTERVAL_MONO,
    FSRS6CostConditionedADRPolicy,
)
from simulator.math.fsrs import Bounds
from simulator.scheduler_catalog import fsrs6_cost_adr_action_space_for_feature_version
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.fsrs6_cost_conditioned_adr import (
    FSRS6CostConditionedADRBatchSchedulerOps,
)
from simulator.short_term_config import resolve_short_term_config


PARAMETER_COUNT = 24
MAX_INTERVAL_DAYS = 36500.0
REFERENCE_MARGIN_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class CostADRTrainingResult:
    baseline_desired_retention_values: tuple[float, ...]
    baseline_metrics: list[CandidateMetrics]
    baseline_hypervolume: float
    reference_point: ObjectivePoint
    best_cost_weight_metrics: list[CandidateMetrics]
    best_coefficients: torch.Tensor
    best_hypervolume: float
    best_hypervolume_delta: float
    history: list[dict[str, float]]
    passed: bool


@dataclass(frozen=True, slots=True)
class CostADRTrainJob:
    user_id: int
    output_dir: Path
    command_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CostADRTrainJobResult:
    job: CostADRTrainJob
    passed: bool
    artifact_paths: tuple[Path, ...]
    progress_path: Path | None
    error: str | None = None


@dataclass(slots=True)
class _PreparedCostADRJob:
    job: CostADRTrainJob
    progress: TrainingProgress
    baseline_drs: tuple[float, ...]
    optimizer_seed: int
    optimizer: Any
    baseline_metrics: list[CandidateMetrics] = field(default_factory=list)
    baseline_points: list[ObjectivePoint] = field(default_factory=list)
    baseline_hypervolume: float = 0.0
    reference_point: ObjectivePoint | None = None
    best_cost_weight_metrics: list[CandidateMetrics] | None = None
    best_coefficients: torch.Tensor | None = None
    best_hypervolume: float = float("-inf")
    best_hypervolume_delta: float = float("-inf")
    history: list[dict[str, float]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FSRS6 cost-conditioned ADR policies with CMA-ES.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-command-path", type=Path, default=None)
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE_PATH,
        help="Path to Anki button usage JSONL.",
    )
    parser.add_argument("--srs-benchmark-root", type=Path, default=None)
    parser.add_argument("--benchmark-result", default=None)
    parser.add_argument("--benchmark-partition", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig.from_toml(args.config)
    results = run_training_jobs(
        jobs=[
            CostADRTrainJob(
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
    )
    return 0 if results and results[0].passed else 1


def optimizer_settings_from_mapping(raw: Mapping[str, Any]) -> CMAESSettings:
    return CMAESSettings.from_mapping(
        raw,
        coefficient_count=PARAMETER_COUNT,
        coefficient_min=-12.0,
        coefficient_max=12.0,
    )


def run_training_batch_jobs(
    *,
    jobs: Sequence[Any],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
) -> list[CostADRTrainJobResult]:
    return run_training_jobs(
        jobs=[
            CostADRTrainJob(
                user_id=job.user_id,
                output_dir=job.output_dir,
                command_record_path=job.command_record_path,
            )
            for job in jobs
        ],
        config=config,
        config_path=config_path,
        repo_root=repo_root,
        button_usage=DEFAULT_BUTTON_USAGE_PATH,
    )


def run_training_jobs(
    *,
    jobs: Sequence[CostADRTrainJob],
    config: ExperimentConfig,
    config_path: Path,
    repo_root: Path,
    button_usage: Path | None = DEFAULT_BUTTON_USAGE_PATH,
    srs_benchmark_root: Path | None = None,
    benchmark_result: str | None = None,
    benchmark_partition: str | None = None,
) -> list[CostADRTrainJobResult]:
    if not jobs:
        return []

    settings = PolicySearchSettings.from_mapping(config.training_policy_search)
    raw_training_policy_search = dict(_read_training_policy_search(config_path))
    optimizer_settings = optimizer_settings_from_mapping(config.training_optimizer)
    cost_weights = DEFAULT_COST_WEIGHTS
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_policy_search.get("learning_steps"),
        relearning_steps=raw_training_policy_search.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    device = torch.device(settings.torch_device)
    benchmark_root = resolve_benchmark_root(repo_root, srs_benchmark_root).resolve()
    overrides = parse_result_overrides(benchmark_result)
    progress_by_job = [
        (job, _progress_for_job(job=job, config_path=config_path)) for job in jobs
    ]
    batched_user_ids = [job.user_id for job in jobs]
    batched_fields = {
        "batched_user_count": len(jobs),
        "batched_user_ids": batched_user_ids,
    }

    states: list[_PreparedCostADRJob] = []
    results_by_job: dict[CostADRTrainJob, CostADRTrainingResult] = {}
    try:
        for job, progress in progress_by_job:
            baseline_drs = _baseline_dr_values_for_user(
                config=config,
                repo_root=repo_root,
                raw_training_policy_search=raw_training_policy_search,
                settings=settings,
                user_id=job.user_id,
            )
            optimizer_seed = _optimizer_seed(
                config=config,
                optimizer_settings=optimizer_settings,
                user_id=job.user_id,
            )
            progress.write(
                "config_loaded",
                settings=asdict(settings),
                optimizer=optimizer_settings.to_dict(),
                optimizer_seed=optimizer_seed,
                feature_version=FEATURE_VERSION_INTERVAL_MONO,
                action_head=ACTION_HEAD_INTERVAL,
                parameter_count=PARAMETER_COUNT,
                cost_weights=list(cost_weights),
                baseline_desired_retention_values=list(baseline_drs),
                simulation=config.simulation.to_dict(),
                seed=config.seed,
                **batched_fields,
            )
            progress.write(
                "device_resolved",
                device=device,
                torch_device=str(device),
                **batched_fields,
            )
            states.append(
                _PreparedCostADRJob(
                    job=job,
                    progress=progress,
                    baseline_drs=baseline_drs,
                    optimizer_seed=optimizer_seed,
                    optimizer=_make_strategy(
                        optimizer_settings=optimizer_settings,
                        optimizer_seed=optimizer_seed,
                    ),
                )
            )

        baseline_lane_user_ids = [
            state.job.user_id for state in states for _dr in state.baseline_drs
        ]
        baseline_bundle = _build_bundle(
            config=config,
            settings=settings,
            lane_user_ids=baseline_lane_user_ids,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        baseline_metrics_by_job = _evaluate_baseline_grids_multiuser(
            config=config,
            settings=settings,
            bundle=baseline_bundle,
            jobs=[state.job for state in states],
            baseline_drs_by_job={state.job: state.baseline_drs for state in states},
            seed=config.seed,
        )
        for state in states:
            baseline_metrics = baseline_metrics_by_job[state.job]
            baseline_points = [
                point_from_metrics(metric) for metric in baseline_metrics
            ]
            hv_reference = reference_point(
                baseline_points,
                margin_fraction=REFERENCE_MARGIN_FRACTION,
            )
            baseline_hv = objective_hypervolume_2d(
                baseline_points,
                reference=hv_reference,
            )
            state.baseline_metrics = baseline_metrics
            state.baseline_points = baseline_points
            state.baseline_hypervolume = baseline_hv
            state.reference_point = hv_reference
            state.progress.write(
                "baseline_grid_evaluated",
                device=baseline_bundle.device,
                effective_lanes=len(baseline_lane_user_ids),
                baseline_hypervolume=baseline_hv,
                reference_point=asdict(hv_reference),
                metrics=[
                    {"baseline_desired_retention": dr, **asdict(metric)}
                    for dr, metric in zip(
                        state.baseline_drs,
                        baseline_metrics,
                        strict=True,
                    )
                ],
                **batched_fields,
            )
        del baseline_bundle
        _clear_cuda_cache(device)

        train_lane_user_ids = [
            state.job.user_id
            for state in states
            for _candidate in range(optimizer_settings.population_size)
            for _weight in cost_weights
        ]
        train_bundle = _build_bundle(
            config=config,
            settings=settings,
            lane_user_ids=train_lane_user_ids,
            benchmark_root=benchmark_root,
            overrides=overrides,
            benchmark_partition=benchmark_partition,
            button_usage=button_usage,
            device=device,
            short_term_source=short_term_source,
            learning_steps=learning_steps,
            relearning_steps=relearning_steps,
        )
        effective_lanes = len(train_lane_user_ids)
        for state in states:
            state.progress.write(
                "train_bundle_built",
                device=train_bundle.device,
                effective_lanes=effective_lanes,
                population_size=optimizer_settings.population_size,
                cost_weight_count=len(cost_weights),
                **batched_fields,
            )
        results_by_job = _run_cmaes_multiuser(
            config=config,
            settings=settings,
            optimizer_settings=optimizer_settings,
            bundle=train_bundle,
            prepared_jobs=states,
            cost_weights=cost_weights,
            effective_lanes=effective_lanes,
            batched_user_ids=batched_user_ids,
        )
        for state in states:
            result = results_by_job[state.job]
            state.progress.write(
                "cmaes_completed",
                device=train_bundle.device,
                best_hypervolume=result.best_hypervolume,
                best_hypervolume_delta=result.best_hypervolume_delta,
                generations=len(result.history),
                passed=result.passed,
                effective_lanes=effective_lanes,
                **batched_fields,
            )
    except Exception as exc:  # noqa: BLE001 - mark every job in this batch failed.
        error = str(exc)
        for _job, progress in progress_by_job:
            progress.write("failed", error=error, **batched_fields)
        return [
            CostADRTrainJobResult(
                job=job,
                passed=False,
                artifact_paths=(),
                progress_path=progress.path,
                error=error,
            )
            for job, progress in progress_by_job
        ]

    outcomes: list[CostADRTrainJobResult] = []
    for state in states:
        job = state.job
        progress = state.progress
        try:
            result = results_by_job[job]
            policy_path, metrics_path, metadata_path = write_artifact(
                output_dir=job.output_dir,
                config=config,
                config_path=config_path,
                settings=settings,
                user_id=job.user_id,
                training_command_path=job.command_record_path,
                optimizer_settings=optimizer_settings,
                optimizer_seed=state.optimizer_seed,
                cost_weights=cost_weights,
                result=result,
            )
            progress.write(
                "artifacts_written",
                device=train_bundle.device,
                passed=result.passed,
                policy_path=_relative_path_string(policy_path, base=job.output_dir),
                metrics_path=_relative_path_string(metrics_path, base=job.output_dir),
                metadata_path=_relative_path_string(
                    metadata_path,
                    base=job.output_dir,
                ),
                **batched_fields,
            )
            outcomes.append(
                CostADRTrainJobResult(
                    job=job,
                    passed=result.passed,
                    artifact_paths=(metadata_path,),
                    progress_path=progress.path,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve batch outcomes.
            progress.write("failed", error=str(exc), **batched_fields)
            outcomes.append(
                CostADRTrainJobResult(
                    job=job,
                    passed=False,
                    artifact_paths=(),
                    progress_path=progress.path,
                    error=str(exc),
                )
            )
    return outcomes


def _baseline_dr_values_for_user(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    raw_training_policy_search: Mapping[str, Any],
    settings: PolicySearchSettings,
    user_id: int,
) -> tuple[float, ...]:
    if config.baseline_dr_selection.manifest is None:
        return _baseline_dr_values(raw_training_policy_search, settings)
    manifest_path = config.baseline_dr_selection.manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    manifest = load_baseline_dr_manifest(
        manifest_path,
        target_count=config.baseline_dr_selection.target_count,
        user_ids=[user_id],
        tolerance=config.baseline_dr_selection.tolerance,
    )
    values = manifest.values_for_user(user_id)
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "baseline_dr_selection manifest values must be inside "
                "training.policy_search retention bounds."
            )
    return values


def _make_strategy(
    *,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
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
            "seed": optimizer_seed,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
        },
    )


def _evaluate_baseline_grids_multiuser(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    jobs: Sequence[CostADRTrainJob],
    baseline_drs_by_job: Mapping[CostADRTrainJob, tuple[float, ...]],
    seed: int,
) -> dict[CostADRTrainJob, list[CandidateMetrics]]:
    flat_drs = [dr for job in jobs for dr in baseline_drs_by_job[job]]
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            flat_drs,
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
    if len(stats) != len(flat_drs):
        raise RuntimeError(
            "simulate_multiuser returned an unexpected baseline lane count: "
            f"{len(stats)} != {len(flat_drs)}."
        )
    metrics = [_metrics_from_stats(item) for item in stats]
    by_job: dict[CostADRTrainJob, list[CandidateMetrics]] = {}
    offset = 0
    for job in jobs:
        baseline_drs = baseline_drs_by_job[job]
        next_offset = offset + len(baseline_drs)
        by_job[job] = metrics[offset:next_offset]
        offset = next_offset
    return by_job


def _run_cmaes_multiuser(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    optimizer_settings: CMAESSettings,
    bundle: Any,
    prepared_jobs: Sequence[_PreparedCostADRJob],
    cost_weights: tuple[float, ...],
    effective_lanes: int,
    batched_user_ids: Sequence[int],
) -> dict[CostADRTrainJob, CostADRTrainingResult]:
    jobs = [state.job for state in prepared_jobs]
    batched_fields = {
        "batched_user_count": len(prepared_jobs),
        "batched_user_ids": list(batched_user_ids),
    }
    for generation in range(optimizer_settings.generations):
        solutions_by_job: list[list[list[float]]] = []
        for state in prepared_jobs:
            solutions = [list(map(float, item)) for item in state.optimizer.ask()]
            if len(solutions) != optimizer_settings.population_size:
                raise RuntimeError(
                    "CMA-ES returned an unexpected population size: "
                    f"{len(solutions)} != {optimizer_settings.population_size}."
                )
            solutions_by_job.append(solutions)

        coefficients_by_job = torch.tensor(
            solutions_by_job,
            device=bundle.device,
            dtype=torch.float32,
        )
        metrics_by_job = _evaluate_cost_adr_candidates_multiuser(
            config=config,
            settings=settings,
            bundle=bundle,
            jobs=jobs,
            coefficients_by_job=coefficients_by_job,
            cost_weights=cost_weights,
            seed=config.seed,
        )
        for job_index, state in enumerate(prepared_jobs):
            reference = state.reference_point
            if reference is None:
                raise RuntimeError("Cost ADR baseline reference was not initialized.")
            metrics_by_candidate = metrics_by_job[state.job]
            scores = [
                objective_hypervolume_2d(
                    [
                        *state.baseline_points,
                        *[point_from_metrics(metric) for metric in candidate_metrics],
                    ],
                    reference=reference,
                )
                - state.baseline_hypervolume
                for candidate_metrics in metrics_by_candidate
            ]
            state.optimizer.tell(
                solutions_by_job[job_index],
                [-score for score in scores],
            )
            generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
            generation_best_score = float(scores[generation_best_idx])
            generation_best_hv = state.baseline_hypervolume + generation_best_score
            if generation_best_score > state.best_hypervolume_delta:
                state.best_hypervolume_delta = generation_best_score
                state.best_hypervolume = generation_best_hv
                state.best_coefficients = (
                    coefficients_by_job[job_index, generation_best_idx].detach().clone()
                )
                state.best_cost_weight_metrics = metrics_by_candidate[
                    generation_best_idx
                ]
            history_entry = {
                "generation": float(generation),
                "sigma": float(state.optimizer.sigma),
                "best_hypervolume_delta": float(state.best_hypervolume_delta),
                "generation_best_hypervolume_delta": generation_best_score,
                "mean_hypervolume_delta": float(sum(scores) / max(len(scores), 1)),
                "generation_best_hypervolume": generation_best_hv,
                "baseline_hypervolume": state.baseline_hypervolume,
            }
            state.history.append(history_entry)
            state.progress.write(
                "cmaes_generation",
                device=bundle.device,
                effective_lanes=effective_lanes,
                population_size=optimizer_settings.population_size,
                cost_weight_count=len(cost_weights),
                **batched_fields,
                **history_entry,
            )
    return {state.job: _training_result_from_state(state) for state in prepared_jobs}


def _training_result_from_state(
    state: _PreparedCostADRJob,
) -> CostADRTrainingResult:
    if (
        state.reference_point is None
        or state.best_coefficients is None
        or state.best_cost_weight_metrics is None
    ):
        raise RuntimeError("CMA-ES did not evaluate any cost ADR candidates.")
    return CostADRTrainingResult(
        baseline_desired_retention_values=state.baseline_drs,
        baseline_metrics=state.baseline_metrics,
        baseline_hypervolume=state.baseline_hypervolume,
        reference_point=state.reference_point,
        best_cost_weight_metrics=state.best_cost_weight_metrics,
        best_coefficients=state.best_coefficients.detach().cpu(),
        best_hypervolume=state.best_hypervolume,
        best_hypervolume_delta=state.best_hypervolume_delta,
        history=state.history,
        passed=state.best_hypervolume_delta > 0.0,
    )


def _evaluate_cost_adr_candidates_multiuser(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    jobs: Sequence[CostADRTrainJob],
    coefficients_by_job: torch.Tensor,
    cost_weights: tuple[float, ...],
    seed: int,
) -> dict[CostADRTrainJob, list[list[CandidateMetrics]]]:
    job_count = int(coefficients_by_job.shape[0])
    population_size = int(coefficients_by_job.shape[1])
    if job_count != len(jobs):
        raise ValueError("coefficients_by_job first dimension must match jobs.")
    flat_coefficients = (
        coefficients_by_job[:, :, None, :]
        .expand(
            job_count,
            population_size,
            len(cost_weights),
            coefficients_by_job.shape[2],
        )
        .reshape(
            job_count * population_size * len(cost_weights),
            coefficients_by_job.shape[2],
        )
    )
    policy = _policy_template(settings=settings)
    sched_ops = FSRS6CostConditionedADRBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=policy,
        goal_cost_weight=torch.tensor(
            [
                cost_weight
                for _job in jobs
                for _candidate in range(population_size)
                for cost_weight in cost_weights
            ],
            device=bundle.device,
            dtype=torch.float32,
        ),
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
    expected_lanes = len(jobs) * population_size * len(cost_weights)
    if len(stats) != expected_lanes:
        raise RuntimeError(
            "simulate_multiuser returned an unexpected cost ADR lane count: "
            f"{len(stats)} != {expected_lanes}."
        )
    metrics = [_metrics_from_stats(item) for item in stats]
    by_job: dict[CostADRTrainJob, list[list[CandidateMetrics]]] = {}
    offset = 0
    for job in jobs:
        candidate_metrics: list[list[CandidateMetrics]] = []
        for _candidate in range(population_size):
            next_offset = offset + len(cost_weights)
            candidate_metrics.append(metrics[offset:next_offset])
            offset = next_offset
        by_job[job] = candidate_metrics
    return by_job


def _policy_template(
    *, settings: PolicySearchSettings
) -> FSRS6CostConditionedADRPolicy:
    return FSRS6CostConditionedADRPolicy(
        coefficients=(0.0,) * PARAMETER_COUNT,
        action_head=ACTION_HEAD_INTERVAL,
        feature_version=FEATURE_VERSION_INTERVAL_MONO,
        cost_weight_min=min(DEFAULT_COST_WEIGHTS),
        cost_weight_max=max(DEFAULT_COST_WEIGHTS),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        max_interval_days=MAX_INTERVAL_DAYS,
        bounds=Bounds(),
        title="FSRS6 cost-conditioned ADR CMA-ES template",
    )


def write_artifact(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    user_id: int,
    training_command_path: Path | None,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
    cost_weights: tuple[float, ...],
    result: CostADRTrainingResult,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = FSRS6CostConditionedADRPolicy(
        coefficients=tuple(float(v) for v in result.best_coefficients.tolist()),
        action_head=ACTION_HEAD_INTERVAL,
        feature_version=FEATURE_VERSION_INTERVAL_MONO,
        cost_weight_min=min(cost_weights),
        cost_weight_max=max(cost_weights),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        max_interval_days=MAX_INTERVAL_DAYS,
        bounds=Bounds(),
        title=f"fsrs6_cost_adr_cmaes_u{user_id}",
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)
    optimizer = {
        **optimizer_settings.to_dict(),
        "seed_resolved": optimizer_seed,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(
        metrics_path,
        {
            "passed_overfit_gate": result.passed,
            "training_objective": "hypervolume_delta",
            "baseline_hypervolume": result.baseline_hypervolume,
            "best_hypervolume": result.best_hypervolume,
            "best_hypervolume_delta": result.best_hypervolume_delta,
            "reference_point": asdict(result.reference_point),
            "baseline_desired_retention_values": list(
                result.baseline_desired_retention_values
            ),
            "baseline_metrics": [
                {
                    "baseline_desired_retention": dr,
                    **asdict(metric),
                }
                for dr, metric in zip(
                    result.baseline_desired_retention_values,
                    result.baseline_metrics,
                    strict=True,
                )
            ],
            "cost_weights": list(cost_weights),
            "selected_cost_weight_rollout_points": [
                {"goal_cost_weight": weight, **asdict(metric)}
                for weight, metric in zip(
                    cost_weights,
                    result.best_cost_weight_metrics,
                    strict=True,
                )
            ],
            "feature_version": FEATURE_VERSION_INTERVAL_MONO,
            "action_head": ACTION_HEAD_INTERVAL,
            "parameter_count": PARAMETER_COUNT,
            "optimizer": optimizer,
            "settings": asdict(settings),
            "history": result.history,
        },
    )
    metadata_path = output_dir / "metadata.json"
    metadata_dir = metadata_path.parent
    action_space = fsrs6_cost_adr_action_space_for_feature_version(
        FEATURE_VERSION_INTERVAL_MONO
    )
    _write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "scheduler-policy",
            "artifact_id": f"fsrs6-cost-adr-user-{user_id}-seed-{config.seed}",
            "family": config.family,
            "scheduler_name": "fsrs6_cost_adr",
            "environment": config.simulation.environment,
            "engine": config.simulation.engine,
            "review_markov_transition": config.simulation.review_markov_transition,
            "training_user_ids": [user_id],
            "validation_user_ids": list(config.users.validation),
            "seed": config.seed,
            "policy_path": "policy.json",
            "feature_version": FEATURE_VERSION_INTERVAL_MONO,
            "action_space": action_space,
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "code_commit": _git_commit(),
            "lambda_value": None,
            "baseline_desired_retention": None,
            "training_objective": "hypervolume_delta",
            "config_snapshot_path": _relative_path_string(
                config_path,
                base=metadata_dir,
            ),
            "training_command_path": _relative_path_string(
                training_command_path,
                base=metadata_dir,
            )
            if training_command_path
            else None,
            "metrics_path": "metrics.json",
            "optimizer": "cma_es",
            "optimizer_settings": optimizer,
            "cost_weights": list(cost_weights),
            "best_hypervolume_delta": result.best_hypervolume_delta,
            "capabilities": ["event", "batched"],
        },
    )
    return policy_path, metrics_path, metadata_path


def _optimizer_seed(
    *,
    config: ExperimentConfig,
    optimizer_settings: CMAESSettings,
    user_id: int,
) -> int:
    return _adr_optimizer_seed(
        config=config,
        settings=optimizer_settings,
        user_id=user_id,
        lambda_value=0.0,
    )


def _progress_for_job(*, job: CostADRTrainJob, config_path: Path) -> TrainingProgress:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress(job.output_dir / "training_progress.jsonl")
    progress.write(
        "started",
        config_path=_relative_path_string(config_path, base=job.output_dir),
        user_id=job.user_id,
    )
    return progress


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

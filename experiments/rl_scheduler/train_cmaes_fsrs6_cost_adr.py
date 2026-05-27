from __future__ import annotations

import argparse
import math
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
    objective_non_dominated_indices,
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
HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION = "union_contribution"
HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE = "scheduler_vs_baseline"
SUPPORTED_HYPERVOLUME_DELTA_MODES = frozenset(
    {
        HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
        HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE,
    }
)
INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1 = "first8_distill24_mean_v1"
SUPPORTED_INITIAL_MEAN_SOURCES = frozenset(
    {INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1}
)
FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS = (
    0.00812541801376,
    -0.20082676596,
    -0.3526779501,
    0.173066326435,
    7.15815268972,
    0.146136612706,
    -5.87601533206,
    6.87199212214,
    -0.640635395638,
    -1.73525428091,
    -7.05468459538,
    -0.593849847649,
    -7.02425749479,
    20.8058840713,
    2.09084002135,
    -3.48101100781,
    -18.7692899902,
    0.440663756256,
    -7.29791357225,
    22.6746907186,
    1.55593044161,
    -0.90047226616,
    -8.21368244117,
    -0.638482992454,
)


@dataclass(frozen=True, slots=True)
class CoverageObjectiveSettings:
    enabled: bool = False
    min_budget_span_coverage: float = 0.90
    min_target_span_coverage: float = 0.90
    penalty_weight: float = 0.05
    filter_baseline_dominated: bool = False
    dominated_point_penalty_weight: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CoverageObjectiveSettings:
        defaults = cls()
        return cls(
            enabled=_bool_setting(
                raw.get("coverage_objective_enabled", defaults.enabled),
                "training.policy_search.coverage_objective_enabled",
            ),
            min_budget_span_coverage=_fraction_setting(
                raw.get(
                    "coverage_min_budget_span",
                    defaults.min_budget_span_coverage,
                ),
                "training.policy_search.coverage_min_budget_span",
            ),
            min_target_span_coverage=_fraction_setting(
                raw.get(
                    "coverage_min_target_span",
                    defaults.min_target_span_coverage,
                ),
                "training.policy_search.coverage_min_target_span",
            ),
            penalty_weight=_nonnegative_float_setting(
                raw.get("coverage_penalty_weight", defaults.penalty_weight),
                "training.policy_search.coverage_penalty_weight",
            ),
            filter_baseline_dominated=_bool_setting(
                raw.get(
                    "coverage_filter_baseline_dominated",
                    defaults.filter_baseline_dominated,
                ),
                "training.policy_search.coverage_filter_baseline_dominated",
            ),
            dominated_point_penalty_weight=_nonnegative_float_setting(
                raw.get(
                    "coverage_dominated_point_penalty_weight",
                    defaults.dominated_point_penalty_weight,
                ),
                "training.policy_search.coverage_dominated_point_penalty_weight",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_budget_span_coverage": self.min_budget_span_coverage,
            "min_target_span_coverage": self.min_target_span_coverage,
            "penalty_weight": self.penalty_weight,
            "filter_baseline_dominated": self.filter_baseline_dominated,
            "dominated_point_penalty_weight": self.dominated_point_penalty_weight,
        }


@dataclass(frozen=True, slots=True)
class CostADRCoverageDiagnostics:
    candidate_count: int
    baseline_dominated_candidate_count: int
    coverage_candidate_count: int
    budget_count: int
    covered_budget_count: int
    total_budget_span: float
    covered_budget_span: float
    budget_span_coverage_percent: float
    target_count: int
    covered_target_count: int
    total_target_span: float
    covered_target_span: float
    target_span_coverage_percent: float


@dataclass(frozen=True, slots=True)
class CostADRCandidateScore:
    hypervolume: float
    hypervolume_delta: float
    objective_score: float
    coverage_penalty: float
    dominance_penalty: float
    coverage_diagnostics: CostADRCoverageDiagnostics


@dataclass(frozen=True, slots=True)
class InitialPolicySettings:
    mean_source: str | None = None
    policy: Path | None = None
    policy_root: Path | None = None
    train_run_root: Path | None = None
    policy_template: str | None = None
    required: bool = False
    expand_bounds: bool = True
    bounds_padding: float = 4.0
    evaluate_in_generation_zero: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> InitialPolicySettings:
        defaults = cls()
        return cls(
            mean_source=_optional_initial_mean_source_setting(
                raw.get("initial_mean_source"),
                "training.policy_search.initial_mean_source",
            ),
            policy=_optional_path_setting(
                raw.get("initial_policy"),
                "training.policy_search.initial_policy",
            ),
            policy_root=_optional_path_setting(
                raw.get("initial_policy_root"),
                "training.policy_search.initial_policy_root",
            ),
            train_run_root=_optional_path_setting(
                raw.get("initial_policy_train_run_root"),
                "training.policy_search.initial_policy_train_run_root",
            ),
            policy_template=_optional_template_setting(
                raw.get("initial_policy_template"),
                "training.policy_search.initial_policy_template",
            ),
            required=_bool_setting(
                raw.get("initial_policy_required", defaults.required),
                "training.policy_search.initial_policy_required",
            ),
            expand_bounds=_bool_setting(
                raw.get("initial_policy_expand_bounds", defaults.expand_bounds),
                "training.policy_search.initial_policy_expand_bounds",
            ),
            bounds_padding=_nonnegative_float_setting(
                raw.get("initial_policy_bounds_padding", defaults.bounds_padding),
                "training.policy_search.initial_policy_bounds_padding",
            ),
            evaluate_in_generation_zero=_bool_setting(
                raw.get(
                    "initial_policy_evaluate_in_generation_zero",
                    defaults.evaluate_in_generation_zero,
                ),
                "training.policy_search.initial_policy_evaluate_in_generation_zero",
            ),
        )

    def __post_init__(self) -> None:
        configured_sources = [
            self.mean_source is not None,
            self.policy is not None,
            self.policy_root is not None,
            self.train_run_root is not None,
            self.policy_template is not None,
        ]
        if sum(1 for configured in configured_sources if configured) > 1:
            raise ValueError(
                "Only one Cost-ADR initial policy source may be configured."
            )

    @property
    def configured(self) -> bool:
        return (
            self.mean_source is not None
            or self.policy is not None
            or self.policy_root is not None
            or self.train_run_root is not None
            or self.policy_template is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_mean_source": self.mean_source,
            "policy": str(self.policy) if self.policy is not None else None,
            "policy_root": str(self.policy_root)
            if self.policy_root is not None
            else None,
            "train_run_root": str(self.train_run_root)
            if self.train_run_root is not None
            else None,
            "policy_template": self.policy_template,
            "required": self.required,
            "expand_bounds": self.expand_bounds,
            "bounds_padding": self.bounds_padding,
            "evaluate_in_generation_zero": self.evaluate_in_generation_zero,
        }


@dataclass(frozen=True, slots=True)
class InitialPolicyInfo:
    path: Path | None
    source: str | None
    coefficients: tuple[float, ...]
    title: str
    cost_weight_min: float
    cost_weight_max: float
    retention_min: float
    retention_max: float
    max_interval_days: float | None

    def to_dict(self, *, base: Path | None = None) -> dict[str, Any]:
        if self.path is None:
            path = None
        else:
            path = (
                _relative_path_string(self.path, base=base)
                if base is not None
                else self.path.as_posix()
            )
        return {
            "path": path,
            "source": self.source,
            "title": self.title,
            "parameter_count": len(self.coefficients),
            "coefficient_min": min(self.coefficients),
            "coefficient_max": max(self.coefficients),
            "cost_weight_min": self.cost_weight_min,
            "cost_weight_max": self.cost_weight_max,
            "retention_min": self.retention_min,
            "retention_max": self.retention_max,
            "max_interval_days": self.max_interval_days,
        }


@dataclass(frozen=True, slots=True)
class _RangeCoverage:
    total_count: int
    covered_count: int
    total_span: float
    covered_span: float
    span_coverage_percent: float


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
    best_objective_score: float
    best_coverage_diagnostics: CostADRCoverageDiagnostics
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
    optimizer_settings: CMAESSettings
    optimizer: Any
    initial_policy: InitialPolicyInfo | None = None
    evaluate_initial_policy_in_generation_zero: bool = False
    baseline_metrics: list[CandidateMetrics] = field(default_factory=list)
    baseline_points: list[ObjectivePoint] = field(default_factory=list)
    baseline_hypervolume: float = 0.0
    reference_point: ObjectivePoint | None = None
    best_cost_weight_metrics: list[CandidateMetrics] | None = None
    best_coefficients: torch.Tensor | None = None
    best_hypervolume: float = float("-inf")
    best_hypervolume_delta: float = float("-inf")
    best_objective_score: float = float("-inf")
    best_coverage_diagnostics: CostADRCoverageDiagnostics | None = None
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


def optimizer_settings_from_mapping(
    raw: Mapping[str, Any],
    *,
    settings: PolicySearchSettings | None = None,
) -> CMAESSettings:
    coefficient_min = -12.0 if settings is None else settings.coefficient_min
    coefficient_max = 12.0 if settings is None else settings.coefficient_max
    return CMAESSettings.from_mapping(
        raw,
        coefficient_count=PARAMETER_COUNT,
        coefficient_min=coefficient_min,
        coefficient_max=coefficient_max,
    )


def cost_weights_from_mapping(raw: Mapping[str, Any]) -> tuple[float, ...]:
    raw_weights = raw.get("cost_weights")
    if raw_weights is None:
        return DEFAULT_COST_WEIGHTS
    if isinstance(raw_weights, str) or not isinstance(raw_weights, Sequence):
        raise ValueError("training.policy_search.cost_weights must be an array.")
    weights = tuple(
        _nonnegative_float_setting(
            item,
            f"training.policy_search.cost_weights[{index}]",
        )
        for index, item in enumerate(raw_weights)
    )
    if not weights:
        raise ValueError("training.policy_search.cost_weights must not be empty.")
    if len(set(weights)) != len(weights):
        raise ValueError(
            "training.policy_search.cost_weights must not contain duplicates."
        )
    return weights


def hypervolume_delta_mode_from_mapping(raw: Mapping[str, Any]) -> str:
    value = raw.get(
        "hypervolume_delta_mode",
        HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "training.policy_search.hypervolume_delta_mode must be a non-empty string."
        )
    mode = value.strip()
    if mode not in SUPPORTED_HYPERVOLUME_DELTA_MODES:
        allowed = ", ".join(sorted(SUPPORTED_HYPERVOLUME_DELTA_MODES))
        raise ValueError(
            f"training.policy_search.hypervolume_delta_mode must be one of: {allowed}."
        )
    return mode


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
    optimizer_settings = optimizer_settings_from_mapping(
        config.training_optimizer,
        settings=settings,
    )
    cost_weights = cost_weights_from_mapping(raw_training_policy_search)
    hypervolume_delta_mode = hypervolume_delta_mode_from_mapping(
        raw_training_policy_search
    )
    coverage_settings = CoverageObjectiveSettings.from_mapping(
        raw_training_policy_search
    )
    initial_policy_settings = InitialPolicySettings.from_mapping(
        raw_training_policy_search
    )
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
            initial_policy = _load_initial_policy_for_user(
                settings=initial_policy_settings,
                user_id=job.user_id,
                repo_root=repo_root,
                cost_weights=cost_weights,
                policy_search_settings=settings,
            )
            job_optimizer_settings = _optimizer_settings_for_initial_policy(
                optimizer_settings=optimizer_settings,
                initial_policy=initial_policy,
                initial_policy_settings=initial_policy_settings,
            )
            progress.write(
                "config_loaded",
                settings=asdict(settings),
                optimizer=job_optimizer_settings.to_dict(),
                optimizer_seed=optimizer_seed,
                feature_version=FEATURE_VERSION_INTERVAL_MONO,
                action_head=ACTION_HEAD_INTERVAL,
                parameter_count=PARAMETER_COUNT,
                cost_weights=list(cost_weights),
                hypervolume_delta_mode=hypervolume_delta_mode,
                coverage_objective=coverage_settings.to_dict(),
                initial_policy_settings=initial_policy_settings.to_dict(),
                initial_policy=initial_policy.to_dict(base=job.output_dir)
                if initial_policy is not None
                else None,
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
                    optimizer_settings=job_optimizer_settings,
                    optimizer=_make_strategy(
                        optimizer_settings=job_optimizer_settings,
                        optimizer_seed=optimizer_seed,
                    ),
                    initial_policy=initial_policy,
                    evaluate_initial_policy_in_generation_zero=(
                        initial_policy is not None
                        and initial_policy_settings.evaluate_in_generation_zero
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
            coverage_settings=coverage_settings,
            hypervolume_delta_mode=hypervolume_delta_mode,
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
                best_objective_score=result.best_objective_score,
                best_coverage=asdict(result.best_coverage_diagnostics),
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
                optimizer_settings=state.optimizer_settings,
                optimizer_seed=state.optimizer_seed,
                initial_policy_settings=initial_policy_settings,
                initial_policy=state.initial_policy,
                coverage_settings=coverage_settings,
                hypervolume_delta_mode=hypervolume_delta_mode,
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


def _load_initial_policy_for_user(
    *,
    settings: InitialPolicySettings,
    user_id: int,
    repo_root: Path,
    cost_weights: Sequence[float],
    policy_search_settings: PolicySearchSettings,
) -> InitialPolicyInfo | None:
    if settings.mean_source is not None:
        return _built_in_initial_mean(
            source=settings.mean_source,
            cost_weights=cost_weights,
            policy_search_settings=policy_search_settings,
        )
    policy_path = _resolve_initial_policy_path(
        settings=settings,
        user_id=user_id,
        repo_root=repo_root,
    )
    if policy_path is None:
        return None
    if not policy_path.exists():
        raise FileNotFoundError(f"Cost-ADR initial policy not found: {policy_path}")
    policy = FSRS6CostConditionedADRPolicy.from_json(policy_path)
    if policy.action_head != ACTION_HEAD_INTERVAL:
        raise ValueError("Cost-ADR initial policy must use action_head='interval'.")
    if policy.feature_version != FEATURE_VERSION_INTERVAL_MONO:
        raise ValueError(
            "Cost-ADR initial policy feature_version must be "
            f"{FEATURE_VERSION_INTERVAL_MONO!r}."
        )
    if policy.parameter_count != PARAMETER_COUNT:
        raise ValueError(
            "Cost-ADR initial policy coefficient count must be "
            f"{PARAMETER_COUNT}, got {policy.parameter_count}."
        )
    for index, coefficient in enumerate(policy.coefficients):
        if not math.isfinite(coefficient):
            raise ValueError(
                "Cost-ADR initial policy coefficients must be finite; "
                f"coefficient {index} is {coefficient!r}."
            )
    expected_cost_min = min(cost_weights)
    expected_cost_max = max(cost_weights)
    if abs(policy.cost_weight_min - expected_cost_min) > 1e-9:
        raise ValueError(
            "Cost-ADR initial policy cost_weight_min must match the training "
            f"cost weight grid minimum ({expected_cost_min})."
        )
    if abs(policy.cost_weight_max - expected_cost_max) > 1e-9:
        raise ValueError(
            "Cost-ADR initial policy cost_weight_max must match the training "
            f"cost weight grid maximum ({expected_cost_max})."
        )
    if abs(policy.retention_min - policy_search_settings.retention_min) > 1e-9:
        raise ValueError(
            "Cost-ADR initial policy retention_min must match "
            "training.policy_search.retention_min."
        )
    if abs(policy.retention_max - policy_search_settings.retention_max) > 1e-9:
        raise ValueError(
            "Cost-ADR initial policy retention_max must match "
            "training.policy_search.retention_max."
        )
    return InitialPolicyInfo(
        path=policy_path,
        source="policy_json",
        coefficients=policy.coefficients,
        title=policy.title,
        cost_weight_min=policy.cost_weight_min,
        cost_weight_max=policy.cost_weight_max,
        retention_min=policy.retention_min,
        retention_max=policy.retention_max,
        max_interval_days=policy.max_interval_days,
    )


def _built_in_initial_mean(
    *,
    source: str,
    cost_weights: Sequence[float],
    policy_search_settings: PolicySearchSettings,
) -> InitialPolicyInfo:
    if source == INITIAL_MEAN_SOURCE_FIRST8_DISTILL24_MEAN_V1:
        coefficients = FIRST8_DISTILL24_MEAN_V1_COEFFICIENTS
        title = "FSRS6 Cost-ADR first8 distill24 mean initializer v1"
    else:
        allowed = ", ".join(sorted(SUPPORTED_INITIAL_MEAN_SOURCES))
        raise ValueError(
            f"training.policy_search.initial_mean_source must be one of: {allowed}."
        )
    if len(coefficients) != PARAMETER_COUNT:
        raise ValueError(
            f"Built-in Cost-ADR initial mean {source!r} must have "
            f"{PARAMETER_COUNT} coefficients."
        )
    for index, coefficient in enumerate(coefficients):
        if not math.isfinite(coefficient):
            raise ValueError(
                f"Built-in Cost-ADR initial mean {source!r} coefficient "
                f"{index} is not finite: {coefficient!r}."
            )
    return InitialPolicyInfo(
        path=None,
        source=source,
        coefficients=coefficients,
        title=title,
        cost_weight_min=min(cost_weights),
        cost_weight_max=max(cost_weights),
        retention_min=policy_search_settings.retention_min,
        retention_max=policy_search_settings.retention_max,
        max_interval_days=MAX_INTERVAL_DAYS,
    )


def _resolve_initial_policy_path(
    *,
    settings: InitialPolicySettings,
    user_id: int,
    repo_root: Path,
) -> Path | None:
    if settings.policy_template is not None:
        if "{user_id}" not in settings.policy_template:
            raise ValueError(
                "training.policy_search.initial_policy_template must contain {user_id}."
            )
        return _resolve_repo_path(
            Path(settings.policy_template.format(user_id=int(user_id))),
            repo_root=repo_root,
        )
    if settings.policy_root is not None:
        return _resolve_repo_path(
            settings.policy_root / f"user_{int(user_id)}" / "policy.json",
            repo_root=repo_root,
        )
    if settings.train_run_root is not None:
        return _resolve_repo_path(
            settings.train_run_root
            / "train-overfit"
            / "train_outputs"
            / f"user_{int(user_id)}"
            / "policy.json",
            repo_root=repo_root,
        )
    if settings.policy is not None:
        return _resolve_repo_path(settings.policy, repo_root=repo_root)
    if settings.required:
        raise ValueError(
            "training.policy_search.initial_policy_required=true requires an "
            "initial policy source."
        )
    return None


def _optimizer_settings_for_initial_policy(
    *,
    optimizer_settings: CMAESSettings,
    initial_policy: InitialPolicyInfo | None,
    initial_policy_settings: InitialPolicySettings,
) -> CMAESSettings:
    if initial_policy is None:
        return optimizer_settings
    initial_mean = tuple(float(value) for value in initial_policy.coefficients)
    lower = list(optimizer_settings.bounds[0])
    upper = list(optimizer_settings.bounds[1])
    out_of_bounds = [
        index
        for index, value in enumerate(initial_mean)
        if value < lower[index] or value > upper[index]
    ]
    if out_of_bounds and not initial_policy_settings.expand_bounds:
        first = out_of_bounds[0]
        raise ValueError(
            "Cost-ADR initial policy coefficients are outside "
            "training.optimizer.bounds and "
            "initial_policy_expand_bounds=false; first out-of-bounds "
            f"coefficient {first}={initial_mean[first]!r}."
        )
    if initial_policy_settings.expand_bounds:
        padding = initial_policy_settings.bounds_padding
        for index, value in enumerate(initial_mean):
            lower[index] = min(lower[index], value - padding)
            upper[index] = max(upper[index], value + padding)
    return CMAESSettings(
        name=optimizer_settings.name,
        population_size=optimizer_settings.population_size,
        generations=optimizer_settings.generations,
        sigma0=optimizer_settings.sigma0,
        initial_mean=initial_mean,
        bounds=(tuple(lower), tuple(upper)),
        seed=optimizer_settings.seed,
    )


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_root / expanded).resolve()


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
    coverage_settings: CoverageObjectiveSettings,
    hypervolume_delta_mode: str,
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
            if generation == 0 and state.evaluate_initial_policy_in_generation_zero:
                solutions[0] = list(state.optimizer_settings.initial_mean)
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
            candidate_scores = [
                _score_candidate(
                    baseline_metrics=state.baseline_metrics,
                    baseline_points=state.baseline_points,
                    baseline_hypervolume=state.baseline_hypervolume,
                    reference=reference,
                    candidate_metrics=candidate_metrics,
                    coverage_settings=coverage_settings,
                    hypervolume_delta_mode=hypervolume_delta_mode,
                )
                for candidate_metrics in metrics_by_candidate
            ]
            state.optimizer.tell(
                solutions_by_job[job_index],
                [-score.objective_score for score in candidate_scores],
            )
            generation_best_idx = max(
                range(len(candidate_scores)),
                key=lambda index: candidate_scores[index].objective_score,
            )
            generation_best = candidate_scores[generation_best_idx]
            if generation_best.objective_score > state.best_objective_score:
                state.best_objective_score = generation_best.objective_score
                state.best_hypervolume_delta = generation_best.hypervolume_delta
                state.best_hypervolume = generation_best.hypervolume
                state.best_coverage_diagnostics = generation_best.coverage_diagnostics
                state.best_coefficients = (
                    coefficients_by_job[job_index, generation_best_idx].detach().clone()
                )
                state.best_cost_weight_metrics = metrics_by_candidate[
                    generation_best_idx
                ]
            hypervolume_deltas = [score.hypervolume_delta for score in candidate_scores]
            objective_scores = [score.objective_score for score in candidate_scores]
            history_entry = {
                "generation": float(generation),
                "sigma": float(state.optimizer.sigma),
                "best_objective_score": float(state.best_objective_score),
                "generation_best_objective_score": float(
                    generation_best.objective_score
                ),
                "best_hypervolume_delta": float(state.best_hypervolume_delta),
                "generation_best_hypervolume_delta": float(
                    generation_best.hypervolume_delta
                ),
                "mean_objective_score": float(
                    sum(objective_scores) / max(len(objective_scores), 1)
                ),
                "mean_hypervolume_delta": float(
                    sum(hypervolume_deltas) / max(len(hypervolume_deltas), 1)
                ),
                "generation_best_hypervolume": float(generation_best.hypervolume),
                "generation_best_coverage_penalty": float(
                    generation_best.coverage_penalty
                ),
                "generation_best_dominance_penalty": float(
                    generation_best.dominance_penalty
                ),
                "generation_best_baseline_dominated_candidate_count": float(
                    generation_best.coverage_diagnostics.baseline_dominated_candidate_count
                ),
                "generation_best_coverage_candidate_count": float(
                    generation_best.coverage_diagnostics.coverage_candidate_count
                ),
                "generation_best_budget_span_coverage_percent": float(
                    generation_best.coverage_diagnostics.budget_span_coverage_percent
                ),
                "generation_best_target_span_coverage_percent": float(
                    generation_best.coverage_diagnostics.target_span_coverage_percent
                ),
                "baseline_hypervolume": state.baseline_hypervolume,
                "initial_policy_evaluated_in_generation_zero": float(
                    generation == 0 and state.evaluate_initial_policy_in_generation_zero
                ),
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
        or state.best_coverage_diagnostics is None
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
        best_objective_score=state.best_objective_score,
        best_coverage_diagnostics=state.best_coverage_diagnostics,
        history=state.history,
        passed=state.best_hypervolume_delta > 0.0,
    )


def _score_candidate(
    *,
    baseline_metrics: Sequence[CandidateMetrics],
    baseline_points: Sequence[ObjectivePoint],
    baseline_hypervolume: float,
    reference: ObjectivePoint,
    candidate_metrics: Sequence[CandidateMetrics],
    coverage_settings: CoverageObjectiveSettings,
    hypervolume_delta_mode: str = HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION,
) -> CostADRCandidateScore:
    candidate_points = [point_from_metrics(metric) for metric in candidate_metrics]
    if hypervolume_delta_mode == HYPERVOLUME_DELTA_MODE_UNION_CONTRIBUTION:
        hypervolume = objective_hypervolume_2d(
            [*baseline_points, *candidate_points],
            reference=reference,
        )
    elif hypervolume_delta_mode == HYPERVOLUME_DELTA_MODE_SCHEDULER_VS_BASELINE:
        hypervolume = objective_hypervolume_2d(candidate_points, reference=reference)
    else:
        allowed = ", ".join(sorted(SUPPORTED_HYPERVOLUME_DELTA_MODES))
        raise ValueError(f"hypervolume_delta_mode must be one of: {allowed}.")
    hypervolume_delta = hypervolume - baseline_hypervolume
    coverage = _coverage_diagnostics(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        filter_baseline_dominated=coverage_settings.filter_baseline_dominated,
    )
    budget_shortfall = max(
        0.0,
        coverage_settings.min_budget_span_coverage
        - coverage.budget_span_coverage_percent / 100.0,
    )
    target_shortfall = max(
        0.0,
        coverage_settings.min_target_span_coverage
        - coverage.target_span_coverage_percent / 100.0,
    )
    coverage_penalty = (
        baseline_hypervolume
        * coverage_settings.penalty_weight
        * (budget_shortfall + target_shortfall)
        if coverage_settings.enabled
        else 0.0
    )
    dominated_fraction = (
        coverage.baseline_dominated_candidate_count / coverage.candidate_count
        if coverage.candidate_count > 0
        else 0.0
    )
    dominance_penalty = (
        baseline_hypervolume
        * coverage_settings.dominated_point_penalty_weight
        * dominated_fraction
    )
    return CostADRCandidateScore(
        hypervolume=hypervolume,
        hypervolume_delta=hypervolume_delta,
        objective_score=hypervolume_delta - coverage_penalty - dominance_penalty,
        coverage_penalty=coverage_penalty,
        dominance_penalty=dominance_penalty,
        coverage_diagnostics=coverage,
    )


def _coverage_diagnostics(
    *,
    baseline_metrics: Sequence[CandidateMetrics],
    candidate_metrics: Sequence[CandidateMetrics],
    filter_baseline_dominated: bool = False,
) -> CostADRCoverageDiagnostics:
    baseline_points = [point_from_metrics(metric) for metric in baseline_metrics]
    candidate_points = [point_from_metrics(metric) for metric in candidate_metrics]
    baseline_dominated = [
        _is_dominated_by_any_baseline(point, baseline_points)
        for point in candidate_points
    ]
    coverage_candidate_metrics = [
        metric
        for metric, dominated in zip(
            candidate_metrics,
            baseline_dominated,
            strict=True,
        )
        if not filter_baseline_dominated or not dominated
    ]
    baseline_time_frontier = _frontier_metrics(baseline_metrics, sort_key="time")
    candidate_time_frontier = _frontier_metrics(
        coverage_candidate_metrics,
        sort_key="time",
    )
    baseline_memory_frontier = _frontier_metrics(baseline_metrics, sort_key="memory")
    candidate_memory_frontier = _frontier_metrics(
        coverage_candidate_metrics,
        sort_key="memory",
    )
    budget = _range_coverage(
        baseline_values=[metric.time_average for metric in baseline_time_frontier],
        candidate_values=[metric.time_average for metric in candidate_time_frontier],
    )
    target = _range_coverage(
        baseline_values=[
            metric.memorized_average for metric in baseline_memory_frontier
        ],
        candidate_values=[
            metric.memorized_average for metric in candidate_memory_frontier
        ],
    )
    return CostADRCoverageDiagnostics(
        candidate_count=len(candidate_metrics),
        baseline_dominated_candidate_count=sum(
            1 for value in baseline_dominated if value
        ),
        coverage_candidate_count=len(coverage_candidate_metrics),
        budget_count=budget.total_count,
        covered_budget_count=budget.covered_count,
        total_budget_span=budget.total_span,
        covered_budget_span=budget.covered_span,
        budget_span_coverage_percent=budget.span_coverage_percent,
        target_count=target.total_count,
        covered_target_count=target.covered_count,
        total_target_span=target.total_span,
        covered_target_span=target.covered_span,
        target_span_coverage_percent=target.span_coverage_percent,
    )


def _is_dominated_by_any_baseline(
    point: ObjectivePoint,
    baseline_points: Sequence[ObjectivePoint],
) -> bool:
    return any(
        _objective_dominates(baseline_point, point)
        for baseline_point in baseline_points
    )


def _objective_dominates(lhs: ObjectivePoint, rhs: ObjectivePoint) -> bool:
    no_worse = (
        lhs.memorized_average >= rhs.memorized_average
        and lhs.negative_time_average >= rhs.negative_time_average
    )
    strictly_better = (
        lhs.memorized_average > rhs.memorized_average
        or lhs.negative_time_average > rhs.negative_time_average
    )
    return no_worse and strictly_better


def _frontier_metrics(
    metrics: Sequence[CandidateMetrics],
    *,
    sort_key: str,
) -> list[CandidateMetrics]:
    points = [point_from_metrics(metric) for metric in metrics]
    indices = objective_non_dominated_indices(points)
    frontier = [metrics[index] for index in indices]
    if sort_key == "time":
        return sorted(
            frontier,
            key=lambda metric: (metric.time_average, metric.memorized_average),
        )
    if sort_key == "memory":
        return sorted(
            frontier,
            key=lambda metric: (metric.memorized_average, -metric.time_average),
        )
    raise ValueError(f"Unknown frontier sort key: {sort_key!r}.")


def _range_coverage(
    *,
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
) -> _RangeCoverage:
    baseline = sorted({float(value) for value in baseline_values})
    candidate = sorted({float(value) for value in candidate_values})
    total_span = max(baseline) - min(baseline) if len(baseline) > 1 else 0.0
    covered_span = 0.0
    covered_count = 0
    if baseline and candidate and total_span > 0.0:
        start = max(min(baseline), min(candidate))
        end = min(max(baseline), max(candidate))
        if end > start:
            covered_span = end - start
            covered_count = sum(
                1
                for value in baseline
                if (start < value < end)
                or value == start
                or value == end
                or abs(value - start) <= 1e-9
                or abs(value - end) <= 1e-9
            )
    span_coverage_percent = (
        (covered_span / total_span) * 100.0 if total_span > 0.0 else 0.0
    )
    return _RangeCoverage(
        total_count=len(baseline),
        covered_count=covered_count,
        total_span=float(total_span),
        covered_span=float(covered_span),
        span_coverage_percent=float(span_coverage_percent),
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
    policy = _policy_template(settings=settings, cost_weights=cost_weights)
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
    *, settings: PolicySearchSettings, cost_weights: Sequence[float]
) -> FSRS6CostConditionedADRPolicy:
    return FSRS6CostConditionedADRPolicy(
        coefficients=(0.0,) * PARAMETER_COUNT,
        action_head=ACTION_HEAD_INTERVAL,
        feature_version=FEATURE_VERSION_INTERVAL_MONO,
        cost_weight_min=min(cost_weights),
        cost_weight_max=max(cost_weights),
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
    initial_policy_settings: InitialPolicySettings,
    initial_policy: InitialPolicyInfo | None,
    coverage_settings: CoverageObjectiveSettings,
    hypervolume_delta_mode: str,
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
    initial_policy_record = (
        initial_policy.to_dict(base=output_dir) if initial_policy is not None else None
    )
    metrics_path = output_dir / "metrics.json"
    if coverage_settings.enabled and (
        coverage_settings.filter_baseline_dominated
        or coverage_settings.dominated_point_penalty_weight > 0.0
    ):
        training_objective = "quality_aware_coverage_hypervolume_delta"
    elif coverage_settings.enabled:
        training_objective = "coverage_aware_hypervolume_delta"
    else:
        training_objective = "hypervolume_delta"
    _write_json(
        metrics_path,
        {
            "passed_overfit_gate": result.passed,
            "training_objective": training_objective,
            "hypervolume_delta_mode": hypervolume_delta_mode,
            "coverage_objective": coverage_settings.to_dict(),
            "baseline_hypervolume": result.baseline_hypervolume,
            "best_hypervolume": result.best_hypervolume,
            "best_hypervolume_delta": result.best_hypervolume_delta,
            "best_objective_score": result.best_objective_score,
            "best_coverage": asdict(result.best_coverage_diagnostics),
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
            "initial_policy_settings": initial_policy_settings.to_dict(),
            "initial_policy": initial_policy_record,
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
            "training_objective": training_objective,
            "hypervolume_delta_mode": hypervolume_delta_mode,
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
            "initial_policy_settings": initial_policy_settings.to_dict(),
            "initial_policy": initial_policy_record,
            "cost_weights": list(cost_weights),
            "coverage_objective": coverage_settings.to_dict(),
            "best_hypervolume_delta": result.best_hypervolume_delta,
            "best_objective_score": result.best_objective_score,
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


def _bool_setting(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _nonnegative_float_setting(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field_name} must be >= 0.")
    return result


def _fraction_setting(value: Any, field_name: str) -> float:
    result = _nonnegative_float_setting(value, field_name)
    if result > 1.0:
        raise ValueError(f"{field_name} must be <= 1.")
    return result


def _optional_path_setting(value: Any, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    return Path(value)


def _optional_template_setting(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    return value


def _optional_initial_mean_source_setting(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    source = value.strip()
    if source not in SUPPORTED_INITIAL_MEAN_SOURCES:
        allowed = ", ".join(sorted(SUPPORTED_INITIAL_MEAN_SOURCES))
        raise ValueError(f"{field_name} must be one of: {allowed}.")
    return source


if __name__ == "__main__":
    raise SystemExit(main())

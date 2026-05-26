from __future__ import annotations

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
    outcomes: list[CostADRTrainJobResult] = []
    for job in jobs:
        progress = _progress_for_job(job=job, config_path=config_path)
        try:
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
            )
            progress.write("device_resolved", device=device, torch_device=str(device))
            baseline_bundle = _build_bundle(
                config=config,
                settings=settings,
                lane_user_ids=[job.user_id for _dr in baseline_drs],
                benchmark_root=benchmark_root,
                overrides=overrides,
                benchmark_partition=benchmark_partition,
                button_usage=button_usage,
                device=device,
                short_term_source=short_term_source,
                learning_steps=learning_steps,
                relearning_steps=relearning_steps,
            )
            baseline_metrics = _evaluate_baseline_grid(
                config=config,
                settings=settings,
                bundle=baseline_bundle,
                baseline_drs=baseline_drs,
                seed=config.seed,
            )
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
            progress.write(
                "baseline_grid_evaluated",
                device=baseline_bundle.device,
                effective_lanes=len(baseline_drs),
                baseline_hypervolume=baseline_hv,
                reference_point=asdict(hv_reference),
                metrics=[
                    {"baseline_desired_retention": dr, **asdict(metric)}
                    for dr, metric in zip(baseline_drs, baseline_metrics, strict=True)
                ],
            )
            del baseline_bundle
            _clear_cuda_cache(device)

            train_bundle = _build_bundle(
                config=config,
                settings=settings,
                lane_user_ids=[
                    job.user_id
                    for _candidate in range(optimizer_settings.population_size)
                    for _weight in cost_weights
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
            progress.write(
                "train_bundle_built",
                device=train_bundle.device,
                effective_lanes=optimizer_settings.population_size * len(cost_weights),
            )
            result = _run_cmaes(
                config=config,
                settings=settings,
                optimizer_settings=optimizer_settings,
                optimizer_seed=optimizer_seed,
                bundle=train_bundle,
                baseline_drs=baseline_drs,
                baseline_metrics=baseline_metrics,
                baseline_hv=baseline_hv,
                reference=hv_reference,
                cost_weights=cost_weights,
                progress=progress,
            )
            progress.write(
                "cmaes_completed",
                device=train_bundle.device,
                best_hypervolume=result.best_hypervolume,
                best_hypervolume_delta=result.best_hypervolume_delta,
                generations=len(result.history),
                passed=result.passed,
            )
            policy_path, metrics_path, metadata_path = write_artifact(
                output_dir=job.output_dir,
                config=config,
                config_path=config_path,
                settings=settings,
                user_id=job.user_id,
                training_command_path=job.command_record_path,
                optimizer_settings=optimizer_settings,
                optimizer_seed=optimizer_seed,
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
            progress.write("failed", error=str(exc))
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


def _evaluate_baseline_grid(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    baseline_drs: Sequence[float],
    seed: int,
) -> list[CandidateMetrics]:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            list(baseline_drs),
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


def _run_cmaes(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    optimizer_settings: CMAESSettings,
    optimizer_seed: int,
    bundle: Any,
    baseline_drs: tuple[float, ...],
    baseline_metrics: list[CandidateMetrics],
    baseline_hv: float,
    reference: ObjectivePoint,
    cost_weights: tuple[float, ...],
    progress: TrainingProgress,
) -> CostADRTrainingResult:
    es = cma.CMAEvolutionStrategy(
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
    baseline_points = [point_from_metrics(metric) for metric in baseline_metrics]
    best_coefficients: torch.Tensor | None = None
    best_metrics: list[CandidateMetrics] | None = None
    best_hv = float("-inf")
    best_hv_delta = float("-inf")
    history: list[dict[str, float]] = []
    for generation in range(optimizer_settings.generations):
        solutions = [list(map(float, item)) for item in es.ask()]
        if len(solutions) != optimizer_settings.population_size:
            raise RuntimeError(
                "CMA-ES returned an unexpected population size: "
                f"{len(solutions)} != {optimizer_settings.population_size}."
            )
        coefficients = torch.tensor(
            solutions,
            device=bundle.device,
            dtype=torch.float32,
        )
        metrics_by_candidate = _evaluate_cost_adr_candidates(
            config=config,
            settings=settings,
            bundle=bundle,
            coefficients=coefficients,
            cost_weights=cost_weights,
            seed=config.seed,
        )
        scores = [
            objective_hypervolume_2d(
                [
                    *baseline_points,
                    *[point_from_metrics(metric) for metric in candidate_metrics],
                ],
                reference=reference,
            )
            - baseline_hv
            for candidate_metrics in metrics_by_candidate
        ]
        es.tell(solutions, [-score for score in scores])
        generation_best_idx = max(range(len(scores)), key=scores.__getitem__)
        generation_best_score = float(scores[generation_best_idx])
        generation_best_hv = baseline_hv + generation_best_score
        if generation_best_score > best_hv_delta:
            best_hv_delta = generation_best_score
            best_hv = generation_best_hv
            best_coefficients = coefficients[generation_best_idx].detach().clone()
            best_metrics = metrics_by_candidate[generation_best_idx]
        history_entry = {
            "generation": float(generation),
            "sigma": float(es.sigma),
            "best_hypervolume_delta": float(best_hv_delta),
            "generation_best_hypervolume_delta": generation_best_score,
            "mean_hypervolume_delta": float(sum(scores) / max(len(scores), 1)),
            "generation_best_hypervolume": generation_best_hv,
            "baseline_hypervolume": baseline_hv,
        }
        history.append(history_entry)
        progress.write(
            "cmaes_generation",
            device=bundle.device,
            effective_lanes=optimizer_settings.population_size * len(cost_weights),
            **history_entry,
        )
    if best_coefficients is None or best_metrics is None:
        raise RuntimeError("CMA-ES did not evaluate any candidates.")
    return CostADRTrainingResult(
        baseline_desired_retention_values=baseline_drs,
        baseline_metrics=baseline_metrics,
        baseline_hypervolume=baseline_hv,
        reference_point=reference,
        best_cost_weight_metrics=best_metrics,
        best_coefficients=best_coefficients.detach().cpu(),
        best_hypervolume=best_hv,
        best_hypervolume_delta=best_hv_delta,
        history=history,
        passed=best_hv_delta > 0.0,
    )


def _evaluate_cost_adr_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: Any,
    coefficients: torch.Tensor,
    cost_weights: tuple[float, ...],
    seed: int,
) -> list[list[CandidateMetrics]]:
    population_size = int(coefficients.shape[0])
    flat_coefficients = (
        coefficients[:, None, :]
        .expand(population_size, len(cost_weights), coefficients.shape[1])
        .reshape(population_size * len(cost_weights), coefficients.shape[1])
    )
    policy = _policy_template(settings=settings)
    sched_ops = FSRS6CostConditionedADRBatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=policy,
        goal_cost_weight=torch.tensor(
            [
                cost_weight
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
    metrics = [_metrics_from_stats(item) for item in stats]
    return [
        metrics[index * len(cost_weights) : (index + 1) * len(cost_weights)]
        for index in range(population_size)
    ]


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

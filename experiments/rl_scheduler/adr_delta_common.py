from __future__ import annotations

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
    RELATIVE_GAIN_GATE_FLOOR,
    PolicySearchSettings,
    SimulationBundle,
    _float_token,
    _git_commit,
    _metrics_from_stats,
    _passes_relative_gain_fraction_gate,
    _relative_gain,
    _relative_gain_fraction_gate_metrics,
    _relative_gain_point_passes,
    _write_json,
)
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.math.fsrs import Bounds
from simulator.fsrs6_adr_delta_policy import (
    FEATURE_VERSION,
    FSRS6ADRDeltaPolicy,
    feature_count,
)
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.fsrs6_adr_delta import FSRS6ADRDeltaBatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    metrics_by_dr: list[CandidateMetrics]
    relative_memorized_gains: list[float]
    relative_efficiency_gains: list[float]
    mean_relative_memorized_gain: float
    mean_relative_efficiency_gain: float
    min_relative_memorized_gain: float
    min_relative_efficiency_gain: float
    passed_overfit_gate: bool
    score: float


@dataclass(frozen=True, slots=True)
class DRConditionedTrainingResult:
    baseline_desired_retention_values: tuple[float, ...]
    baselines: list[CandidateMetrics]
    best_coefficients: torch.Tensor
    best: CandidateEvaluation
    history: list[dict[str, float]]
    passed: bool


def _baseline_dr_values(
    raw_training_policy: Mapping[str, Any],
    settings: PolicySearchSettings,
) -> tuple[float, ...]:
    raw_values = raw_training_policy.get("baseline_desired_retention_values")
    if raw_values is None:
        values = (settings.baseline_desired_retention,)
    else:
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise ValueError(
                "training.policy_search.baseline_desired_retention_values must be an array."
            )
        values = tuple(
            _float(item, "training.policy_search.baseline_desired_retention_values")
            for item in raw_values
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "training.policy_search.baseline_desired_retention_values must not contain duplicates."
        )
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "training.policy_search.baseline_desired_retention_values must be inside "
                "the retention bounds."
            )
    return values


def _dr_batch_size(raw_training_policy: Mapping[str, Any], value_count: int) -> int:
    default = min(max(value_count, 1), 4)
    return min(
        _int_like(
            raw_training_policy.get("dr_batch_size", default),
            "training.policy_search.dr_batch_size",
            1,
        ),
        value_count,
    )


def _policy_feature_version(raw_training_policy: Mapping[str, Any]) -> str:
    raw_value = raw_training_policy.get("feature_version", FEATURE_VERSION)
    if not isinstance(raw_value, str):
        raise ValueError("training.policy_search.feature_version must be a string.")
    feature_count(raw_value)
    return raw_value


def _iter_dr_chunks(
    values: tuple[float, ...],
    baselines: list[CandidateMetrics],
    chunk_size: int,
) -> Sequence[tuple[tuple[float, ...], list[CandidateMetrics]]]:
    return [
        (
            values[start : start + chunk_size],
            baselines[start : start + chunk_size],
        )
        for start in range(0, len(values), chunk_size)
    ]


def _pad_tuple(values: tuple[float, ...], size: int) -> tuple[float, ...]:
    if not values:
        raise ValueError("Cannot pad an empty DR chunk.")
    if len(values) >= size:
        return values
    return (*values, *(values[-1] for _ in range(size - len(values))))


def _evaluate_fsrs6_baselines(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    seed: int,
) -> list[CandidateMetrics]:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=torch.tensor(
            baseline_dr_values,
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


def _evaluate_dr_conditioned_candidates(
    *,
    config: ExperimentConfig,
    settings: PolicySearchSettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    dr_batch_size: int,
    baselines: list[CandidateMetrics],
    coefficients: torch.Tensor,
    lambda_value: float,
    feature_version: str,
    seed: int,
) -> list[CandidateEvaluation]:
    dr_count = len(baseline_dr_values)
    candidate_count = int(coefficients.shape[0])
    template = FSRS6ADRDeltaPolicy.baseline(
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    coefficient_count = template.feature_count
    if int(coefficients.shape[1]) != coefficient_count:
        raise ValueError(
            "DR-conditioned coefficients must have shape "
            f"(candidates, {coefficient_count})."
        )
    metrics_by_candidate: list[list[CandidateMetrics]] = [
        [] for _ in range(candidate_count)
    ]
    for chunk_dr_values, chunk_baselines in _iter_dr_chunks(
        baseline_dr_values,
        baselines,
        dr_batch_size,
    ):
        actual_count = len(chunk_dr_values)
        padded_dr_values = _pad_tuple(chunk_dr_values, dr_batch_size)
        desired_retention = torch.tensor(
            [dr for _candidate in range(candidate_count) for dr in padded_dr_values],
            device=bundle.device,
            dtype=torch.float32,
        )
        lane_coefficients = (
            coefficients[:, None, :]
            .expand(candidate_count, dr_batch_size, coefficient_count)
            .reshape(candidate_count * dr_batch_size, coefficient_count)
        )
        sched_ops = FSRS6ADRDeltaBatchSchedulerOps(
            weights=bundle.scheduler_weights,
            desired_retention=desired_retention,
            policy=template,
            coefficients=lane_coefficients,
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
        for candidate_index in range(candidate_count):
            start = candidate_index * dr_batch_size
            candidate_metrics = metrics[start : start + actual_count]
            metrics_by_candidate[candidate_index].extend(candidate_metrics)
    evaluations: list[CandidateEvaluation] = []
    for candidate_index in range(candidate_count):
        candidate_metrics = metrics_by_candidate[candidate_index]
        if len(candidate_metrics) != dr_count:
            raise AssertionError("DR-conditioned evaluation produced missing metrics.")
        evaluations.append(
            _candidate_evaluation(candidate_metrics, baselines, lambda_value)
        )
    return evaluations


def _candidate_evaluation(
    metrics_by_dr: list[CandidateMetrics],
    baselines: Sequence[CandidateMetrics],
    lambda_value: float,
) -> CandidateEvaluation:
    if len(metrics_by_dr) != len(baselines):
        raise ValueError("metrics_by_dr and baselines must have the same length.")
    if not metrics_by_dr:
        raise ValueError("DR-conditioned evaluation requires at least one DR.")
    relative_memorized_gains = [
        _relative_gain(metric.memorized_average, baseline.memorized_average)
        for metric, baseline in zip(metrics_by_dr, baselines, strict=True)
    ]
    relative_efficiency_gains = [
        _relative_gain(metric.memorized_per_minute, baseline.memorized_per_minute)
        for metric, baseline in zip(metrics_by_dr, baselines, strict=True)
    ]
    mean_relative_memorized_gain = sum(relative_memorized_gains) / len(
        relative_memorized_gains
    )
    mean_relative_efficiency_gain = sum(relative_efficiency_gains) / len(
        relative_efficiency_gains
    )
    return CandidateEvaluation(
        metrics_by_dr=list(metrics_by_dr),
        relative_memorized_gains=relative_memorized_gains,
        relative_efficiency_gains=relative_efficiency_gains,
        mean_relative_memorized_gain=mean_relative_memorized_gain,
        mean_relative_efficiency_gain=mean_relative_efficiency_gain,
        min_relative_memorized_gain=min(relative_memorized_gains),
        min_relative_efficiency_gain=min(relative_efficiency_gains),
        passed_overfit_gate=_dr_grid_passed_relative_gains(
            relative_memorized_gains,
            relative_efficiency_gains,
        ),
        score=_score_dr_grid_relative_gains(
            relative_memorized_gains,
            relative_efficiency_gains,
            lambda_value,
        ),
    )


def _dr_grid_passed_relative_gains(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
) -> bool:
    _validate_relative_gain_grid(relative_memorized_gains, relative_efficiency_gains)
    return _passes_relative_gain_fraction_gate(
        relative_memorized_gains,
        relative_efficiency_gains,
    )


def _score_dr_grid_relative_gains(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
    lambda_value: float,
) -> float:
    _validate_relative_gain_grid(relative_memorized_gains, relative_efficiency_gains)
    if _dr_grid_passed_relative_gains(
        relative_memorized_gains,
        relative_efficiency_gains,
    ):
        return sum(
            (
                (1.0 - lambda_value) * rel_mem
                + lambda_value * rel_eff
                - RELATIVE_GAIN_GATE_FLOOR
            )
            for rel_mem, rel_eff in zip(
                relative_memorized_gains,
                relative_efficiency_gains,
                strict=True,
            )
        ) / len(relative_memorized_gains)
    return -sum(
        max(0.0, RELATIVE_GAIN_GATE_FLOOR - rel_mem)
        + max(0.0, RELATIVE_GAIN_GATE_FLOOR - rel_eff)
        for rel_mem, rel_eff in zip(
            relative_memorized_gains,
            relative_efficiency_gains,
            strict=True,
        )
    )


def _validate_relative_gain_grid(
    relative_memorized_gains: Sequence[float],
    relative_efficiency_gains: Sequence[float],
) -> None:
    if len(relative_memorized_gains) != len(relative_efficiency_gains):
        raise ValueError("Relative gain arrays must have the same length.")
    if not relative_memorized_gains:
        raise ValueError("Relative gain arrays must not be empty.")


def _write_artifact(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: PolicySearchSettings,
    user_id: int,
    lambda_value: float,
    training_command_path: Path | None,
    feature_version: str,
    result: DRConditionedTrainingResult,
) -> tuple[Path, Path, Path]:
    policy = FSRS6ADRDeltaPolicy(
        coefficients=tuple(float(v) for v in result.best_coefficients.tolist()),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        title=f"fsrs6_adr_delta_u{user_id}_lambda_{lambda_value:g}",
        feature_version=feature_version,
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)

    per_dr = []
    for dr, baseline, best in zip(
        result.baseline_desired_retention_values,
        result.baselines,
        result.best.metrics_by_dr,
        strict=True,
    ):
        relative_memorized_gain = _relative_gain(
            best.memorized_average,
            baseline.memorized_average,
        )
        relative_efficiency_gain = _relative_gain(
            best.memorized_per_minute,
            baseline.memorized_per_minute,
        )
        per_dr.append(
            {
                "baseline_desired_retention": dr,
                "baseline": asdict(baseline),
                "best": asdict(best),
                "relative_memorized_gain": relative_memorized_gain,
                "relative_efficiency_gain": relative_efficiency_gain,
                "memorized_average_gt_baseline": relative_memorized_gain > 0.0,
                "memorized_per_minute_gt_baseline": relative_efficiency_gain > 0.0,
                "passed_overfit_gate": _relative_gain_point_passes(
                    relative_memorized_gain,
                    relative_efficiency_gain,
                ),
            }
        )
    fraction_gate = _relative_gain_fraction_gate_metrics(
        result.best.relative_memorized_gains,
        result.best.relative_efficiency_gains,
    )
    metrics = {
        "passed_overfit_gate": result.passed,
        "gate": {
            **fraction_gate,
            "all_desired_retention_memorized_average_gt_baseline": (
                result.best.min_relative_memorized_gain > 0.0
            ),
            "all_desired_retention_memorized_per_minute_gt_baseline": (
                result.best.min_relative_efficiency_gain > 0.0
            ),
            "min_relative_memorized_gain": result.best.min_relative_memorized_gain,
            "min_relative_efficiency_gain": result.best.min_relative_efficiency_gain,
            "mean_relative_memorized_gain": (result.best.mean_relative_memorized_gain),
            "mean_relative_efficiency_gain": (
                result.best.mean_relative_efficiency_gain
            ),
        },
        "best_score": result.best.score,
        "settings": {
            **asdict(settings),
            "feature_version": feature_version,
            "baseline_desired_retention_values": list(
                result.baseline_desired_retention_values
            ),
        },
        "per_desired_retention": per_dr,
        "history": result.history,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "scheduler-policy",
        "artifact_id": _artifact_id(user_id, lambda_value, config.seed),
        "family": config.family,
        "scheduler_name": "fsrs6_adr_delta",
        "environment": config.simulation.environment,
        "engine": config.simulation.engine,
        "training_user_ids": [user_id],
        "validation_user_ids": list(config.users.validation),
        "seed": config.seed,
        "policy_path": "policy.json",
        "feature_version": feature_version,
        "action_space": "sddr_logit_retention_adjustment",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "code_commit": _git_commit(),
        "lambda_value": lambda_value,
        "baseline_desired_retention": None,
        "config_snapshot_path": str(config_path.resolve()),
        "training_command_path": str(training_command_path)
        if training_command_path
        else None,
        "metrics_path": "metrics.json",
        "capabilities": ["event", "vectorized", "batched"],
    }
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    return policy_path, metrics_path, metadata_path


def _artifact_id(user_id: int, lambda_value: float, seed: int) -> str:
    lambda_token = _float_token(lambda_value)
    return f"fsrs6-adr-delta-user-{user_id}-lambda-{lambda_token}-seed-{seed}"


def _float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must contain only numbers.")
    return float(value)


def _int_like(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

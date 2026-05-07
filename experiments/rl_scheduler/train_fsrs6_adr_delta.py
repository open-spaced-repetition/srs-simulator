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

from experiments.rl_scheduler.train_fsrs6_adr_direct import (
    CandidateMetrics,
    SASettings,
    SimulationBundle,
    TrainingProgress,
    _build_bundle,
    _clamp_coefficients,
    _float_token,
    _git_commit,
    _metrics_from_stats,
    _read_training_sa,
    _relative_gain,
    _score_from_relative_gains,
    _temperature,
    _write_json,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
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
class ChainEvaluation:
    metrics_by_dr: list[CandidateMetrics]
    mean_relative_memorized_gain: float
    mean_relative_efficiency_gain: float
    score: float


@dataclass(frozen=True, slots=True)
class DRConditionedTrainingResult:
    baseline_desired_retention_values: tuple[float, ...]
    baselines: list[CandidateMetrics]
    best_coefficients: torch.Tensor
    best: ChainEvaluation
    history: list[dict[str, float]]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one DR-conditioned FSRS6 ADR Direct scheduler policy over a desired "
            "retention grid."
        ),
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
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress(output_dir / "training_progress.jsonl")
    progress.write(
        "started",
        config_path=str(args.config),
        user_id=args.user_id,
        lambda_value=args.lambda_value,
    )

    config = ExperimentConfig.from_toml(args.config)
    settings = SASettings.from_mapping(config.training_sa)
    raw_training_sa = _read_training_sa(args.config)
    policy_feature_version = _policy_feature_version(raw_training_sa)
    baseline_dr_values = _baseline_dr_values(raw_training_sa, settings)
    dr_batch_size = _dr_batch_size(raw_training_sa, len(baseline_dr_values))
    progress.write(
        "config_loaded",
        settings=asdict(settings),
        feature_version=policy_feature_version,
        simulation=config.simulation.to_dict(),
        seed=config.seed,
        baseline_desired_retention_values=list(baseline_dr_values),
        dr_batch_size=dr_batch_size,
    )

    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    short_term_args = argparse.Namespace(
        short_term_source=config.simulation.short_term_source,
        learning_steps=raw_training_sa.get("learning_steps"),
        relearning_steps=raw_training_sa.get("relearning_steps"),
    )
    short_term_source, learning_steps, relearning_steps = resolve_short_term_config(
        short_term_args
    )
    device = torch.device(settings.torch_device)
    progress.write("device_resolved", device=device, torch_device=str(device))

    baseline_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=len(baseline_dr_values),
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baselines = _evaluate_fsrs6_baselines(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        baseline_dr_values=baseline_dr_values,
        seed=config.seed,
    )
    progress.write(
        "baselines_evaluated",
        device=baseline_bundle.device,
        effective_lanes=len(baseline_dr_values),
        metrics=[
            {
                "baseline_desired_retention": dr,
                **asdict(metrics),
            }
            for dr, metrics in zip(baseline_dr_values, baselines, strict=True)
        ],
    )
    del baseline_bundle
    _clear_cuda_cache(device)

    train_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=dr_batch_size * settings.chains,
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=args.benchmark_partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    progress.write(
        "train_bundle_built",
        device=train_bundle.device,
        effective_lanes=dr_batch_size * settings.chains,
        grid_lanes=len(baseline_dr_values) * settings.chains,
    )
    result = _anneal_dr_conditioned(
        config=config,
        settings=settings,
        bundle=train_bundle,
        baseline_dr_values=baseline_dr_values,
        dr_batch_size=dr_batch_size,
        baselines=baselines,
        lambda_value=args.lambda_value,
        feature_version=policy_feature_version,
        progress=progress,
    )
    progress.write(
        "annealing_completed",
        device=train_bundle.device,
        best_score=result.best.score,
        mean_relative_memorized_gain=result.best.mean_relative_memorized_gain,
        mean_relative_efficiency_gain=result.best.mean_relative_efficiency_gain,
        iterations=len(result.history),
    )

    policy_path, metrics_path, metadata_path = _write_artifact(
        output_dir=output_dir,
        config=config,
        config_path=args.config,
        settings=settings,
        user_id=args.user_id,
        lambda_value=args.lambda_value,
        training_command_path=args.training_command_path,
        feature_version=policy_feature_version,
        result=result,
    )
    progress.write(
        "artifacts_written",
        device=train_bundle.device,
        passed=result.passed,
        policy_path=str(policy_path),
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
    )
    return 0 if result.passed else 1


def _baseline_dr_values(
    raw_training_sa: Mapping[str, Any],
    settings: SASettings,
) -> tuple[float, ...]:
    raw_values = raw_training_sa.get("baseline_desired_retention_values")
    if raw_values is None:
        values = (settings.baseline_desired_retention,)
    else:
        if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
            raise ValueError(
                "training.sa.baseline_desired_retention_values must be an array."
            )
        values = tuple(
            _float(item, "training.sa.baseline_desired_retention_values")
            for item in raw_values
        )
    if len(set(values)) != len(values):
        raise ValueError(
            "training.sa.baseline_desired_retention_values must not contain duplicates."
        )
    for value in values:
        if not (settings.retention_min <= value <= settings.retention_max):
            raise ValueError(
                "training.sa.baseline_desired_retention_values must be inside "
                "the retention bounds."
            )
    return values


def _dr_batch_size(raw_training_sa: Mapping[str, Any], value_count: int) -> int:
    default = min(max(value_count, 1), 4)
    return min(
        _int_like(
            raw_training_sa.get("dr_batch_size", default),
            "training.sa.dr_batch_size",
            1,
        ),
        value_count,
    )


def _policy_feature_version(raw_training_sa: Mapping[str, Any]) -> str:
    raw_value = raw_training_sa.get("feature_version", FEATURE_VERSION)
    if not isinstance(raw_value, str):
        raise ValueError("training.sa.feature_version must be a string.")
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
    settings: SASettings,
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


def _anneal_dr_conditioned(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    dr_batch_size: int,
    baselines: list[CandidateMetrics],
    lambda_value: float,
    feature_version: str,
    progress: TrainingProgress,
) -> DRConditionedTrainingResult:
    device = bundle.device
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    coefficient_count = feature_count(feature_version)
    current = _initial_coefficients(
        settings=settings,
        coefficient_count=coefficient_count,
        device=device,
        generator=generator,
    )
    current_evaluations = _evaluate_sa_dr_chains(
        config=config,
        settings=settings,
        bundle=bundle,
        baseline_dr_values=baseline_dr_values,
        dr_batch_size=dr_batch_size,
        baselines=baselines,
        coefficients=current,
        lambda_value=lambda_value,
        feature_version=feature_version,
        seed=config.seed,
    )
    current_scores = torch.tensor(
        [evaluation.score for evaluation in current_evaluations],
        device=device,
        dtype=torch.float32,
    )
    best_idx = int(torch.argmax(current_scores).item())
    best_coefficients = current[best_idx].detach().clone()
    best_evaluation = current_evaluations[best_idx]
    best_score = float(current_scores[best_idx].item())
    history: list[dict[str, float]] = []
    progress.write(
        "initial_candidates_evaluated",
        device=device,
        effective_lanes=len(baseline_dr_values) * settings.chains,
        max_batch_lanes=dr_batch_size * settings.chains,
        best_score=best_score,
        best_mean_relative_memorized_gain=(
            best_evaluation.mean_relative_memorized_gain
        ),
        best_mean_relative_efficiency_gain=(
            best_evaluation.mean_relative_efficiency_gain
        ),
    )

    for iteration in range(settings.iterations):
        temp = _temperature(settings, iteration)
        proposal = _clamp_coefficients(
            current
            + torch.randn(current.shape, device=device, generator=generator)
            * settings.proposal_scale,
            settings,
        )
        proposal_evaluations = _evaluate_sa_dr_chains(
            config=config,
            settings=settings,
            bundle=bundle,
            baseline_dr_values=baseline_dr_values,
            dr_batch_size=dr_batch_size,
            baselines=baselines,
            coefficients=proposal,
            lambda_value=lambda_value,
            feature_version=feature_version,
            seed=config.seed,
        )
        proposal_scores = torch.tensor(
            [evaluation.score for evaluation in proposal_evaluations],
            device=device,
            dtype=torch.float32,
        )
        delta = proposal_scores - current_scores
        accept_prob = torch.exp(delta / max(temp, 1e-9))
        accept = (delta >= 0) | (
            torch.rand(delta.shape, device=device, generator=generator) < accept_prob
        )
        accepted_count = int(accept.sum().item())
        if accept.any():
            current[accept] = proposal[accept]
            current_scores[accept] = proposal_scores[accept]
            for idx in torch.nonzero(accept, as_tuple=False).flatten().tolist():
                current_evaluations[int(idx)] = proposal_evaluations[int(idx)]

        iteration_best_idx = int(torch.argmax(current_scores).item())
        iteration_best_score = float(current_scores[iteration_best_idx].item())
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_coefficients = current[iteration_best_idx].detach().clone()
            best_evaluation = current_evaluations[iteration_best_idx]

        history_entry = {
            "iteration": float(iteration),
            "temperature": float(temp),
            "best_score": best_score,
            "best_mean_relative_memorized_gain": (
                best_evaluation.mean_relative_memorized_gain
            ),
            "best_mean_relative_efficiency_gain": (
                best_evaluation.mean_relative_efficiency_gain
            ),
        }
        history.append(history_entry)
        progress.write(
            "annealing_iteration",
            device=device,
            accepted_count=accepted_count,
            effective_lanes=len(baseline_dr_values) * settings.chains,
            max_batch_lanes=dr_batch_size * settings.chains,
            **history_entry,
        )

    passed = (
        best_evaluation.mean_relative_memorized_gain > 0.0
        and best_evaluation.mean_relative_efficiency_gain > 0.0
    )
    return DRConditionedTrainingResult(
        baseline_desired_retention_values=baseline_dr_values,
        baselines=baselines,
        best_coefficients=best_coefficients.detach().cpu(),
        best=best_evaluation,
        history=history,
        passed=passed,
    )


def _initial_coefficients(
    *,
    settings: SASettings,
    coefficient_count: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    current = torch.zeros(
        (settings.chains, coefficient_count),
        dtype=torch.float32,
        device=device,
    )
    if settings.chains > 1:
        current[1:] = _clamp_coefficients(
            torch.randn(
                current[1:].shape,
                device=device,
                generator=generator,
            )
            * settings.proposal_scale,
            settings,
        )
    return current


def _evaluate_sa_dr_chains(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    baseline_dr_values: tuple[float, ...],
    dr_batch_size: int,
    baselines: list[CandidateMetrics],
    coefficients: torch.Tensor,
    lambda_value: float,
    feature_version: str,
    seed: int,
) -> list[ChainEvaluation]:
    dr_count = len(baseline_dr_values)
    chains = int(coefficients.shape[0])
    template = FSRS6ADRDeltaPolicy.baseline(
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        feature_version=feature_version,
    )
    coefficient_count = template.feature_count
    if int(coefficients.shape[1]) != coefficient_count:
        raise ValueError(
            "DR-conditioned coefficients must have shape "
            f"(chains, {coefficient_count})."
        )
    metrics_by_chain: list[list[CandidateMetrics]] = [[] for _ in range(chains)]
    rel_mem_sums = [0.0 for _ in range(chains)]
    rel_eff_sums = [0.0 for _ in range(chains)]
    for chunk_dr_values, chunk_baselines in _iter_dr_chunks(
        baseline_dr_values,
        baselines,
        dr_batch_size,
    ):
        actual_count = len(chunk_dr_values)
        padded_dr_values = _pad_tuple(chunk_dr_values, dr_batch_size)
        desired_retention = torch.tensor(
            [dr for _chain in range(chains) for dr in padded_dr_values],
            device=bundle.device,
            dtype=torch.float32,
        )
        lane_coefficients = (
            coefficients[:, None, :]
            .expand(chains, dr_batch_size, coefficient_count)
            .reshape(chains * dr_batch_size, coefficient_count)
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
        for chain in range(chains):
            start = chain * dr_batch_size
            chain_metrics = metrics[start : start + actual_count]
            metrics_by_chain[chain].extend(chain_metrics)
            for metric, baseline in zip(
                chain_metrics,
                chunk_baselines,
                strict=True,
            ):
                rel_mem_sums[chain] += _relative_gain(
                    metric.memorized_average,
                    baseline.memorized_average,
                )
                rel_eff_sums[chain] += _relative_gain(
                    metric.memorized_per_minute,
                    baseline.memorized_per_minute,
                )
    evaluations: list[ChainEvaluation] = []
    for chain in range(chains):
        chain_metrics = metrics_by_chain[chain]
        if len(chain_metrics) != dr_count:
            raise AssertionError("DR-conditioned evaluation produced missing metrics.")
        mean_rel_mem = rel_mem_sums[chain] / max(dr_count, 1)
        mean_rel_eff = rel_eff_sums[chain] / max(dr_count, 1)
        evaluations.append(
            ChainEvaluation(
                metrics_by_dr=chain_metrics,
                mean_relative_memorized_gain=mean_rel_mem,
                mean_relative_efficiency_gain=mean_rel_eff,
                score=_score_from_relative_gains(
                    mean_rel_mem,
                    mean_rel_eff,
                    lambda_value,
                ),
            )
        )
    return evaluations


def _write_artifact(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    config_path: Path,
    settings: SASettings,
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
        per_dr.append(
            {
                "baseline_desired_retention": dr,
                "baseline": asdict(baseline),
                "best": asdict(best),
                "relative_memorized_gain": _relative_gain(
                    best.memorized_average,
                    baseline.memorized_average,
                ),
                "relative_efficiency_gain": _relative_gain(
                    best.memorized_per_minute,
                    baseline.memorized_per_minute,
                ),
            }
        )
    metrics = {
        "passed_overfit_gate": result.passed,
        "gate": {
            "mean_memorized_average_gt_baseline": (
                result.best.mean_relative_memorized_gain > 0.0
            ),
            "mean_memorized_per_minute_gt_baseline": (
                result.best.mean_relative_efficiency_gain > 0.0
            ),
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


if __name__ == "__main__":
    raise SystemExit(main())

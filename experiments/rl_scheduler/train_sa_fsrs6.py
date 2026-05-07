from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batched_sweep.behavior_cost import build_behavior_cost, load_usage
from simulator.batched_sweep.weights import (
    build_default_fsrs6_weights,
    load_fsrs6_weights,
    resolve_lstm_paths,
)
from simulator.benchmark_loader import parse_result_overrides, resolve_benchmark_root
from simulator.button_usage import DEFAULT_BUTTON_USAGE_PATH
from simulator.defaults import DEFAULT_LEARN_LIMIT, DEFAULT_SHORT_TERM_LOOPS_LIMIT
from simulator.experiment_infra.schemas import ExperimentConfig, SCHEMA_VERSION
from simulator.math.fsrs import Bounds
from simulator.models.fsrs import FSRS6BatchEnvOps
from simulator.models.lstm_batch import LSTMBatchedEnvOps, PackedLSTMWeights
from simulator.sa_fsrs6_policy import FEATURE_VERSION, SAFSRS6Policy
from simulator.schedulers.fsrs import FSRS6BatchSchedulerOps
from simulator.schedulers.sa_fsrs6 import SAFSRS6BatchSchedulerOps
from simulator.short_term_config import resolve_short_term_config
from simulator.vectorized.multiuser_engine import simulate_multiuser
from simulator.vectorized.multiuser_types import MultiUserBehavior, MultiUserCost


@dataclass(frozen=True, slots=True)
class SASettings:
    chains: int = 4
    iterations: int = 8
    initial_temp: float = 0.05
    final_temp: float = 0.005
    proposal_scale: float = 0.35
    coefficient_min: float = -8.0
    coefficient_max: float = 8.0
    retention_min: float = 0.70
    retention_max: float = 0.98
    baseline_desired_retention: float = 0.90
    torch_device: str = "cpu"
    short_term_threshold: float = 0.5
    short_term_loops_limit: int = DEFAULT_SHORT_TERM_LOOPS_LIMIT

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SASettings:
        return cls(
            chains=_int(raw.get("chains", cls.chains), "training.sa.chains", 1),
            iterations=_int(
                raw.get("iterations", cls.iterations),
                "training.sa.iterations",
                0,
            ),
            initial_temp=_float(
                raw.get("initial_temp", cls.initial_temp),
                "training.sa.initial_temp",
                0.0,
            ),
            final_temp=_float(
                raw.get("final_temp", cls.final_temp),
                "training.sa.final_temp",
                0.0,
            ),
            proposal_scale=_float(
                raw.get("proposal_scale", cls.proposal_scale),
                "training.sa.proposal_scale",
                0.0,
            ),
            coefficient_min=_float(
                raw.get("coefficient_min", cls.coefficient_min),
                "training.sa.coefficient_min",
            ),
            coefficient_max=_float(
                raw.get("coefficient_max", cls.coefficient_max),
                "training.sa.coefficient_max",
            ),
            retention_min=_float(
                raw.get("retention_min", cls.retention_min),
                "training.sa.retention_min",
            ),
            retention_max=_float(
                raw.get("retention_max", cls.retention_max),
                "training.sa.retention_max",
            ),
            baseline_desired_retention=_float(
                raw.get(
                    "baseline_desired_retention",
                    cls.baseline_desired_retention,
                ),
                "training.sa.baseline_desired_retention",
            ),
            torch_device=_str(
                raw.get("torch_device", cls.torch_device),
                "training.sa.torch_device",
            ),
            short_term_threshold=_float(
                raw.get("short_term_threshold", cls.short_term_threshold),
                "training.sa.short_term_threshold",
                0.0,
            ),
            short_term_loops_limit=_int(
                raw.get("short_term_loops_limit", cls.short_term_loops_limit),
                "training.sa.short_term_loops_limit",
                0,
            ),
        )

    def __post_init__(self) -> None:
        if self.coefficient_min >= self.coefficient_max:
            raise ValueError("coefficient_min must be less than coefficient_max.")
        if not (0.0 < self.retention_min < self.retention_max < 1.0):
            raise ValueError("retention bounds must satisfy 0 < min < max < 1.")
        if not (
            self.retention_min <= self.baseline_desired_retention <= self.retention_max
        ):
            raise ValueError(
                "baseline_desired_retention must be inside the retention bounds."
            )


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    memorized_average: float
    time_average: float
    memorized_per_minute: float
    total_reviews: int
    total_lapses: int
    total_cost: float


@dataclass(frozen=True, slots=True)
class SimulationBundle:
    env_ops: Any
    scheduler_weights: torch.Tensor
    behavior: MultiUserBehavior
    cost_model: MultiUserCost
    device: torch.device
    short_term_source: str | None
    learning_steps: list[float]
    relearning_steps: list[float]


class TrainingProgress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        device: torch.device | None = None,
        **payload: Any,
    ) -> None:
        record: dict[str, Any] = {
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "elapsed_seconds": time.monotonic() - self._started,
            "event": event,
            **payload,
        }
        gpu = _progress_gpu_snapshot(device)
        if gpu is not None:
            record["gpu"] = gpu
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SA FSRS-6 log-polynomial scheduler policy.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_value", type=float, required=True)
    parser.add_argument(
        "--baseline-desired-retention",
        type=float,
        default=None,
        help="Override training.sa.baseline_desired_retention for DR-grid runs.",
    )
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
    if args.baseline_desired_retention is not None:
        settings = replace(
            settings,
            baseline_desired_retention=args.baseline_desired_retention,
        )
        settings.__post_init__()
    raw_training_sa = _read_training_sa(args.config)
    progress.write(
        "config_loaded",
        settings=asdict(settings),
        simulation=config.simulation.to_dict(),
        seed=config.seed,
    )

    benchmark_root = resolve_benchmark_root(
        REPO_ROOT, args.srs_benchmark_root
    ).resolve()
    overrides = parse_result_overrides(args.benchmark_result)
    partition = args.benchmark_partition
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
        lanes=1,
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    baseline_metrics = _evaluate_fsrs6_baseline(
        config=config,
        settings=settings,
        bundle=baseline_bundle,
        seed=config.seed,
    )
    progress.write(
        "baseline_evaluated",
        device=baseline_bundle.device,
        effective_lanes=1,
        metrics=asdict(baseline_metrics),
    )

    train_bundle = _build_bundle(
        config=config,
        settings=settings,
        user_id=args.user_id,
        lanes=settings.chains,
        benchmark_root=benchmark_root,
        overrides=overrides,
        benchmark_partition=partition,
        button_usage=args.button_usage,
        device=device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )
    progress.write(
        "train_bundle_built",
        device=train_bundle.device,
        effective_lanes=settings.chains,
    )
    best_coefficients, best_metrics, history = _anneal(
        config=config,
        settings=settings,
        bundle=train_bundle,
        lambda_value=args.lambda_value,
        baseline=baseline_metrics,
        progress=progress,
    )
    progress.write(
        "annealing_completed",
        device=train_bundle.device,
        best=asdict(best_metrics),
        iterations=len(history),
    )
    rel_mem = _relative_gain(
        best_metrics.memorized_average,
        baseline_metrics.memorized_average,
    )
    rel_eff = _relative_gain(
        best_metrics.memorized_per_minute,
        baseline_metrics.memorized_per_minute,
    )
    passed = rel_mem > 0.0 and rel_eff > 0.0

    policy = SAFSRS6Policy(
        coefficients=tuple(float(v) for v in best_coefficients.tolist()),
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
        baseline_desired_retention=settings.baseline_desired_retention,
        title=(
            f"sa_fsrs6_u{args.user_id}_dr_"
            f"{settings.baseline_desired_retention:.2f}_lambda_{args.lambda_value:g}"
        ),
    )
    policy_path = output_dir / "policy.json"
    policy.write_json(policy_path)

    metrics = {
        "passed_overfit_gate": passed,
        "gate": {
            "memorized_average_gt_baseline": rel_mem > 0.0,
            "memorized_per_minute_gt_baseline": rel_eff > 0.0,
            "relative_memorized_gain": rel_mem,
            "relative_efficiency_gain": rel_eff,
        },
        "baseline": asdict(baseline_metrics),
        "best": asdict(best_metrics),
        "settings": asdict(settings),
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "scheduler-policy",
        "artifact_id": _artifact_id(
            args.user_id,
            args.lambda_value,
            settings.baseline_desired_retention,
            config.seed,
        ),
        "family": config.family,
        "scheduler_name": "sa_fsrs6",
        "environment": config.simulation.environment,
        "engine": config.simulation.engine,
        "training_user_ids": [args.user_id],
        "validation_user_ids": list(config.users.validation),
        "seed": config.seed,
        "policy_path": "policy.json",
        "feature_version": FEATURE_VERSION,
        "action_space": "sd_retention_function",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "code_commit": _git_commit(),
        "lambda_value": args.lambda_value,
        "baseline_desired_retention": settings.baseline_desired_retention,
        "config_snapshot_path": str(args.config.resolve()),
        "training_command_path": str(args.training_command_path)
        if args.training_command_path
        else None,
        "metrics_path": "metrics.json",
        "capabilities": ["event", "vectorized", "batched"],
    }
    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    progress.write(
        "artifacts_written",
        device=train_bundle.device,
        passed=passed,
        policy_path=str(policy_path),
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
    )
    return 0 if passed else 1


def _build_bundle(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    user_id: int | None = None,
    lanes: int | None = None,
    lane_user_ids: Sequence[int] | None = None,
    benchmark_root: Path,
    overrides: dict[str, str],
    benchmark_partition: str | None,
    button_usage: Path | None,
    device: torch.device,
    short_term_source: str | None,
    learning_steps: list[float],
    relearning_steps: list[float],
) -> SimulationBundle:
    if lane_user_ids is None:
        if user_id is None or lanes is None:
            raise ValueError("user_id and lanes are required without lane_user_ids.")
        user_ids = [user_id for _ in range(lanes)]
    else:
        user_ids = [int(item) for item in lane_user_ids]
        lanes = len(user_ids)
        if lanes < 1:
            raise ValueError("lane_user_ids must not be empty.")
    short_term = bool(short_term_source)
    scheduler_weights, kept_users = load_fsrs6_weights(
        repo_root=REPO_ROOT,
        user_ids=user_ids,
        benchmark_root=benchmark_root,
        benchmark_partition=benchmark_partition,
        overrides=overrides,
        short_term=short_term,
        device=device,
    )
    if scheduler_weights is None or len(kept_users) != lanes:
        raise SystemExit(f"Missing FSRS-6 scheduler weights for user {user_id}.")

    environment = config.simulation.environment
    if environment == "lstm":
        lstm_paths, kept_lstm = resolve_lstm_paths(
            user_ids, benchmark_root, short_term=short_term
        )
        if len(kept_lstm) != lanes:
            raise SystemExit(f"Missing LSTM environment weights for user {user_id}.")
        packed = PackedLSTMWeights.from_paths(
            lstm_paths,
            use_duration_feature=False,
            device=device,
            dtype=torch.float32,
        )
        env_ops = LSTMBatchedEnvOps(
            packed,
            device=packed.process_0_weight.device,
            dtype=torch.float32,
        )
    elif environment == "fsrs6":
        env_weights, kept_env = load_fsrs6_weights(
            repo_root=REPO_ROOT,
            user_ids=user_ids,
            benchmark_root=benchmark_root,
            benchmark_partition=benchmark_partition,
            overrides=overrides,
            short_term=short_term,
            device=device,
        )
        if env_weights is None or len(kept_env) != lanes:
            raise SystemExit(f"Missing FSRS-6 environment weights for user {user_id}.")
        env_ops = FSRS6BatchEnvOps(
            weights=env_weights.to(device),
            bounds=Bounds(),
            device=device,
            dtype=torch.float32,
        )
    elif environment == "fsrs6_default":
        env_weights = build_default_fsrs6_weights(user_ids=user_ids, device=device)
        env_ops = FSRS6BatchEnvOps(
            weights=env_weights,
            bounds=Bounds(),
            device=device,
            dtype=torch.float32,
        )
    else:
        raise SystemExit(
            "SA FSRS-6 trainer supports lstm, fsrs6, and fsrs6_default environments."
        )

    (
        learn_costs,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        learning_rating_prob,
        relearning_rating_prob,
        state_rating_costs,
        review_markov_success_weights,
    ) = load_usage(user_ids, button_usage)
    behavior, cost_model = build_behavior_cost(
        lanes,
        deck_size=config.simulation.deck,
        learn_limit=config.simulation.learn_limit or DEFAULT_LEARN_LIMIT,
        review_limit=config.simulation.review_limit,
        cost_limit_minutes=config.simulation.cost_limit_minutes,
        learn_costs=learn_costs.to(env_ops.device),
        review_costs=review_costs.to(env_ops.device),
        first_rating_prob=first_rating_prob.to(env_ops.device),
        review_rating_prob=review_rating_prob.to(env_ops.device),
        learning_rating_prob=learning_rating_prob.to(env_ops.device),
        relearning_rating_prob=relearning_rating_prob.to(env_ops.device),
        state_rating_costs=state_rating_costs.to(env_ops.device),
        review_markov_success_weights=review_markov_success_weights.to(env_ops.device),
        short_term=short_term,
    )
    return SimulationBundle(
        env_ops=env_ops,
        scheduler_weights=scheduler_weights.to(env_ops.device),
        behavior=behavior,
        cost_model=cost_model,
        device=env_ops.device,
        short_term_source=short_term_source,
        learning_steps=learning_steps,
        relearning_steps=relearning_steps,
    )


def _evaluate_fsrs6_baseline(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    seed: int,
) -> CandidateMetrics:
    sched_ops = FSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        desired_retention=settings.baseline_desired_retention,
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
    return _metrics_from_stats(stats[0])


def _anneal(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    lambda_value: float,
    baseline: CandidateMetrics,
    progress: TrainingProgress,
) -> tuple[torch.Tensor, CandidateMetrics, list[dict[str, float]]]:
    device = bundle.device
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    base_policy = SAFSRS6Policy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
    )
    current = torch.tensor(
        base_policy.coefficients,
        dtype=torch.float32,
        device=device,
    ).repeat(settings.chains, 1)
    if settings.chains > 1:
        current[1:] = _clamp_coefficients(
            current[1:]
            + torch.randn(
                current[1:].shape,
                device=device,
                generator=generator,
            )
            * settings.proposal_scale,
            settings,
        )
    current_metrics = _evaluate_sa_candidates(
        config=config,
        settings=settings,
        bundle=bundle,
        coefficients=current,
        seed=config.seed,
    )
    current_scores = torch.tensor(
        [_score(metrics, baseline, lambda_value) for metrics in current_metrics],
        device=device,
        dtype=torch.float32,
    )
    best_idx = int(torch.argmax(current_scores).item())
    best_coefficients = current[best_idx].detach().clone()
    best_metrics = current_metrics[best_idx]
    best_score = float(current_scores[best_idx].item())
    history: list[dict[str, float]] = []
    progress.write(
        "initial_candidates_evaluated",
        device=device,
        effective_lanes=settings.chains,
        best_score=best_score,
        best=asdict(best_metrics),
        best_relative_memorized_gain=_relative_gain(
            best_metrics.memorized_average,
            baseline.memorized_average,
        ),
        best_relative_efficiency_gain=_relative_gain(
            best_metrics.memorized_per_minute,
            baseline.memorized_per_minute,
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
        proposal_metrics = _evaluate_sa_candidates(
            config=config,
            settings=settings,
            bundle=bundle,
            coefficients=proposal,
            seed=config.seed,
        )
        proposal_scores = torch.tensor(
            [_score(metrics, baseline, lambda_value) for metrics in proposal_metrics],
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
                current_metrics[int(idx)] = proposal_metrics[int(idx)]

        iteration_best_idx = int(torch.argmax(current_scores).item())
        iteration_best_score = float(current_scores[iteration_best_idx].item())
        if iteration_best_score > best_score:
            best_score = iteration_best_score
            best_coefficients = current[iteration_best_idx].detach().clone()
            best_metrics = current_metrics[iteration_best_idx]
        history.append(
            {
                "iteration": float(iteration),
                "temperature": float(temp),
                "best_score": best_score,
                "best_relative_memorized_gain": _relative_gain(
                    best_metrics.memorized_average,
                    baseline.memorized_average,
                ),
                "best_relative_efficiency_gain": _relative_gain(
                    best_metrics.memorized_per_minute,
                    baseline.memorized_per_minute,
                ),
            }
        )
        progress.write(
            "annealing_iteration",
            device=device,
            iteration=iteration,
            temperature=float(temp),
            accepted_count=accepted_count,
            effective_lanes=settings.chains,
            best_score=best_score,
            best_relative_memorized_gain=_relative_gain(
                best_metrics.memorized_average,
                baseline.memorized_average,
            ),
            best_relative_efficiency_gain=_relative_gain(
                best_metrics.memorized_per_minute,
                baseline.memorized_per_minute,
            ),
        )
    return best_coefficients.cpu(), best_metrics, history


def _evaluate_sa_candidates(
    *,
    config: ExperimentConfig,
    settings: SASettings,
    bundle: SimulationBundle,
    coefficients: torch.Tensor,
    seed: int,
) -> list[CandidateMetrics]:
    template = SAFSRS6Policy.baseline(
        desired_retention=settings.baseline_desired_retention,
        retention_min=settings.retention_min,
        retention_max=settings.retention_max,
    )
    sched_ops = SAFSRS6BatchSchedulerOps(
        weights=bundle.scheduler_weights,
        policy=template,
        coefficients=coefficients,
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


def _metrics_from_stats(stats: Any) -> CandidateMetrics:
    time_average = (
        sum(stats.daily_cost) / len(stats.daily_cost) / 60.0
        if stats.daily_cost
        else 0.0
    )
    memorized_average = (
        sum(stats.daily_memorized) / len(stats.daily_memorized)
        if stats.daily_memorized
        else 0.0
    )
    return CandidateMetrics(
        memorized_average=float(memorized_average),
        time_average=float(time_average),
        memorized_per_minute=float(
            memorized_average / time_average if time_average > 0.0 else 0.0
        ),
        total_reviews=int(stats.total_reviews),
        total_lapses=int(stats.total_lapses),
        total_cost=float(stats.total_cost),
    )


def _score(
    metrics: CandidateMetrics,
    baseline: CandidateMetrics,
    lambda_value: float,
) -> float:
    rel_mem = _relative_gain(metrics.memorized_average, baseline.memorized_average)
    rel_eff = _relative_gain(
        metrics.memorized_per_minute,
        baseline.memorized_per_minute,
    )
    score = (1.0 - lambda_value) * rel_mem + lambda_value * rel_eff
    if rel_mem <= 0.0 or rel_eff <= 0.0:
        score -= 10.0 + 10.0 * abs(min(rel_mem, rel_eff, 0.0))
    return float(score)


def _relative_gain(value: float, baseline: float) -> float:
    denom = max(abs(baseline), 1e-9)
    return (float(value) - float(baseline)) / denom


def _temperature(settings: SASettings, iteration: int) -> float:
    if settings.iterations <= 1:
        return settings.final_temp
    ratio = iteration / float(settings.iterations - 1)
    if settings.initial_temp <= 0.0:
        return settings.final_temp
    return settings.initial_temp * (
        (settings.final_temp / settings.initial_temp) ** ratio
    )


def _clamp_coefficients(
    coefficients: torch.Tensor, settings: SASettings
) -> torch.Tensor:
    return torch.clamp(
        coefficients,
        min=settings.coefficient_min,
        max=settings.coefficient_max,
    )


def _progress_gpu_snapshot(device: torch.device | None) -> dict[str, int] | None:
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "current_allocated_memory_bytes": int(torch.cuda.memory_allocated(device)),
        "current_reserved_memory_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _read_training_sa(config_path: Path) -> Mapping[str, Any]:
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw.get("training", {})
    if not isinstance(training, Mapping):
        return {}
    sa = training.get("sa", {})
    return sa if isinstance(sa, Mapping) else {}


def _artifact_id(
    user_id: int,
    lambda_value: float,
    baseline_desired_retention: float,
    seed: int,
) -> str:
    lambda_token = _float_token(lambda_value)
    dr_token = _float_token(baseline_desired_retention)
    return f"sa-fsrs6-user-{user_id}-dr-{dr_token}-lambda-{lambda_token}-seed-{seed}"


def _float_token(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.SubprocessError:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _int(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return value


def _float(value: Any, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field_name} must be a number.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    return result


def _str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
